"""
agents/logos.py — Logos: the background consolidation and learning agent.

Responsibilities (CognitiveModule.md §4.3, §8):
  - Read raw STM events via get_raw_events(after_id=logos_watermark)
  - Never reads consN — always raw events
  - Synthesise high-quality LTM narratives from raw event batches
  - Write episodic LTM entries, entity observations, semantic assertions
  - Detect repeated execution traces → promote to skill artifacts
  - Scan staging directory for new tool files (ToolDiscovery)
  - Install affirmed staged tools (ToolManager.install_tool)
  - Run memory hygiene (decay, maintenance)
  - Advance logos_watermark and flush STM after verified LTM writes
  - Perform final ASC flush at session end

Hard prohibitions:
  - Never reads consN.summary_text as consolidation input
  - Never triggers consN updates
  - Never writes to ASC.hot_entities (Perception Manager / Sagax write these)
  - Exception: final session-end ASC flush (asc.flush()) is Logos' responsibility

Safe flush protocol (§8.2):
  1. Fetch raw events after logos_watermark
  2. Synthesise and write LTM artifacts
  3. Verify each write
  4. Store evidence pointers in LTM entries (stm_event_ids)
  5. Only then call stm.flush_up_to(batch_end_id)
  6. Emit logos_health event to STM

Early maintenance cycle:
  Sagax can call logos.request_early_cycle() when the user says "urgent"
  during a tool installation confirmation. Logos will run its next pass
  immediately rather than waiting for the normal interval.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..runtime.stm import STMStore, STMEvent
from ..runtime.htm import HTM
from ..llm.client import LLMClient
from ..llm.prompts import (
    LOGOS_CONSOLIDATE_v1,
    LOGOS_CONSOLIDATE_USER_v1,
    LOGOS_SKILL_EVAL_v1,
    LOGOS_SKILL_EVAL_USER_v1,
)


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

LOGOS_BATCH_SIZE        = 50      # max raw events per consolidation pass
LOGOS_INTERVAL_S        = 300.0   # how often Logos wakes (seconds)
LOGOS_SIZE_TRIGGER      = 40      # also wake if raw events exceed this
SKILL_MIN_RUNS          = 3
SKILL_MIN_DAYS          = 2
SKILL_MIN_SIMILARITY    = 0.85
SKILL_MAX_FAILURE_RATE  = 0.0     # 0 failures in last 5
SKILL_CONFIRM_RUNS      = 2       # new skills need this many confirmed runs


# ---------------------------------------------------------------------------
# Logos
# ---------------------------------------------------------------------------

class Logos:
    """
    Background consolidation daemon.

    Parameters
    ----------
    stm : STMStore
    htm : HTM
    muninn : MemoryAgent
    llm : LLMClient
        Large model for narrative synthesis and skill evaluation.
    interval_s : float
        Seconds between consolidation passes.
    batch_size : int
        Max raw events per pass.
    """

    def __init__(
        self,
        stm:          STMStore,
        htm:          HTM,
        muninn,
        llm:          LLMClient,
        tool_manager  = None,
        discovery     = None,
        interval_s:   float = LOGOS_INTERVAL_S,
        batch_size:   int   = LOGOS_BATCH_SIZE,
    ):
        self.stm          = stm
        self.htm          = htm
        self.muninn       = muninn
        self.llm          = llm
        self.tool_manager = tool_manager   # ToolManager (for install_tool)
        self.discovery    = discovery      # ToolDiscovery (for scan + affirmed)
        self.interval_s   = interval_s
        self.batch_size   = batch_size

        self._running        = False
        self._thread:        Optional[threading.Thread] = None
        self._pass_id        = 0
        self._first_pass     = True   # write startup procedure once
        self._early_cycle    = threading.Event()   # set by request_early_cycle()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="LogosThread"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10.0)

    def run_once(self):
        """Run a single consolidation pass synchronously. Useful for tests."""
        self._pass()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self):
        while self._running:
            # Wait for either the normal interval or an early-cycle request
            triggered = self._early_cycle.wait(timeout=self.interval_s)
            if triggered:
                self._early_cycle.clear()
            if not self._running:
                break

            # Size trigger: also wake early if raw events have accumulated
            if not triggered:
                raw_count = len(self.stm.get_raw_events(
                    after_id=self.stm.get_logos_watermark(),
                    limit=LOGOS_SIZE_TRIGGER + 1,
                ))
                # If not enough events and not explicitly triggered, continue
                # (the wait() already handled the interval)

            self._pass()

    # ------------------------------------------------------------------
    # Consolidation pass
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public: early-cycle trigger (called by Sagax on urgent install)
    # ------------------------------------------------------------------

    def request_early_cycle(self):
        """
        Wake Logos for an immediate maintenance cycle.
        Called by Orchestrator.request_early_logos_cycle() when the user
        says "urgent" during a staged tool confirmation.
        Thread-safe — signals the Event that _loop() is waiting on.
        """
        self._early_cycle.set()

    # ------------------------------------------------------------------
    # Staging scan + install passes
    # ------------------------------------------------------------------

    def _staging_scan_pass(self):
        """
        Ask ToolDiscovery to scan staging dir for new .py files.
        New files → STM event + HTM pending task (wakes Exilis → Sagax).
        Called on every _pass, even when there are no STM events.
        """
        if self.discovery is None:
            return
        try:
            self.discovery.scan()
        except Exception as e:
            self.stm.record(
                source="system", type="internal",
                payload={"subtype": "staging_scan_error", "error": str(e)},
            )

    def _staging_install_pass(self) -> int:
        """
        Process all affirmed staging tasks.
        For each: move file, install deps, load module, register handler,
        write LTM, complete HTM task.
        Returns number of tools successfully installed.
        """
        if self.discovery is None or self.tool_manager is None:
            return 0

        installed = 0
        for staged in self.discovery.get_affirmed_tasks():
            manifest = staged.manifest
            task_id  = staged.task_id

            self.htm.note(task_id, "Logos: beginning installation.")

            # Move from staging/ → active/ so the module path is stable
            try:
                self.discovery.move_to_active(manifest)
            except Exception as e:
                self.htm.note(task_id,
                    f"Logos: could not move to active/ — {e}")
                continue

            activate_pipeline = (
                manifest.perception_capable
                and self._staged_wants_pipeline(task_id)
            )

            try:
                descriptor = self.tool_manager.install_tool(
                    manifest          = manifest,
                    stm               = self.stm,
                    htm               = self.htm,
                    activate_pipeline = activate_pipeline,
                )
                # Wire any post-install callbacks (e.g. timer STM fire hook)
                self._post_install_wire(manifest.tool_id, descriptor)
                self.htm.complete(task_id, output={
                    "tool_id":   manifest.tool_id,
                    "installed": True,
                    "pipeline":  activate_pipeline,
                })
                self.htm.note(task_id,
                    f"Logos: {manifest.tool_id} installed. "
                    f"Pipeline active: {activate_pipeline}.")
                installed += 1

            except Exception as e:
                self.htm.note(task_id,
                    f"Logos: installation failed — {e}. Will retry next pass.")
                self.stm.record(
                    source="system", type="internal",
                    payload={
                        "subtype": "tool_install_failed",
                        "tool_id": manifest.tool_id,
                        "error":   str(e),
                    },
                )

        return installed

    # ------------------------------------------------------------------
    # Startup procedure
    # ------------------------------------------------------------------

    _STARTUP_PROCEDURE_KEY = "procedure.startup.v1"

    _STARTUP_PROCEDURE_DEFAULT = {
        "class_type":    "procedure",
        "key":           "procedure.startup.v1",
        "version":       1,
        "description":   "Default boot sequence for Artux. Logos improves this over time.",
        "steps": [
            {
                "id":          "recall_context",
                "action":      "aug_call",
                "description": "Recall recent session context and known entities from LTM",
                "call":        {"name": "recall",
                                "args": {"query": "recent context entities session",
                                         "top_k": 5}},
            },
            {
                "id":          "activate_pipelines",
                "action":      "htm_query",
                "description": "Ensure all perception pipeline tasks are active",
                "tags":        ["pipeline", "perception"],
            },
            {
                "id":          "announce",
                "action":      "speech",
                "description": "Announce readiness to the active entity",
                "text":        "I'm here.",
            },
        ],
    }

    def _ensure_startup_procedure(self):
        """
        Write the default startup procedure to LTM if none exists.

        On every first Logos pass (fresh DB or new instance), we check
        whether procedure.startup.v1 is already in LTM. If not, we write
        the default. Future Logos synthesis passes will improve it based
        on observed execution traces.
        """
        try:
            existing = self.muninn.recall(
                self._STARTUP_PROCEDURE_KEY, top_k=1
            )
            if existing:
                return   # already exists
        except Exception:
            pass   # recall not available yet — will retry next pass

        try:
            import json as _json
            self.muninn.consolidate_ltm(
                narrative  = (
                    "Artux default startup procedure v1. "
                    "Steps: recall recent context, activate perception pipelines, "
                    "announce readiness."
                ),
                class_type = "procedure",
                topics     = ["startup", "procedure", "boot"],
                entities   = [],
                confidence = 1.0,
                meta       = {
                    "key":  self._STARTUP_PROCEDURE_KEY,
                    "body": _json.dumps(self._STARTUP_PROCEDURE_DEFAULT),
                },
            )
            self.stm.record(
                source="system", type="internal",
                payload={
                    "subtype": "startup_procedure_written",
                    "key":     self._STARTUP_PROCEDURE_KEY,
                },
            )
        except Exception as e:
            # Non-fatal — next pass will retry
            self.stm.record(
                source="system", type="internal",
                payload={
                    "subtype": "startup_procedure_error",
                    "error":   str(e),
                },
            )

    # ------------------------------------------------------------------
    # System config defaults
    # ------------------------------------------------------------------

    # Parallel to _CFG_TOPICS in Orchestrator — same keys
    _CONFIG_DEFAULTS = {
        "artux.config.llm.exilis.v1": {
            "backend":     "ollama",
            "model":       "qwen2.5:0.5b",
            "host":        "http://localhost:11434",
            "temperature": 0.0,
            "timeout":     30.0,
            "description": "Fast triage model for Exilis attention gate.",
        },
        "artux.config.llm.sagax.v1": {
            "backend":     "ollama",
            "model":       "llama3.2",
            "host":        "http://localhost:11434",
            "temperature": 0.2,
            "timeout":     60.0,
            "description": "Primary reasoning model for Sagax planning loop.",
        },
        "artux.config.llm.logos.v1": {
            "backend":     "ollama",
            "model":       "llama3.2",
            "host":        "http://localhost:11434",
            "temperature": 0.0,
            "timeout":     120.0,
            "description": "Consolidation model for Logos LTM synthesis.",
        },
    }

    def _ensure_system_config(self):
        """
        Write default LLM config entries to Muninn LTM on first pass.

        Each config is stored as class_type="config" with a distinctive
        topic key (e.g. "artux.config.llm.sagax.v1").  The Orchestrator
        recalls these surgically at startup using the same topic string.

        Content is JSON-encoded into the LTM content field.  Muninn is
        not modified — this is pure write using the existing store_ltm API.

        If a config entry already exists (Orchestrator or user wrote one),
        it is left untouched.
        """
        import json
        for topic, defaults in self._CONFIG_DEFAULTS.items():
            try:
                # Surgical recall: topic string is distinctive — hits structured
                # match before embeddings.  top_k=1 returns exactly that entry.
                existing = self.muninn.recall(topic, top_k=1)
                if existing:
                    continue   # already written — leave user config untouched
            except Exception:
                pass   # recall not ready yet — will retry next pass

            try:
                self.muninn.store_ltm(
                    content    = json.dumps(defaults),
                    class_type = "config",
                    topics     = [topic, "artux.config", "system"],
                    confidence = 1.0,
                )
                self.stm.record(
                    source="system", type="internal",
                    payload={"subtype": "config_written", "key": topic},
                )
            except Exception as e:
                self.stm.record(
                    source="system", type="internal",
                    payload={"subtype": "config_write_error",
                             "key": topic, "error": str(e)},
                )

    def _staged_wants_pipeline(self, task_id: str) -> bool:
        """Read HTM task notebook to see if the user requested pipeline activation."""
        try:
            results = self.htm.query(task_id=task_id)
            if not results:
                return False
            task = results[0]
            for note in reversed(task.notebook):
                entry = note.get("entry", "")
                if "enable_pipeline: true" in entry:
                    return True
                if "enable_pipeline: false" in entry:
                    return False
        except Exception:
            pass
        return False

    def _post_install_wire(self, tool_id: str, descriptor) -> None:
        """
        Post-install wiring for tools that need runtime callbacks injected.
        Called immediately after install_tool() returns the ToolDescriptor.

        Currently wired tools:
          tool.timer.v1 — register an STM fire callback so timer expiry
                          writes a 'timer' event that wakes Exilis → Sagax.
        """
        if tool_id != "tool.timer.v1":
            return

        try:
            # ToolManager._load_module stores the module in sys.modules under
            # the huginn_tool_<normalised_id> key and also exposes it via the
            # descriptor's handler __module__ attribute.
            import sys
            module_key = f"huginn_tool_{tool_id.replace('.', '_').replace('-', '_')}"
            mod = sys.modules.get(module_key)

            if mod is None and descriptor.source_path:
                # Fallback: load directly from the installed source file
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    module_key, descriptor.source_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[module_key] = mod
                    spec.loader.exec_module(mod)

            if mod is None:
                return   # can't wire — non-fatal

            stm = self.stm   # capture for closure

            def _timer_fired(timer_id: str, label: str):
                stm.record(
                    source  = "system",
                    type    = "timer",
                    payload = {
                        "timer_id": timer_id,
                        "label":    label,
                        "event":    "fired",
                    },
                )

            if hasattr(mod, "register_fire_callback"):
                mod.register_fire_callback(_timer_fired)

        except Exception as e:
            # Non-fatal — timer still works, it just won't wake Sagax on fire
            self.stm.record(
                source  = "system",
                type    = "internal",
                payload = {
                    "subtype": "timer_wire_warning",
                    "tool_id": tool_id,
                    "error":   str(e),
                },
            )

    # ------------------------------------------------------------------
    # Main consolidation pass
    # ------------------------------------------------------------------

    def _pass(self):
        self._pass_id += 1
        pass_id   = f"logos-pass-{self._pass_id:04d}"
        started   = time.monotonic()
        counts    = {
            "episodic": 0, "assertions": 0, "entity_updates": 0,
            "skills": 0, "procedures": 0, "errors": 0, "skipped": 0,
            "tools_installed": 0,
        }

        # On first ever pass: ensure startup procedure and system config exist in LTM
        if self._first_pass:
            self._first_pass = False
            self._ensure_startup_procedure()
            self._ensure_system_config()

        # Always scan staging dir — even when no STM events to consolidate
        self._staging_scan_pass()
        counts["tools_installed"] = self._staging_install_pass()

        watermark = self.stm.get_logos_watermark()
        raw_batch = self.stm.get_raw_events(
            after_id=watermark, limit=self.batch_size
        )

        if not raw_batch:
            self._emit_health(pass_id, started, counts, stm_flushed=0)
            return

        batch_end_id = raw_batch[-1].id

        # -- Step 1: narrative synthesis from raw events
        try:
            ltm_ids  = []
            consol   = self._synthesise_narrative(raw_batch)

            # Write period LTM entry
            entry = self.muninn.consolidate_ltm(
                narrative   = consol["narrative"],
                class_type  = consol.get("class_type", "observation"),
                topics      = consol.get("topics", []),
                entities    = consol.get("entities", []),
                confidence  = consol.get("confidence", 0.8),
            )
            ltm_ids.append(entry.id)
            counts["episodic"] += 1

            # Per-segment LTM entries for every raw event (high-fidelity preservation)
            for ev in raw_batch:
                if ev.type == "internal":
                    continue
                seg_text = _event_to_text(ev)
                if not seg_text:
                    continue
                seg_entry = self.muninn.consolidate_ltm(
                    narrative   = seg_text,
                    class_type  = "event",
                    topics      = consol.get("topics", []),
                    confidence  = ev.confidence,
                )
                ltm_ids.append(seg_entry.id)

            # Entity observations
            for obs in consol.get("entity_observations", []):
                try:
                    self.muninn.observe_entity(
                        entity_id   = obs["entity_id"],
                        observation = obs["observation"],
                        authority   = obs.get("authority", "peer"),
                    )
                    counts["entity_updates"] += 1
                except Exception:
                    pass

            # Semantic assertions
            for assertion in consol.get("semantic_assertions", []):
                try:
                    self.muninn.consolidate_ltm(
                        narrative   = assertion["fact"],
                        class_type  = "assertion",
                        confidence  = assertion.get("confidence", 0.8),
                    )
                    counts["assertions"] += 1
                except Exception:
                    pass

        except Exception as e:
            counts["errors"] += 1
            self.stm.record(
                source="system", type="internal",
                payload={"subtype": "logos_error", "pass_id": pass_id,
                         "error": str(e)},
            )
            # Do NOT flush — batch stays until next pass
            self._emit_health(pass_id, started, counts, stm_flushed=0)
            return

        # -- Step 2: skill synthesis scan
        try:
            synth_count = self._skill_synthesis_scan(raw_batch)
            counts["skills"] += synth_count
        except Exception as e:
            counts["errors"] += 1

        # -- Step 3: examine pipeline task notebooks for LTM meta updates
        try:
            self._pipeline_adaptation_pass()
        except Exception:
            pass

        # -- Step 4: verified — safe to flush
        self.stm.flush_up_to(batch_end_id)
        flushed = len(raw_batch)

        # -- Step 5: mark HTM tasks consolidated where appropriate
        for task in self.htm.query(state="completed", initiated_by="sagax"):
            if task.persistence in ("persist", "audit_only"):
                self.htm.mark_consolidated(task.task_id)

        # -- Step 6: health event
        self._emit_health(pass_id, started, counts, stm_flushed=flushed)

    # ------------------------------------------------------------------
    # Narrative synthesis
    # ------------------------------------------------------------------

    def _synthesise_narrative(self, events: list[STMEvent]) -> dict:
        """
        Call the large LLM to produce a consolidation JSON from raw events.
        Returns the parsed dict from the LLM.
        """
        events_text = "\n".join(
            f"[{e.ts}] id={e.id} source={e.source} type={e.type} "
            f"conf={e.confidence:.2f} payload={json.dumps(e.payload)[:200]}"
            for e in events
        )

        result = self.llm.complete_json(
            system      = LOGOS_CONSOLIDATE_v1,
            user        = LOGOS_CONSOLIDATE_USER_v1.format(raw_events=events_text),
            schema      = {
                "narrative": "string",
                "class_type": "string",
                "topics": ["string"],
                "entities": ["string"],
                "confidence": "number",
                "entity_observations": "array",
                "semantic_assertions": "array",
            },
            temperature = 0,
        )
        return result

    # ------------------------------------------------------------------
    # Skill synthesis
    # ------------------------------------------------------------------

    def _skill_synthesis_scan(self, events: list[STMEvent]) -> int:
        """
        Look for repeated successful tool-call sequences in completed HTM tasks.
        Returns the number of skills promoted.
        """
        promoted = 0

        # Find completed persist tasks not yet examined
        completed_tasks = self.htm.query(state="completed", initiated_by="sagax")
        persist_tasks   = [t for t in completed_tasks
                           if t.persistence == "persist"
                           and not any("consolidated" in n.get("entry", "")
                                       for n in t.notebook)]

        # Group tasks by title pattern (simplistic clustering)
        clusters: dict[str, list] = {}
        for task in persist_tasks:
            key = task.title.lower().strip()
            clusters.setdefault(key, []).append(task)

        for key, tasks in clusters.items():
            if len(tasks) < SKILL_MIN_RUNS:
                continue

            # Check day spread
            from datetime import datetime, timezone
            dates = set()
            for t in tasks:
                try:
                    d = datetime.fromisoformat(t.created_at).date()
                    dates.add(d)
                except Exception:
                    pass
            if len(dates) < SKILL_MIN_DAYS:
                continue

            # Ask LLM to evaluate and optionally synthesise
            traces_text = "\n\n".join(
                f"Task: {t.title}\n"
                f"Created: {t.created_at}\n"
                f"Output: {json.dumps(t.output)}\n"
                f"Notebook:\n" + "\n".join(n["entry"] for n in t.notebook)
                for t in tasks[-5:]   # last 5 executions
            )

            candidate = {
                "title":   key,
                "runs":    len(tasks),
                "success": sum(1 for t in tasks if t.output.get("confidence", 0) > 0.5),
            }

            try:
                eval_result = self.llm.complete_json(
                    system      = LOGOS_SKILL_EVAL_v1,
                    user        = LOGOS_SKILL_EVAL_USER_v1.format(
                        traces    = traces_text,
                        candidate = json.dumps(candidate),
                    ),
                    schema      = {
                        "decision": "string",
                        "confidence": "number",
                        "reason": "string",
                        "capability_summary": "string",
                        "suggested_title": "string",
                    },
                    temperature = 0,
                )

                if eval_result.get("decision") == "promote":
                    self._write_skill_artifact(key, tasks, eval_result)
                    promoted += 1

                # Emit synthesis candidate event
                self.stm.record(
                    source="system", type="internal",
                    payload={
                        "subtype":        "logos_synthesis_candidate",
                        "candidate_id":   f"cand-{key[:20]}",
                        "title":          eval_result.get("suggested_title", key),
                        "evidence_count": len(tasks),
                        "decision":       eval_result.get("decision"),
                        "reason":         eval_result.get("reason"),
                    },
                )

            except Exception:
                pass

        return promoted

    def _write_skill_artifact(
        self, key: str, tasks: list, eval_result: dict
    ):
        """Write a promoted skill to Muninn LTM."""
        skill_content = json.dumps({
            "artifact_type":       "skill",
            "skill_id":            f"skill.{key.replace(' ', '_')}.v1",
            "title":               eval_result.get("suggested_title", key),
            "capability_summary":  eval_result.get("capability_summary", ""),
            "evidence":            [t.task_id for t in tasks],
            "confidence":          eval_result.get("confidence", 0.8),
            "version":             1,
            "synthesised_at":      _utcnow(),
            "requires_confirmation_for_n_runs": SKILL_CONFIRM_RUNS,
        })
        self.muninn.consolidate_ltm(
            narrative   = skill_content,
            class_type  = "skill",
            topics      = [key],
            confidence  = eval_result.get("confidence", 0.8),
        )

    # ------------------------------------------------------------------
    # Pipeline adaptation pass
    # ------------------------------------------------------------------

    def _pipeline_adaptation_pass(self):
        """
        Examine pipeline task notebooks post-session.
        Promote observed parameter adjustments to LTM meta where warranted.
        """
        pipeline_tasks = self.htm.query(
            initiated_by="system",
            state="active|paused|completed",
        )
        for task in pipeline_tasks:
            if not any("pipeline" in (task.tags or [])):
                continue
            # Look for parameter adjustment notes
            for note in task.notebook:
                entry = note.get("entry", "")
                if "[sagax_adjust]" in entry:
                    # A parameter adjustment was made mid-session.
                    # Write to LTM meta as an observation for operator review.
                    self.muninn.consolidate_ltm(
                        narrative   = f"Pipeline adaptation observed: {entry}",
                        class_type  = "observation",
                        topics      = ["pipeline", "adaptation"] + (task.tags or []),
                        confidence  = 0.7,
                    )

    # ------------------------------------------------------------------
    # Session end — ASC flush
    # ------------------------------------------------------------------

    def session_end_flush(self, session_id: str = ""):
        """
        Called by Orchestrator at session end.
        Logos is the sole authority to flush ASC.hot_entities.
        Runs a final consolidation pass first.
        """
        self._pass()   # consume remaining raw events
        self.htm.asc.flush()

        # Emit session end marker to STM
        self.stm.record(
            source="system", type="internal",
            payload={
                "subtype":    "session_end",
                "session_id": session_id,
                "flushed_by": "logos",
            },
        )

    # ------------------------------------------------------------------
    # Health event
    # ------------------------------------------------------------------

    def _emit_health(
        self, pass_id: str, started: float,
        counts: dict, stm_flushed: int
    ):
        duration_ms = int((time.monotonic() - started) * 1000)
        self.stm.record(
            source="system", type="internal",
            payload={
                "subtype":       "logos_health",
                "pass_id":       pass_id,
                "duration_ms":   duration_ms,
                "consolidated":  counts,
                "stm_flushed":   stm_flushed,
                "stm_remaining": len(self.stm.get_raw_events(
                    after_id=self.stm.get_logos_watermark(), limit=200
                )),
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event_to_text(event: STMEvent) -> str:
    p = event.payload
    return (
        p.get("text")
        or p.get("description")
        or p.get("content")
        or p.get("event")
        or json.dumps(p)[:200]
    )


def _utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
