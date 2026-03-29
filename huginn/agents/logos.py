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
            try:
                from memory_module.recall import RecallQuery
                _q = RecallQuery(topics=["startup", "procedure", "boot"], top_k=1)
            except ImportError:
                _q = self._STARTUP_PROCEDURE_KEY
            existing = self.muninn.recall(_q, top_k=1)
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
            "provider":    "tool.llm.ollama.v1",
            "model":       "qwen2.5:0.5b",
            "host":        "http://localhost:11434",
            "temperature": 0.0,
            "timeout":     30.0,
            "description": "Exilis triage agent LLM config.",
        },
        "artux.config.llm.sagax.v1": {
            "provider":    "tool.llm.ollama.v1",
            "model":       "llama3.2",
            "host":        "http://localhost:11434",
            "temperature": 0.2,
            "timeout":     60.0,
            "description": "Sagax planning agent LLM config.",
        },
        "artux.config.llm.logos.v1": {
            "provider":    "tool.llm.ollama.v1",
            "model":       "llama3.2",
            "host":        "http://localhost:11434",
            "temperature": 0.0,
            "timeout":     120.0,
            "description": "Logos consolidation agent LLM config.",
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
                try:
                    from memory_module.recall import RecallQuery as _RQ
                    _q = _RQ(topics=[topic], top_k=1)
                except ImportError:
                    _q = topic
                existing = self.muninn.recall(_q, top_k=1)
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

    # ------------------------------------------------------------------
    # Instruction artifact defaults
    # ------------------------------------------------------------------

    # Topic key → constant name in prompts.py
    # Written on first pass; never overwritten if already in LTM.
    _INSTRUCTION_TOPICS = {
        "instruction.htm_tasks":       "INSTRUCTION_HTM_TASKS_v1",
        "instruction.skill_execution": "INSTRUCTION_SKILL_EXECUTION_v1",
        "instruction.memory":          "INSTRUCTION_MEMORY_v1",
        "instruction.states":          "INSTRUCTION_STATES_v1",
        "instruction.live_tools":      "INSTRUCTION_LIVE_TOOLS_v1",
        "instruction.staging":         "INSTRUCTION_STAGING_v1",
        "instruction.entities":        "INSTRUCTION_ENTITIES_v1",
        "instruction.speech_step":     "INSTRUCTION_SPEECH_STEP_v1",
    }

    def _ensure_instruction_defaults(self):
        """
        Write Sagax instruction artifacts to Muninn LTM on first pass.

        Each artifact is stored as class_type="instruction" with topics:
            ["instruction", "instruction.<n>", "artux.instruction"]

        The content is the raw text of the corresponding INSTRUCTION_*_v1
        constant from prompts.py — so it is always in sync with the codebase.

        If an artifact already exists (operator updated it), it is left alone.
        """
        try:
            from huginn.llm.prompts import (
                INSTRUCTION_HTM_TASKS_v1,
                INSTRUCTION_SKILL_EXECUTION_v1,
                INSTRUCTION_MEMORY_v1,
                INSTRUCTION_STATES_v1,
                INSTRUCTION_LIVE_TOOLS_v1,
                INSTRUCTION_STAGING_v1,
                INSTRUCTION_ENTITIES_v1,
                INSTRUCTION_SPEECH_STEP_v1,
            )
        except ImportError as e:
            self.stm.record(
                source="system", type="internal",
                payload={"subtype": "instruction_defaults_import_error",
                         "error": str(e)},
            )
            return

        _constants = {
            "instruction.htm_tasks":       INSTRUCTION_HTM_TASKS_v1,
            "instruction.skill_execution": INSTRUCTION_SKILL_EXECUTION_v1,
            "instruction.memory":          INSTRUCTION_MEMORY_v1,
            "instruction.states":          INSTRUCTION_STATES_v1,
            "instruction.live_tools":      INSTRUCTION_LIVE_TOOLS_v1,
            "instruction.staging":         INSTRUCTION_STAGING_v1,
            "instruction.entities":        INSTRUCTION_ENTITIES_v1,
            "instruction.speech_step":     INSTRUCTION_SPEECH_STEP_v1,
        }

        written = 0
        for topic, content in _constants.items():
            # Check whether this instruction already exists in LTM
            try:
                try:
                    from memory_module.recall import RecallQuery as _RQ
                    _q = _RQ(topics=[topic, "instruction"], top_k=1)
                except ImportError:
                    _q = topic
                existing = self.muninn.recall(_q, top_k=1)
                if existing:
                    continue   # operator may have customised it — leave alone
            except Exception:
                pass   # recall not ready — will retry next pass

            try:
                self.muninn.store_ltm(
                    content    = content,
                    class_type = "instruction",
                    topics     = [topic, "instruction", "artux.instruction"],
                    confidence = 1.0,
                )
                written += 1
            except Exception as e:
                self.stm.record(
                    source="system", type="internal",
                    payload={"subtype": "instruction_write_error",
                             "topic": topic, "error": str(e)},
                )

        if written:
            self.stm.record(
                source="system", type="internal",
                payload={"subtype": "instruction_defaults_written",
                         "count": written},
            )

    # ------------------------------------------------------------------
    # Builtin default tool installation
    # ------------------------------------------------------------------

    def _ensure_default_tools(self):
        """
        Auto-install builtin default tools on first pass.

        Builtin tools (in tools/builtin/) ship with Huginn and are
        installed without staging confirmation — they are the packaged
        defaults, not operator extensions. This mirrors how instruction
        artifacts and system config are written on first boot.

        Tools installed:
          tool.llm.llamacpp.v1   — in-process GGUF inference
          tool.tts.kokoro.v1     — Kokoro ONNX TTS daemon
          tool.asr.moonshine.v1  — Moonshine ONNX ASR daemon
          tool.ui.text.v1        — terminal text I/O

        Each tool is only installed if:
          1. Not already registered in ToolManager
          2. Its source file exists in tools/builtin/
          3. Its dependencies are importable (skipped gracefully if not)
        """
        import os, pathlib
        from huginn.runtime.tool_discovery import parse_manifest

        builtin_dir = pathlib.Path(__file__).parent.parent.parent / "tools" / "builtin"
        if not builtin_dir.exists():
            return

        for tool_file in sorted(builtin_dir.glob("*.py")):
            if tool_file.name.startswith("_"):
                continue
            try:
                source = tool_file.read_text()
                manifest = parse_manifest(source, str(tool_file))
                if manifest is None:
                    continue
                tool_id = manifest.tool_id

                # Skip if already installed
                if self.tool_manager.get_descriptor(tool_id) is not None:
                    continue

                # Skip if dependencies are missing
                if not self._deps_available(manifest.dependencies):
                    continue

                self.tool_manager.install_tool(
                    manifest = manifest,
                    stm      = self.stm,
                    htm      = self.htm,
                )
                self.stm.record(
                    source="system", type="internal",
                    payload={
                        "subtype": "builtin_tool_installed",
                        "tool_id": tool_id,
                    },
                )
            except Exception as e:
                self.stm.record(
                    source="system", type="internal",
                    payload={
                        "subtype": "builtin_tool_install_error",
                        "file":    tool_file.name,
                        "error":   str(e),
                    },
                )

    def _deps_available(self, dependencies: list) -> bool:
        """Return True if all pip dependencies are importable."""
        for dep in dependencies:
            pkg = dep.strip().split("[")[0].replace("-", "_")
            if not pkg:
                continue
            try:
                __import__(pkg)
            except ImportError:
                return False
        return True

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
            self._ensure_instruction_defaults()
            self._ensure_default_tools()

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

        # -- Step 1: segment batch into coherent arcs, write one LTM entry per arc
        try:
            ltm_ids = []
            arcs    = self._segment_and_synthesise(raw_batch)

            for arc in arcs:
                # One LTM entry per arc — never a monolithic batch dump
                entry = self.muninn.consolidate_ltm(
                    narrative   = arc["narrative"],
                    class_type  = arc.get("class_type", "observation"),
                    topics      = arc.get("topics", []),
                    concepts    = arc.get("concepts", []),
                    entities    = arc.get("entities", []),
                    confidence  = arc.get("confidence", 0.8),
                )
                ltm_ids.append(entry.id)
                counts["episodic"] += 1

                # Entity observations extracted from this arc
                for obs in arc.get("entity_observations", []):
                    try:
                        self.muninn.observe_entity(
                            entity_id   = obs["entity_id"],
                            observation = obs["observation"],
                            authority   = obs.get("authority", "peer"),
                        )
                        counts["entity_updates"] += 1
                    except Exception:
                        pass

                # Semantic assertions extracted from this arc — each gets its
                # own LTM entry so it can be recalled independently
                for assertion in arc.get("semantic_assertions", []):
                    try:
                        self.muninn.consolidate_ltm(
                            narrative   = assertion["fact"],
                            class_type  = "assertion",
                            topics      = assertion.get("topics", arc.get("topics", [])),
                            confidence  = assertion.get("confidence", 0.85),
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

        # -- Step 2: skill synthesis — observe patterns, manage evaluation tasks
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

    def _segment_and_synthesise(self, events: list[STMEvent]) -> list[dict]:
        """
        Call the large LLM to segment raw events into coherent arcs and
        produce one consolidation JSON entry per arc.

        Returns a list of entry dicts, each with:
          narrative, class_type, topics, concepts, entities, confidence,
          event_ids, entity_observations, semantic_assertions

        Falls back to a single minimal entry if parsing fails.
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
                "entries": "array",
            },
            temperature = 0,
        )

        entries = result.get("entries", [])

        # Validate: must be a non-empty list of dicts with at least a narrative
        if not isinstance(entries, list) or not entries:
            # Fallback: treat the whole result as a single entry if it has a narrative
            if isinstance(result, dict) and result.get("narrative"):
                return [result]
            # Hard fallback: minimal entry from event payloads
            fallback_text = " | ".join(
                e.payload.get("text", e.payload.get("description", ""))
                for e in events
                if e.payload.get("text") or e.payload.get("description")
            ) or f"Batch of {len(events)} events"
            return [{
                "narrative":  fallback_text,
                "class_type": "observation",
                "topics":     [],
                "concepts":   [],
                "entities":   [],
                "confidence": 0.7,
                "event_ids":  [events[0].id, events[-1].id] if events else [],
                "entity_observations": [],
                "semantic_assertions": [],
            }]

        # Ensure each entry is a dict with a narrative
        valid = []
        for e in entries:
            if isinstance(e, dict) and e.get("narrative"):
                valid.append(e)
        return valid if valid else [{
            "narrative":  f"Batch of {len(events)} events",
            "class_type": "observation",
            "topics": [], "concepts": [], "entities": [],
            "confidence": 0.7,
            "event_ids": [events[0].id, events[-1].id] if events else [],
            "entity_observations": [],
            "semantic_assertions": [],
        }]

    # ------------------------------------------------------------------
    # Skill synthesis
    # ------------------------------------------------------------------
    # Skill synthesis — observation-driven evaluation tasks
    # ------------------------------------------------------------------
    #
    # Design principle (Q4 resolution):
    #   There are no hardcoded thresholds. Logos observes execution
    #   patterns from the workbook and task notebooks, creates HTM
    #   evaluation tasks for promising candidates, accumulates evidence
    #   across sessions, and proposes skills when the evidence is strong.
    #
    #   The evaluation task IS the evidence record — its notebook
    #   accumulates Logos's observations over multiple passes. When
    #   Logos judges the evidence sufficient, it proposes the skill to
    #   the user through the staging workflow.
    #
    #   Example: Sagax goes through 5 tools to find the kitchen light
    #   switch. Logos observes the friction, creates a candidate task
    #   tagged ["synthesis_candidate", "lighting", "kitchen"], notes
    #   the pattern in the notebook. On the third similar session Logos
    #   proposes "skill.switch_kitchen_store_lights.v1" with the
    #   observation that the store lights only go off after asking the
    #   user to close all cabinets.
    # ------------------------------------------------------------------

    def _skill_synthesis_scan(self, events: list[STMEvent]) -> int:
        """
        Observe execution patterns and manage synthesis evaluation tasks.
        Returns the number of new evaluation tasks created or updated.
        """
        changed = 0
        changed += self._identify_synthesis_candidates(events)
        changed += self._advance_evaluation_tasks()
        return changed

    def _identify_synthesis_candidates(self, events: list[STMEvent]) -> int:
        """
        Scan completed persist tasks and raw events for patterns worth
        synthesising. For each novel pattern, create an evaluation task.
        For existing evaluation tasks, append new evidence.
        """
        created = 0
        completed_tasks = self.htm.query(state="completed", initiated_by="sagax")
        persist_tasks   = [t for t in completed_tasks
                           if t.persistence == "persist"
                           and not any("[logos_examined]" in n.get("entry", "")
                                       for n in t.notebook)]

        if not persist_tasks:
            return 0

        # Ask Logos LLM to identify patterns across this batch of tasks
        tasks_text = "\n\n".join(
            f"task_id: {t.task_id}\n"
            f"title: {t.title}\n"
            f"created: {t.created_at}\n"
            f"output confidence: {t.output.get('confidence', 'unknown')}\n"
            f"notebook entries: {len(t.notebook)}\n"
            + "\n".join(
                f"  - {n.get('entry', '')[:200]}"
                for n in t.notebook
            )
            for t in persist_tasks[:10]   # cap at 10 per pass
        )

        synthesis_prompt = (
            "Examine these completed Sagax tasks and identify:\n"
            "1. Repeated friction patterns (multiple tool attempts, fallbacks, corrections)\n"
            "2. Sequences that worked well and might be reusable\n"
            "3. Gaps where a skill or shortcut would reduce future friction\n\n"
            "Return a JSON object with key 'candidates': a list of objects, each with:\n"
            "  domain: string (short capability domain, e.g. 'kitchen_lighting')\n"
            "  pattern: string (what repeated pattern was observed)\n"
            "  friction: string (what made it hard, or 'none')\n"
            "  opportunity: string (what a skill could do better)\n"
            "  evidence_task_ids: list of task_ids that show this pattern\n"
            "  worth_tracking: boolean\n\n"
            "TASKS:\n" + tasks_text
        )
        try:
            observations = self.llm.complete_json(
                system      = LOGOS_SKILL_EVAL_v1,
                user        = synthesis_prompt,
                schema      = {"candidates": "array"},
                temperature = 0,
            )
        except Exception:
            # Mark tasks as examined even if LLM failed
            for t in persist_tasks[:10]:
                self.htm.note(t.task_id, "[logos_examined] LLM unavailable this pass")
            return 0

        candidates = observations.get("candidates", [])

        for c in candidates:
            if not c.get("worth_tracking", False):
                continue

            domain  = c.get("domain", "unknown").lower().replace(" ", "_")[:40]
            pattern = c.get("pattern", "")[:200]
            tag     = f"synthesis_candidate.{domain}"

            # Check if an evaluation task already exists for this domain
            existing = self.htm.query(
                tags_any=[tag], state="active|paused|waiting"
            )

            if existing:
                # Append new evidence to the existing evaluation task
                task = existing[0]
                entry = (
                    f"[evidence] New execution pattern observed: {pattern}. "
                    f"Friction: {c.get('friction', 'unknown')}. "
                    f"Opportunity: {c.get('opportunity', '')}. "
                    f"Evidence tasks: {c.get('evidence_task_ids', [])}"
                )
                self.htm.note(task.task_id, entry)
            else:
                # Create a new evaluation task for this synthesis candidate
                task_id = self.htm.create(
                    title       = f"Skill synthesis candidate: {domain}",
                    initiated_by= "logos",
                    persistence = "persist",
                    tags        = ["synthesis_candidate", tag, domain],
                    progress    = f"Pattern identified: {pattern[:100]}",
                )
                self.htm.note(task_id,
                    f"[created] Initial observation: {pattern}. "
                    f"Opportunity: {c.get('opportunity', '')}."
                )
                self.stm.record(
                    source="system", type="internal",
                    payload={
                        "subtype": "logos_synthesis_candidate",
                        "domain":  domain,
                        "pattern": pattern[:100],
                        "task_id": task_id,
                    },
                )
                created += 1

        # Mark examined tasks so they don't recur
        for t in persist_tasks[:10]:
            self.htm.note(t.task_id, "[logos_examined]")

        return created

    def _advance_evaluation_tasks(self) -> int:
        """
        For each active synthesis evaluation task, re-examine accumulated
        evidence and decide: gather more, propose to user, or close as
        insufficient.

        When evidence is strong, creates a staging-style HTM task for
        user confirmation — same flow as tool installation.
        """
        advanced = 0
        eval_tasks = self.htm.query(
            tags_any=["synthesis_candidate"], state="active|paused|waiting"
        )

        for task in eval_tasks:
            # Count evidence entries in notebook
            evidence_entries = [
                n for n in task.notebook
                if "[evidence]" in n.get("entry", "")
            ]
            # Count sessions spanned
            sessions = len({
                n.get("entry", "")[:10]  # approximate by timestamp prefix
                for n in task.notebook
            })

            # Not enough evidence yet — continue accumulating
            if len(evidence_entries) < 2:
                continue

            # Ask Logos to evaluate readiness
            notebook_text = "\n".join(
                n.get("entry", "") for n in task.notebook[-20:]
            )
            try:
                eval_prompt = (
                    "Evaluate this synthesis candidate:\n"
                    f"Domain: {task.title}\n"
                    f"Evidence entries: {len(evidence_entries)}\n"
                    f"Notebook:\n{notebook_text}\n\n"
                    "Return JSON with:\n"
                    "  decision: 'propose' | 'gather_more' | 'insufficient'\n"
                    "  confidence: 0.0-1.0\n"
                    "  skill_title: proposed skill name (if propose)\n"
                    "  capability_summary: one sentence (if propose)\n"
                    "  guidance_steps: list of step strings (if propose)\n"
                    "  reason: why this decision"
                )
                verdict = self.llm.complete_json(
                    system      = LOGOS_SKILL_EVAL_v1,
                    user        = eval_prompt,
                    schema      = {"decision": "string", "confidence": "number"},
                    temperature = 0,
                )
            except Exception:
                continue

            decision = verdict.get("decision", "gather_more")

            if decision == "propose":
                self._propose_skill(task, verdict)
                self.htm.update(task.task_id, state="paused",
                                progress="Proposed to user — awaiting confirmation")
                advanced += 1

            elif decision == "insufficient":
                self.htm.complete(task.task_id,
                    output={"result": "insufficient_evidence",
                            "reason": verdict.get("reason", "")},
                    confidence=0.0,
                    note="[logos] Closed: insufficient evidence for synthesis",
                )
                advanced += 1

            # gather_more → leave task active, append note
            else:
                self.htm.note(task.task_id,
                    f"[logos_eval] Gather more: {verdict.get('reason', '')}"
                )

        return advanced

    def _propose_skill(self, eval_task, verdict: dict) -> None:
        """
        Write a skill proposal to LTM and create a confirmation HTM task
        (same staging workflow as tool installation — user confirms via Sagax).
        """
        domain      = eval_task.title.replace("Skill synthesis candidate: ", "")
        skill_id    = f"skill.{domain.replace(' ','_').replace('-','_')}.v1"
        skill_title = verdict.get("skill_title", domain)
        capability  = verdict.get("capability_summary", "")
        steps       = verdict.get("guidance_steps", [])
        confidence  = float(verdict.get("confidence", 0.8))

        skill_content = json.dumps({
            "artifact_type":       "skill",
            "skill_id":            skill_id,
            "title":               skill_title,
            "capability_summary":  capability,
            "steps": [
                {"order": i+1, "guidance": s, "interactive": False}
                for i, s in enumerate(steps)
            ],
            "polarity":            "write",
            "evidence":            [eval_task.task_id],
            "confidence":          confidence,
            "version":             1,
            "synthesised_at":      _utcnow(),
            "requires_confirmation_for_n_runs": SKILL_CONFIRM_RUNS,
            "status":              "proposed",   # not yet confirmed
        })

        # Write to LTM as proposed (not active) — becomes active on confirmation
        self.muninn.store_ltm(
            content    = skill_content,
            class_type = "skill",
            topics     = ["skill", skill_id, domain, "proposed"],
            confidence = confidence,
        )

        # Create a user-facing confirmation task (same pattern as staging)
        proposal_task_id = self.htm.create(
            title       = f"Skill proposal: {skill_title}",
            initiated_by= "logos",
            persistence = "persist",
            tags        = ["skill_proposal", skill_id, domain],
            progress    = f"Proposed skill ready for review: {skill_id}",
        )
        self.htm.note(proposal_task_id,
            f"skill_id: {skill_id}\n"
            f"capability: {capability}\n"
            f"steps: {len(steps)}\n"
            f"confidence: {confidence:.2f}\n"
            f"evidence_task: {eval_task.task_id}"
        )

        self.stm.record(
            source="system", type="internal",
            payload={
                "subtype":    "logos_skill_proposed",
                "skill_id":   skill_id,
                "title":      skill_title,
                "confidence": confidence,
                "task_id":    proposal_task_id,
            },
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
        1. Final consolidation pass (consume remaining raw events)
        2. Persist any dirty HTM.states back to Muninn LTM config
        3. Flush ASC
        4. Emit session end marker
        """
        self._pass()   # consume remaining raw events
        self._persist_dirty_states()
        self.htm.asc.flush()

        self.stm.record(
            source="system", type="internal",
            payload={
                "subtype":    "session_end",
                "session_id": session_id,
                "flushed_by": "logos",
            },
        )

    def _persist_dirty_states(self):
        """
        Write dirty HTM.states values back to Muninn LTM config entries.

        Only namespaces that map to known LTM config topics are persisted.
        Unknown namespaces (e.g. tool-specific states) are written as a
        single config entry under "artux.config.{namespace}.v1".

        This is the path by which a Sagax-initiated state change (e.g.
        "update sagax.model to phi4") becomes durable across reboots.
        """
        import json

        dirty = self.htm.states.flush_dirty()
        if not dirty:
            return

        # Group by namespace prefix
        by_ns: dict[str, dict] = {}
        for key, value in dirty.items():
            ns, _, param = key.partition(".")
            if not param:
                continue   # top-level key without namespace — skip
            by_ns.setdefault(ns, {})[param] = value

        for ns, updates in by_ns.items():
            topic = f"artux.config.{ns}.v1"
            try:
                # Read existing config entry for this namespace
                try:
                    from memory_module.recall import RecallQuery as _RQ
                    _q = _RQ(topics=[topic], top_k=1)
                except ImportError:
                    _q = topic
                results = self.muninn.recall(_q, top_k=1)
                existing: dict = {}
                if results:
                    raw = getattr(getattr(results[0], "entry", results[0]), "content", "")
                    if raw:
                        try:
                            existing = json.loads(raw)
                        except Exception:
                            pass

                # Merge updates into existing config
                merged = {**existing, **updates}

                self.muninn.store_ltm(
                    content    = json.dumps(merged),
                    class_type = "config",
                    topics     = [topic, "artux.config", "system"],
                    confidence = 1.0,
                )
                self.stm.record(
                    source="system", type="internal",
                    payload={
                        "subtype":    "states_persisted",
                        "namespace":  ns,
                        "keys":       list(updates.keys()),
                    },
                )
            except Exception as e:
                self.stm.record(
                    source="system", type="internal",
                    payload={
                        "subtype":   "states_persist_error",
                        "namespace": ns,
                        "error":     str(e),
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
