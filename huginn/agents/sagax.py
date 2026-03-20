"""
agents/sagax.py — Sagax: the interactive planning and reasoning agent.

Responsibilities (CognitiveModule.md §4.2, §6):
  - Read STM working context (consN + new-event window)
  - Read HTM for active/paused tasks (resume points)
  - Execute the recall-driven planning loop
  - Produce the Narrator structured token stream
  - Submit tool calls to the Orchestrator
  - Update consN when new-event window grows large
  - Write its own output events to STM (contemplation, speech)
  - Park tasks in HTM on interruption; resume from notebook

Hard prohibitions:
  - Never writes systemic LTM directly (Logos' job)
  - Never flushes STM (Logos' job)
  - Never synthesises or modifies skills
  - Never calls a tool outside a <tool_call> or <aug_call> block

Narrator grammar (consumed by Orchestrator):
  <thinking>          model scratchpad — debug log only, never stored
  <contemplation>     world reasoning — stored in STM, Logos reads this
  <speech target="id"> streamed to TTS, stored in STM
  <tool_call>         write-capable, dispatched async via permission gate
  <aug_call>          read-only, dispatched inline (Sagax pauses for result)
  <aug_result>        injected by Orchestrator — Sagax reads this inline
  <task_update>       HTM task create/update/complete
  <projection>        structured data sent to UI
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Optional

from ..runtime.stm import STMStore, STMEvent
from ..runtime.htm import HTM
from ..llm.client import LLMClient
from ..llm.prompts import (
    SAGAX_PLAN_v2 as SAGAX_PLAN_v1,
    SAGAX_PLAN_USER_v1,
    CONS_N_SUMMARISE_v1,
    CONS_N_SUMMARISE_USER_v1,
)


# ---------------------------------------------------------------------------
# Wake signal (from Orchestrator/Exilis)
# ---------------------------------------------------------------------------

@dataclass
class WakeSignal:
    priority: str = "normal"   # "normal" | "urgent"
    event:    Optional[STMEvent] = None


# ---------------------------------------------------------------------------
# Sagax
# ---------------------------------------------------------------------------

class Sagax:
    """
    Interactive planning agent. Sleeps until woken by the Orchestrator.

    Parameters
    ----------
    stm : STMStore
    htm : HTM
    muninn : MemoryAgent
    llm : LLMClient
        Medium/large model for planning and Narrator output.
    orchestrator : Orchestrator
        For submitting tool calls and receiving results.
    entity_id : str
        Confirmed session entity. Empty on cold start.
    permission_scope : list[str]
        What this session is allowed to do.
    on_narrator_token : callable(token: str)
        Called with each token from the Narrator stream.
        The Orchestrator provides this and routes tokens in real time.
    """

    MAX_PLAN_STEPS  = 20
    MAX_RECALL_ITER = 3
    STUCK_THRESHOLD = 2

    def __init__(
        self,
        stm:                STMStore,
        htm:                HTM,
        muninn,
        llm:                LLMClient,
        orchestrator,
        entity_id:          str = "",
        permission_scope:   list[str] = None,
        on_narrator_token:  Optional[Callable] = None,
    ):
        self.stm               = stm
        self.htm               = htm
        self.muninn            = muninn
        self.llm               = llm
        self.orchestrator      = orchestrator
        self.entity_id         = entity_id
        self.permission_scope  = permission_scope or []
        self.on_narrator_token = on_narrator_token

        self._wake_queue: queue.Queue = queue.Queue()
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        self._messages: list[dict] = []   # Sagax's own conversation history
        self._speech_step_var:   str   = ""    # pending speech_step variable name
        self._speech_step_event         = None  # threading.Event from Orchestrator

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="SagaxThread"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        self._wake_queue.put(None)   # unblock
        if self._thread:
            self._thread.join(timeout=5.0)

    def wake(self, signal: WakeSignal = None):
        """Queue a wake signal. Called by Orchestrator."""
        self._wake_queue.put(signal or WakeSignal())

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self):
        while self._running:
            try:
                signal = self._wake_queue.get(timeout=1.0)
                if signal is None:
                    break
                self._cycle(signal)
            except queue.Empty:
                pass
            except Exception as e:
                self.stm.record(
                    source="system", type="internal",
                    payload={"subtype": "sagax_error", "error": str(e)},
                )

    def _cycle(self, signal: WakeSignal):
        """
        One Sagax reasoning cycle:
          1. Read STM context + HTM tasks (including pending staging tasks)
          2. Decide: resume parked task, or handle new event, or idle
          3. Produce Narrator token stream
          4. Update consN if window is large
        """
        context      = self.stm.get_stm_window()
        active_tasks = self.htm.query(
            initiated_by="sagax", state="active|paused|due"
        )
        # Include pending staging tasks so Sagax knows to ask about them
        staging_tasks = self.htm.query(
            tags_any=["tool_staging"], state="waiting"
        )

        # Build user prompt
        user_prompt = SAGAX_PLAN_USER_v1.format(
            stm_context      = _format_stm_context(context),
            active_tasks     = _format_tasks(active_tasks),
            state_snapshot   = self.htm.states.summary(),
            staging_tools    = _format_staging_tasks(staging_tasks),
            entity_id        = self.entity_id or "unknown",
            permission_scope = ", ".join(self.permission_scope) or "none",
        )

        # Add to conversation history
        self._messages.append({"role": "user", "content": user_prompt})

        # Stream the Narrator response
        narrator_text = ""
        try:
            for chunk in self.llm.stream(
                system   = SAGAX_PLAN_v1,
                messages = self._messages,
                temperature = 0.2,
            ):
                narrator_text += chunk.delta
                if self.on_narrator_token:
                    self.on_narrator_token(chunk.delta)

        except Exception as e:
            self.stm.record(
                source="system", type="internal",
                payload={"subtype": "sagax_stream_error", "error": str(e)},
            )
            narrator_text = f"<contemplation>I encountered an error: {e}</contemplation>"
            if self.on_narrator_token:
                for ch in narrator_text:
                    self.on_narrator_token(ch)

        # Add assistant turn to history
        self._messages.append({"role": "assistant", "content": narrator_text})

        # Trim history to avoid unbounded growth (keep last 20 turns)
        if len(self._messages) > 40:
            self._messages = self._messages[-40:]

        # Update consN if new-event window has grown large
        if self.stm.should_update_cons_n():
            self._update_cons_n()

    # ------------------------------------------------------------------
    # consN update — Sagax is the sole consN author
    # ------------------------------------------------------------------

    def _update_cons_n(self):
        """
        Trigger a consN rolling narrative update.
        Uses the same LLM model as Sagax (shared, per spec P-3).
        """
        current = self.stm.get_cons_n()
        window  = self.stm.get_stm_window()
        new_events = window["new_events"]

        if not new_events:
            return

        # Build summarise() call
        existing = current.summary_text if current else ""
        events_text = "\n".join(
            f"[{e.ts}] {e.source}/{e.type}: "
            + (e.payload.get("text") or e.payload.get("description")
               or json.dumps(e.payload)[:100])
            for e in new_events
        )

        user_prompt = CONS_N_SUMMARISE_USER_v1.format(
            existing_narrative = existing or "(none yet)",
            new_events         = events_text,
        )

        try:
            resp = self.llm.complete(
                system      = CONS_N_SUMMARISE_v1,
                user        = user_prompt,
                temperature = 0,
            )
            # Inject the summarise_fn into STMStore for this update
            new_text = resp.text.strip()

            # Temporarily override the summarise fn to return our result
            original_fn = self.stm._summarise
            self.stm._summarise = lambda _existing, _events: new_text
            self.stm.update_cons_n(force=True)
            self.stm._summarise = original_fn

        except Exception as e:
            # Non-fatal: consN update failed, will retry next cycle
            self.stm.record(
                source="system", type="internal",
                payload={"subtype": "cons_n_update_error", "error": str(e)},
            )

    # ------------------------------------------------------------------
    # Task management helpers (used by Orchestrator when parsing
    # <task_update> blocks from the Narrator stream)
    # ------------------------------------------------------------------

    def park_current_task(self, task_id: str, resume_at: str, note: str = ""):
        """Park a task on interruption. Called by Orchestrator on nudge."""
        self.htm.update(
            task_id,
            state     = "paused",
            resume_at = resume_at,
            note      = note or "Paused: interruption received.",
        )

    def resume_task(self, task_id: str) -> Optional[str]:
        """Return the resume_at pointer for a paused task."""
        tasks = self.htm.query(task_id=task_id)
        if tasks:
            return tasks[0].resume_at
        return None

    # ------------------------------------------------------------------
    # Convenience: inject an aug_result into the stream
    # Called by Orchestrator after dispatching aug_call tools
    # ------------------------------------------------------------------

    def inject_aug_result(self, result: dict):
        """
        Append an <aug_result> to the pending message.
        Orchestrator calls this after resolving aug_call tools.
        Sagax's generation resumes after this injection.
        """
        result_text = (
            f"<aug_result>{json.dumps(result)}</aug_result>"
        )
        # Appended to last assistant message (streaming continuation)
        if self._messages and self._messages[-1]["role"] == "assistant":
            self._messages[-1]["content"] += result_text
        else:
            self._messages.append({
                "role": "assistant",
                "content": result_text,
            })

    # ------------------------------------------------------------------
    # Boot: execute the startup procedure recalled from LTM
    # ------------------------------------------------------------------

    def execute_startup_procedure(self):
        """
        Execute the startup procedure recalled from LTM.

        Called by HuginnInstance.start() after all components are running.
        Sagax recalls procedure.startup.v1 from Muninn and emits Narrator
        tokens to execute each step.

        If no procedure exists yet (fresh install, first boot before Logos
        has written it), we emit a minimal default boot sequence inline.
        """
        import json as _json

        PROCEDURE_KEY = "procedure.startup.v1"

        # Try to recall the procedure from LTM
        procedure = None
        try:
            try:
                from memory_module.recall import RecallQuery as _RQ
                _q = _RQ(topics=["startup", "procedure", "boot"], top_k=1)
            except ImportError:
                _q = PROCEDURE_KEY
            results = self.muninn.recall(_q, top_k=1)
            if results:
                meta = getattr(results[0], "meta", {}) or {}
                body = meta.get("body", "")
                if body:
                    procedure = _json.loads(body)
        except Exception:
            pass

        if procedure is None:
            # Logos hasn't written it yet (first-ever boot) — use inline default
            procedure = {
                "steps": [
                    {"id": "announce", "action": "speech",
                     "text": "I'm here."},
                ]
            }

        # Record boot start in STM
        self.stm.record(
            source="system", type="internal",
            payload={"subtype": "boot_start", "procedure": PROCEDURE_KEY},
        )

        # Execute each step
        for step in procedure.get("steps", []):
            action = step.get("action", "")
            try:
                if action == "speech":
                    text = step.get("text", "")
                    if text and self.on_narrator_token:
                        entity = self.entity_id or "user"
                        for ch in f'<speech target="{entity}">{text}</speech>':
                            self.on_narrator_token(ch)

                elif action == "aug_call":
                    call = step.get("call", {})
                    if call and self.on_narrator_token:
                        payload = _json.dumps(call)
                        for ch in f'<aug_call timeout_ms="600">{payload}</aug_call>':
                            self.on_narrator_token(ch)

                elif action == "htm_query":
                    tags = step.get("tags", [])
                    tasks = self.htm.query(tags_any=tags, state="active")
                    self.stm.record(
                        source="system", type="internal",
                        payload={
                            "subtype":    "boot_pipeline_check",
                            "tags":       tags,
                            "task_count": len(tasks),
                        },
                    )
            except Exception as e:
                self.stm.record(
                    source="system", type="internal",
                    payload={
                        "subtype": "boot_step_error",
                        "step_id": step.get("id", ""),
                        "error":   str(e),
                    },
                )

        self.stm.record(
            source="system", type="internal",
            payload={"subtype": "boot_complete", "procedure": PROCEDURE_KEY},
        )

    # ------------------------------------------------------------------
    # speech_step: pause/resume Sagax generation for user response
    # ------------------------------------------------------------------

    def pause_for_speech_step(self, var: str, step_event: "threading.Event"):
        """
        Called by Orchestrator when a <speech_step> block closes.

        Records the pending variable name and the event that will be set
        when the user's response arrives.  Sagax's _cycle() / stream loop
        must check _speech_step_event before continuing to yield tokens.
        We use a simple threading.Event stored on self; the Orchestrator's
        generation thread blocks on it via inject_speech_step_result().
        """
        import threading
        self._speech_step_var   = var
        self._speech_step_event = step_event   # the same Event the Orchestrator owns

    def inject_speech_step_result(self, var: str, value: str):
        """
        Called by Orchestrator after receive_speech_step_response().

        Appends a <speech_step_result> to the assistant message stream
        so the LLM resumes with the user's answer bound to `var`.
        The threading.Event has already been set by the Orchestrator before
        this is called, so generation can continue.
        """
        result_text = (
            f'<speech_step_result var="{var}">{value}</speech_step_result>'
        )
        if self._messages and self._messages[-1]["role"] == "assistant":
            self._messages[-1]["content"] += result_text
        else:
            self._messages.append({
                "role":    "assistant",
                "content": result_text,
            })
        # Clear local state
        self._speech_step_var   = ""
        self._speech_step_event = None

    # ------------------------------------------------------------------
    # Direct chat (for tests and minimal setups without the full loop)
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> str:
        """
        Synchronous single-turn chat. For tests and minimal setups.
        Does not use the Narrator stream grammar — returns plain text.
        """
        self.stm.record(
            source="user", type="speech",
            payload={"text": user_input},
        )

        context       = self.stm.get_stm_window()
        active_tasks  = self.htm.query(
            initiated_by="sagax", state="active|paused|due"
        )
        staging_tasks = self.htm.query(
            tags_any=["tool_staging"], state="waiting"
        )

        user_prompt = SAGAX_PLAN_USER_v1.format(
            stm_context      = _format_stm_context(context),
            active_tasks     = _format_tasks(active_tasks),
            state_snapshot   = self.htm.states.summary(),
            staging_tools    = _format_staging_tasks(staging_tasks),
            entity_id        = self.entity_id or "unknown",
            permission_scope = ", ".join(self.permission_scope) or "none",
        )

        self._messages.append({"role": "user", "content": user_prompt})

        resp = self.llm.complete(
            system   = SAGAX_PLAN_v1,
            messages = self._messages,
        )
        self._messages.append({"role": "assistant", "content": resp.text})

        self.stm.record(
            source="system", type="output",
            payload={"subtype": "speech", "text": resp.text, "status": "complete"},
        )

        return resp.text


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_stm_context(context: dict) -> str:
    parts = []
    if context.get("cold_start"):
        parts.append("[COLD START — no prior narrative]")
    else:
        parts.append(f"[CONTEXT NARRATIVE (consN v{context.get('cons_n_version', 0)})]")
        parts.append(context.get("cons_n_text", "") or "(empty)")
        parts.append("")

    new_events = context.get("new_events", [])
    if new_events:
        parts.append(f"[NEW EVENTS — {len(new_events)} since last consN update]")
        for e in new_events[-12:]:   # last 12 events in full detail
            payload_text = (
                e.payload.get("text")
                or e.payload.get("description")
                or e.payload.get("event")
                or json.dumps(e.payload)[:150]
            )
            parts.append(
                f"  {e.ts}  {e.source}/{e.type}  "
                f"entity={e.payload.get('entity_id', '?')}  "
                f"{payload_text!r}"
            )
    return "\n".join(parts)


def _format_tasks(tasks: list) -> str:
    if not tasks:
        return "(none)"
    lines = []
    for t in tasks:
        lines.append(
            f"  [{t.state}] {t.title}  "
            f"resume_at={t.resume_at!r}  "
            f"tags={t.tags}"
        )
        # Include last notebook entry if present
        if t.notebook:
            last = t.notebook[-1]
            lines.append(f"    last note: {last['entry']!r}")
    return "\n".join(lines)


def _format_staging_tasks(staging_tasks: list) -> str:
    """
    Format pending staging tasks for the SAGAX_PLAN_USER prompt.
    Each entry shows the tool name, capability summary, permissions,
    dependencies, and whether it is perception-capable — enough for Sagax
    to give the user a full picture before asking for confirmation.
    """
    if not staging_tasks:
        return "(none)"

    import json
    lines = []
    for t in staging_tasks:
        try:
            info = json.loads(t.progress or "{}")
        except Exception:
            info = {}

        tool_id    = info.get("tool_id", t.title)
        polarity   = info.get("polarity", "?")
        scope      = ", ".join(info.get("permission_scope", [])) or "none"
        deps       = info.get("deps", [])
        perc       = info.get("perception_capable", False)
        src        = info.get("source_file", "")

        lines.append(f"  task_id={t.task_id}")
        lines.append(f"    tool_id:            {tool_id}")
        lines.append(f"    polarity:           {polarity}")
        lines.append(f"    permission_scope:   {scope}")
        lines.append(f"    perception_capable: {perc}")
        if deps:
            lines.append(f"    dependencies:       {', '.join(deps)}")
        if src:
            lines.append(f"    source_file:        {src}")
        if t.notebook:
            last = t.notebook[-1]
            lines.append(f"    last note: {last['entry']!r}")

    return "\n".join(lines)

