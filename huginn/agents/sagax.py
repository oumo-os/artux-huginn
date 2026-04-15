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
    SAGAX_PLAN_v2 as SAGAX_PLAN_FULL,   # original 764-token prompt, kept for large models
    SAGAX_PLAN_USER_v2 as SAGAX_PLAN_USER_FULL,
    CONS_N_SUMMARISE_v1,
    CONS_N_SUMMARISE_USER_v1,
)
from ..llm.sagax_prompts import (
    # Mode-specific system prompts (~150 tokens each)
    SAGAX_RESPOND_v1,       SAGAX_RESPOND_USER_v1,
    SAGAX_PLAN_v1,          SAGAX_PLAN_USER_v1,
    SAGAX_RESUME_v1,        SAGAX_RESUME_USER_v1,
    SAGAX_STAGE_v1,         SAGAX_STAGE_USER_v1,
    SAGAX_MODE_ROUTER_v1,   SAGAX_MODE_ROUTER_USER_v1,
    NARRATOR_GRAMMAR_REF_v1,
    # Compact instruction artifacts for on-demand injection
    INSTRUCTION_RECALL_COMPACT_v1,
    INSTRUCTION_TASK_COMPACT_v1,
    INSTRUCTION_SKILL_COMPACT_v1,
    INSTRUCTION_STATES_COMPACT_v1,
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

    # Keywords that signal a tool or action is likely needed.
    # Used by _select_mode() so simple questions route to RESPOND, not PLAN.
    _TOOL_SIGNAL_WORDS = frozenset([
        "set", "turn", "start", "stop", "play", "install", "schedule",
        "remind", "lights", "weather", "lock", "volume", "timer", "alarm",
        "check", "find", "search", "open", "close", "run", "show", "kettle",
        "camera", "switch", "change", "enable", "disable", "restart", "list",
    ])

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
        use_mode_routing:   bool = True,
        max_context_tokens: int  = 800,
    ):
        """
        Parameters
        ----------
        use_mode_routing : bool
            When True (default), each cycle selects an atomised role-specific
            system prompt instead of the monolithic SAGAX_PLAN_FULL.
            Recommended for all models; essential for <4B / CPU inference.
            Set to False only to compare against the original behaviour.

        max_context_tokens : int
            Soft cap on user-context tokens per cycle.
              600  for 2K models (Phi-3.5-mini, Qwen2.5-1.5B)
              800  for 3-4B models (default)
              1200 for 7-8B models with 4K context
        """
        self.stm               = stm
        self.htm               = htm
        self.muninn            = muninn
        self.llm               = llm
        self.orchestrator      = orchestrator
        self._orchestrator     = orchestrator
        self.entity_id         = entity_id
        self.permission_scope  = permission_scope or []
        self.on_narrator_token = on_narrator_token

        self.use_mode_routing   = use_mode_routing
        self.max_context_tokens = max_context_tokens

        self._wake_queue: queue.Queue = queue.Queue()
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        self._messages: list[dict] = []
        self._speech_step_var:   str   = ""
        self._speech_step_event         = None

        # Session task — Sagax's own HTM task that tracks agent state
        # across cycles. Created on first cycle, persisted for the session.
        self._session_task_id:   str   = ""
        self._current_session_id: str  = ""   # set by Orchestrator.new_session()

        # Background executor for async consN updates
        import concurrent.futures as _cf
        self._executor = _cf.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="SagaxConsN"
        )

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
        One Sagax reasoning cycle.

        When use_mode_routing=True (default):
          1. Read STM context + HTM tasks
          2. _select_mode() picks the tightest applicable role prompt
          3. _build_mode_context() assembles the matching user template
          4. Stream Narrator response using the mode-specific system prompt
          5. Update consN if window is large

        When use_mode_routing=False:
          Falls back to the original monolithic SAGAX_PLAN_FULL prompt.
          Useful for large models or for A/B comparison.
        """
        context       = self.stm.get_stm_window()
        active_tasks  = self.htm.query(initiated_by="sagax", state="active|paused|due")
        staging_tasks = self.htm.query(tags_any=["tool_staging"], state="waiting")

        # Reset per-cycle message buffer. Each cycle is a single inference
        # call with fresh context from STM + HTM. Cross-cycle history would
        # duplicate what's already in consN and add stale noise.
        # Within-cycle injections (aug_result, speech_step) still work —
        # they append to this list before the stream resumes.
        self._messages = []

        # Ensure the session task exists and mark Sagax as active
        self._ensure_session_task()
        self._update_session_task_state("active")

        mode = "plan"   # default, overridden below
        if self.use_mode_routing:
            mode = self._select_mode(context, active_tasks, staging_tasks, signal)
            system_prompt, user_prompt = self._build_mode_context(
                mode, context, active_tasks, staging_tasks
            )
        else:
            # Legacy path — original monolithic prompt
            system_prompt = SAGAX_PLAN_FULL
            user_prompt   = SAGAX_PLAN_USER_FULL.format(
                stm_context      = _format_stm_context(context),
                active_tasks     = _format_tasks(active_tasks),
                state_snapshot   = self.htm.states.summary(),
                staging_tools    = _format_staging_tasks(staging_tasks),
                entity_id        = self.entity_id or "unknown",
                permission_scope = ", ".join(self.permission_scope) or "none",
            )

        self._messages.append({"role": "user", "content": user_prompt})

        narrator_text = ""
        try:
            for chunk in self.llm.stream(
                system      = system_prompt,
                messages    = self._messages,
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

        self._messages.append({"role": "assistant", "content": narrator_text})

        # Write a notebook entry to the session task.
        # This is the per-cycle record Logos reads for intent context.
        # Python extracts what the model wrote (cycle_note) and adds
        # structural metadata — mode, event count, task state.
        self._write_session_notebook_entry(
            mode         = mode,
            narrator_text = narrator_text,
            context      = context,
            active_tasks = active_tasks,
        )
        self._update_session_task_state("waiting")

        # If the model didn't write a <cycle_note>, write a minimal fallback
        # so the consN feed always has input. Detection is a quick string check
        # on the already-streamed text — no extra LLM call.
        if "<cycle_note>" not in narrator_text:
            self._write_fallback_cycle_note(
                mode if self.use_mode_routing else "legacy"
            )

        # Update consN after every cycle using cycle notes as input.
        # Runs async so Exilis can triage the next event immediately.
        # Skipped only if the model emitted nothing worth compressing.
        if self.stm.event_count_since_cons_n() >= 2:
            self._executor.submit(self._update_cons_n_from_cycle_note)

    # ------------------------------------------------------------------
    # Mode selection — heuristic, free (no LLM call)
    # ------------------------------------------------------------------

    def _select_mode(
        self,
        context:       dict,
        active_tasks:  list,
        staging_tasks: list,
        signal:        WakeSignal,
    ) -> str:
        """
        Classify the current cycle into one of four modes.
        All logic is pure Python — zero LLM cost.

        Priority order (first match wins):
          urgent signal      → respond  (interrupt, stop mid-speech)
          staging tasks      → stage    (one pending tool confirmation)
          paused task        → resume   (known resume_at step)
          tool signals       → plan
          simple user query  → respond
          default            → plan
        """
        # 1. Urgent interrupt — respond immediately, no tools
        if getattr(signal, "priority", "normal") == "urgent":
            return "respond"

        # 2. Tool staging — highest priority non-urgent task
        if staging_tasks:
            return "stage"

        # 3. Paused task with a clear resume point
        paused = [t for t in active_tasks if t.state == "paused" and t.resume_at]
        if paused:
            return "resume"

        # 4. Analyse the last user event
        new_events = context.get("new_events", [])
        last_user = next(
            (e for e in reversed(new_events) if e.source == "user"), None
        )

        if last_user:
            user_text = last_user.payload.get("text", "").lower()
            words = set(user_text.split())
            has_tool_signal = bool(words & self._TOOL_SIGNAL_WORDS)

            # Simple question, no active tasks, no tool words → respond
            if not has_tool_signal and not active_tasks:
                return "respond"

        # 5. Active tasks needing continuation → plan
        if active_tasks:
            return "plan"

        # 6. Default
        return "plan"

    # ------------------------------------------------------------------
    # Mode context builder — assembles system + user prompt for the mode
    # ------------------------------------------------------------------

    def _build_mode_context(
        self,
        mode:          str,
        context:       dict,
        active_tasks:  list,
        staging_tasks: list,
    ) -> tuple[str, str]:
        """
        Return (system_prompt, user_prompt) for the selected mode.

        Each user prompt is kept within self.max_context_tokens by
        limiting event history depth and consN snippet length.
        """
        entity_id   = self.entity_id or "unknown"
        grants      = ", ".join(self.permission_scope) or "none"
        cons_n      = _format_cons_n_snippet(context, self.max_context_tokens // 3)
        recent_evts = _format_recent_events(context, max_events=6)
        state_snap  = _format_state_snapshot_compact(self.htm.states)

        if mode == "respond":
            user = SAGAX_RESPOND_USER_v1.format(
                cons_n_snippet   = cons_n,
                recent_events    = recent_evts,
                entity_id        = entity_id,
                permission_scope = grants,
            )
            return SAGAX_RESPOND_v1, user

        if mode == "resume":
            paused = [t for t in active_tasks if t.state == "paused" and t.resume_at]
            task   = paused[0] if paused else (active_tasks[0] if active_tasks else None)
            instr  = self._fetch_compact_instruction("task")
            user = SAGAX_RESUME_USER_v1.format(
                task_block           = _format_task_block(task),
                recent_events        = recent_evts,
                state_snapshot       = state_snap,
                entity_id            = entity_id,
                permission_scope     = grants,
                injected_instruction = instr,
            )
            return SAGAX_RESUME_v1, user

        if mode == "stage":
            tool_block = _format_staging_tool_block(staging_tasks[0])
            user = SAGAX_STAGE_USER_v1.format(
                staging_tool_block = tool_block,
                recent_events      = recent_evts,
                entity_id          = entity_id,
                permission_scope   = grants,
            )
            return SAGAX_STAGE_v1, user

        # mode == "plan" (default)
        instr = self._fetch_compact_instruction("recall")
        user = SAGAX_PLAN_USER_v1.format(
            cons_n_snippet       = cons_n,
            recent_events        = recent_evts,
            active_tasks         = _format_tasks(active_tasks),
            state_snapshot       = state_snap,
            staging_tools        = _format_staging_tasks(staging_tasks),
            entity_id            = entity_id,
            permission_scope     = grants,
            injected_instruction = instr,
        )
        return SAGAX_PLAN_v1, user

    # ------------------------------------------------------------------
    # Compact instruction pre-fetch
    # Pre-fetched by Python so the model doesn't have to ask.
    # ------------------------------------------------------------------

    def _fetch_compact_instruction(self, topic: str) -> str:
        """
        Return a compact instruction artifact for injection into the user turn.

        Tries Muninn LTM first (respects any user customisations written there).
        Falls back to the hardcoded constants in sagax_prompts.py.

        topic: "recall" | "task" | "skill" | "states"
        """
        LTM_TOPIC_MAP = {
            "recall": "instruction.compact.recall.v1",
            "task":   "instruction.compact.task.v1",
            "skill":  "instruction.compact.skill.v1",
            "states": "instruction.compact.states.v1",
        }
        FALLBACK_MAP = {
            "recall": INSTRUCTION_RECALL_COMPACT_v1,
            "task":   INSTRUCTION_TASK_COMPACT_v1,
            "skill":  INSTRUCTION_SKILL_COMPACT_v1,
            "states": INSTRUCTION_STATES_COMPACT_v1,
        }

        ltm_topic = LTM_TOPIC_MAP.get(topic)
        if ltm_topic and self.muninn is not None:
            try:
                try:
                    from memory_module.recall import RecallQuery as _RQ
                    _q = _RQ(topics=[ltm_topic], top_k=1)
                except ImportError:
                    _q = ltm_topic
                results = self.muninn.recall(_q, top_k=1)
                if results:
                    raw = getattr(results[0], "entry", results[0])
                    content = getattr(raw, "content", "")
                    if content:
                        return content
            except Exception:
                pass

        return FALLBACK_MAP.get(topic, "")

    # ------------------------------------------------------------------
    # consN update — Sagax is the sole consN author
    # ------------------------------------------------------------------

    def _update_cons_n_from_cycle_note(self):
        """
        Update consN using the cycle note Sagax just wrote as the primary
        compression input, rather than raw events.

        Why cycle notes over raw events:
          Raw events require a full LLM compression pass to extract meaning.
          A cycle note is already at the right abstraction — it's Sagax's
          own statement of intent and outcome, one level above raw events
          but below the world narrative. Folding it into consN is cheap
          (the input is already compressed) and more accurate (Sagax wrote
          it with full context of what it actually decided).

        If no cycle note exists for this cycle (model forgot, or cycle was
        too trivial), falls back to the raw-event path.
        """
        current = self.stm.get_cons_n()
        existing = current.summary_text if current else ""

        # Prefer cycle notes written since the last consN update
        cycle_notes = self._get_cycle_notes_since_last_cons_n()

        if cycle_notes:
            # Cycle notes are already compressed — just fold them in.
            # Each note is 1–3 sentences, so this is a cheap call.
            notes_text = "\n".join(
                f"[{n.ts}] {n.payload.get('text', '')}"
                for n in cycle_notes
            )
            new_events_text = notes_text
            source_label    = f"cycle_notes ({len(cycle_notes)} notes)"
        else:
            # No cycle notes — fall back to raw event compression
            window     = self.stm.get_stm_window()
            raw_events = window.get("new_events", [])
            if not raw_events:
                return
            new_events_text = "\n".join(
                f"[{e.ts}] {e.source}/{e.type}: "
                + (e.payload.get("text") or e.payload.get("description")
                   or json.dumps(e.payload)[:100])
                for e in raw_events
            )
            source_label = f"raw_events ({len(raw_events)} events)"

        user_prompt = CONS_N_SUMMARISE_USER_v1.format(
            existing_narrative = existing or "(none yet)",
            new_events         = new_events_text,
        )

        try:
            resp = self.llm.complete(
                system      = CONS_N_SUMMARISE_v1,
                user        = user_prompt,
                temperature = 0,
            )
            new_text = resp.text.strip()

            original_fn = self.stm._summarise
            self.stm._summarise = lambda _existing, _events: new_text
            new_cons = self.stm.update_cons_n(force=True)
            self.stm._summarise = original_fn

            if new_cons is not None and self._orchestrator is not None:
                try:
                    self._orchestrator.on_consn_updated(new_cons.summary_text)
                except Exception:
                    pass

            self.stm.record(
                source="system", type="internal",
                payload={"subtype": "cons_n_updated",
                         "source": source_label,
                         "new_len": len(new_text)},
            )

        except Exception as e:
            self.stm.record(
                source="system", type="internal",
                payload={"subtype": "cons_n_update_error", "error": str(e)},
            )

    def _update_cons_n(self):
        """Backward-compat alias. Delegates to the cycle-note path."""
        self._update_cons_n_from_cycle_note()

    def _get_cycle_notes_since_last_cons_n(self) -> list:
        """
        Return cycle_note STM events written since the last consN update.
        These are Sagax's own per-cycle summaries — the primary input for
        rolling narrative compression.
        """
        current     = self.stm.get_cons_n()
        watermark   = current.last_event_id if current else "0"
        all_since   = self.stm.get_events_after(watermark)
        return [
            e for e in all_since
            if e.source == "sagax" and e.type == "cycle_note"
        ]

    # ------------------------------------------------------------------
    # Session task — Sagax's self-monitoring HTM task
    # ------------------------------------------------------------------

    def _ensure_session_task(self) -> None:
        """
        Create the sagax.session HTM task if it doesn't exist yet.
        Called at the start of every cycle. Idempotent.

        The session task serves two roles:
          1. Notebook: per-cycle intent summaries for Logos
          2. State signal: 'active' while streaming, 'waiting' between
             cycles — Exilis reads this to calibrate triage threshold.
        """
        if self._session_task_id:
            return   # already created

        # Check if one already exists from a previous cycle this session
        existing = self.htm.query(tags_all=["sagax_session"], state="active|waiting|paused")
        if existing:
            self._session_task_id = existing[0].task_id
            return

        session_id = getattr(self, '_current_session_id', '')
        self._session_task_id = self.htm.create(
            title        = "sagax.session",
            initiated_by = "sagax",
            persistence  = "persist",
            tags         = ["sagax_session", "agent_state"],
            session_id   = session_id,
        )

    def _update_session_task_state(self, state: str) -> None:
        """
        Update the session task state. Non-fatal if task doesn't exist.

        States:
          active   — Sagax is streaming a Narrator response right now.
                     Exilis: raise the urgent bar, queue act for natural pause.
          waiting  — Between cycles. Safe to wake Sagax normally.
          paused   — Interrupted mid-cycle by urgent signal.
        """
        if not self._session_task_id:
            return
        try:
            self.htm.update(self._session_task_id, state=state)
        except Exception:
            pass   # non-fatal

    def invalidate_session_task(self) -> None:
        """
        Called by Orchestrator on new_session() to clear the cached task id
        so a fresh task is created for the new session.
        """
        if self._session_task_id:
            try:
                self.htm.update(self._session_task_id, state="completed",
                                note="session ended")
            except Exception:
                pass
        self._session_task_id = ""

    def _write_session_notebook_entry(
        self,
        mode:          str,
        narrator_text: str,
        context:       dict,
        active_tasks:  list,
    ) -> None:
        """
        Write a per-cycle notebook entry to the sagax.session task.

        If the model wrote a <cycle_note> block, its text is the primary
        content. Python adds structural metadata regardless.

        This is the record Logos reads for intent context — distinct from:
          - <contemplation>: in-flight reasoning, written to STM
          - HTM workbook:    technical tool/speech trace for skill gaps
          - consN:           world narrative for Exilis triage
        """
        if not self._session_task_id:
            return

        # Extract model's cycle_note if present
        import re as _re
        cycle_note_match = _re.search(
            r'<cycle_note>(.*?)</cycle_note>',
            narrator_text, _re.DOTALL
        )
        model_note = cycle_note_match.group(1).strip() if cycle_note_match else ""

        # Structural metadata Python derives
        new_events   = context.get("new_events", [])
        n_events     = len(new_events)
        open_tasks   = [t.title for t in active_tasks
                        if t.state in ("active", "paused")
                        and t.task_id != self._session_task_id]
        has_tool     = "<tool_call>" in narrator_text
        has_speech   = "<speech" in narrator_text

        parts = [f"mode={mode}  events={n_events}"]
        if has_tool:    parts.append("tool_call=yes")
        if has_speech:  parts.append("spoke=yes")
        if open_tasks:  parts.append(f"open_tasks=[{', '.join(open_tasks[:3])}]")

        entry = "  ".join(parts)
        if model_note:
            entry += f"\n{model_note}"

        try:
            self.htm.update(self._session_task_id, note=entry)
        except Exception:
            pass

    def _write_fallback_cycle_note(self, mode: str) -> None:
        """
        If the model produced output but no <cycle_note> block, write a
        minimal fallback so the consN feed always has something to work with.
        Called at the end of _cycle when no cycle_note was detected.
        """
        self.stm.record(
            source="sagax", type="cycle_note",
            payload={
                "text":    f"Cycle complete (mode={mode}). No summary provided.",
                "fallback": True,
            },
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

        # chat() is a single-turn call — use a local list, not self._messages,
        # so it doesn't pollute or depend on cross-cycle state.
        messages = [{"role": "user", "content": user_prompt}]

        resp = self.llm.complete(
            system   = SAGAX_PLAN_v1,
            messages = messages,
        )

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


def _format_cons_n_snippet(context: dict, max_chars: int = 400) -> str:
    """
    Compact consN snippet for mode-routing user prompts.
    Truncates from the start (oldest) to fit the budget.
    """
    if context.get("cold_start"):
        return "[cold start]"
    text = context.get("cons_n_text", "") or ""
    if len(text) > max_chars:
        text = "..." + text[-max_chars:]
    return text or "(none yet)"


def _format_recent_events(context: dict, max_events: int = 6) -> str:
    """
    Last N events, compact single-line format for mode-routing user prompts.
    Tighter than _format_stm_context — no full JSON payloads.
    """
    new_events = context.get("new_events", [])
    selected   = new_events[-max_events:]
    if not selected:
        return "(none)"
    lines = []
    for e in selected:
        payload_text = (
            e.payload.get("text")
            or e.payload.get("subtype")
            or e.payload.get("tool_id")
            or e.payload.get("key")
            or ""
        )
        extra = ""
        if e.payload.get("success") is False:
            extra = " [FAILED]"
        lines.append(
            f"  {e.ts[11:19]}  {e.source}/{e.type}  {payload_text!r}{extra}"
        )
    return "\n".join(lines)


def _format_state_snapshot_compact(states) -> str:
    """
    Key operational states only — model/provider pairs and session flags.
    Omits tool-specific internal states that don't affect planning.
    """
    try:
        all_states = states.list("") if hasattr(states, "list") else {}
    except Exception:
        return "(unavailable)"

    important_prefixes = ("sagax.", "exilis.", "logos.", "session.")
    lines = []
    for k, v in all_states.items():
        if any(k.startswith(p) for p in important_prefixes):
            lines.append(f"  {k}={v}")
    return "\n".join(lines) or "(empty)"


def _format_task_block(task) -> str:
    """
    Full task detail for RESUME mode — model needs everything to pick up correctly.
    """
    if task is None:
        return "(no task found)"
    lines = [
        f"  task_id:    {task.task_id}",
        f"  title:      {task.title}",
        f"  state:      {task.state}",
        f"  resume_at:  {task.resume_at!r}",
        f"  progress:   {task.progress!r}",
        f"  tags:       {task.tags}",
    ]
    if task.notebook:
        lines.append("  notebook (last 3 entries):")
        for entry in task.notebook[-3:]:
            lines.append(f"    [{entry.get('ts','?')[:19]}] {entry.get('entry','')}")
    return "\n".join(lines)


def _format_staging_tool_block(task) -> str:
    """
    Single staging task details for STAGE mode.
    """
    import json as _json
    try:
        info = _json.loads(task.progress or "{}")
    except Exception:
        info = {}

    tool_id  = info.get("tool_id",  task.title)
    scope    = ", ".join(info.get("permission_scope", [])) or "none"
    summary  = info.get("capability_summary", "")
    deps     = ", ".join(info.get("deps", [])) or "none"
    perc     = info.get("perception_capable", False)
    polarity = info.get("polarity", "?")

    lines = [
        f"  task_id:            {task.task_id}",
        f"  tool_id:            {tool_id}",
        f"  capability_summary: {summary}",
        f"  polarity:           {polarity}",
        f"  permission_scope:   {scope}",
        f"  perception_capable: {perc}",
        f"  dependencies:       {deps}",
    ]
    return "\n".join(lines)

