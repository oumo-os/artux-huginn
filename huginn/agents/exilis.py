"""
agents/exilis.py — Exilis: the attention gate.

What Exilis IS (CognitiveModule_Addendum.md §3):
  A passive triage agent. It wakes when the Orchestrator notifies it
  that a new event has been written to STM. It reads the working context
  (consN + new-event window), makes one small LLM call, emits one of
  three signals, and sleeps.

What Exilis IS NOT:
  - Not an ingestion agent
  - Does not own any I/O
  - Does not write to STM
  - Does not resolve signatures (that's the Perception Manager)
  - Does not contain hardcoded rules, regex, or classifiers

The entire agent is one LLM call inside a notification handler.

Loop behaviour:
  Exilis runs in a tight continuous loop. Each iteration checks
  stm.events_pending(last_event_id) — a single EXISTS query with no
  row loading. If nothing new, the loop yields (os.sched_yield or
  time.sleep(0)) and continues immediately. When events are pending,
  it batches them all into one triage call. This means:
    - No fixed polling interval — inference fires exactly when needed
    - Detection latency = one EXISTS query (< 1 ms) not the poll interval
    - No wasted inference when the environment is quiet
    - All
  available new events into a single LLM call per cycle — one call
  covers however many events arrived since the last check.

Shared context:
  Exilis reads the same consN that Sagax writes. Same model, same
  priors. This is intentional: Exilis should triage as Sagax would.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Optional

from ..runtime.stm import STMStore
from ..runtime.htm import HTM
from ..runtime.denoise import DenoiseEngine
from ..llm.client import LLMClient
from ..llm.prompts import EXILIS_TRIAGE_v1, EXILIS_TRIAGE_USER_v1


# ---------------------------------------------------------------------------
# Triage signal
# ---------------------------------------------------------------------------

class TriageLabel:
    HOLD      = "hold"
    ACT       = "act"
    INTERRUPT = "interrupt"


class TriageSignal:
    def __init__(self, label: str, reason: str = ""):
        self.label  = label
        self.reason = reason

    def __repr__(self):
        return f"TriageSignal({self.label!r}: {self.reason!r})"


# ---------------------------------------------------------------------------
# Exilis
# ---------------------------------------------------------------------------

class Exilis:
    """
    Attention gate. Poll loop + one LLM triage call per batch.

    Parameters
    ----------
    stm : STMStore
    htm : HTM
    llm : LLMClient
        Small/fast model (Qwen2.5-0.5B, Llama-3.2-1B, or Claude Haiku).
    on_act : callable
        Called when triage == "act". Orchestrator queues Sagax wake.
    on_urgent : callable(event)
        Called when triage == "interrupt". Orchestrator issues nudge.
    idle_yield_s : float
        Seconds between polls when no new events. Default 0.005 (5 ms).
    """

    def __init__(
        self,
        stm:             STMStore,
        htm:             HTM,
        llm:             LLMClient,
        on_act:          Callable,
        on_urgent:       Callable,
        idle_yield_s: float = 0.0,
        # idle_yield_s: seconds to yield when no events are pending.
        # 0.0 = os.sched_yield (cooperative multitasking, near-zero latency).
        # Set to 0.001 on systems where busy-waiting causes thermal issues.
    ):
        self.stm             = stm
        self.htm             = htm
        self.llm             = llm
        self.on_act          = on_act
        self.on_urgent       = on_urgent
        self.idle_yield_s = idle_yield_s
        self.denoise_engine = DenoiseEngine()

        self._last_processed_id: str = ""
        self._cursor_lock = threading.Lock()   # guards _last_processed_id
        self._running = False
        self._thread:  Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the poll loop in a background thread."""
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="ExilisThread"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    def _loop(self):
        """
        Tight continuous loop.

        Each iteration:
          1. events_pending() — one EXISTS query, < 1 ms, no rows loaded
          2. If nothing new: yield and continue (no inference)
          3. If new events: claim them atomically, run one triage LLM call

        This replaces the fixed-interval sleep. Detection latency is
        now bounded by one EXISTS query instead of the poll interval.
        """
        import os as _os
        while self._running:
            try:
                # Fast gate: is there anything to process?
                if not self.stm.events_pending(self._last_processed_id):
                    # Nothing new — yield to other threads without sleeping
                    if self.idle_yield_s > 0:
                        time.sleep(self.idle_yield_s)
                    else:
                        try:
                            _os.sched_yield()
                        except AttributeError:
                            time.sleep(0)   # Windows fallback
                    continue

                self._tick()

            except Exception as e:
                try:
                    self.stm.record(
                        source="system", type="internal",
                        payload={"subtype": "exilis_error", "error": str(e)},
                    )
                except Exception:
                    pass
                # Brief pause after error to avoid tight error loops
                time.sleep(0.01)

    def _claim_events(self) -> list:
        """
        Atomically claim the unprocessed event batch and advance the cursor.

        Both the poll loop (_tick) and the push path (on_new_event) run on
        different threads. Reading and advancing the shared cursor happens
        under one lock, so every batch is claimed exactly once — no missed
        wake, no double triage.
        """
        with self._cursor_lock:
            events = self.stm.get_events_after(self._last_processed_id)
            if events:
                # Advance cursor regardless of triage outcome
                self._last_processed_id = events[-1].id
        return events

    def _tick(self):
        """
        One Exilis cycle:
          1. Claim all new events since last processed
          2. If none: sleep and return
          3. Read working context (consN + window)
          4. Make one LLM triage call
          5. Dispatch signal
        """
        new_events = self._claim_events()
        if not new_events:
            return

        context      = self.stm.get_stm_window()
        active_tasks = self.htm.query(state="active|paused", initiated_by="sagax")

        # Derive Sagax foreground state from its session task.
        # 'active'  — Sagax is mid-stream right now → raise interrupt bar
        # 'paused'  — interrupted mid-cycle → act is fine, interrupt only if critical
        # 'waiting' — between cycles → normal triage thresholds apply
        sagax_state = "waiting"
        for t in active_tasks:
            if "sagax_session" in (t.tags or []):
                sagax_state = t.state
                break

        # Fast pre-LLM checks, both pure Python, both microseconds:
        #   reflex_match — does new speech match an active skill reflex pattern?
        #   self_addressed — does new speech mention a self name claim?
        # These are fed to the triage LLM as weight signals, not hard decisions.
        reflex_match   = self._check_reflex(new_events)
        self_addressed = self._check_self_addressed(new_events)

        # Denoise new events for context (preserves frequency info)
        cons_n = self.stm.get_cons_n()
        last_ctx_id = cons_n.last_event_id if cons_n else ""
        denoised_events = self.denoise_engine.denoise(new_events, last_ctx_id)

        signal = self._triage(context, active_tasks, denoised_events,
                              sagax_state, self_addressed, reflex_match)
        self._dispatch_signal(signal, new_events[-1])
        # hold → nothing

    # ------------------------------------------------------------------
    # LLM triage call
    # ------------------------------------------------------------------

    def _triage(
        self,
        context:       dict,
        active_tasks:  list,
        new_events:    list,
        sagax_state:   str  = "waiting",
        self_addressed: bool = False,
        reflex_match:   str  = "",
    ) -> TriageSignal:
        """
        One LLM call. Returns a TriageSignal.
        Falls back to TriageLabel.ACT on any failure.

        Exilis sees everything Sagax sees (full context), but must be fast
        and always in the present. It focuses triage on new events while
        using the full context for awareness.
        """
        # Build context with all layers (fading + active + new events)
        cons_n_text = context.get("cons_n_text") or ""

        user_prompt = EXILIS_TRIAGE_USER_v1.format(
            cons_n_text    = cons_n_text or "No prior context (cold start).",
            active_tasks   = _format_tasks(active_tasks),
            new_events     = _format_events(new_events),
            sagax_state    = sagax_state,
            self_addressed = "yes — name claim matched in speech" if self_addressed else "no",
            reflex_match   = f"yes — {reflex_match}" if reflex_match else "no",
        )

        try:
            result = self.llm.complete_json(
                system      = EXILIS_TRIAGE_v1,
                user        = user_prompt,
                schema      = {"triage": "string", "reason": "string"},
                temperature = 0,
            )
            label  = result.get("triage", TriageLabel.ACT)
            reason = result.get("reason", "")
            if label not in (TriageLabel.HOLD, TriageLabel.ACT, TriageLabel.INTERRUPT):
                label = TriageLabel.ACT
            return TriageSignal(label=label, reason=reason)

        except Exception as e:
            # LLM failure → safe default: act (Sagax wakes, reasons properly)
            return TriageSignal(
                label=TriageLabel.ACT,
                reason=f"triage_llm_error: {e}",
            )

    # ------------------------------------------------------------------
    # Reflex pre-check — fast regex against active skill reflex patterns
    # ------------------------------------------------------------------

    def _check_reflex(self, new_events: list) -> str:
        """
        Fast pre-LLM regex check against active skill reflex patterns.
        Returns skill_id if matched, empty string otherwise.

        Not a hard classifier — just a weight signal for the triage LLM.
        """
        import re as _re
        reflex_tools = self.htm.asc.get_reflex_tools()
        if not reflex_tools:
            return ""
        for ev in new_events:
            if ev.type not in ("speech", "text"):
                continue
            text = (ev.payload.get("text") or ev.payload.get("content") or "").lower()
            if not text:
                continue
            for t in reflex_tools:
                try:
                    if t.reflex_pattern and _re.search(t.reflex_pattern, text, _re.IGNORECASE):
                        return t.reflex_skill_id or t.tool_id
                except _re.error:
                    pass
        return ""

    def _check_self_addressed(self, new_events: list) -> bool:
        """
        Scan new speech/text events for any name claim matching entity-self.

        This is a fast word-level check — not NLP, not embedding similarity.
        Checks whether any word in incoming speech matches a known self name
        claim (case-insensitive). Returns True if a match is found.

        Intentionally coarse — false positives are better than misses here.
        The Exilis LLM triage prompt uses this as a weight signal, not a
        hard decision. "Sebastian in the kitchen" still only gets act-level
        attention unless there is also an imperative in the utterance.
        """
        # Get self name claims from hot entity cache
        self_entity = self.htm.asc.hot_entities.get("entity-self")
        if not self_entity:
            return False
        claims = {
            c.lower().strip()
            for c in (self_entity.name_claims or [])
            if c.strip()
        }
        if not claims:
            return False

        # Scan speech and text events only — not sensor data
        for ev in new_events:
            if ev.type not in ("speech", "text", "output"):
                continue
            text = (
                ev.payload.get("text") or
                ev.payload.get("content") or ""
            ).lower()
            if not text:
                continue
            # Word-level check — avoid substring false positives
            # ("art" matching "AIDER" in "start the music")
            words = set(text.replace(",","").replace(".","").replace("?","").split())
            if claims.intersection(words):
                return True
        return False

    # ------------------------------------------------------------------
    # Called directly by Orchestrator (push notification path)
    # ------------------------------------------------------------------

    def on_new_event(self) -> Optional[TriageSignal]:
        """
        Synchronous single-event triage for cases where the Orchestrator
        wants an immediate answer (e.g. tool result arriving mid-Sagax).
        Used when async polling is replaced by push notification.

        Dispatches the triage result itself: every producer of this callback
        (PerceptionManager, Executor) discards the return value, so returning
        alone would never raise sagax.signal_act(). Shared cursor advancement is
        atomic with the poll loop (see _claim_events), guaranteeing exactly
        one claim per event batch.
        """
        new_events = self._claim_events()
        if not new_events:
            return None
        context      = self.stm.get_stm_window()
        active_tasks = self.htm.query(state="active|paused", initiated_by="sagax")
        sagax_state  = "waiting"
        for t in active_tasks:
            if "sagax_session" in (t.tags or []):
                sagax_state = t.state
                break
        reflex_match   = self._check_reflex(new_events)
        self_addressed = self._check_self_addressed(new_events)

        # Denoise new events for context (preserves frequency info)
        cons_n = self.stm.get_cons_n()
        last_ctx_id = cons_n.last_event_id if cons_n else ""
        denoised_events = self.denoise_engine.denoise(new_events, last_ctx_id)

        signal = self._triage(context, active_tasks, denoised_events,
                              sagax_state, self_addressed, reflex_match)
        self._dispatch_signal(signal, new_events[-1])
        return signal


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_events(events: list) -> str:
    if not events:
        return "(none)"
    lines = []
    for e in events:
        payload_text = (
            e.payload.get("text")
            or e.payload.get("description")
            or e.payload.get("event")
            or json.dumps(e.payload)[:120]
        )
        lines.append(
            f"[{e.ts}] {e.source}/{e.type}  "
            f"entity={e.payload.get('entity_id', '?')}  "
            f"conf={e.confidence:.2f}  "
            f"content={payload_text!r}"
        )
    return "\n".join(lines)


def _format_tasks(tasks: list) -> str:
    if not tasks:
        return "(none)"
    lines = []
    for t in tasks:
        lines.append(f"[{t.state}] {t.title}  resume_at={t.resume_at!r}")
    return "\n".join(lines)
