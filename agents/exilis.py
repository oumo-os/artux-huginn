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
  Exilis runs in a tight poll loop (5 ms by default). It batches all
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
from ..llm.client import LLMClient
from ..llm.prompts import EXILIS_TRIAGE_v1, EXILIS_TRIAGE_USER_v1


# ---------------------------------------------------------------------------
# Triage signal
# ---------------------------------------------------------------------------

class TriageLabel:
    IGNORE = "ignore"
    ACT    = "act"
    URGENT = "urgent"


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
        Called when triage == "urgent". Orchestrator issues nudge.
    poll_interval_s : float
        Seconds between polls when no new events. Default 0.005 (5 ms).
    """

    def __init__(
        self,
        stm:             STMStore,
        htm:             HTM,
        llm:             LLMClient,
        on_act:          Callable,
        on_urgent:       Callable,
        poll_interval_s: float = 0.005,
    ):
        self.stm             = stm
        self.htm             = htm
        self.llm             = llm
        self.on_act          = on_act
        self.on_urgent       = on_urgent
        self.poll_interval_s = poll_interval_s

        self._last_processed_id: str = ""
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
        while self._running:
            try:
                self._tick()
            except Exception as e:
                # Log to STM as internal error, keep running
                try:
                    self.stm.record(
                        source="system", type="internal",
                        payload={"subtype": "exilis_error", "error": str(e)},
                    )
                except Exception:
                    pass
            time.sleep(self.poll_interval_s)

    def _tick(self):
        """
        One Exilis cycle:
          1. Get all new events since last processed
          2. If none: sleep and return
          3. Read working context (consN + window)
          4. Make one LLM triage call
          5. Dispatch signal
        """
        new_events = self.stm.get_events_after(self._last_processed_id)
        if not new_events:
            return

        # Advance cursor regardless of triage outcome
        self._last_processed_id = new_events[-1].id

        context      = self.stm.get_stm_window()
        active_tasks = self.htm.query(state="active|paused", initiated_by="sagax")

        signal = self._triage(context, active_tasks, new_events)

        if signal.label == TriageLabel.URGENT:
            self.on_urgent(new_events[-1])
        elif signal.label == TriageLabel.ACT:
            self.on_act()
        # ignore → nothing

    # ------------------------------------------------------------------
    # LLM triage call
    # ------------------------------------------------------------------

    def _triage(
        self,
        context:      dict,
        active_tasks: list,
        new_events:   list,
    ) -> TriageSignal:
        """
        One LLM call. Returns a TriageSignal.
        Falls back to TriageLabel.ACT on any failure.
        """
        user_prompt = EXILIS_TRIAGE_USER_v1.format(
            cons_n_text   = context.get("cons_n_text") or "No prior context (cold start).",
            active_tasks  = _format_tasks(active_tasks),
            new_events    = _format_events(new_events),
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
            if label not in (TriageLabel.IGNORE, TriageLabel.ACT, TriageLabel.URGENT):
                label = TriageLabel.ACT
            return TriageSignal(label=label, reason=reason)

        except Exception as e:
            # LLM failure → safe default: act (Sagax wakes, reasons properly)
            return TriageSignal(
                label=TriageLabel.ACT,
                reason=f"triage_llm_error: {e}",
            )

    # ------------------------------------------------------------------
    # Called directly by Orchestrator (push notification path)
    # ------------------------------------------------------------------

    def on_new_event(self) -> Optional[TriageSignal]:
        """
        Synchronous single-event triage for cases where the Orchestrator
        wants an immediate answer (e.g. tool result arriving mid-Sagax).
        Used when async polling is replaced by push notification.
        """
        new_events = self.stm.get_events_after(self._last_processed_id)
        if not new_events:
            return None
        self._last_processed_id = new_events[-1].id
        context      = self.stm.get_stm_window()
        active_tasks = self.htm.query(state="active|paused", initiated_by="sagax")
        return self._triage(context, active_tasks, new_events)


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
