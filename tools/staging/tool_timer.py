"""
HUGINN_MANIFEST
tool_id:            tool.timer.v1
title:              Timer
capability_summary: >
  Set countdown timers and alarms. Use when the user asks to be reminded
  about something after a delay ("remind me in 10 minutes", "timer for
  pasta", "alarm at 7am"). Timers fire an STM event when they expire,
  which wakes Sagax to deliver the reminder.
polarity:           write
permission_scope:   []
inputs:
  action:       {type: string, enum: [set, cancel, list], description: "set | cancel | list"}
  duration_s:   {type: number,  default: 0,   description: "Countdown duration in seconds. Used when action=set."}
  label:        {type: string,  default: "",  description: "Human label for the timer, e.g. 'pasta timer'"}
  timer_id:     {type: string,  default: "",  description: "Timer ID to cancel. Used when action=cancel."}
outputs:
  timer_id:     {type: string,  description: "Unique ID for the new timer (action=set)"}
  label:        {type: string}
  fires_at:     {type: string,  description: "ISO timestamp when timer will fire (action=set)"}
  timers:       {type: array,   description: "List of active timers (action=list)"}
  cancelled:    {type: boolean, description: "True if the timer was found and cancelled (action=cancel)"}
  status:       {type: string}
dependencies: []
perception_capable: false
handler:            handle
END_MANIFEST

Timer tool with STM fire notification.

When a timer expires, it fires an STM event by calling the registered
fire callback. Huginn's ToolManager wires this up automatically via
register_fire_callback() after installation.

If no callback is wired (e.g. during testing), expired timers are logged
to stderr and can be retrieved via action=list (state="fired").

Wiring (done automatically by build_huginn / ToolManager):
    import tools.active.tool_timer as timer_mod
    timer_mod.register_fire_callback(
        lambda timer_id, label: stm.record(
            source="system", type="timer",
            payload={"timer_id": timer_id, "label": label, "event": "fired"},
        )
    )

Multiple callbacks can be registered (they all fire on expiry).
"""

from __future__ import annotations

import threading
import time
import uuid
import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Timer registry
# ---------------------------------------------------------------------------

@dataclass
class _Timer:
    timer_id:  str
    label:     str
    fires_at:  float          # time.monotonic() target
    fires_iso: str            # human-readable ISO string
    state:     str = "active" # active | fired | cancelled
    _handle:   Optional[Any] = field(default=None, repr=False)


_timers: dict[str, _Timer] = {}
_lock   = threading.Lock()

# Registered fire callbacks — wired by ToolManager after install
_fire_callbacks: list[Callable[[str, str], None]] = []


def register_fire_callback(fn: Callable[[str, str], None]):
    """
    Register a function to call when any timer fires.
    Signature: fn(timer_id: str, label: str)

    The ToolManager calls this automatically after install to wire in
    the STM write callback.
    """
    _fire_callbacks.append(fn)


# ---------------------------------------------------------------------------
# Internal fire handler
# ---------------------------------------------------------------------------

def _on_fire(timer_id: str):
    with _lock:
        t = _timers.get(timer_id)
        if t is None or t.state != "active":
            return
        t.state = "fired"

    label = t.label if t else timer_id
    for cb in _fire_callbacks:
        try:
            cb(timer_id, label)
        except Exception:
            pass

    if not _fire_callbacks:
        import sys
        print(f"[timer] FIRED: {timer_id!r} — {label!r}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

def handle(
    action:    str   = "list",
    duration_s: float = 0,
    label:     str   = "",
    timer_id:  str   = "",
) -> dict:
    """
    Set, cancel, or list timers.

    action=set    — create a new countdown timer
    action=cancel — cancel an existing timer by timer_id
    action=list   — return all active (and recently fired) timers
    """
    if action == "set":
        return _set(duration_s, label)
    elif action == "cancel":
        return _cancel(timer_id)
    elif action == "list":
        return _list()
    else:
        return {"status": "error", "message": f"Unknown action: {action!r}"}


def _set(duration_s: float, label: str) -> dict:
    if duration_s <= 0:
        return {"status": "error", "message": "duration_s must be > 0"}

    tid      = f"tmr_{uuid.uuid4().hex[:8]}"
    label    = label.strip() or f"{int(duration_s)}s timer"
    fires_at = time.monotonic() + duration_s
    fires_dt = datetime.datetime.now() + datetime.timedelta(seconds=duration_s)
    fires_iso = fires_dt.isoformat(timespec="seconds")

    handle_obj = threading.Timer(duration_s, _on_fire, args=(tid,))
    handle_obj.daemon = True

    t = _Timer(
        timer_id  = tid,
        label     = label,
        fires_at  = fires_at,
        fires_iso = fires_iso,
        _handle   = handle_obj,
    )

    with _lock:
        _timers[tid] = t
    handle_obj.start()

    return {
        "timer_id": tid,
        "label":    label,
        "fires_at": fires_iso,
        "status":   "set",
    }


def _cancel(timer_id: str) -> dict:
    with _lock:
        t = _timers.get(timer_id)
        if t is None:
            return {"cancelled": False, "status": "not_found",
                    "timer_id": timer_id}
        if t.state != "active":
            return {"cancelled": False, "status": t.state,
                    "timer_id": timer_id}
        if t._handle:
            t._handle.cancel()
        t.state = "cancelled"

    return {"cancelled": True, "status": "cancelled",
            "timer_id": timer_id, "label": t.label}


def _list() -> dict:
    now = time.monotonic()
    with _lock:
        timers = [
            {
                "timer_id":    t.timer_id,
                "label":       t.label,
                "fires_at":    t.fires_iso,
                "remaining_s": max(0, round(t.fires_at - now, 1)),
                "state":       t.state,
            }
            for t in _timers.values()
            if t.state in ("active", "fired")
        ]
    return {"timers": timers, "status": "ok"}
