"""
tool.ui.text.v1 — Text I/O interface for Huginn.

Provides a minimal terminal interface:
  • Input:  reads stdin line by line → writes STM speech events
  • Output: subscribes to ActuationBus speech events → prints to stdout

This is both the simplest user interface and the universal test harness.
If you can drive the system from a terminal, everything is testable without
hardware. It works alongside Moonshine ASR and Kokoro TTS — the text
output is informational when TTS is active.

HUGINN_MANIFEST
tool_id:            tool.ui.text.v1
title:              Text UI
capability_summary: >
  Minimal terminal interface. Reads user input from stdin and writes it
  to STM as speech events. Prints Artux speech output to stdout. Works
  standalone (no hardware required) or alongside ASR/TTS. Use as the
  development interface, test harness, or fallback when audio is unavailable.
polarity:           read
permission_scope:   []
mode:               service
direction:          io
subscriptions:
  - type: output
    target: speech
    complete: full
  - type: output
    target: contemplation
    complete: full
  - type: output
    target: display
    complete: full
inputs: {}
outputs: {}
states:
  show_contemplation:
    default: false
    type: boolean
    description: Print <contemplation> blocks (useful for debugging)
  prompt:
    default: "You"
    type: string
    description: Input prompt label shown to the user
  output_prefix:
    default: "Artux"
    type: string
    description: Label prepended to Artux speech output
  quiet_mode:
    default: false
    type: boolean
    description: Suppress output printing (useful when TTS handles output)
dependencies: []
END_MANIFEST

Design notes
------------
The text UI deliberately does not call huginn.sagax.chat() — it writes
directly to STM as a speech event. This keeps the path identical to ASR:
  typed text → STM speech event → Exilis triage → Sagax wake

The output subscriber reads full speech events from the ActuationBus.
When Kokoro TTS is active, the speech has already been played as audio;
the text print is then secondary/informational. When TTS is not active,
the text print is the primary output channel.

Projected/display events are also printed if they arrive, useful for
seeing what Sagax is projecting to the UI.
"""

from __future__ import annotations

import sys
import threading
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_stop_event   = threading.Event()
_stm_writer:  Optional[Callable] = None
_htm_ref      = None
_entity_id    = ""


# ---------------------------------------------------------------------------
# Service interface
# ---------------------------------------------------------------------------

def start(config: dict, _htm=None, _stm=None, _session=None) -> None:
    """
    Start the input thread. Output is handled by handle() via ActuationBus.

    _htm, _stm, _session are injected by ToolManager.
    """
    global _stop_event, _stm_writer, _htm_ref, _entity_id

    _stm_writer = _stm.record if _stm is not None else None
    _htm_ref    = _htm
    _entity_id  = getattr(_session, "entity_id", "user") if _session else "user"

    _stop_event.clear()

    prompt = (_htm.states.get("tool.ui.text.v1.prompt") if _htm else None) or "You"

    t = threading.Thread(
        target   = _input_loop,
        args     = (prompt,),
        daemon   = True,
        name     = "TextUIInput",
    )
    t.start()


def stop() -> None:
    _stop_event.set()


def handle(event: dict, _htm=None) -> None:
    """
    Receive an ActuationBus output event and print it.
    Called from the ActuationManager dispatch thread.
    """
    if _htm is not None:
        global _htm_ref
        _htm_ref = _htm

    quiet = _read_state("quiet_mode", False, _bool)
    if quiet:
        return

    target     = event.get("target", "")
    text       = event.get("text", "").strip()
    interrupted = event.get("interrupted", False)

    if not text:
        return

    if target == "speech":
        prefix = _read_state("output_prefix", "Artux", str)
        suffix = " [interrupted]" if interrupted else ""
        print(f"\n{prefix}: {text}{suffix}\n", flush=True)

    elif target == "contemplation":
        if _read_state("show_contemplation", False, _bool):
            print(f"[thinking] {text[:120]}", flush=True)

    elif target in ("display", "projection"):
        print(f"[display] {text[:200]}", flush=True)


# ---------------------------------------------------------------------------
# Input loop
# ---------------------------------------------------------------------------

def _input_loop(prompt: str) -> None:
    """
    Read stdin line by line and write speech events to STM.
    Handles EOF gracefully (piped input or Ctrl-D).
    """
    print(f"Text UI active. Type your message and press Enter.", flush=True)

    while not _stop_event.is_set():
        try:
            line = input(f"{prompt}: ").strip()
        except EOFError:
            # Piped input exhausted — wait for stop signal
            _stop_event.wait()
            break
        except KeyboardInterrupt:
            break

        if not line:
            continue

        if line.lower() in ("quit", "exit", "bye"):
            print("(Exiting text UI)", flush=True)
            _stop_event.set()
            break

        if _stm_writer is not None:
            _stm_writer(
                source     = "user",
                type       = "speech",
                payload    = {
                    "text":     line,
                    "modality": "text",
                    "tool":     "tool.ui.text.v1",
                    "entity_id": _entity_id,
                },
                confidence = 1.0,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_state(key: str, default: Any, cast=str) -> Any:
    if _htm_ref is not None:
        v = _htm_ref.states.get(f"tool.ui.text.v1.{key}")
        if v is not None:
            try:
                return cast(v)
            except Exception:
                pass
    return default


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes")
