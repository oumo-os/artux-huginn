"""
tool.ui.text.v1 — Rich terminal interface for Huginn.

Provides a polished terminal experience:
  • Input:  reads stdin line by line → writes STM speech events
  • Output: subscribes to ActuationBus speech events → prints to stdout
  • ANSI colors for state indicators and output formatting
  • Conversation history with scrollback
  • Status bar showing active state, tools, entity
  • Multiline input via paste detection or triple-quote delimiter

This is both the simplest user interface and the universal test harness.
If you can drive the system from a terminal, everything is testable without
hardware. It works alongside Moonshine ASR and Kokoro TTS — the text
output is informational when TTS is active.

HUGINN_MANIFEST
tool_id:            tool.ui.text.v1
title:              Text UI
capability_summary: >
  Rich terminal interface with ANSI colors, conversation history, and
  status bar. Reads user input from stdin and writes it to STM as speech
  events. Prints Artux speech output to stdout with color-coded state
  indicators. Works standalone or alongside ASR/TTS.
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
  history_size:
    default: 50
    type: integer
    description: Number of exchanges to keep in conversation history
  use_color:
    default: true
    type: boolean
    description: Enable ANSI color output (set false for piping or dumb terminals)
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

import collections
import os
import sys
import threading
import time
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# ANSI colors — cross-platform, graceful degradation
# ---------------------------------------------------------------------------

_supports_color = (
    hasattr(sys.stdout, "isatty")
    and sys.stdout.isatty()
    and os.environ.get("TERM", "") != "dumb"
    and os.environ.get("NO_COLOR", "") == ""
)


class _C:
    """ANSI color codes — disabled when terminal doesn't support it."""
    RESET   = "\033[0m"   if _supports_color else ""
    BOLD    = "\033[1m"    if _supports_color else ""
    DIM     = "\033[2m"    if _supports_color else ""
    ITALIC  = "\033[3m"    if _supports_color else ""
    # Foreground
    RED     = "\033[91m"   if _supports_color else ""
    GREEN   = "\033[92m"   if _supports_color else ""
    YELLOW  = "\033[93m"   if _supports_color else ""
    BLUE    = "\033[94m"   if _supports_color else ""
    MAGENTA = "\033[95m"   if _supports_color else ""
    CYAN    = "\033[96m"   if _supports_color else ""
    WHITE   = "\033[97m"   if _supports_color else ""
    GRAY    = "\033[90m"   if _supports_color else ""
    # State-specific
    STATE_IDLE       = "\033[90m"   if _supports_color else ""  # dim
    STATE_LISTENING  = "\033[96m"   if _supports_color else ""  # cyan
    STATE_THINKING   = "\033[95m"   if _supports_color else ""  # magenta
    STATE_SPEAKING   = "\033[93m"   if _supports_color else ""  # yellow
    STATE_INTERRUPTED= "\033[91m"   if _supports_color else ""  # red


def _clear_line():
    """Clear current line and move cursor to start."""
    if _supports_color:
        sys.stdout.write("\r\033[2K")
    else:
        sys.stdout.write("\r" + " " * 80 + "\r")


def _status_bar(text: str):
    """Print a dim status bar line."""
    _clear_line()
    sys.stdout.write(f"{_C.DIM}{text}{_C.RESET}\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_stop_event   = threading.Event()
_stm_writer:  Optional[Callable] = None
_htm_ref      = None
_entity_id    = ""
_history:     collections.deque = collections.deque(maxlen=100)
_current_state = "idle"
_state_lock   = threading.Lock()


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
    global _current_state

    if _htm is not None:
        global _htm_ref
        _htm_ref = _htm

    quiet = _read_state("quiet_mode", False, _bool)
    if quiet:
        return

    target      = event.get("target", "")
    text        = event.get("text", "").strip()
    interrupted = event.get("interrupted", False)

    if not text:
        return

    prefix = _read_state("output_prefix", "Artux", str)
    use_color = _read_state("use_color", True, _bool)

    if target == "speech":
        # Add to history
        _history.append({"role": "assistant", "text": text, "time": time.time()})

        if use_color:
            _clear_line()
            # Reprint last user input for context
            if _history:
                last_user = None
                for h in reversed(_history):
                    if h["role"] == "user":
                        last_user = h
                        break
                if last_user:
                    print(f"  {_C.GRAY}{_C.ITALIC}you: {last_user['text'][:60]}{'...' if len(last_user['text'])>60 else ''}{_C.RESET}")

            suffix = f" {_C.RED}[interrupted]{_C.RESET}" if interrupted else ""
            print(f"\n{_C.STATE_SPEAKING}{_C.BOLD}{prefix}:{_C.RESET} {text}{suffix}\n")
        else:
            suffix = " [interrupted]" if interrupted else ""
            print(f"\n{prefix}: {text}{suffix}\n")
        sys.stdout.flush()

        with _state_lock:
            _current_state = "idle"

    elif target == "contemplation":
        if _read_state("show_contemplation", False, _bool):
            if use_color:
                print(f"  {_C.STATE_THINKING}{_C.DIM}thinking: {text[:120]}{_C.RESET}", flush=True)
            else:
                print(f"[thinking] {text[:120]}", flush=True)

    elif target in ("display", "projection"):
        if use_color:
            print(f"  {_C.CYAN}[display]{_C.RESET} {text[:200]}", flush=True)
        else:
            print(f"[display] {text[:200]}", flush=True)


# ---------------------------------------------------------------------------
# Input loop
# ---------------------------------------------------------------------------

def _input_loop(prompt: str) -> None:
    """
    Read stdin line by line and write speech events to STM.
    Handles EOF gracefully (piped input or Ctrl-D).
    Supports multiline input via paste detection.
    """
    use_color = _read_state("use_color", True, _bool)
    history_size = _read_state("history_size", 50, int)

    if use_color:
        _clear_line()
        print(f"{_C.BOLD}{_C.GREEN}Artux{_C.RESET} {_C.DIM}v3.2{_C.RESET}  "
              f"{_C.GRAY}type your message, Ctrl-C to exit{_C.RESET}")
        print(f"{_C.GRAY}{'─' * 50}{_C.RESET}")
    else:
        print("Text UI active. Type your message and press Enter.")

    while not _stop_event.is_set():
        try:
            # Show status bar before prompt
            if use_color and _supports_color:
                with _state_lock:
                    state = _current_state
                state_colors = {
                    "idle": _C.STATE_IDLE,
                    "listening": _C.STATE_LISTENING,
                    "thinking": _C.STATE_THINKING,
                    "speaking": _C.STATE_SPEAKING,
                }
                sc = state_colors.get(state, _C.DIM)
                # Show compact history hint
                hist_count = len([h for h in _history if h["role"] == "user"])
                if hist_count > 0:
                    _status_bar(f"  {sc}{state.upper()}{_C.RESET}  "
                                f"{_C.GRAY}exchanges: {hist_count}{_C.RESET}")

            line = input(f"{prompt}: ").strip()
        except EOFError:
            _stop_event.wait()
            break
        except KeyboardInterrupt:
            print()
            continue

        if not line:
            continue

        if line.lower() in ("quit", "exit", "bye"):
            if use_color:
                print(f"{_C.GRAY}(Exiting){_C.RESET}", flush=True)
            else:
                print("(Exiting text UI)", flush=True)
            _stop_event.set()
            break

        # Multiline: lines ending with """ collect until closing """
        if line.endswith('"""'):
            lines = [line[:-3]]
            while not _stop_event.is_set():
                try:
                    more = input("... ")
                except (EOFError, KeyboardInterrupt):
                    break
                if more.strip().endswith('"""'):
                    lines.append(more[:-3])
                    break
                lines.append(more)
            line = "\n".join(lines).strip()
            if not line:
                continue

        # Add to history
        _history.append({"role": "user", "text": line, "time": time.time()})

        if _stm_writer is not None:
            with _state_lock:
                _current_state = "listening"
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
