"""
tool.capture.microphone.v1 — Microphone audio capture for Huginn.

Captures raw PCM audio from the system microphone and pushes CaptureFrames
into a queue that PerceptionManager dispatches to processor tools
(ASR, voice identity, ambient sound detection, etc.).

This tool has NO OPINION about what the audio means. It just captures.
Processing is done by whatever processor tools are installed.

HUGINN_MANIFEST
tool_id:            tool.capture.microphone.v1
title:              Microphone Capture
capability_summary: >
  Continuous microphone capture. Produces audio_chunk CaptureFrames
  dispatched by PerceptionManager to ASR, voice identity, and any other
  processor tool that consumes audio_chunk. Does not write to STM.
  sounddevice is the only hardware dependency in the audio pipeline.
polarity:           read
permission_scope:   [microphone, audio.input]
mode:               service
direction:          capture
emits:
  - audio_chunk
inputs: {}
outputs: {}
states:
  sample_rate:
    default: 16000
    type: integer
    description: Sample rate in Hz (Moonshine and ECAPA both expect 16 kHz)
  chunk_secs:
    default: 5.0
    type: float
    description: Duration of each captured chunk in seconds
  device:
    default: ""
    type: string
    description: Input device name or index (empty = system default)
  enabled:
    default: true
    type: boolean
    description: Set to false to pause capture without stopping the tool
dependencies:
  sounddevice
  numpy
END_MANIFEST
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

_stop_event = threading.Event()
_q:         queue.Queue = queue.Queue(maxsize=64)
_htm_ref    = None
_running    = False


# ---------------------------------------------------------------------------
# Service interface
# ---------------------------------------------------------------------------

def start(config: dict, _htm=None) -> None:
    global _stop_event, _htm_ref, _running
    _htm_ref = _htm
    _stop_event.clear()
    _running = True
    threading.Thread(
        target=_capture_loop, daemon=True, name="MicCapture"
    ).start()


def stop() -> None:
    global _running
    _running = False
    _stop_event.set()
    _q.put(None)   # unblock any waiting consumer


def get_queue() -> queue.Queue:
    """PM calls this to get the CaptureFrame queue."""
    return _q


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------

def _capture_loop() -> None:
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        return

    while not _stop_event.is_set():
        # Read live config on every chunk
        sample_rate = _read("sample_rate", 16000, int)
        chunk_secs  = _read("chunk_secs",  5.0,   float)
        device      = _read("device",      "",    str) or None
        enabled     = _read("enabled",     True,  _bool)

        if not enabled:
            time.sleep(0.5)
            continue

        n_samples = int(chunk_secs * sample_rate)
        try:
            audio = sd.rec(
                n_samples,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=device,
            )
            sd.wait()
            data = audio.flatten()
        except Exception:
            time.sleep(1.0)
            continue

        # Import here to avoid circular deps at module level
        try:
            from huginn.runtime.perception import CaptureFrame
        except ImportError:
            # Fallback: plain dict if CaptureFrame not importable yet
            class CaptureFrame:  # type: ignore
                def __init__(self, kind, data, meta):
                    self.kind = kind
                    self.data = data
                    self.meta = meta

        frame = CaptureFrame(
            kind="audio_chunk",
            data=data,
            meta={"sample_rate": sample_rate, "duration_s": chunk_secs,
                  "device": str(device)},
        )
        try:
            _q.put_nowait(frame)
        except queue.Full:
            pass   # drop oldest would be better but simple drop-newest is fine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(key: str, default: Any, cast) -> Any:
    if _htm_ref is not None:
        v = _htm_ref.states.get(f"tool.capture.microphone.v1.{key}")
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
