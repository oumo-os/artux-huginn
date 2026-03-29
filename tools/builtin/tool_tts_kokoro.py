"""
tool.tts.kokoro.v1 — Kokoro ONNX text-to-speech daemon for Huginn.

Runs as a live service tool. Subscribes to speech chunk events on the
ActuationBus and synthesises audio in real time using kokoro-onnx.

HUGINN_MANIFEST
tool_id:            tool.tts.kokoro.v1
title:              Kokoro TTS
capability_summary: >
  Real-time text-to-speech using the Kokoro ONNX model. Runs entirely
  in-process — no external server. Subscribes to speech chunk events and
  synthesises audio phrase by phrase for natural prosody. Supports multiple
  voices and live speed adjustment without restart.
polarity:           write
permission_scope:   [audio.output]
mode:               service
direction:          output
subscriptions:
  - type: output
    target: speech
    complete: chunk
  - type: output
    target: speech
    complete: full
inputs: {}
outputs: {}
states:
  voice:
    default: "af_bella"
    type: string
    description: Voice name (af_bella, af_sarah, bf_emma, am_adam, bm_george, ...)
  speed:
    default: 1.0
    type: float
    description: Speech speed multiplier (0.5 = slow, 1.0 = normal, 1.5 = fast)
  sample_rate:
    default: 24000
    type: integer
    description: Output audio sample rate in Hz
  device:
    default: ""
    type: string
    description: sounddevice output device name or index (empty = system default)
dependencies:
  kokoro-onnx
  sounddevice
  numpy
END_MANIFEST

Installation
------------
  pip install kokoro-onnx sounddevice numpy

  Kokoro downloads its ONNX model (~300 MB) on first use from HuggingFace.
  Set KOKORO_CACHE_DIR to control where models are stored.

Configuration via HTM.states (live, no restart needed):
  <task_update>{"action":"state_set","key":"tool.tts.kokoro.v1.speed","value":1.2}</task_update>
  <task_update>{"action":"state_set","key":"tool.tts.kokoro.v1.voice","value":"af_sarah"}</task_update>

Available voices (Kokoro v1.0):
  American female: af_bella (default), af_sarah, af_nicole, af_sky
  American male:   am_adam, am_michael
  British female:  bf_emma, bf_isabella
  British male:    bm_george, bm_lewis
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_kokoro    = None   # Kokoro model instance
_lock      = threading.Lock()
_audio_q:  queue.Queue = queue.Queue(maxsize=32)
_play_thread: Optional[threading.Thread] = None
_running   = False


# ---------------------------------------------------------------------------
# Service interface: start / stop / handle
# ---------------------------------------------------------------------------

def start(config: dict) -> None:
    """
    Initialise Kokoro model and start the audio playback thread.
    config is read from HTM.states at activation time.
    """
    global _kokoro, _running, _play_thread

    try:
        from kokoro_onnx import Kokoro
    except ImportError:
        raise ImportError(
            "kokoro-onnx not installed.\npip install kokoro-onnx"
        )

    with _lock:
        if _kokoro is None:
            _kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

    _running = True
    _play_thread = threading.Thread(
        target=_playback_loop,
        daemon=True,
        name="KokoroPlayback",
    )
    _play_thread.start()


def stop() -> None:
    """Drain the audio queue and stop the playback thread."""
    global _running
    _running = False
    _audio_q.put(None)   # sentinel


def handle(event: dict) -> None:
    """
    Receive an ActuationBus event and enqueue synthesis.

    For 'chunk' events: synthesise and play immediately.
    For 'full' events with interrupted=True: do nothing (chunk already played).
    For 'full' events: synthesise any remaining text not yet covered by chunks.
    """
    if event.get("interrupted"):
        return   # chunk events handled it; interrupted full is a no-op

    text = event.get("text", "").strip()
    if not text:
        return

    complete = event.get("complete", "full")
    if complete == "chunk":
        # Enqueue synthesis of this phrase chunk
        _audio_q.put(text)
    # 'full' events: if Kokoro is active, chunks already handled it.
    # If no TTS chunks arrived (e.g. very short utterance), synthesise now.
    elif complete == "full":
        # Only enqueue if queue is empty (chunks already handled the text)
        if _audio_q.empty():
            _audio_q.put(text)


# ---------------------------------------------------------------------------
# Internal playback loop
# ---------------------------------------------------------------------------

def _playback_loop() -> None:
    """
    Drain the synthesis queue and play audio via sounddevice.
    Reads HTM.states on each item for live config changes.
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        return

    while _running:
        try:
            text = _audio_q.get(timeout=0.5)
        except queue.Empty:
            continue

        if text is None:   # stop sentinel
            break

        with _lock:
            if _kokoro is None:
                continue

        # Read live config (applied per-phrase — no restart needed)
        # We access HTM.states through the module-level handle to avoid
        # threading complexity. Use module defaults if not set.
        voice       = _current_voice()
        speed       = _current_speed()
        sample_rate = _current_sample_rate()

        try:
            samples, sr = _kokoro.create(text, voice=voice, speed=speed,
                                          lang="en-us")
            if samples is None or len(samples) == 0:
                continue
            sd.play(samples, samplerate=sr or sample_rate, blocking=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Live config reads (module-level cache updated by handle via _htm)
# ---------------------------------------------------------------------------

_current_config: dict = {
    "voice":       "af_bella",
    "speed":       1.0,
    "sample_rate": 24000,
}


def _update_config(htm) -> None:
    """Called from handle() with _htm if it declares _htm parameter."""
    if htm is None:
        return
    _current_config["voice"]       = htm.states.get("tool.tts.kokoro.v1.voice",       "af_bella")
    _current_config["speed"]       = float(htm.states.get("tool.tts.kokoro.v1.speed", 1.0))
    _current_config["sample_rate"] = int(htm.states.get("tool.tts.kokoro.v1.sample_rate", 24000))


def handle(event: dict, _htm=None) -> None:   # noqa: F811 (redefinition with _htm injection)
    """handle() with optional _htm injection for live config reads."""
    if _htm is not None:
        _update_config(_htm)

    if event.get("interrupted"):
        return

    text     = event.get("text", "").strip()
    complete = event.get("complete", "full")

    if not text:
        return

    if complete == "chunk":
        _audio_q.put(text)
    elif complete == "full" and _audio_q.empty():
        _audio_q.put(text)


def _current_voice()       -> str:   return _current_config["voice"]
def _current_speed()       -> float: return _current_config["speed"]
def _current_sample_rate() -> int:   return _current_config["sample_rate"]
