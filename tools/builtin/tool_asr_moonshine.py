"""
tool.asr.moonshine.v1 — Moonshine ONNX speech recognition daemon for Huginn.

Runs as a live service tool. Continuously captures microphone audio,
applies voice activity detection, transcribes via Moonshine ONNX, and
writes speech events to STM. Typed text input and Moonshine output are
identical event types — Sagax cannot tell the difference.

HUGINN_MANIFEST
tool_id:            tool.asr.moonshine.v1
title:              Moonshine ASR
capability_summary: >
  Real-time speech-to-text using the Moonshine ONNX model. Purpose-built
  for on-device transcription: 5x faster than Whisper-tiny at equivalent
  accuracy on short utterances. Runs entirely in-process on CPU — no GPU
  required. Writes transcribed speech as STM speech events indistinguishable
  from typed text input. Supports base (100 MB) and tiny (50 MB) model sizes.
polarity:           read
permission_scope:   [audio.input, microphone]
mode:               service
direction:          input
subscriptions: []
inputs: {}
outputs:
  text:  {type: string, description: Transcribed speech text}
states:
  model:
    default: "moonshine/base"
    type: string
    description: Model variant (moonshine/tiny or moonshine/base)
  sample_rate:
    default: 16000
    type: integer
    description: Microphone sample rate in Hz (Moonshine expects 16 kHz)
  chunk_secs:
    default: 5
    type: float
    description: Audio capture chunk duration in seconds
  vad_threshold:
    default: 0.008
    type: float
    description: RMS energy threshold below which audio is treated as silence
  device:
    default: ""
    type: string
    description: sounddevice input device name or index (empty = system default)
dependencies:
  moonshine-onnx
  sounddevice
  numpy
END_MANIFEST

Installation
------------
  pip install moonshine-onnx sounddevice numpy

  Moonshine downloads ONNX weights (~50–100 MB) on first use.
  Model sizes:
    moonshine/tiny  — ~50 MB, fastest, lower accuracy
    moonshine/base  — ~100 MB, good balance (default)
    moonshine/small — ~200 MB, best accuracy

Configuration via HTM.states:
  <task_update>{"action":"state_set","key":"tool.asr.moonshine.v1.vad_threshold","value":0.01}</task_update>
  <task_update>{"action":"state_set","key":"tool.asr.moonshine.v1.model","value":"moonshine/tiny"}</task_update>
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_moonshine    = None
_model_name   = ""
_lock         = threading.Lock()
_stop_event   = threading.Event()
_stm_writer: Optional[Callable] = None   # injected: stm.record(...)
_htm_ref      = None


# ---------------------------------------------------------------------------
# Service interface: start / stop
# No handle() — this is an input tool, it pushes to STM rather than
# receiving from the ActuationBus.
# ---------------------------------------------------------------------------

def start(config: dict, _htm=None, _stm=None) -> None:
    """
    Initialise Moonshine and start the capture/transcribe loop.

    _htm and _stm are injected by ToolManager because this tool declares
    them as parameters. This is how an input service tool writes to STM
    without going through the normal world-tool dispatch path.
    """
    global _moonshine, _model_name, _stm_writer, _htm_ref, _stop_event

    _htm_ref    = _htm
    _stm_writer = _stm.record if _stm is not None else None

    model_name = (
        (_htm.states.get("tool.asr.moonshine.v1.model") if _htm else None)
        or config.get("model", "moonshine/base")
    )

    # Load or reload model if name changed
    with _lock:
        if _moonshine is None or model_name != _model_name:
            try:
                from moonshine_onnx import MoonshineOnnxModel
            except ImportError:
                raise ImportError(
                    "moonshine-onnx not installed.\npip install moonshine-onnx"
                )
            _moonshine  = MoonshineOnnxModel(model_name=model_name)
            _model_name = model_name

    _stop_event.clear()

    t = threading.Thread(
        target=_capture_loop,
        daemon=True,
        name="MoonshineASR",
    )
    t.start()


def stop() -> None:
    _stop_event.set()


# ---------------------------------------------------------------------------
# Capture / transcribe loop
# ---------------------------------------------------------------------------

def _capture_loop() -> None:
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        return

    while not _stop_event.is_set():
        try:
            _capture_and_process(sd, np)
        except Exception:
            time.sleep(1.0)


def _capture_and_process(sd, np) -> None:
    # Read live config
    sample_rate = _read_state("sample_rate", 16000, int)
    chunk_secs  = _read_state("chunk_secs",  5,     float)
    vad_thresh  = _read_state("vad_threshold", 0.008, float)
    device      = _read_state("device", "", str) or None

    n_samples = int(chunk_secs * sample_rate)
    audio = sd.rec(
        n_samples,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    audio = audio.flatten()

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < vad_thresh:
        return   # silence — skip transcription

    with _lock:
        if _moonshine is None:
            return
        tokens = _moonshine.generate(audio[np.newaxis, :])
        texts  = _moonshine.tokenizer.decode_batch(tokens)

    text = texts[0].strip() if texts else ""
    if not text:
        return

    # Write to STM as a speech event — same format as typed input
    if _stm_writer is not None:
        _stm_writer(
            source     = "user",
            type       = "speech",
            payload    = {
                "text":       text,
                "modality":   "audio",
                "tool":       "moonshine",
                "model":      _model_name,
                "confidence": 1.0,
            },
            confidence = 1.0,
        )


def _read_state(key: str, default: Any, cast=str) -> Any:
    if _htm_ref is not None:
        v = _htm_ref.states.get(f"tool.asr.moonshine.v1.{key}")
        if v is not None:
            try:
                return cast(v)
            except Exception:
                pass
    return default
