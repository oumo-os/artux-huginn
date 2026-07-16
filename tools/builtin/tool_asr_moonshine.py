"""
tool.asr.moonshine.v1 — Moonshine ONNX ASR processor for Huginn.

Receives audio_chunk CaptureFrames from PerceptionManager and transcribes
them using the Moonshine ONNX model. Writes speech events to STM.

This tool has NO microphone dependency — it processes whatever audio bytes
PerceptionManager sends it. The microphone is owned by tool.capture.microphone.v1.

HUGINN_MANIFEST
tool_id:            tool.asr.moonshine.v1
title:              Moonshine ASR
capability_summary: >
  Real-time speech-to-text via Moonshine ONNX. Receives audio_chunk
  CaptureFrames from PerceptionManager and writes transcribed speech
  events to STM. Indistinguishable from typed text input to Exilis and
  Sagax. Compatible with moonshine/tiny (50 MB) and moonshine/base (100 MB).
polarity:           write
permission_scope:   []
mode:               service
direction:          process
consumes:
  - audio_chunk
inputs: {}
outputs:
  text: {type: string}
states:
  model:
    default: "moonshine/base"
    type: string
    description: Model variant (moonshine/tiny or moonshine/base)
  vad_threshold:
    default: 0.008
    type: float
    description: RMS energy below which audio is treated as silence
  enabled:
    default: true
    type: boolean
    description: Set to false to pause transcription
dependencies:
  moonshine-onnx
  numpy
END_MANIFEST
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Optional

_moonshine   = None
_model_name  = ""
_lock        = threading.Lock()
_proc_queue: queue.Queue = queue.Queue(maxsize=32)
_stop_event  = threading.Event()
_stm_writer: Optional[Callable] = None
_htm_ref     = None


def start(config: dict, _stm=None, _htm=None) -> None:
    global _stm_writer, _htm_ref, _stop_event
    _stm_writer = _stm.record if _stm else None
    _htm_ref    = _htm
    model_name  = (
        (_htm.states.get("tool.asr.moonshine.v1.model") if _htm else None)
        or config.get("model", "moonshine/base")
    )
    _load_model(model_name)
    _stop_event.clear()
    threading.Thread(target=_process_loop, daemon=True, name="MoonshineASR").start()


def stop() -> None:
    _stop_event.set()
    _proc_queue.put(None)


def push(frame, _stm=None, _htm=None) -> None:
    """Called by PerceptionManager for each audio_chunk CaptureFrame."""
    if not _read("enabled", True, _bool):
        return
    try:
        _proc_queue.put_nowait(frame)
    except queue.Full:
        pass


def _process_loop() -> None:
    while not _stop_event.is_set():
        try:
            frame = _proc_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if frame is None:
            break
        try:
            _transcribe(frame)
        except Exception:
            pass


def _transcribe(frame) -> None:
    import numpy as np
    audio = frame.data
    if audio is None or len(audio) == 0:
        return
    vad_thresh = _read("vad_threshold", 0.008, float)
    if float(np.sqrt(np.mean(audio ** 2))) < vad_thresh:
        return
    with _lock:
        if _moonshine is None:
            return
        tokens = _moonshine.generate(audio[np.newaxis, :])
        texts  = _moonshine.tokenizer.decode_batch(tokens)
    text = texts[0].strip() if texts else ""
    if not text or _stm_writer is None:
        return
    _stm_writer(
        source="user", type="speech",
        payload={
            "text": text, "modality": "audio",
            "tool": "tool.asr.moonshine.v1", "model": _model_name,
        },
        confidence=1.0,
    )


def _load_model(model_name: str) -> None:
    global _moonshine, _model_name
    with _lock:
        if _moonshine is not None and model_name == _model_name:
            return
        try:
            from moonshine_onnx import MoonshineOnnxModel
            _moonshine  = MoonshineOnnxModel(model_name=model_name)
            _model_name = model_name
        except ImportError:
            raise ImportError("moonshine-onnx not installed.\npip install moonshine-onnx")


def _read(key: str, default: Any, cast) -> Any:
    if _htm_ref is not None:
        v = _htm_ref.states.get(f"tool.asr.moonshine.v1.{key}")
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
