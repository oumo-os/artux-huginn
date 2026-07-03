"""
tool.identity.voice.v1 — Voice identity processor for Huginn.

Receives audio_chunk CaptureFrames from PerceptionManager, extracts
192-dim ECAPA-TDNN speaker embeddings via SpeechBrain, and writes
voiceprint sensor events to STM for PerceptionManager signature resolution.

This tool and tool.asr.moonshine.v1 both consume audio_chunk — they
receive the same frame independently, no duplication of capture.

HUGINN_MANIFEST
tool_id:            tool.identity.voice.v1
title:              Voice Identity
capability_summary: >
  Speaker identity via ECAPA-TDNN voice embeddings. Receives the same
  audio_chunk CaptureFrames as Moonshine ASR — one capture, two processors.
  Writes voiceprint sensor events to STM for PerceptionManager to resolve
  against registered entity signatures. Works alongside ASR without any
  additional microphone usage.
polarity:           write
permission_scope:   [microphone]
mode:               service
direction:          process
consumes:
  - audio_chunk
inputs: {}
outputs:
  entity_id: {type: string}
states:
  model_source:
    default: "speechbrain/spkrec-ecapa-voxceleb"
    type: string
    description: SpeechBrain model source (HuggingFace path or local dir)
  min_rms:
    default: 0.01
    type: float
    description: Minimum RMS energy — below this, skip identity (silence/noise)
  enabled:
    default: true
    type: boolean
    description: Set to false to pause identity without stopping the tool
dependencies:
  speechbrain
  torch
  numpy
END_MANIFEST
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Any, Callable, Optional

_classifier  = None
_lock        = threading.Lock()
_proc_queue: queue.Queue = queue.Queue(maxsize=16)
_stop_event  = threading.Event()
_stm_writer: Optional[Callable] = None
_htm_ref     = None


def start(config: dict, _stm=None, _htm=None) -> None:
    global _stm_writer, _htm_ref, _stop_event
    _stm_writer = _stm.record if _stm else None
    _htm_ref    = _htm
    _load_model(config.get("model_source",
        (_htm.states.get("tool.identity.voice.v1.model_source") if _htm else None)
        or "speechbrain/spkrec-ecapa-voxceleb"
    ))
    _stop_event.clear()
    threading.Thread(target=_process_loop, daemon=True, name="VoiceIdentity").start()


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
            _extract_and_write(frame)
        except Exception:
            pass


def _extract_and_write(frame) -> None:
    import numpy as np
    audio = frame.data
    if audio is None or len(audio) == 0:
        return

    # Skip noise/silence — no point extracting an embedding from nothing
    min_rms = _read("min_rms", 0.01, float)
    if float(np.sqrt(np.mean(audio ** 2))) < min_rms:
        return

    with _lock:
        if _classifier is None:
            return
        try:
            import torch
            signal = torch.tensor(audio[np.newaxis, :])
            with torch.no_grad():
                emb = _classifier.encode_batch(signal)
            embedding = emb.squeeze(0).squeeze(0).cpu().numpy().tolist()
        except Exception:
            return

    if _stm_writer is None:
        return

    # Write as a sensor event with a voiceprint signature.
    # PerceptionManager's _resolve_signature picks this up and
    # matches it against registered entity voiceprints in Muninn.
    _stm_writer(
        source="sensor", type="vision",   # reuses vision type for sensor events
        payload={
            "modality": "voiceprint",
            "signature": {
                "kind":       "voiceprint",
                "embedding":  embedding,
                "confidence": 1.0,
            },
            "sample_rate": frame.meta.get("sample_rate", 16000),
            "tool":        "tool.identity.voice.v1",
        },
        confidence=1.0,
    )


def _load_model(source: str) -> None:
    global _classifier
    with _lock:
        if _classifier is not None:
            return
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            cache = os.path.join(os.path.expanduser("~"), ".cache/speechbrain")
            _classifier = EncoderClassifier.from_hparams(
                source=source, savedir=cache,
                run_opts={"device": "cpu"},
            )
        except ImportError:
            raise ImportError(
                "speechbrain not installed.\npip install speechbrain torch"
            )


def _read(key: str, default: Any, cast) -> Any:
    if _htm_ref is not None:
        v = _htm_ref.states.get(f"tool.identity.voice.v1.{key}")
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
