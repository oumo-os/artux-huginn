"""
HUGINN_MANIFEST
tool_id:            tool.voice.moonshine.v1
title:              Voice Input — Moonshine ASR
capability_summary: >
  Capture a chunk of microphone audio and transcribe it to text using
  Moonshine ONNX. Fully offline, CPU-native, no PyTorch required.
  Use as a perception pipeline step for continuous speech-to-STM.
polarity:           read
permission_scope:   [microphone]
inputs:
  sample_rate:      {type: integer, default: 16000}
  chunk_secs:       {type: number,  default: 2.5,  description: "Recording window in seconds"}
  silence_rms:      {type: number,  default: 0.01, description: "RMS threshold below which audio is treated as silence"}
  model:            {type: string,  default: "moonshine/base", description: "moonshine/tiny | moonshine/base | moonshine/small"}
outputs:
  text:             {type: string,  description: "Transcribed text, empty string if silence"}
  rms:              {type: number,  description: "RMS energy of the recorded chunk"}
  is_speech:        {type: boolean, description: "True if RMS exceeded the silence threshold"}
  duration_ms:      {type: integer, description: "Transcription time in ms"}
dependencies:
  - moonshine-onnx>=0.2
  - sounddevice>=0.4
  - numpy>=1.24
perception_capable: true
handler:            handle
END_MANIFEST

Moonshine ONNX voice perception pipeline step.

Drop this file into tools/staging/ and Huginn will ask whether to install it.
Say yes, and optionally enable it as an always-on perception pipeline.

Once active, the Orchestrator calls handle() on every pipeline tick.
The function records chunk_secs of audio, runs VAD, and transcribes with
Moonshine if speech is detected. An empty text field means silence — the
Perception Manager discards those silently.

Model tradeoffs:
  moonshine/tiny  ~50 MB   RTF < 0.1    fastest, good for short commands
  moonshine/base  ~100 MB  RTF < 0.15   best accuracy/speed balance (default)
  moonshine/small ~200 MB  RTF < 0.25   near Whisper-small quality on CPU

All models download automatically from HuggingFace on first use.

Configuration via environment variables (override manifest defaults):
  HUGINN_MOONSHINE_MODEL=moonshine/base
  HUGINN_MICROPHONE_DEVICE=None   (None = system default)
  HUGINN_AUDIO_CHUNK_SECS=2.5
  HUGINN_SILENCE_RMS=0.01
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Module-level singletons — loaded lazily on first handle() call
# ---------------------------------------------------------------------------

_model        = None          # MoonshineOnnxModel instance
_model_name   = None          # name used to load _model (for re-init on change)

_SAMPLE_RATE  = 16000         # Moonshine expects 16 kHz mono float32

# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------

def _get_model(model_name: str):
    """Return a cached MoonshineOnnxModel, re-initialising if model_name changed."""
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from moonshine_onnx import MoonshineOnnxModel
        _model      = MoonshineOnnxModel(model_name=model_name)
        _model_name = model_name
    return _model


def _transcribe(audio: np.ndarray, model) -> str:
    """Transcribe float32 16 kHz mono array. Returns empty string on failure."""
    try:
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]       # (1, samples) shape Moonshine expects
        tokens = model.generate(audio)
        texts  = model.tokenizer.decode_batch(tokens)
        return texts[0].strip() if texts else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

def handle(
    sample_rate: int   = 16000,
    chunk_secs:  float = 2.5,
    silence_rms: float = 0.01,
    model:       str   = "moonshine/base",
) -> dict:
    """
    Record chunk_secs of audio and transcribe.

    Returns:
        text       — transcribed string (empty = silence or transcription failed)
        rms        — audio RMS energy
        is_speech  — True if RMS > silence_rms
        duration_ms — time spent transcribing
    """
    import sounddevice as sd

    # Allow env overrides (useful without redeploying the file)
    model       = os.environ.get("HUGINN_MOONSHINE_MODEL", model)
    chunk_secs  = float(os.environ.get("HUGINN_AUDIO_CHUNK_SECS", chunk_secs))
    silence_rms = float(os.environ.get("HUGINN_SILENCE_RMS", silence_rms))
    device      = os.environ.get("HUGINN_MICROPHONE_DEVICE", None)
    if device == "None":
        device = None

    n_samples = int(chunk_secs * sample_rate)

    # Record
    audio = sd.rec(
        n_samples,
        samplerate = sample_rate,
        channels   = 1,
        dtype      = "float32",
        device     = device,
    )
    sd.wait()
    audio = audio.flatten()

    rms = float(np.sqrt(np.mean(audio ** 2)))

    if rms < silence_rms:
        return {"text": "", "rms": rms, "is_speech": False, "duration_ms": 0}

    # Transcribe
    m = _get_model(model)
    t0   = time.monotonic()
    text = _transcribe(audio, m)
    dur  = int((time.monotonic() - t0) * 1000)

    return {
        "text":        text,
        "rms":         rms,
        "is_speech":   True,
        "duration_ms": dur,
    }
