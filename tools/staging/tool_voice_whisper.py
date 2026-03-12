"""
HUGINN_MANIFEST
tool_id:            tool.voice.whisper.v1
title:              Voice Input — faster-whisper ASR
capability_summary: >
  Capture microphone audio and transcribe it with faster-whisper.
  Higher accuracy than Moonshine, especially for longer utterances and
  non-English speech. Requires PyTorch or CTranslate2. Use as a perception
  pipeline step for continuous speech-to-STM.
polarity:           read
permission_scope:   [microphone]
inputs:
  sample_rate:      {type: integer, default: 16000}
  chunk_secs:       {type: number,  default: 3.0}
  silence_rms:      {type: number,  default: 0.01}
  model:            {type: string,  default: "base", description: "tiny|base|small|medium|large-v3"}
  language:         {type: string,  default: "en",   description: "BCP-47 language code, or 'auto'"}
  device:           {type: string,  default: "cpu",  description: "cpu | cuda"}
  compute_type:     {type: string,  default: "int8", description: "int8 | float16 | float32"}
outputs:
  text:             {type: string}
  language:         {type: string,  description: "Detected language code"}
  language_prob:    {type: number,  description: "Language detection confidence 0-1"}
  rms:              {type: number}
  is_speech:        {type: boolean}
  duration_ms:      {type: integer}
dependencies:
  - faster-whisper>=1.0
  - sounddevice>=0.4
  - numpy>=1.24
perception_capable: true
handler:            handle
END_MANIFEST

faster-whisper voice pipeline step.

Uses CTranslate2 under the hood — significantly faster than openai-whisper
at equivalent accuracy. Models are cached in ~/.cache/huggingface/hub/.

Model tradeoffs:
  tiny     ~75 MB    fastest, English-only quality
  base     ~150 MB   good balance                  (default)
  small    ~500 MB   better accuracy, multi-lingual
  medium   ~1.5 GB   high accuracy
  large-v3 ~3 GB     near-human accuracy

Set device=cuda and compute_type=float16 for GPU acceleration.

Configuration via environment variables:
  HUGINN_WHISPER_MODEL=base
  HUGINN_WHISPER_LANGUAGE=en      (set to 'auto' for language detection)
  HUGINN_WHISPER_DEVICE=cpu
  HUGINN_MICROPHONE_DEVICE=None
"""

from __future__ import annotations

import os
import time

import numpy as np

# Module-level singletons
_model       = None
_model_name  = None
_model_device = None

_SAMPLE_RATE = 16000


def _get_model(model_name: str, device: str, compute_type: str):
    global _model, _model_name, _model_device
    key = (model_name, device, compute_type)
    if _model is None or (_model_name, _model_device) != (model_name, device):
        from faster_whisper import WhisperModel
        _model        = WhisperModel(model_name, device=device,
                                     compute_type=compute_type)
        _model_name   = model_name
        _model_device = device
    return _model


def handle(
    sample_rate:  int   = 16000,
    chunk_secs:   float = 3.0,
    silence_rms:  float = 0.01,
    model:        str   = "base",
    language:     str   = "en",
    device:       str   = "cpu",
    compute_type: str   = "int8",
) -> dict:
    """
    Record chunk_secs of audio, run VAD, transcribe with faster-whisper.

    Returns text (empty string if silence), detected language, and RMS.
    """
    import sounddevice as sd

    model        = os.environ.get("HUGINN_WHISPER_MODEL",    model)
    language     = os.environ.get("HUGINN_WHISPER_LANGUAGE", language)
    device       = os.environ.get("HUGINN_WHISPER_DEVICE",   device)
    chunk_secs   = float(os.environ.get("HUGINN_AUDIO_CHUNK_SECS", chunk_secs))
    silence_rms  = float(os.environ.get("HUGINN_SILENCE_RMS",      silence_rms))
    mic_device   = os.environ.get("HUGINN_MICROPHONE_DEVICE", None)
    if mic_device == "None":
        mic_device = None

    n_samples = int(chunk_secs * sample_rate)
    audio = sd.rec(n_samples, samplerate=sample_rate,
                   channels=1, dtype="float32", device=mic_device)
    sd.wait()
    audio = audio.flatten()

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < silence_rms:
        return {"text": "", "language": language, "language_prob": 1.0,
                "rms": rms, "is_speech": False, "duration_ms": 0}

    m  = _get_model(model, device, compute_type)
    t0 = time.monotonic()

    lang_arg = None if language == "auto" else language
    segments, info = m.transcribe(audio, language=lang_arg, beam_size=5)
    text = " ".join(s.text for s in segments).strip()
    dur  = int((time.monotonic() - t0) * 1000)

    return {
        "text":          text,
        "language":      info.language,
        "language_prob": round(float(info.language_probability), 3),
        "rms":           rms,
        "is_speech":     True,
        "duration_ms":   dur,
    }
