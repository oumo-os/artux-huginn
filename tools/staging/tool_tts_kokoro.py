"""
HUGINN_MANIFEST
tool_id:            tool.tts.kokoro.v1
title:              Text-to-Speech — Kokoro Neural TTS
capability_summary: >
  Speak text aloud using Kokoro, a lightweight high-quality neural TTS
  model (~330 MB, runs on CPU). Natural-sounding, fast, fully offline.
  Falls back to pyttsx3 if Kokoro is not available. Use when Artux needs
  to say something to the user through speakers.
polarity:           write
permission_scope:   [speaker]
inputs:
  text:         {type: string,  description: "Text to speak"}
  voice:        {type: string,  default: "af_heart", description: "Kokoro voice ID. See voice list below."}
  speed:        {type: number,  default: 1.0,   description: "Playback speed multiplier (0.5–2.0)"}
  lang:         {type: string,  default: "en-us", description: "Language code for phonemisation"}
outputs:
  spoken:       {type: boolean}
  backend:      {type: string,  description: "kokoro | pyttsx3 | espeak"}
  duration_ms:  {type: integer}
  error:        {type: string}
dependencies:
  - kokoro>=0.9
  - sounddevice>=0.4
  - numpy>=1.24
perception_capable: false
handler:            handle
END_MANIFEST

Kokoro TTS — lightweight neural text-to-speech.

Kokoro (kokoro-82M) is an 82M parameter TTS model with near-human
quality on English, achieving scores competitive with models 10× its size.
It runs on CPU in real time and requires no GPU.

Model size: ~330 MB (downloads once from HuggingFace on first call).
Inference: ~150–300 ms for a short sentence on a modern CPU.

Voice selection:
  English:
    af_heart  — warm female         ← default
    af_bella  — soft female
    af_nicole — clear female
    am_adam   — natural male
    am_michael— deep male
    bf_emma   — British female
    bm_george — British male
  Other languages (set lang= accordingly):
    ff_siwis  — French female       (lang=fr-fr)
    hf_alpha  — Hindi female        (lang=hi)
    jf_alpha  — Japanese female     (lang=ja)
    zf_xiaobei— Chinese female      (lang=cmn)
    ef_dora   — Spanish female      (lang=es)

  Full list: https://huggingface.co/hexgrad/Kokoro-82M

Fallback chain (if Kokoro unavailable):
  pyttsx3 → espeak-ng → silent fail with error message

Environment overrides:
  HUGINN_TTS_VOICE=af_heart
  HUGINN_TTS_SPEED=1.0
  HUGINN_TTS_LANG=en-us
  HUGINN_TTS_BACKEND=kokoro        (force a backend: kokoro|pyttsx3|espeak)
"""

from __future__ import annotations

import os
import time
import threading

import numpy as np

_kokoro_pipeline = None
_kokoro_lock     = threading.Lock()


def _get_kokoro(lang: str):
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        with _kokoro_lock:
            if _kokoro_pipeline is None:
                from kokoro import KPipeline
                _kokoro_pipeline = KPipeline(lang_code=lang[:2])
    return _kokoro_pipeline


def handle(
    text:  str   = "",
    voice: str   = "af_heart",
    speed: float = 1.0,
    lang:  str   = "en-us",
) -> dict:
    """Speak text using Kokoro neural TTS with pyttsx3 fallback."""
    text  = text.strip()
    if not text:
        return {"spoken": False, "backend": "none", "duration_ms": 0,
                "error": "empty text"}

    voice   = os.environ.get("HUGINN_TTS_VOICE",   voice)
    speed   = float(os.environ.get("HUGINN_TTS_SPEED", speed))
    lang    = os.environ.get("HUGINN_TTS_LANG",    lang)
    backend = os.environ.get("HUGINN_TTS_BACKEND", "kokoro")

    if backend != "kokoro":
        return _fallback(text, backend)

    t0 = time.monotonic()
    try:
        return _speak_kokoro(text, voice, speed, lang, t0)
    except Exception as e:
        # Try pyttsx3 before giving up
        try:
            return _speak_pyttsx3(text, t0)
        except Exception:
            dur = int((time.monotonic() - t0) * 1000)
            return {"spoken": False, "backend": "kokoro",
                    "duration_ms": dur, "error": str(e)}


def _speak_kokoro(text: str, voice: str, speed: float, lang: str, t0: float) -> dict:
    import sounddevice as sd

    pipeline = _get_kokoro(lang)

    # KPipeline.generate() yields (gs, ps, audio_array) chunks
    # Collect all audio chunks and concatenate
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        if audio is not None and len(audio) > 0:
            chunks.append(audio)

    if not chunks:
        raise RuntimeError("Kokoro returned no audio")

    audio   = np.concatenate(chunks, axis=-1).flatten()
    sr      = 24000   # Kokoro outputs at 24 kHz

    # Adjust speed via resampling if speed != 1.0 (simple linear resample)
    if abs(speed - 1.0) > 0.05:
        import scipy.signal
        target_len = int(len(audio) / speed)
        audio      = scipy.signal.resample(audio, target_len).astype(np.float32)

    sd.play(audio, samplerate=sr)
    sd.wait()

    dur = int((time.monotonic() - t0) * 1000)
    return {"spoken": True, "backend": "kokoro", "duration_ms": dur, "error": ""}


def _speak_pyttsx3(text: str, t0: float) -> dict:
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    dur = int((time.monotonic() - t0) * 1000)
    return {"spoken": True, "backend": "pyttsx3", "duration_ms": dur, "error": ""}


def _fallback(text: str, backend: str) -> dict:
    t0 = time.monotonic()
    try:
        if backend == "pyttsx3":
            return _speak_pyttsx3(text, t0)
        elif backend == "espeak":
            import subprocess, shutil
            bin_ = shutil.which("espeak-ng") or shutil.which("espeak")
            if not bin_:
                raise RuntimeError("espeak not found")
            subprocess.run([bin_, text], check=True, capture_output=True)
            dur = int((time.monotonic() - t0) * 1000)
            return {"spoken": True, "backend": "espeak",
                    "duration_ms": dur, "error": ""}
    except Exception as e:
        dur = int((time.monotonic() - t0) * 1000)
        return {"spoken": False, "backend": backend,
                "duration_ms": dur, "error": str(e)}
    return {"spoken": False, "backend": backend, "duration_ms": 0,
            "error": f"unknown backend {backend!r}"}
