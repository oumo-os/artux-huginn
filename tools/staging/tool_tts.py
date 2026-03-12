"""
HUGINN_MANIFEST
tool_id:            tool.tts.v1
title:              Text-to-Speech
capability_summary: >
  Speak text aloud using a local TTS engine. Use when Artux needs to say
  something to the user through speakers. Fully offline. Supports pyttsx3
  (cross-platform), piper (neural, high quality), and espeak-ng (lightweight).
  Preferred backend is tried in that order.
polarity:           write
permission_scope:   [speaker]
inputs:
  text:         {type: string,  description: "Text to speak"}
  backend:      {type: string,  default: "auto", description: "auto | pyttsx3 | piper | espeak"}
  rate:         {type: integer, default: 170,  description: "Speech rate words-per-minute (pyttsx3 only)"}
  volume:       {type: number,  default: 1.0,  description: "Volume 0.0-1.0 (pyttsx3 only)"}
  voice_id:     {type: string,  default: "",   description: "pyttsx3 voice id or piper model path"}
outputs:
  spoken:       {type: boolean, description: "True if speech was produced"}
  backend_used: {type: string}
  duration_ms:  {type: integer}
  error:        {type: string,  description: "Non-empty if speech failed"}
dependencies:
  - pyttsx3>=2.90
perception_capable: false
handler:            handle
END_MANIFEST

Text-to-speech output tool.

Backend selection order (with 'auto'):
  1. pyttsx3 — cross-platform, wraps OS native TTS (SAPI5/NSSpeechSynthesizer/eSpeak).
               Zero-latency start. Good enough for most uses.
  2. piper   — neural TTS, much higher quality. Requires 'piper-tts' package and
               a voice model file. Set HUGINN_TTS_PIPER_VOICE to the .onnx model path.
  3. espeak  — fallback on Linux. Robotic but reliable.

For the Narrator pipeline, Sagax emits <speech> tokens which the Orchestrator
streams directly to TTS without going through tool_call. This tool is for cases
where Sagax wants to speak from within a skill step, or a tool result should be
read aloud explicitly.

Environment overrides:
  HUGINN_TTS_BACKEND=auto|pyttsx3|piper|espeak
  HUGINN_TTS_RATE=170
  HUGINN_TTS_VOLUME=1.0
  HUGINN_TTS_PIPER_VOICE=/path/to/voice.onnx    (for piper backend)
  HUGINN_TTS_VOICE_ID=                           (pyttsx3 voice id)
"""

from __future__ import annotations

import os
import time


def handle(
    text:     str   = "",
    backend:  str   = "auto",
    rate:     int   = 170,
    volume:   float = 1.0,
    voice_id: str   = "",
) -> dict:
    """Speak text using the best available local TTS backend."""
    text    = text.strip()
    if not text:
        return {"spoken": False, "backend_used": "none",
                "duration_ms": 0, "error": "empty text"}

    backend  = os.environ.get("HUGINN_TTS_BACKEND",  backend)
    rate     = int(os.environ.get("HUGINN_TTS_RATE",   rate))
    volume   = float(os.environ.get("HUGINN_TTS_VOLUME", volume))
    voice_id = os.environ.get("HUGINN_TTS_VOICE_ID", voice_id)

    if backend == "auto":
        for b in ("pyttsx3", "piper", "espeak"):
            result = _try_backend(b, text, rate, volume, voice_id)
            if result["spoken"]:
                return result
        return {"spoken": False, "backend_used": "none",
                "duration_ms": 0, "error": "no TTS backend available"}

    return _try_backend(backend, text, rate, volume, voice_id)


def _try_backend(
    backend:  str,
    text:     str,
    rate:     int,
    volume:   float,
    voice_id: str,
) -> dict:
    t0 = time.monotonic()
    try:
        if backend == "pyttsx3":
            return _speak_pyttsx3(text, rate, volume, voice_id, t0)
        elif backend == "piper":
            return _speak_piper(text, voice_id, t0)
        elif backend == "espeak":
            return _speak_espeak(text, rate, t0)
        else:
            return {"spoken": False, "backend_used": backend,
                    "duration_ms": 0, "error": f"unknown backend {backend!r}"}
    except Exception as e:
        dur = int((time.monotonic() - t0) * 1000)
        return {"spoken": False, "backend_used": backend,
                "duration_ms": dur, "error": str(e)}


def _speak_pyttsx3(text, rate, volume, voice_id, t0) -> dict:
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate",   rate)
    engine.setProperty("volume", volume)
    if voice_id:
        engine.setProperty("voice", voice_id)
    engine.say(text)
    engine.runAndWait()
    return {"spoken": True, "backend_used": "pyttsx3",
            "duration_ms": int((time.monotonic() - t0) * 1000), "error": ""}


def _speak_piper(text, voice_id, t0) -> dict:
    import subprocess, tempfile, shutil
    voice = voice_id or os.environ.get("HUGINN_TTS_PIPER_VOICE", "")
    if not voice:
        raise RuntimeError("HUGINN_TTS_PIPER_VOICE not set")
    piper_bin = shutil.which("piper") or "piper"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    subprocess.run(
        [piper_bin, "--model", voice, "--output_file", wav_path],
        input=text.encode(), check=True, capture_output=True,
    )
    # Play via aplay (Linux) or afplay (macOS)
    player = "afplay" if os.uname().sysname == "Darwin" else "aplay"
    subprocess.run([player, wav_path], check=True, capture_output=True)
    os.unlink(wav_path)
    return {"spoken": True, "backend_used": "piper",
            "duration_ms": int((time.monotonic() - t0) * 1000), "error": ""}


def _speak_espeak(text, rate, t0) -> dict:
    import subprocess, shutil
    bin_ = shutil.which("espeak-ng") or shutil.which("espeak")
    if not bin_:
        raise RuntimeError("espeak-ng / espeak not found on PATH")
    subprocess.run([bin_, "-s", str(rate), text],
                   check=True, capture_output=True)
    return {"spoken": True, "backend_used": "espeak",
            "duration_ms": int((time.monotonic() - t0) * 1000), "error": ""}
