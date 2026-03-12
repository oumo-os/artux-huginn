"""
HUGINN_MANIFEST
tool_id:            tool.audio.listener.v1
title:              Audio Listener — Transcription, Diarisation, Ambient Analysis
capability_summary: >
  Record audio and return word-level transcription with speaker attribution,
  ambient sound classification, and optional emotion inference. Speakers are
  identified by voiceprint embeddings and matched against known entities in
  Muninn. Returns structured segments that the Perception Manager can use
  to resolve and attribute speech to specific people.
polarity:           read
permission_scope:   [microphone]
inputs:
  chunk_secs:         {type: number,  default: 5.0,  description: "Recording window in seconds"}
  sample_rate:        {type: integer, default: 16000}
  silence_rms:        {type: number,  default: 0.01}
  asr_model:          {type: string,  default: "base", description: "faster-whisper model: tiny|base|small|medium"}
  language:           {type: string,  default: "en"}
  asr_device:         {type: string,  default: "cpu"}
  embed_window_s:     {type: number,  default: 1.5,  description: "Speaker embedding window in seconds"}
  min_segment_s:      {type: number,  default: 0.3,  description: "Minimum diarised segment duration"}
  analyse_ambient:    {type: boolean, default: true,  description: "Include ambient sound classification"}
outputs:
  segments:           {type: array, description: "[{speaker_id, text, words, start_s, end_s, embedding, confidence}]"}
  ambient:            {type: object, description: "{dominant: speech|music|noise|silence, energy_db, spectral_centroid_hz, has_music, has_speech, has_noise}"}
  speaker_count:      {type: integer}
  total_speech_s:     {type: number}
  is_speech:          {type: boolean}
  rms:                {type: number}
dependencies:
  - faster-whisper>=1.0
  - resemblyzer>=0.1.1
  - sounddevice>=0.4
  - numpy>=1.24
  - scipy>=1.11
perception_capable: true
handler:            handle
END_MANIFEST

Audio listener with diarisation and voiceprint-based speaker attribution.

Diarisation pipeline:
  1. Record chunk_secs of audio.
  2. VAD: skip if below silence_rms.
  3. faster-whisper: transcribe with word-level timestamps.
  4. resemblyzer: extract speaker embeddings over embed_window_s sliding windows.
  5. Cluster embeddings using cosine similarity (simple greedy nearest-neighbour).
     No ML clustering model needed — works with 2 or 3 speakers reliably.
  6. Assign each word to its nearest speaker cluster → diarised segments.
  7. Return per-segment {speaker_id, text, words, embedding}.

Speaker IDs in the output (e.g. "SPK_0", "SPK_1") are session-local cluster
labels. The Perception Manager resolves these against known entity voiceprints
in Muninn by calling _resolve_signature() on each segment's embedding, which
may promote "SPK_0" to "entity:john" if the embedding matches.

Ambient analysis (if analyse_ambient=True):
  Uses numpy FFT — no extra dependencies. Classifies the chunk as:
    speech  — high energy in 85–3000 Hz band, human fundamental frequencies
    music   — sustained harmonic content across multiple octaves
    noise   — broadband energy without clear harmonic structure
    silence — below silence_rms threshold

Overlapping speech:
  When two speakers overlap, the window embeddings will show low cluster
  confidence. Such windows are tagged with speaker_id="SPK_OVERLAP" and
  confidence < 0.6. The text is still transcribed (as a mixed segment).

Dependency note:
  resemblyzer (~30 MB model, downloads on first call from HuggingFace).
  faster-whisper and resemblyzer share no model format — they coexist cleanly.

Environment overrides:
  HUGINN_AUDIO_ASR_MODEL=base
  HUGINN_AUDIO_LANGUAGE=en
  HUGINN_AUDIO_CHUNK_SECS=5.0
  HUGINN_SILENCE_RMS=0.01
  HUGINN_MICROPHONE_DEVICE=None
"""

from __future__ import annotations

import os
import time
import threading
from typing import Optional

import numpy as np

# ─── Module-level singletons ────────────────────────────────────────────────

_asr_model      = None
_asr_model_name = None
_asr_device     = None
_encoder        = None   # resemblyzer VoiceEncoder
_encoder_lock   = threading.Lock()

_SR = 16000   # resemblyzer and faster-whisper both expect 16 kHz


# ─── Lazy loaders ───────────────────────────────────────────────────────────

def _get_asr(model_name: str, device: str, compute_type: str = "int8"):
    global _asr_model, _asr_model_name, _asr_device
    if _asr_model is None or _asr_model_name != model_name or _asr_device != device:
        from faster_whisper import WhisperModel
        _asr_model      = WhisperModel(model_name, device=device,
                                        compute_type=compute_type)
        _asr_model_name = model_name
        _asr_device     = device
    return _asr_model


def _get_encoder():
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                from resemblyzer import VoiceEncoder
                _encoder = VoiceEncoder()
    return _encoder


# ─── Tool handler ────────────────────────────────────────────────────────────

def handle(
    chunk_secs:      float = 5.0,
    sample_rate:     int   = 16000,
    silence_rms:     float = 0.01,
    asr_model:       str   = "base",
    language:        str   = "en",
    asr_device:      str   = "cpu",
    embed_window_s:  float = 1.5,
    min_segment_s:   float = 0.3,
    analyse_ambient: bool  = True,
) -> dict:
    """
    Record audio, transcribe with word timestamps, diarise by speaker,
    and classify ambient sound.
    """
    import sounddevice as sd

    # Env overrides
    asr_model   = os.environ.get("HUGINN_AUDIO_ASR_MODEL",  asr_model)
    language    = os.environ.get("HUGINN_AUDIO_LANGUAGE",   language)
    chunk_secs  = float(os.environ.get("HUGINN_AUDIO_CHUNK_SECS", chunk_secs))
    silence_rms = float(os.environ.get("HUGINN_SILENCE_RMS",      silence_rms))
    mic_device  = os.environ.get("HUGINN_MICROPHONE_DEVICE", None)
    if mic_device == "None":
        mic_device = None

    n_samples = int(chunk_secs * sample_rate)
    audio = sd.rec(n_samples, samplerate=sample_rate,
                   channels=1, dtype="float32", device=mic_device)
    sd.wait()
    audio = audio.flatten()

    rms = float(np.sqrt(np.mean(audio ** 2)))
    ambient = _analyse_ambient(audio, sample_rate) if analyse_ambient \
              else {"dominant": "silence", "energy_db": -80.0,
                    "spectral_centroid_hz": 0.0, "has_music": False,
                    "has_speech": False, "has_noise": False}

    if rms < silence_rms:
        return {
            "segments": [], "ambient": ambient, "speaker_count": 0,
            "total_speech_s": 0.0, "is_speech": False, "rms": rms,
        }

    # ── Step 1: Transcribe with word timestamps ──────────────────────────────
    lang_arg = None if language == "auto" else language
    asr = _get_asr(asr_model, asr_device)
    whisper_segments, _ = asr.transcribe(
        audio, language=lang_arg, beam_size=5,
        word_timestamps=True, vad_filter=True,
    )

    words = []
    for seg in whisper_segments:
        for w in (seg.words or []):
            words.append({
                "word":  w.word.strip(),
                "start": w.start,
                "end":   w.end,
            })

    if not words:
        return {
            "segments": [], "ambient": ambient, "speaker_count": 0,
            "total_speech_s": 0.0, "is_speech": True, "rms": rms,
        }

    # ── Step 2: Speaker embeddings (sliding windows) ─────────────────────────
    encoder     = _get_encoder()
    window_size = int(embed_window_s * sample_rate)
    step_size   = max(1, window_size // 2)

    # Build (timestamp, embedding) pairs at window centres
    window_embeds: list[tuple[float, np.ndarray]] = []
    for start in range(0, len(audio) - window_size, step_size):
        window  = audio[start: start + window_size]
        if np.sqrt(np.mean(window ** 2)) < silence_rms * 0.5:
            continue
        try:
            emb = encoder.embed_utterance(window)
            centre_s = (start + window_size / 2) / sample_rate
            window_embeds.append((centre_s, emb))
        except Exception:
            continue

    # ── Step 3: Greedy cosine speaker clustering ─────────────────────────────
    # No ML clustering model — just nearest-neighbour assignment.
    # Works well for 1–3 distinct speakers in short chunks.
    SIMILARITY_THRESHOLD = 0.82   # tune: lower = more speakers detected
    clusters: list[np.ndarray] = []   # centroid per speaker

    window_labels: list[tuple[float, int, float]] = []  # (ts, spk_idx, confidence)

    for ts, emb in window_embeds:
        if not clusters:
            clusters.append(emb.copy())
            window_labels.append((ts, 0, 1.0))
            continue

        sims = [float(_cosine(emb, c)) for c in clusters]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= SIMILARITY_THRESHOLD:
            # Update centroid (running mean)
            n = sum(1 for _, l, _ in window_labels if l == best_idx)
            clusters[best_idx] = (clusters[best_idx] * n + emb) / (n + 1)
            window_labels.append((ts, best_idx, best_sim))
        elif len(clusters) < 6:   # max 6 speakers
            clusters.append(emb.copy())
            window_labels.append((ts, len(clusters) - 1, 1.0))
        else:
            # Force-assign to best even if below threshold
            window_labels.append((ts, best_idx, best_sim))

    # ── Step 4: Assign each word to its nearest speaker window ───────────────
    def _speaker_at(t: float) -> tuple[int, float]:
        """Return (speaker_idx, confidence) for time t."""
        if not window_labels:
            return 0, 1.0
        # Find the closest window centre
        closest = min(window_labels, key=lambda x: abs(x[0] - t))
        return closest[1], closest[2]

    annotated_words = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        spk_idx, conf = _speaker_at(mid)
        overlap = conf < 0.6
        annotated_words.append({
            **w,
            "speaker_idx": spk_idx,
            "confidence":  conf,
            "overlap":     overlap,
        })

    # ── Step 5: Group consecutive same-speaker words into segments ────────────
    segments = _words_to_segments(annotated_words, clusters, min_segment_s)

    total_speech = sum(s["end_s"] - s["start_s"] for s in segments)

    return {
        "segments":      segments,
        "ambient":       ambient,
        "speaker_count": len(clusters),
        "total_speech_s": round(total_speech, 2),
        "is_speech":     True,
        "rms":           rms,
    }


def _words_to_segments(
    words: list[dict],
    clusters: list[np.ndarray],
    min_segment_s: float,
) -> list[dict]:
    """Merge consecutive same-speaker words into segments."""
    if not words:
        return []

    segments = []
    current_spk = words[0]["speaker_idx"]
    current_words = [words[0]]

    for w in words[1:]:
        if w["speaker_idx"] == current_spk and not w.get("overlap"):
            current_words.append(w)
        else:
            seg = _make_segment(current_spk, current_words, clusters)
            if seg:
                segments.append(seg)
            current_spk  = w["speaker_idx"]
            current_words = [w]

    if current_words:
        seg = _make_segment(current_spk, current_words, clusters)
        if seg:
            segments.append(seg)

    # Filter very short segments
    segments = [s for s in segments
                if s["end_s"] - s["start_s"] >= min_segment_s]
    return segments


def _make_segment(
    spk_idx: int,
    words: list[dict],
    clusters: list[np.ndarray],
) -> Optional[dict]:
    if not words:
        return None

    is_overlap = all(w.get("overlap") for w in words)
    spk_id     = "SPK_OVERLAP" if is_overlap else f"SPK_{spk_idx}"
    text       = " ".join(w["word"] for w in words).strip()
    confidence = float(np.mean([w["confidence"] for w in words]))
    embedding  = clusters[spk_idx].tolist() if spk_idx < len(clusters) else []

    return {
        "speaker_id": spk_id,
        "text":       text,
        "words":      [{"word": w["word"], "start": w["start"], "end": w["end"]}
                       for w in words],
        "start_s":    words[0]["start"],
        "end_s":      words[-1]["end"],
        "embedding":  embedding,       # for Perception Manager → entity resolution
        "confidence": round(confidence, 3),
    }


# ─── Ambient sound analysis (numpy only) ─────────────────────────────────────

def _analyse_ambient(audio: np.ndarray, sr: int) -> dict:
    """
    Classify ambient sound using FFT spectral analysis.
    No ML model — pure numpy.

    Speech band:    85–3000 Hz (human fundamentals + formants)
    Music band:     200–8000 Hz with harmonic structure (sustained peaks)
    Noise:          broadband, no clear harmonic pattern
    """
    if len(audio) < sr // 4:
        return _silent_ambient()

    # RMS and dB
    rms    = float(np.sqrt(np.mean(audio ** 2)))
    eps    = 1e-9
    db     = float(20 * np.log10(rms + eps))

    # FFT magnitude spectrum
    n      = len(audio)
    freqs  = np.fft.rfftfreq(n, d=1.0 / sr)
    mag    = np.abs(np.fft.rfft(audio))

    # Spectral centroid
    total  = np.sum(mag) + eps
    centroid = float(np.sum(freqs * mag) / total)

    # Band energies
    def band_energy(lo, hi):
        mask = (freqs >= lo) & (freqs <= hi)
        return float(np.sum(mag[mask]) / (total))

    speech_energy = band_energy(85,  3000)
    music_energy  = band_energy(200, 8000)
    sub_energy    = band_energy(20,  85)

    # Simple harmonic structure: look for periodically spaced peaks in speech band
    speech_mask = (freqs >= 85) & (freqs <= 1000)
    speech_mag  = mag[speech_mask]
    # Count peaks relative to mean — harmonic content has many distinct peaks
    if len(speech_mag) > 10:
        peaks = int(np.sum(speech_mag > np.mean(speech_mag) * 2))
        harmonic = peaks > 5
    else:
        harmonic = False

    has_speech = speech_energy > 0.15 and centroid < 3000
    has_music  = music_energy  > 0.25 and harmonic and centroid > 800
    has_noise  = (not has_speech) and (not has_music) and db > -40

    if db < -50:
        dominant = "silence"
    elif has_music:
        dominant = "music"
    elif has_speech:
        dominant = "speech"
    elif has_noise:
        dominant = "noise"
    else:
        dominant = "silence"

    return {
        "dominant":           dominant,
        "energy_db":          round(db, 1),
        "spectral_centroid_hz": round(centroid, 1),
        "has_music":          has_music,
        "has_speech":         has_speech,
        "has_noise":          has_noise,
    }


def _silent_ambient() -> dict:
    return {
        "dominant": "silence", "energy_db": -80.0,
        "spectral_centroid_hz": 0.0,
        "has_music": False, "has_speech": False, "has_noise": False,
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0
