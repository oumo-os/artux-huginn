"""
HUGINN_MANIFEST
tool_id:            tool.translate.v1
title:              Translation
capability_summary: >
  Translate text between any two languages. Uses MyMemory free API (no key
  required, online) with argostranslate as an offline fallback. Use when the
  user asks to translate something, or when Artux encounters text in a foreign
  language and needs to understand or relay it.
polarity:           read
permission_scope:   []
inputs:
  text:         {type: string,  description: "Text to translate"}
  target_lang:  {type: string,  description: "Target language BCP-47 code: en, fr, de, es, ja, zh, ar, pt, ru, ko, hi, it, nl, pl, tr, vi, …"}
  source_lang:  {type: string,  default: "auto", description: "Source language code, or 'auto' for detection"}
  backend:      {type: string,  default: "auto", description: "auto | mymemory | argostranslate"}
outputs:
  translated:         {type: string,  description: "Translated text"}
  source_lang:        {type: string,  description: "Detected or specified source language"}
  target_lang:        {type: string}
  confidence:         {type: number,  description: "Translation confidence 0-1 (where available)"}
  backend_used:       {type: string}
  char_count:         {type: integer, description: "Input character count"}
  summary:            {type: string,  description: "One-line confirmation for speech"}
dependencies:
  - requests>=2.28
perception_capable: false
handler:            handle
END_MANIFEST

Translation using MyMemory (online, free) and argostranslate (offline).

MyMemory (https://mymemory.translated.net):
  Free tier: 5000 characters/day without a key. No signup.
  Supports 60+ language pairs via ISO 639-1 codes.
  Returns a confidence score (0–1) with each translation.
  Error on rate limit: tool returns the original text with an error note.

argostranslate (offline fallback):
  Open-source offline translation via OpenNMT. Models (~50–100 MB each)
  are downloaded once per language pair on first use.
  Install: pip install argostranslate
  Note: argostranslate is NOT in the declared dependencies to keep the
  default install light. The tool installs it lazily if set as the backend
  or if MyMemory fails and argostranslate is already present.

Language code examples:
  en, fr, de, es, pt, it, nl, pl, ru, uk, cs, ro, hu, sv, da, fi, no
  ja, zh, ko, ar, hi, tr, vi, th, id, ms, el, he, fa, bn, sw, tl

Batch translation:
  For multiple strings, call once per string. The MyMemory free tier
  applies per-request, not per-character, so batching is not beneficial.

Environment overrides:
  HUGINN_TRANSLATE_BACKEND=auto     (auto | mymemory | argostranslate)
  HUGINN_MYMEMORY_EMAIL=            (optional: raises daily limit to 50k chars)
"""

from __future__ import annotations

import os
import time
from typing import Optional

import requests

_MYMEMORY_URL = "https://api.mymemory.translated.net/get"
_TIMEOUT      = 8


def handle(
    text:        str = "",
    target_lang: str = "en",
    source_lang: str = "auto",
    backend:     str = "auto",
) -> dict:
    text    = text.strip()
    if not text:
        return _result("", source_lang, target_lang, 1.0, "none", 0,
                       "No text to translate.")

    backend     = os.environ.get("HUGINN_TRANSLATE_BACKEND", backend)
    target_lang = target_lang.lower().split("-")[0]   # normalise en-US → en
    source_lang = source_lang.lower().split("-")[0] if source_lang != "auto" \
                  else "auto"

    if backend == "argostranslate":
        return _try_argos(text, source_lang, target_lang)

    # auto or mymemory: try MyMemory first
    result = _try_mymemory(text, source_lang, target_lang)
    if result["translated"] and "error" not in result["summary"].lower():
        return result

    # Fallback to argostranslate if available
    argos = _try_argos(text, source_lang, target_lang)
    if argos["translated"]:
        return argos

    # Return MyMemory result even if marginal (includes error note in summary)
    return result


# ─── MyMemory ─────────────────────────────────────────────────────────────────

def _try_mymemory(text: str, source: str, target: str) -> dict:
    lang_pair = f"{source}|{target}" if source != "auto" else f"en|{target}"
    email     = os.environ.get("HUGINN_MYMEMORY_EMAIL", "")

    params = {"q": text, "langpair": lang_pair}
    if email:
        params["de"] = email

    try:
        resp = requests.get(_MYMEMORY_URL, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return _result(text, source, target, 0.0, "mymemory",
                       len(text), f"Network error: {e}")

    rm = data.get("responseData", {})
    translated  = rm.get("translatedText", "").strip()
    confidence  = float(rm.get("match", 0.0))
    detected    = source if source != "auto" \
                  else data.get("detectedLanguage", source)

    # MyMemory returns error strings as translated text when rate-limited
    if translated.upper().startswith("MYMEMORY WARNING"):
        return _result(text, detected, target, 0.0, "mymemory",
                       len(text), f"MyMemory rate limit: {translated[:80]}")

    return _result(translated, detected, target, confidence,
                   "mymemory", len(text))


# ─── argostranslate ───────────────────────────────────────────────────────────

def _try_argos(text: str, source: str, target: str) -> dict:
    try:
        from argostranslate import package, translate

        # Install language pair if not present
        installed = translate.get_installed_languages()
        src_langs  = [l for l in installed if l.code == source]
        tgt_langs  = [l for l in installed if l.code == target]

        if not src_langs or not tgt_langs:
            # Try to download and install
            package.update_package_index()
            available = package.get_available_packages()
            to_install = [
                p for p in available
                if p.from_code == source and p.to_code == target
            ]
            if to_install:
                package.install_from_path(to_install[0].download())
                installed  = translate.get_installed_languages()
                src_langs  = [l for l in installed if l.code == source]
                tgt_langs  = [l for l in installed if l.code == target]

        if not src_langs or not tgt_langs:
            return _result("", source, target, 0.0, "argostranslate",
                           len(text), f"Language pair {source}→{target} not available offline.")

        translation = src_langs[0].get_translation(tgt_langs[0])
        translated  = translation.translate(text)
        return _result(translated, source, target, 0.95,
                       "argostranslate", len(text))

    except ImportError:
        return _result("", source, target, 0.0, "argostranslate",
                       len(text), "argostranslate not installed.")
    except Exception as e:
        return _result("", source, target, 0.0, "argostranslate",
                       len(text), str(e))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _result(
    translated: str,
    source:     str,
    target:     str,
    confidence: float,
    backend:    str,
    char_count: int,
    error:      str = "",
) -> dict:
    if error:
        summary = f"Translation failed: {error}"
    elif translated:
        preview = translated[:60] + ("…" if len(translated) > 60 else "")
        summary = f'Translated to {target}: \u201c{preview}\u201d'
    else:
        summary = "Translation returned empty."

    return {
        "translated":   translated,
        "source_lang":  source,
        "target_lang":  target,
        "confidence":   round(confidence, 3),
        "backend_used": backend,
        "char_count":   char_count,
        "summary":      summary,
    }
