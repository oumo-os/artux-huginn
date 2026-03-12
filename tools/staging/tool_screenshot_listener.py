"""
HUGINN_MANIFEST
tool_id:            tool.screenshot.listener.v1
title:              Screenshot Listener — Screen → VLM Description
capability_summary: >
  Capture a screenshot of any display and produce a natural-language
  description of what is on screen using SmolVLM. Useful for ambient
  awareness of what the user is working on, UI state, on-screen text
  extraction, and activity logging. Fully offline.
polarity:           read
permission_scope:   [screen]
inputs:
  monitor:          {type: integer, default: 1,    description: "Monitor index (1 = primary, 0 = all monitors combined)"}
  region:           {type: object,  default: null, description: "Optional {top,left,width,height} crop in pixels"}
  vlm_model_id:     {type: string,  default: "HuggingFaceTB/SmolVLM-500M-Instruct"}
  vlm_max_tokens:   {type: integer, default: 256}
  vlm_prompt:       {type: string,  default: ""}
  captures_dir:     {type: string,  default: "./captures"}
  jpeg_quality:     {type: integer, default: 75, description: "JPEG quality for saved screenshot"}
  extract_text:     {type: boolean, default: false, description: "Ask VLM to extract visible text (uses a different prompt)"}
outputs:
  description:      {type: string}
  screenshot_path:  {type: string}
  width:            {type: integer}
  height:           {type: integer}
  monitor:          {type: integer}
  vlm_ms:           {type: integer}
dependencies:
  - mss>=9.0
  - Pillow>=10.0
  - transformers>=4.40
  - torch>=2.1
  - accelerate>=0.27
  - numpy>=1.24
perception_capable: true
handler:            handle
END_MANIFEST

Screenshot listener — mss (very lightweight) + SmolVLM.

mss (Multiple Screenshots) is a pure-Python library that uses ctypes
directly against OS display APIs. It has no native extension to compile,
no X11 dependency on Linux, and works on macOS, Windows, and Linux.
It is the lightest cross-platform screenshot tool available.

Use cases:
  • Ambient screen awareness — what is the user looking at?
  • Activity logging — which app is active, what document is open?
  • On-screen text extraction — receipts, articles, code on screen
  • UI state monitoring — is a specific window open?
  • Accessibility — describe what is on screen for voice feedback

Text extraction mode (extract_text=True):
  The VLM prompt is replaced with a text-extraction instruction.
  Output will be prose listing all visible text in reading order.
  Not a replacement for OCR but effective for SmolVLM-scale content.

Region capture:
  Pass region as {"top": 0, "left": 0, "width": 800, "height": 600}
  to capture only part of the screen. Coordinates are in pixels,
  origin top-left.

Environment overrides:
  HUGINN_SCREEN_MONITOR=1
  HUGINN_VLM_MODEL=HuggingFaceTB/SmolVLM-500M-Instruct
  HUGINN_CAPTURES_DIR=./captures
"""

from __future__ import annotations

import datetime
import os
import threading
import time
from pathlib import Path
from typing import Optional

_vlm_processor  = None
_vlm_model      = None
_vlm_model_id   = None
_vlm_lock       = threading.Lock()

_DEFAULT_PROMPT = (
    "Describe what is visible on this computer screen. "
    "Include the main application, open windows or documents, visible text headings, "
    "active content, and any notable UI elements. Be concise and factual. 2–4 sentences."
)

_TEXT_PROMPT = (
    "Extract all visible text from this screen screenshot in reading order "
    "(top to bottom, left to right). Include headings, body text, labels, "
    "button text, and any other readable content. Separate sections with newlines."
)


def _get_vlm(model_id: str):
    global _vlm_processor, _vlm_model, _vlm_model_id
    if _vlm_model is not None and _vlm_model_id == model_id:
        return _vlm_processor, _vlm_model
    with _vlm_lock:
        if _vlm_model is not None and _vlm_model_id == model_id:
            return _vlm_processor, _vlm_model
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq
        device = "cuda" if _cuda_ok() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32
        proc   = AutoProcessor.from_pretrained(model_id)
        mdl    = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            mdl = mdl.to("cpu")
        _vlm_processor = proc
        _vlm_model     = mdl
        _vlm_model_id  = model_id
    return _vlm_processor, _vlm_model


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def handle(
    monitor:        int            = 1,
    region:         Optional[dict] = None,
    vlm_model_id:   str            = "HuggingFaceTB/SmolVLM-500M-Instruct",
    vlm_max_tokens: int            = 256,
    vlm_prompt:     str            = "",
    captures_dir:   str            = "./captures",
    jpeg_quality:   int            = 75,
    extract_text:   bool           = False,
) -> dict:
    """
    Capture a screenshot and describe it with SmolVLM.
    """
    import mss
    from PIL import Image

    monitor      = int(os.environ.get("HUGINN_SCREEN_MONITOR", monitor))
    vlm_model_id = os.environ.get("HUGINN_VLM_MODEL",     vlm_model_id)
    captures_dir = os.environ.get("HUGINN_CAPTURES_DIR",  captures_dir)

    if extract_text:
        prompt = _TEXT_PROMPT
    else:
        prompt = vlm_prompt or _DEFAULT_PROMPT

    # ── Screenshot ───────────────────────────────────────────────────────────
    try:
        with mss.mss() as sct:
            if region:
                grab_target = {
                    "top":    region.get("top",    0),
                    "left":   region.get("left",   0),
                    "width":  region.get("width",  800),
                    "height": region.get("height", 600),
                }
            else:
                mons = sct.monitors   # [0] = all combined, [1] = primary
                idx  = min(monitor, len(mons) - 1)
                grab_target = mons[idx]

            shot = sct.grab(grab_target)
            pil  = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    except Exception as e:
        return {"description": f"[screenshot failed: {e}]",
                "screenshot_path": "", "width": 0, "height": 0,
                "monitor": monitor, "vlm_ms": 0}

    w, h = pil.size

    # ── Save ─────────────────────────────────────────────────────────────────
    path = Path(captures_dir)
    path.mkdir(parents=True, exist_ok=True)
    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    save_path = path / f"screen_{ts}.jpg"
    pil.save(str(save_path), "JPEG", quality=jpeg_quality)

    # ── VLM ──────────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        import torch
        proc, mdl = _get_vlm(vlm_model_id)
        messages  = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        chat_prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs      = proc(text=chat_prompt, images=[pil], return_tensors="pt")
        inputs      = {k: v.to(mdl.device) for k, v in inputs.items()}

        with torch.no_grad():
            gen = mdl.generate(**inputs, max_new_tokens=vlm_max_tokens,
                                do_sample=False)

        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], gen)]
        description = proc.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
    except Exception as e:
        description = f"[VLM error: {e}]"

    vlm_ms = int((time.monotonic() - t0) * 1000)

    return {
        "description":     description,
        "screenshot_path": str(save_path.resolve()),
        "width":           w,
        "height":          h,
        "monitor":         monitor,
        "vlm_ms":          vlm_ms,
    }
