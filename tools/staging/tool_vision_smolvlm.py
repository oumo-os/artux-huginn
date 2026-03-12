"""
HUGINN_MANIFEST
tool_id:            tool.vision.smolvlm.v1
title:              Vision — SmolVLM Webcam Description
capability_summary: >
  Capture a webcam frame and produce a dense natural-language scene description
  using SmolVLM running in-process. Fully offline. Saves the frame to disk and
  returns the description and file path for LTM source attachment.
  Use as a perception pipeline step for continuous visual awareness.
polarity:           read
permission_scope:   [camera]
inputs:
  webcam_index:     {type: integer, default: 0,    description: "OpenCV device index"}
  model_id:         {type: string,  default: "HuggingFaceTB/SmolVLM-Instruct"}
  captures_dir:     {type: string,  default: "./captures", description: "Directory for saved frames"}
  jpeg_quality:     {type: integer, default: 85}
  max_new_tokens:   {type: integer, default: 256}
  prompt:           {type: string,  default: "",   description: "Override default vision prompt"}
outputs:
  description:      {type: string,  description: "Scene description from SmolVLM"}
  frame_path:       {type: string,  description: "Saved JPEG path for LTM source attachment"}
  width:            {type: integer}
  height:           {type: integer}
  duration_ms:      {type: integer, description: "VLM inference time"}
dependencies:
  - transformers>=4.40
  - accelerate>=0.27
  - Pillow>=10.0
  - torch>=2.1
  - opencv-python>=4.8
  - numpy>=1.24
perception_capable: true
handler:            handle
END_MANIFEST

SmolVLM in-process webcam vision pipeline step.

SmolVLM-Instruct (2B) and SmolVLM-500M-Instruct are both supported.
The 500M variant runs on CPU-only machines; the 2B variant benefits from
a GPU but works on CPU. Both download automatically from HuggingFace.

The saved JPEG path is returned so Sagax can call record_source() to
attach it to any LTM entry that references the scene.

Configuration via environment variables:
  HUGINN_VLM_MODEL=HuggingFaceTB/SmolVLM-Instruct
  HUGINN_VLM_500M=1                   (shortcut to use 500M variant)
  HUGINN_CAPTURES_DIR=./captures
  HUGINN_WEBCAM_INDEX=0
"""

from __future__ import annotations

import os
import time
import datetime
import threading
from pathlib import Path
from typing import Optional

# Module-level singletons
_processor  = None
_model      = None
_model_id   = None
_lock       = threading.Lock()

_DEFAULT_PROMPT = (
    "Describe this webcam frame precisely and concisely. "
    "Include all visible objects and their exact positions "
    "(left / right / centre / foreground / background / on top of / next to), "
    "colours, any people and what they are doing, spatial relationships "
    "between objects, and anything unusual or notable. "
    "Be factual and specific. 2–4 sentences."
)


def _ensure_model(model_id: str):
    """Lazy-load SmolVLM. Thread-safe."""
    global _processor, _model, _model_id
    if _model is not None and _model_id == model_id:
        return _processor, _model
    with _lock:
        if _model is not None and _model_id == model_id:
            return _processor, _model
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        device = "cuda" if _cuda_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32

        proc  = AutoProcessor.from_pretrained(model_id)
        mdl   = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype = dtype,
            device_map  = "auto" if device == "cuda" else None,
        )
        if device == "cpu":
            mdl = mdl.to("cpu")

        _processor = proc
        _model     = mdl
        _model_id  = model_id
    return _processor, _model


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def handle(
    webcam_index:   int   = 0,
    model_id:       str   = "HuggingFaceTB/SmolVLM-Instruct",
    captures_dir:   str   = "./captures",
    jpeg_quality:   int   = 85,
    max_new_tokens: int   = 256,
    prompt:         str   = "",
) -> dict:
    """
    Capture one webcam frame, describe it with SmolVLM, save the JPEG.

    Returns description, frame_path, dimensions, and inference time.
    Empty description means the capture failed or the frame was blank.
    """
    import cv2
    import torch
    from PIL import Image
    import numpy as np

    # Env overrides
    if os.environ.get("HUGINN_VLM_500M"):
        model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
    model_id     = os.environ.get("HUGINN_VLM_MODEL",      model_id)
    webcam_index = int(os.environ.get("HUGINN_WEBCAM_INDEX", webcam_index))
    captures_dir = os.environ.get("HUGINN_CAPTURES_DIR",   captures_dir)
    prompt       = prompt or _DEFAULT_PROMPT

    # Capture frame
    cap = cv2.VideoCapture(webcam_index)
    try:
        if not cap.isOpened():
            return {"description": "", "frame_path": "", "width": 0, "height": 0,
                    "duration_ms": 0}
        ret, frame = cap.read()
    finally:
        cap.release()

    if not ret or frame is None:
        return {"description": "", "frame_path": "", "width": 0, "height": 0,
                "duration_ms": 0}

    h, w = frame.shape[:2]

    # Save JPEG
    captures_path = Path(captures_dir)
    captures_path.mkdir(parents=True, exist_ok=True)
    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = captures_path / f"frame_{ts}.jpg"
    cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    # Convert to PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Load model (lazy)
    proc, mdl = _ensure_model(model_id)

    # Build prompt
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]
    chat_prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=chat_prompt, images=[pil_image], return_tensors="pt")
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

    # Inference
    t0 = time.monotonic()
    with torch.no_grad():
        generated_ids = mdl.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False)
    dur = int((time.monotonic() - t0) * 1000)

    # Decode (strip input tokens)
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs["input_ids"], generated_ids)
    ]
    description = proc.batch_decode(
        trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return {
        "description": description,
        "frame_path":  str(save_path.resolve()),
        "width":       w,
        "height":      h,
        "duration_ms": dur,
    }
