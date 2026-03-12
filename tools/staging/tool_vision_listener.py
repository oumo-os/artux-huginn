"""
HUGINN_MANIFEST
tool_id:            tool.vision.listener.v1
title:              Vision Listener — YOLO + VLM Scene Understanding
capability_summary: >
  Capture a webcam frame and run fast object detection (YOLO) and/or a
  rich scene description (SmolVLM). In hybrid mode, YOLO runs every call
  and the VLM fires only when YOLO detects meaningful change — saving GPU/CPU.
  Returns detected objects with bounding boxes, a scene description, and the
  saved frame path for LTM source attachment.
polarity:           read
permission_scope:   [camera]
inputs:
  webcam_index:     {type: integer, default: 0}
  mode:             {type: string,  default: "hybrid", description: "yolo | vlm | hybrid"}
  yolo_model:       {type: string,  default: "yolo11n.pt", description: "YOLO model: yolo11n/s/m/x.pt or yolov8n.pt etc"}
  yolo_conf:        {type: number,  default: 0.35, description: "YOLO confidence threshold 0-1"}
  vlm_model_id:     {type: string,  default: "HuggingFaceTB/SmolVLM-500M-Instruct"}
  vlm_max_tokens:   {type: integer, default: 200}
  vlm_prompt:       {type: string,  default: "", description: "Override default VLM prompt"}
  change_threshold: {type: number,  default: 0.3,  description: "IoU-based scene change score to trigger VLM in hybrid mode (0-1). Lower = more sensitive."}
  captures_dir:     {type: string,  default: "./captures"}
  jpeg_quality:     {type: integer, default: 80}
outputs:
  objects:          {type: array,   description: "[{label, confidence, bbox:[x1,y1,x2,y2], centre:[x,y]}]"}
  object_count:     {type: integer}
  object_labels:    {type: array,   description: "Unique detected class names"}
  description:      {type: string,  description: "VLM scene description (empty if VLM skipped in hybrid mode)"}
  vlm_ran:          {type: boolean, description: "True if VLM was invoked this call"}
  change_score:     {type: number,  description: "Scene change score vs previous call (0-1). 1 = fully changed."}
  frame_path:       {type: string}
  width:            {type: integer}
  height:           {type: integer}
  yolo_ms:          {type: integer}
  vlm_ms:           {type: integer}
dependencies:
  - ultralytics>=8.3
  - opencv-python>=4.8
  - numpy>=1.24
  - transformers>=4.40
  - torch>=2.1
  - Pillow>=10.0
  - accelerate>=0.27
perception_capable: true
handler:            handle
END_MANIFEST

YOLO + VLM hybrid vision pipeline.

YOLO runs every call (~10–50 ms on CPU, <10 ms on GPU).
VLM runs only when scene change exceeds change_threshold in hybrid mode,
or every call in vlm-only mode.

Change detection:
  The previous call's set of (label, bbox_centre) pairs is stored in module
  state. Change score = 1 - (objects retained / max(prev_count, curr_count)).
  A score > change_threshold triggers the VLM. New entities entering the
  frame or existing ones moving significantly both raise the score.

YOLO model selection (least-dependency principle):
  yolo11n.pt  — ~6 MB,   fastest, COCO 80 classes       ← default
  yolo11s.pt  — ~22 MB,  balanced
  yolo11m.pt  — ~52 MB,  high accuracy
  yolo11x.pt  — ~110 MB, maximum accuracy
  Models download automatically from ultralytics on first call.

VLM model selection:
  SmolVLM-500M-Instruct — ~1 GB, CPU-friendly           ← default
  SmolVLM-Instruct      — ~2 GB, higher quality
  Set HUGINN_VLM_MODEL env var to override.

Object class filtering:
  Set HUGINN_VISION_CLASSES to a comma-separated list to restrict YOLO output.
  Example: HUGINN_VISION_CLASSES=person,cat,dog
  Empty = all 80 COCO classes.

Environment overrides:
  HUGINN_VISION_MODE=hybrid
  HUGINN_YOLO_MODEL=yolo11n.pt
  HUGINN_YOLO_CONF=0.35
  HUGINN_VLM_MODEL=HuggingFaceTB/SmolVLM-500M-Instruct
  HUGINN_VISION_CLASSES=
  HUGINN_WEBCAM_INDEX=0
  HUGINN_CAPTURES_DIR=./captures
"""

from __future__ import annotations

import datetime
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ─── Module-level singletons ─────────────────────────────────────────────────

_yolo_model      = None
_yolo_model_name = None
_vlm_processor   = None
_vlm_model       = None
_vlm_model_id    = None
_vlm_lock        = threading.Lock()

# State for change detection across calls
_prev_objects: list[dict] = []

_DEFAULT_VLM_PROMPT = (
    "Describe this scene precisely and concisely. Include all visible people, "
    "objects, their positions (left/right/centre/foreground/background), "
    "colours, and any notable activity. Be factual. 2–3 sentences."
)


# ─── Lazy loaders ────────────────────────────────────────────────────────────

def _get_yolo(model_name: str):
    global _yolo_model, _yolo_model_name
    if _yolo_model is None or _yolo_model_name != model_name:
        from ultralytics import YOLO
        _yolo_model      = YOLO(model_name)
        _yolo_model_name = model_name
    return _yolo_model


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


# ─── Tool handler ─────────────────────────────────────────────────────────────

def handle(
    webcam_index:     int   = 0,
    mode:             str   = "hybrid",
    yolo_model:       str   = "yolo11n.pt",
    yolo_conf:        float = 0.35,
    vlm_model_id:     str   = "HuggingFaceTB/SmolVLM-500M-Instruct",
    vlm_max_tokens:   int   = 200,
    vlm_prompt:       str   = "",
    change_threshold: float = 0.3,
    captures_dir:     str   = "./captures",
    jpeg_quality:     int   = 80,
) -> dict:
    import cv2

    # Env overrides
    mode         = os.environ.get("HUGINN_VISION_MODE",    mode)
    yolo_model   = os.environ.get("HUGINN_YOLO_MODEL",     yolo_model)
    yolo_conf    = float(os.environ.get("HUGINN_YOLO_CONF", yolo_conf))
    vlm_model_id = os.environ.get("HUGINN_VLM_MODEL",      vlm_model_id)
    webcam_index = int(os.environ.get("HUGINN_WEBCAM_INDEX", webcam_index))
    captures_dir = os.environ.get("HUGINN_CAPTURES_DIR",   captures_dir)
    filter_str   = os.environ.get("HUGINN_VISION_CLASSES", "")
    class_filter = {c.strip().lower() for c in filter_str.split(",") if c.strip()}
    vlm_prompt   = vlm_prompt or _DEFAULT_VLM_PROMPT

    _empty = lambda reason: {
        "objects": [], "object_count": 0, "object_labels": [],
        "description": "", "vlm_ran": False, "change_score": 0.0,
        "frame_path": "", "width": 0, "height": 0,
        "yolo_ms": 0, "vlm_ms": 0,
    }

    # ── Capture frame ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(webcam_index)
    try:
        if not cap.isOpened():
            return _empty("webcam not available")
        ret, frame = cap.read()
    finally:
        cap.release()

    if not ret or frame is None:
        return _empty("frame capture failed")

    h, w = frame.shape[:2]

    # ── Save frame ───────────────────────────────────────────────────────────
    path = Path(captures_dir)
    path.mkdir(parents=True, exist_ok=True)
    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    save_path = path / f"vision_{ts}.jpg"
    cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    # ── YOLO ─────────────────────────────────────────────────────────────────
    objects  = []
    yolo_ms  = 0
    if mode in ("yolo", "hybrid"):
        t0 = time.monotonic()
        objects = _run_yolo(frame, yolo_model, yolo_conf, class_filter)
        yolo_ms = int((time.monotonic() - t0) * 1000)

    # ── Change detection ─────────────────────────────────────────────────────
    change_score = _compute_change(objects)
    _update_prev(objects)

    # ── VLM ──────────────────────────────────────────────────────────────────
    description = ""
    vlm_ms      = 0
    vlm_ran     = False

    run_vlm = (
        mode == "vlm"
        or (mode == "hybrid" and change_score >= change_threshold)
    )
    if run_vlm:
        t0 = time.monotonic()
        try:
            description = _run_vlm(frame, vlm_model_id, vlm_prompt,
                                    vlm_max_tokens)
            vlm_ran = True
        except Exception as e:
            description = f"[VLM error: {e}]"
        vlm_ms = int((time.monotonic() - t0) * 1000)

    labels = sorted({o["label"] for o in objects})

    return {
        "objects":       objects,
        "object_count":  len(objects),
        "object_labels": labels,
        "description":   description,
        "vlm_ran":       vlm_ran,
        "change_score":  round(change_score, 3),
        "frame_path":    str(save_path.resolve()),
        "width":         w,
        "height":        h,
        "yolo_ms":       yolo_ms,
        "vlm_ms":        vlm_ms,
    }


# ─── YOLO runner ─────────────────────────────────────────────────────────────

def _run_yolo(frame, model_name: str, conf: float, class_filter: set) -> list[dict]:
    yolo = _get_yolo(model_name)
    results = yolo(frame, conf=conf, verbose=False)
    objects = []
    for r in results:
        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            if class_filter and label.lower() not in class_filter:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            objects.append({
                "label":      label,
                "confidence": round(float(box.conf[0]), 3),
                "bbox":       [round(x1), round(y1), round(x2), round(y2)],
                "centre":     [round(cx), round(cy)],
            })
    return objects


# ─── VLM runner ──────────────────────────────────────────────────────────────

def _run_vlm(frame, model_id: str, prompt: str, max_tokens: int) -> str:
    import cv2
    import torch
    from PIL import Image

    proc, mdl = _get_vlm(model_id)
    rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil        = Image.fromarray(rgb)

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    chat_prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs      = proc(text=chat_prompt, images=[pil], return_tensors="pt")
    inputs      = {k: v.to(mdl.device) for k, v in inputs.items()}

    with torch.no_grad():
        gen = mdl.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], gen)]
    return proc.batch_decode(trimmed, skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)[0].strip()


# ─── Change detection ─────────────────────────────────────────────────────────

def _compute_change(current: list[dict]) -> float:
    """
    Compute scene change score vs previous call.
    Score = 1 - (retained_objects / max(prev, curr)).
    An object is "retained" if the same label appears at a similar centre position.
    """
    global _prev_objects
    if not _prev_objects or not current:
        return 1.0 if current or _prev_objects else 0.0

    matched = 0
    DIST_THRESHOLD = 80   # pixel distance for "same object"

    for prev_obj in _prev_objects:
        for cur_obj in current:
            if prev_obj["label"] != cur_obj["label"]:
                continue
            dx = prev_obj["centre"][0] - cur_obj["centre"][0]
            dy = prev_obj["centre"][1] - cur_obj["centre"][1]
            if (dx ** 2 + dy ** 2) ** 0.5 < DIST_THRESHOLD:
                matched += 1
                break

    denominator = max(len(_prev_objects), len(current))
    return 1.0 - (matched / denominator)


def _update_prev(objects: list[dict]):
    global _prev_objects
    _prev_objects = objects
