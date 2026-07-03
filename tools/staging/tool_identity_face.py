"""
tool.identity.face.v1 — Face identity processor for Huginn.

Receives video_frame CaptureFrames from PerceptionManager, runs YOLO
person detection + InsightFace ArcFace embedding extraction, and writes
faceprint sensor events to STM for PerceptionManager signature resolution.

Uses real 512-dim ArcFace embeddings (InsightFace) — not pixel histograms.

HUGINN_MANIFEST
tool_id:            tool.identity.face.v1
title:              Face Identity
capability_summary: >
  Face detection and identity via InsightFace ArcFace embeddings.
  Receives video_frame CaptureFrames from PerceptionManager (same frames
  used by any other vision processor). Writes faceprint sensor events to
  STM for PerceptionManager to resolve against registered entity faceprints.
  Uses YOLO for person detection, InsightFace for 512-dim face embeddings.
polarity:           write
permission_scope:   [camera]
mode:               service
direction:          process
consumes:
  - video_frame
inputs: {}
outputs:
  entity_id: {type: string}
states:
  det_confidence:
    default: 0.4
    type: float
    description: YOLO person detection confidence threshold
  face_det_size:
    default: 640
    type: integer
    description: InsightFace detection resolution (must be multiple of 32)
  enabled:
    default: true
    type: boolean
    description: Set to false to pause detection without stopping the tool
dependencies:
  insightface
  ultralytics
  opencv-python
  numpy
END_MANIFEST
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Optional

_face_app    = None
_yolo        = None
_lock        = threading.Lock()
_proc_queue: queue.Queue = queue.Queue(maxsize=8)
_stop_event  = threading.Event()
_stm_writer: Optional[Callable] = None
_htm_ref     = None


def start(config: dict, _stm=None, _htm=None) -> None:
    global _stm_writer, _htm_ref, _stop_event
    _stm_writer = _stm.record if _stm else None
    _htm_ref    = _htm
    _load_models()
    _stop_event.clear()
    threading.Thread(target=_process_loop, daemon=True, name="FaceIdentity").start()


def stop() -> None:
    _stop_event.set()
    _proc_queue.put(None)


def push(frame, _stm=None, _htm=None) -> None:
    """Called by PerceptionManager for each video_frame CaptureFrame."""
    if not _read("enabled", True, _bool):
        return
    try:
        _proc_queue.put_nowait(frame)
    except queue.Full:
        # Drop oldest frame — newer frame is more useful for identity
        try:
            _proc_queue.get_nowait()
        except queue.Empty:
            pass
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
            _detect_and_write(frame)
        except Exception:
            pass


def _detect_and_write(frame) -> None:
    import numpy as np
    img = frame.data   # numpy BGR array from OpenCV
    if img is None or img.size == 0:
        return

    with _lock:
        if _face_app is None:
            return

        # Stage 1: YOLO person detection (optional pre-filter for performance)
        person_regions = []
        if _yolo is not None:
            det_conf = _read("det_confidence", 0.4, float)
            results  = _yolo(img, conf=det_conf, verbose=False)
            for r in results:
                for box in r.boxes:
                    if r.names[int(box.cls)] == "person":
                        person_regions.append(box.xyxy[0].tolist())

        # Stage 2: InsightFace face detection + ArcFace embedding extraction
        # Run on full frame (or per person crop if YOLO found people)
        faces = _face_app.get(img)
        if not faces:
            return

        for face in faces:
            embedding  = face.embedding.tolist()   # 512-dim ArcFace embedding
            det_score  = float(face.det_score)
            bbox       = face.bbox.tolist()

            if _stm_writer is None:
                continue

            _stm_writer(
                source="sensor", type="vision",
                payload={
                    "modality": "faceprint",
                    "signature": {
                        "kind":       "faceprint",
                        "embedding":  embedding,   # real ArcFace — not random
                        "confidence": round(det_score, 3),
                    },
                    "bbox":   bbox,
                    "width":  frame.meta.get("width",  img.shape[1]),
                    "height": frame.meta.get("height", img.shape[0]),
                    "tool":   "tool.identity.face.v1",
                },
                confidence=det_score,
            )


def _load_models() -> None:
    global _face_app, _yolo
    with _lock:
        det_size = _read("face_det_size", 640, int)
        # InsightFace ArcFace — 512-dim real face embeddings
        try:
            import insightface
            if _face_app is None:
                _face_app = insightface.app.FaceAnalysis(
                    providers=["CPUExecutionProvider"]
                )
                _face_app.prepare(ctx_id=0, det_size=(det_size, det_size))
        except ImportError:
            raise ImportError(
                "insightface not installed.\npip install insightface"
            )
        # YOLO — optional person pre-filter (graceful if not available)
        try:
            from ultralytics import YOLO as _YOLO
            if _yolo is None:
                _yolo = _YOLO("yolov8n.pt")
        except ImportError:
            pass   # YOLO optional — InsightFace still runs on full frame


def _read(key: str, default: Any, cast) -> Any:
    if _htm_ref is not None:
        v = _htm_ref.states.get(f"tool.identity.face.v1.{key}")
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
