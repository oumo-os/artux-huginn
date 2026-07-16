"""
tool.capture.camera.v1 — Camera frame capture for Huginn.

Captures frames from a connected camera and pushes CaptureFrames into a
queue that PerceptionManager dispatches to processor tools (face identity,
YOLO detection, scene description, OCR, etc.).

No processing happens here. The tool just captures and queues.

HUGINN_MANIFEST
tool_id:            tool.capture.camera.v1
title:              Camera Capture
capability_summary: >
  Continuous camera capture. Produces video_frame CaptureFrames dispatched
  by PerceptionManager to face identity, scene understanding, OCR, and any
  other processor tool that consumes video_frame. Does not write to STM.
  opencv-python is the only hardware dependency in the video pipeline.
polarity:           read
permission_scope:   [camera]
mode:               service
direction:          capture
emits:
  - video_frame
inputs: {}
outputs: {}
states:
  device_index:
    default: 0
    type: integer
    description: Camera device index (0 = first/default camera)
  capture_interval_s:
    default: 2.0
    type: float
    description: Seconds between captured frames (reduce for faster detection)
  resolution_width:
    default: 640
    type: integer
    description: Capture width in pixels
  resolution_height:
    default: 480
    type: integer
    description: Capture height in pixels
  enabled:
    default: true
    type: boolean
    description: Set to false to pause capture without stopping the tool
dependencies:
  opencv-python
  numpy
END_MANIFEST
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

_stop_event = threading.Event()
_q:         queue.Queue = queue.Queue(maxsize=16)
_htm_ref    = None
_cap        = None   # cv2.VideoCapture instance


# ---------------------------------------------------------------------------
# Service interface
# ---------------------------------------------------------------------------

def start(config: dict, _htm=None) -> None:
    global _stop_event, _htm_ref, _cap
    _htm_ref = _htm
    _stop_event.clear()
    threading.Thread(
        target=_capture_loop, daemon=True, name="CamCapture"
    ).start()


def stop() -> None:
    _stop_event.set()
    _q.put(None)
    if _cap is not None:
        try:
            _cap.release()
        except Exception:
            pass


def get_queue() -> queue.Queue:
    return _q


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------

def _capture_loop() -> None:
    global _cap
    try:
        import cv2
    except ImportError:
        return

    device_index = _read("device_index", 0, int)
    w = _read("resolution_width",  640, int)
    h = _read("resolution_height", 480, int)

    _cap = cv2.VideoCapture(device_index)
    _cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    if not _cap.isOpened():
        _stop_event.set()
        return

    while not _stop_event.is_set():
        interval = _read("capture_interval_s", 2.0, float)
        enabled  = _read("enabled", True, _bool)

        if not enabled:
            time.sleep(0.5)
            continue

        ret, frame = _cap.read()
        if not ret:
            time.sleep(1.0)
            continue

        try:
            from huginn.runtime.perception import CaptureFrame
        except ImportError:
            class CaptureFrame:  # type: ignore
                def __init__(self, kind, data, meta):
                    self.kind = kind; self.data = data; self.meta = meta

        cf = CaptureFrame(
            kind="video_frame",
            data=frame,   # numpy BGR array
            meta={
                "width":  frame.shape[1],
                "height": frame.shape[0],
                "device": device_index,
            },
        )
        try:
            _q.put_nowait(cf)
        except queue.Full:
            # Drop the oldest frame and add the new one
            try:
                _q.get_nowait()
            except queue.Empty:
                pass
            try:
                _q.put_nowait(cf)
            except queue.Full:
                pass

        time.sleep(max(0.0, interval))

    _cap.release()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(key: str, default: Any, cast) -> Any:
    if _htm_ref is not None:
        v = _htm_ref.states.get(f"tool.capture.camera.v1.{key}")
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
