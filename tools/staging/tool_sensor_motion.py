"""
tool.sensor.motion.v1 — Webcam motion/presence sensor for AIDER.

Runs as a live service tool. Captures frames from a webcam, computes
frame-to-frame difference, and writes STM sensor events when motion is
detected or clears. No ML required — pure frame differencing.

HUGINN_MANIFEST
tool_id:            tool.sensor.motion.v1
title:              Motion Sensor
capability_summary: >
  Webcam-based motion and presence detection. Captures frames at a
  configurable rate, detects motion via frame differencing, and writes
  motion_detected / motion_cleared events to STM. Works on Pi Camera
  or any USB webcam. No GPU or ML model required.
polarity:           read
permission_scope:   [camera, vision.input]
mode:               service
direction:          input
subscriptions:      []
inputs:             {}
outputs:
  event:        {type: string, description: "motion_detected | motion_cleared | presence_check"}
  motion_level: {type: number, description: "Mean pixel difference 0-255 (normalised to 0-1)"}
  region:       {type: string, description: "Frame region with most motion (tl|tr|bl|br|center|full)"}
  confidence:   {type: number, description: "Detection confidence 0-1"}
states:
  camera_index:
    default:     "0"
    type:        string
    description: Camera device index (0 = first camera, 1 = second, etc.)
  fps:
    default:     "2.0"
    type:        string
    description: Capture rate in frames per second (lower = less CPU)
  motion_threshold:
    default:     "0.015"
    type:        string
    description: >
      Mean pixel difference (0-1) above which motion is declared.
      Lower = more sensitive. 0.015 works well for typical rooms.
  clear_threshold:
    default:     "0.005"
    type:        string
    description: >
      Mean difference below which motion is declared cleared.
      Hysteresis: must be lower than motion_threshold.
  min_motion_secs:
    default:     "0.3"
    type:        string
    description: Minimum duration of motion before firing detected event
  min_clear_secs:
    default:     "2.0"
    type:        string
    description: Minimum duration of stillness before firing cleared event
  resize_width:
    default:     "320"
    type:        string
    description: Resize frames to this width before differencing (lower = faster)
dependencies:
  - opencv-python-headless>=4.8
  - numpy>=1.24
perception_capable: true
handler:            handle
END_MANIFEST

Webcam motion sensor — live service daemon.

Algorithm
---------
Frame differencing is simple and effective for presence/motion detection
without a neural network:

  1. Capture frame at configured FPS
  2. Convert to grayscale, apply Gaussian blur (reduces noise)
  3. Compute absolute diff against the previous frame
  4. Normalise mean diff to 0-1 range
  5. Apply hysteresis thresholds for motion_detected / motion_cleared
  6. Identify the most active quadrant for approximate region localisation

Events written to STM:
  motion_detected — motion level crossed threshold and held for min_motion_secs
  motion_cleared  — level dropped below clear_threshold for min_clear_secs

This is a perception pipeline tool — it writes to STM via the registered
sensor callback, which routes through Perception Manager's write_sensor_event().
Exilis triages every event; consecutive identical motion_detected events
with the same level (rounded to 1dp) are noise-collapsed by Logos before
consolidation.

Pi Camera guidance
------------------
  Pi Camera (v1/v2/v3) via libcamera works with OpenCV when using the
  correct backend:
    cv2.VideoCapture(0, cv2.CAP_V4L2)   # most cases
    cv2.VideoCapture('/dev/video0')       # explicit device path

  Set FPS to 1.0 on Pi Zero, 2.0-4.0 on Pi 4.
  Set resize_width=160 on Pi Zero for minimal CPU.

Calibration
-----------
  Run with a fixed camera pointing at the room for ~30s before relying
  on thresholds. The tool logs the rolling mean_diff to stderr in debug
  mode (set ANIMA_MOTION_DEBUG=1) to help calibrate motion_threshold.
"""

from __future__ import annotations

import time
import threading
from typing import Any, Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_sensor_cb:   Optional[Callable[[str, dict, float], None]] = None
_htm_ref      = None
_running      = False
_thread:      Optional[threading.Thread] = None


# ---------------------------------------------------------------------------
# HUGINN_MANIFEST hooks
# ---------------------------------------------------------------------------

def start(config: dict) -> bool:
    global _running, _thread, _htm_ref
    _htm_ref = config.get("_htm")
    _running = True
    _thread  = threading.Thread(
        target=_loop, daemon=True, name="AnimaMotionSensor"
    )
    _thread.start()
    return True


def stop() -> bool:
    global _running
    _running = False
    return True


def handle(event: dict) -> None:
    """ActuationBus handler — not used for input-only sensors."""
    pass


# ---------------------------------------------------------------------------
# ToolManager wiring
# ---------------------------------------------------------------------------

def register_sensor_callback(fn: Callable[[str, dict, float], None]) -> None:
    """
    Register the function to call when a sensor event fires.

    Signature: fn(event_name: str, data: dict, confidence: float)
      event_name — "motion_detected" | "motion_cleared"
      data       — {motion_level, region, camera_index, source}
      confidence — detection confidence 0-1

    Wired by ToolManager to perception.write_sensor_event() after install.
    """
    global _sensor_cb
    _sensor_cb = fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_state(key: str, default: str) -> str:
    if _htm_ref is not None:
        val = _htm_ref.states.get(f"tool.sensor.motion.v1.{key}", "")
        if val:
            return str(val)
    return default


def _is_debug() -> bool:
    return _read_state("debug", "false").lower() in ("true", "1", "yes")


def _detect_region(diff_frame: np.ndarray) -> str:
    """Return the quadrant with most motion, or 'full' if uniform."""
    h, w  = diff_frame.shape
    mh, mw = h // 2, w // 2
    quadrants = {
        "tl": diff_frame[:mh, :mw],
        "tr": diff_frame[:mh, mw:],
        "bl": diff_frame[mh:, :mw],
        "br": diff_frame[mh:, mw:],
    }
    means = {k: float(v.mean()) for k, v in quadrants.items()}
    top   = max(means, key=means.get)
    # If the winning quadrant isn't 30% stronger than average, call it 'full'
    avg   = sum(means.values()) / 4
    if avg < 0.01 or means[top] < avg * 1.3:
        return "full"
    return top


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------

def _loop() -> None:
    try:
        import cv2
    except ImportError:
        import sys
        print("[motion] OpenCV not installed. pip install opencv-python-headless",
              file=sys.stderr)
        return

    import sys

    prev_gray   = None
    motion_on   = False          # current state
    motion_start = 0.0           # when current motion level started
    clear_start  = 0.0           # when current clear period started

    while _running:
        try:
            cam_idx      = int(_read_state("camera_index",    "0"))
            fps          = float(_read_state("fps",            "2.0"))
            motion_thr   = float(_read_state("motion_threshold", "0.015"))
            clear_thr    = float(_read_state("clear_threshold",  "0.005"))
            min_motion_s = float(_read_state("min_motion_secs",  "0.3"))
            min_clear_s  = float(_read_state("min_clear_secs",   "2.0"))
            resize_w     = int(_read_state("resize_width",    "320"))

            interval = 1.0 / max(fps, 0.1)

            cap = cv2.VideoCapture(cam_idx)
            if not cap.isOpened():
                if _is_debug():
                    print(f"[motion] Cannot open camera {cam_idx}", file=sys.stderr)
                time.sleep(5.0)
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resize_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(resize_w * 0.75))

            frame_t = time.monotonic()

            while _running:
                now   = time.monotonic()
                sleep = interval - (now - frame_t)
                if sleep > 0:
                    time.sleep(sleep)
                frame_t = time.monotonic()

                ret, frame = cap.read()
                if not ret:
                    break

                # Convert and blur
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray  = cv2.GaussianBlur(gray, (7, 7), 0)
                h, w  = gray.shape

                # Resize for consistent processing
                if w != resize_w:
                    scale = resize_w / w
                    gray  = cv2.resize(gray, (resize_w, int(h * scale)))

                if prev_gray is None or prev_gray.shape != gray.shape:
                    prev_gray = gray
                    continue

                # Frame diff
                diff       = cv2.absdiff(prev_gray, gray)
                prev_gray  = gray
                mean_diff  = float(diff.mean()) / 255.0   # normalise 0-1
                level      = round(mean_diff, 4)

                if _is_debug():
                    print(f"[motion] diff={level:.4f}  state={'ON' if motion_on else 'off'}",
                          file=sys.stderr)

                now = time.monotonic()

                if not motion_on:
                    if level >= motion_thr:
                        if motion_start == 0.0:
                            motion_start = now
                        elif now - motion_start >= min_motion_s:
                            # Confirmed motion
                            motion_on   = True
                            clear_start = 0.0
                            region      = _detect_region(diff)
                            confidence  = min(1.0, level / (motion_thr * 3))
                            _fire("motion_detected", {
                                "motion_level":  level,
                                "region":        region,
                                "camera_index":  cam_idx,
                                "source":        "tool.sensor.motion.v1",
                            }, round(confidence, 3))
                    else:
                        motion_start = 0.0
                else:
                    if level <= clear_thr:
                        if clear_start == 0.0:
                            clear_start = now
                        elif now - clear_start >= min_clear_s:
                            # Confirmed clear
                            motion_on    = False
                            motion_start = 0.0
                            _fire("motion_cleared", {
                                "motion_level":  level,
                                "region":        "full",
                                "camera_index":  cam_idx,
                                "source":        "tool.sensor.motion.v1",
                            }, 0.95)
                    else:
                        clear_start = 0.0

            cap.release()

        except KeyboardInterrupt:
            break
        except Exception as e:
            import sys
            print(f"[motion] error: {e}", file=sys.stderr)
            time.sleep(2.0)


def _fire(event_name: str, data: dict, confidence: float) -> None:
    if _sensor_cb is not None:
        _sensor_cb(event_name, data, confidence)
    elif _is_debug():
        import sys
        print(f"[motion] EVENT: {event_name}  {data}  conf={confidence}",
              file=sys.stderr)
