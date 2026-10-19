"""
tool.desktop.notify.v1 — Send OS-native desktop notifications from AIDER.

Non-speech output channel. Use when Sagax wants to surface information
without speaking aloud — reminders, status updates, background task
completions. Works on Linux (libnotify), macOS (osascript), and Windows
(win10toast or plyer fallback).

HUGINN_MANIFEST
tool_id:            tool.desktop.notify.v1
title:              Desktop Notifications
capability_summary: >
  Send a native OS desktop notification with a title, message, and optional
  urgency level. Use for non-speech alerts — reminders, background task
  completions, status updates, or any notification the user should see
  without AIDER speaking aloud. Works on Linux, macOS, and Windows without
  any API keys or cloud services.
polarity:           write
permission_scope:   [notifications]
mode:               callable
direction:          output
states: {}
inputs:
  title:
    type: string
    default: "AIDER"
    description: "Notification title (headline)."
  message:
    type: string
    description: "Notification body text."
  urgency:
    type: string
    default: "normal"
    description: "low | normal | critical. Controls visual prominence on Linux."
  timeout_ms:
    type: integer
    default: 5000
    description: "Auto-dismiss timeout in milliseconds (0 = persistent). Linux only."
  icon:
    type: string
    default: ""
    description: "Path to icon file. Empty = system default. Linux/macOS only."
  sound:
    type: boolean
    default: false
    description: "Play notification sound (macOS only)."
outputs:
  sent:         {type: boolean, description: "True if notification was sent successfully"}
  platform:     {type: string,  description: "Platform backend used: libnotify | osascript | win10toast | plyer | unsupported"}
  error:        {type: string,  description: "Error message if sent=false"}
dependencies:
  - plyer>=2.1   # optional fallback; platform-native is preferred
END_MANIFEST

Platform backends (tried in order):
  Linux   → libnotify (notify-send) — best integration, supports urgency + icons
  Linux   → plyer fallback
  macOS   → osascript display notification — no dependencies
  Windows → win10toast if installed, else plyer
  Any     → plyer as final fallback

Install on Linux if notify-send is not available:
    sudo apt install libnotify-bin
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Any


def handle(args: dict, _htm=None, _session=None) -> dict:
    title      = str(args.get("title",   "AIDER"))
    message    = str(args.get("message", "")).strip()
    urgency    = str(args.get("urgency", "normal")).strip().lower()
    timeout_ms = int(args.get("timeout_ms", 5000))
    icon       = str(args.get("icon",    "")).strip()
    sound      = bool(args.get("sound",  False))

    if not message:
        return {"error": "message is required"}

    if sys.platform.startswith("linux"):
        return _notify_linux(title, message, urgency, timeout_ms, icon)
    elif sys.platform == "darwin":
        return _notify_macos(title, message, sound)
    elif sys.platform == "win32":
        return _notify_windows(title, message, timeout_ms)
    else:
        return _notify_plyer(title, message, timeout_ms)


def _notify_linux(title, message, urgency, timeout_ms, icon) -> dict:
    if shutil.which("notify-send"):
        cmd = ["notify-send", "-u", urgency, "-t", str(timeout_ms)]
        if icon and icon != "":
            cmd += ["-i", icon]
        cmd += [title, message]
        try:
            subprocess.run(cmd, timeout=5, check=True, capture_output=True)
            return {"sent": True, "platform": "libnotify"}
        except subprocess.SubprocessError as e:
            pass  # fall through to plyer

    return _notify_plyer(title, message, timeout_ms)


def _notify_macos(title, message, sound) -> dict:
    sound_str = "default" if sound else ""
    script = (
        f'display notification "{_esc(message)}" '
        f'with title "{_esc(title)}"'
        + (f' sound name "{sound_str}"' if sound_str else "")
    )
    try:
        subprocess.run(
            ["osascript", "-e", script],
            timeout=5, capture_output=True, check=True
        )
        return {"sent": True, "platform": "osascript"}
    except subprocess.SubprocessError as e:
        return _notify_plyer(title, message, 5000)


def _notify_windows(title, message, timeout_ms) -> dict:
    try:
        from win10toast import ToastNotifier
        t = ToastNotifier()
        t.show_toast(title, message, duration=max(1, timeout_ms // 1000))
        return {"sent": True, "platform": "win10toast"}
    except ImportError:
        pass
    return _notify_plyer(title, message, timeout_ms)


def _notify_plyer(title, message, timeout_ms) -> dict:
    try:
        from plyer import notification
        notification.notify(
            title    = title,
            message  = message,
            timeout  = max(1, timeout_ms // 1000),
        )
        return {"sent": True, "platform": "plyer"}
    except Exception as e:
        return {"sent": False, "platform": "unsupported", "error": str(e)}


def _esc(s: str) -> str:
    return s.replace('"', '\\"').replace("'", "\\'")
