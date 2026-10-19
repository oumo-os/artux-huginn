"""
tool.actuation.display.v1 — Terminal/console display output for AIDER.

Subscribes to ActuationBus projection events and renders structured
content to the terminal — tables, key-value snapshots, lists, code blocks.
This is the output counterpart to tool.ui.text.v1's input side.

Also outputs to an optional HTML file that auto-refreshes, useful when
AIDER is connected to a monitor (Pi connected to a TV or display).

HUGINN_MANIFEST
tool_id:            tool.actuation.display.v1
title:              Display Output
capability_summary: >
  Renders structured content from AIDER projections to a terminal or
  HTML file. Use for non-speech output — tables, status snapshots,
  schedules, lists, code. Pairs with tool.ui.text.v1 for a complete
  text-based interface. Optionally writes a self-refreshing HTML file
  for display on a connected screen.
polarity:           write
permission_scope:   [display.output]
mode:               service
direction:          output
subscriptions:
  - type: output
    target: display
    complete: full
  - type: output
    target: main
    complete: full
inputs:             {}
outputs:            {}
states:
  mode:
    default:     "terminal"
    type:        string
    description: "terminal | html | both"
  html_path:
    default:     ""
    type:        string
    description: >
      Path for the auto-refreshing HTML output file.
      Default: ~/anima-display.html  (mode=html or mode=both)
  refresh_secs:
    default:     "3"
    type:        string
    description: HTML auto-refresh interval in seconds
  max_history:
    default:     "20"
    type:        string
    description: Number of display events to keep in the HTML history view
  terminal_width:
    default:     "80"
    type:        string
    description: Terminal column width for wrapping
dependencies:       []
perception_capable: false
handler:            handle
END_MANIFEST

Display output tool — live service daemon.

Projection format
-----------------
Sagax sends <projection> blocks with structured JSON. This tool renders
them. Supported projection types:

  {"type": "text",    "content": "..."}
  {"type": "table",   "headers": [...], "rows": [[...], ...]}
  {"type": "list",    "title": "...", "items": ["...", ...]}
  {"type": "kv",      "title": "...", "data": {"key": "value", ...}}
  {"type": "code",    "language": "...", "content": "..."}
  {"type": "status",  "data": {"key": "value", ...}}
  {"type": "clear"}

All projection events (target="display") are handled here.
Speech events (target="main") are printed as [AIDER]: prefix lines
so the terminal reflects both speech and visual output.

HTML output
-----------
The HTML file auto-refreshes every refresh_secs seconds. Each projection
is appended to a history list. Useful for:
  - Pi connected to a TV/monitor as a home dashboard
  - Remote monitoring via a browser on the LAN
  - Development — see the full output history without scrolling

Access on LAN: python3 -m http.server 8080 --directory ~/ then
open http://[pi-ip]:8080/anima-display.html in a browser.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_htm_ref      = None
_running      = False
_thread:      Optional[threading.Thread] = None
_history:     list[dict] = []   # [(ts, rendered)]
_history_lock = threading.Lock()


# ---------------------------------------------------------------------------
# HUGINN_MANIFEST hooks
# ---------------------------------------------------------------------------

def start(config: dict) -> bool:
    global _running, _htm_ref
    _htm_ref = config.get("_htm")
    _running = True
    return True


def stop() -> bool:
    global _running
    _running = False
    return True


def handle(event: dict) -> None:
    """
    Called by ActuationManager for every matching bus event.
    event shape: ActuationEvent fields as dict
      type, target, complete, text, session_id, meta
    """
    if not _running:
        return

    target   = event.get("target", "")
    complete = event.get("complete", "full")
    text     = event.get("text", "")

    if complete != "full":
        return   # only render complete events

    if target == "display":
        _render_projection(text)
    elif target == "main":
        _render_speech(text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_state(key: str, default: str) -> str:
    if _htm_ref is not None:
        val = _htm_ref.states.get(f"tool.actuation.display.v1.{key}", "")
        if val:
            return str(val)
    return default


def _render_speech(text: str) -> None:
    """Print speech output to terminal with a clear prefix."""
    if not text.strip():
        return
    width = int(_read_state("terminal_width", "80"))
    mode  = _read_state("mode", "terminal")
    ts    = datetime.now().strftime("%H:%M:%S")

    lines = _wrap(f"[AIDER] {text}", width)
    rendered = "\n".join(lines)

    if mode in ("terminal", "both"):
        print(f"\033[36m{rendered}\033[0m")   # cyan for speech

    _append_history(ts, "speech", rendered, mode)


def _render_projection(text: str) -> None:
    """Parse and render a projection block."""
    if not text.strip():
        return

    width = int(_read_state("terminal_width", "80"))
    mode  = _read_state("mode", "terminal")
    ts    = datetime.now().strftime("%H:%M:%S")

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Raw text projection
        data = {"type": "text", "content": text}

    ptype    = data.get("type", "text")
    rendered = _render_type(ptype, data, width)

    if mode in ("terminal", "both"):
        print(rendered)

    _append_history(ts, ptype, rendered, mode)


def _render_type(ptype: str, data: dict, width: int) -> str:
    if ptype == "clear":
        if os.name == "nt":
            os.system("cls")
        else:
            print("\033[2J\033[H", end="")
        return ""

    if ptype == "text":
        return "\n".join(_wrap(data.get("content", ""), width))

    if ptype in ("status", "kv"):
        title = data.get("title", "Status")
        kv    = data.get("data", {})
        lines = [f"── {title} {'─' * max(0, width - len(title) - 4)}"]
        for k, v in kv.items():
            key_str = f"  {k}:"
            val_str = str(v)
            lines.append(f"{key_str:<24}{val_str}")
        lines.append("─" * width)
        return "\n".join(lines)

    if ptype == "list":
        title  = data.get("title", "")
        items  = data.get("items", [])
        lines  = ([f"── {title}"] if title else [])
        lines += [f"  • {item}" for item in items]
        return "\n".join(lines)

    if ptype == "table":
        headers = data.get("headers", [])
        rows    = data.get("rows", [])
        return _render_table(headers, rows, width)

    if ptype == "code":
        lang    = data.get("language", "")
        content = data.get("content", "")
        sep     = "─" * width
        label   = f" {lang}" if lang else ""
        return f"┌{sep}┐\n│{label}\n" + "\n".join(f"│ {l}" for l in content.splitlines()) + f"\n└{sep}┘"

    # Fallback
    return json.dumps(data, indent=2)


def _render_table(headers: list, rows: list, width: int) -> str:
    if not headers and not rows:
        return "(empty table)"
    all_rows  = [headers] + rows if headers else rows
    n_cols    = max(len(r) for r in all_rows)
    all_rows  = [list(r) + [""] * (n_cols - len(r)) for r in all_rows]
    col_widths = [max(len(str(r[i])) for r in all_rows) for i in range(n_cols)]

    # Cap column widths to fit terminal
    total     = sum(col_widths) + n_cols * 3
    if total > width and n_cols:
        scale     = width / total
        col_widths = [max(4, int(w * scale)) for w in col_widths]

    sep  = "─┼─".join("─" * w for w in col_widths)
    sep  = f"─{sep}─"
    lines = []

    for i, row in enumerate(all_rows):
        cells  = [str(row[j])[:col_widths[j]].ljust(col_widths[j]) for j in range(n_cols)]
        lines.append(" │ ".join(cells))
        if i == 0 and headers:
            lines.append(sep)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    """Simple word-wrap."""
    if not text:
        return [""]
    result = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            result.append("")
            continue
        words, line = paragraph.split(), ""
        for word in words:
            if len(line) + len(word) + 1 > width and line:
                result.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            result.append(line)
    return result


def _append_history(ts: str, ptype: str, rendered: str, mode: str) -> None:
    """Append to history and optionally write HTML."""
    max_hist = int(_read_state("max_history", "20"))
    with _history_lock:
        _history.append({"ts": ts, "type": ptype, "text": rendered})
        while len(_history) > max_hist:
            _history.pop(0)
        hist_copy = list(_history)

    if mode in ("html", "both"):
        _write_html(hist_copy)


def _write_html(history: list[dict]) -> None:
    """Write a self-refreshing HTML display file."""
    refresh   = int(_read_state("refresh_secs", "3"))
    html_path = _read_state("html_path", "")
    if not html_path:
        html_path = str(Path.home() / "anima-display.html")

    items_html = ""
    for entry in reversed(history):
        rendered = entry["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        cls  = "speech" if entry["type"] == "speech" else "block"
        items_html += (
            f'<div class="entry {cls}">'
            f'<span class="ts">{entry["ts"]}</span>'
            f'<pre>{rendered}</pre></div>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="{refresh}">
<title>AIDER Display</title>
<style>
  body{{background:#08080a;color:#e8e4dd;font-family:'Cascadia Code','Fira Code',monospace;font-size:13px;padding:20px;margin:0}}
  .entry{{border-bottom:1px solid #1e1e27;padding:8px 0;margin:4px 0}}
  .entry.speech pre{{color:#5b9bd5}}
  .ts{{color:#4a4845;font-size:10px;display:block;margin-bottom:3px}}
  pre{{margin:0;white-space:pre-wrap;word-break:break-word}}
  h1{{color:#c8a45a;font-size:14px;letter-spacing:.1em;border-bottom:1px solid #1e1e27;padding-bottom:8px}}
</style>
</head>
<body>
<h1>ANIMA · AIDER · {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</h1>
{items_html}
</body>
</html>"""

    try:
        Path(html_path).write_text(html, encoding="utf-8")
    except Exception:
        pass
