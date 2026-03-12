"""
HUGINN_MANIFEST
tool_id:            tool.notes.v1
title:              Notes
capability_summary: >
  Create, read, list, and search plain-text notes on disk. Use when the user
  asks Artux to remember something specific as a note ("jot that down", "note
  that", "add to my shopping list"), to read back notes, or to search across
  notes. Notes are plain UTF-8 text files in a configured directory.
polarity:           write
permission_scope:   []
inputs:
  action:       {type: string, enum: [append, read, list, search, delete], description: "append | read | list | search | delete"}
  title:        {type: string, default: "",   description: "Note filename (without .txt). Required for append/read/delete. Auto-generated if empty for append."}
  content:      {type: string, default: "",   description: "Text to append to the note. Used with action=append."}
  query:        {type: string, default: "",   description: "Search string. Used with action=search."}
  notes_dir:    {type: string, default: "",   description: "Override notes directory for this call."}
outputs:
  note_id:      {type: string, description: "Note filename (without .txt)"}
  content:      {type: string, description: "Full note content (action=read)"}
  notes:        {type: array,  description: "[{note_id, title, snippet, modified}] (action=list/search)"}
  count:        {type: integer}
  status:       {type: string}
  summary:      {type: string, description: "Short plain English confirmation for speech"}
dependencies: []
perception_capable: false
handler:            handle
END_MANIFEST

Plain-text notes stored as .txt files in a directory.

Notes directory resolution:
  1. 'notes_dir' argument (per-call override)
  2. HUGINN_NOTES_DIR environment variable
  3. ~/Documents/artux-notes (created on first use)

Note titles are sanitised filenames — spaces become underscores, special
characters are removed. A timestamp suffix is added for auto-generated titles.

Append behaviour:
  If the note already exists, the new content is appended with a timestamp
  separator. If it doesn't exist, it's created with the content as the full body.
  This makes "add butter to shopping list" and "add milk to shopping list"
  accumulate naturally in a single note.

Search:
  Case-insensitive substring match across all .txt file contents.
  Returns notes where the query appears in the title OR the body.
  Snippet shows up to 200 characters of surrounding context.

Environment overrides:
  HUGINN_NOTES_DIR=~/Documents/artux-notes
"""

from __future__ import annotations

import os
import re
import datetime
from pathlib import Path
from typing import Any


_DEFAULT_DIR = Path.home() / "Documents" / "artux-notes"


def handle(
    action:    str = "list",
    title:     str = "",
    content:   str = "",
    query:     str = "",
    notes_dir: str = "",
) -> dict:
    notes_path = _resolve_dir(notes_dir)

    if action == "append":
        return _append(notes_path, title, content)
    elif action == "read":
        return _read(notes_path, title)
    elif action == "list":
        return _list(notes_path)
    elif action == "search":
        return _search(notes_path, query)
    elif action == "delete":
        return _delete(notes_path, title)
    else:
        return {"status": "error", "summary": f"Unknown action: {action!r}",
                "note_id": "", "content": "", "notes": [], "count": 0}


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _append(notes_path: Path, title: str, content: str) -> dict:
    if not content.strip():
        return {"status": "error", "summary": "No content to append.",
                "note_id": "", "content": "", "notes": [], "count": 0}

    note_id  = _sanitise_title(title) if title else _auto_title()
    filepath = notes_path / f"{note_id}.txt"

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    if filepath.exists():
        sep = f"\n\n---\n{ts}\n"
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(sep + content.strip() + "\n")
        verb = "updated"
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {title or note_id}\n{ts}\n\n{content.strip()}\n")
        verb = "created"

    display = title or note_id.replace("_", " ")
    return {
        "note_id": note_id,
        "content": "",
        "notes":   [],
        "count":   1,
        "status":  verb,
        "summary": f"Note '{display}' {verb}.",
    }


def _read(notes_path: Path, title: str) -> dict:
    if not title:
        return {"status": "error", "summary": "Title required for read.",
                "note_id": "", "content": "", "notes": [], "count": 0}

    note_id  = _sanitise_title(title)
    filepath = notes_path / f"{note_id}.txt"

    if not filepath.exists():
        # Try fuzzy match (longest common substring)
        matches = [f.stem for f in notes_path.glob("*.txt")
                   if title.lower() in f.stem.lower()]
        if matches:
            filepath = notes_path / f"{matches[0]}.txt"
            note_id  = matches[0]
        else:
            return {"status": "not_found",
                    "summary": f"No note found matching '{title}'.",
                    "note_id": note_id, "content": "", "notes": [], "count": 0}

    content = filepath.read_text(encoding="utf-8")
    return {
        "note_id": note_id,
        "content": content,
        "notes":   [],
        "count":   1,
        "status":  "ok",
        "summary": f"Note '{note_id}': {content[:120]}{'...' if len(content) > 120 else ''}",
    }


def _list(notes_path: Path) -> dict:
    notes = []
    for f in sorted(notes_path.glob("*.txt"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        content = f.read_text(encoding="utf-8", errors="replace")
        notes.append({
            "note_id":  f.stem,
            "title":    f.stem.replace("_", " "),
            "snippet":  content[:100].replace("\n", " "),
            "modified": datetime.datetime.fromtimestamp(
                f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    if not notes:
        return {"note_id": "", "content": "", "notes": [], "count": 0,
                "status": "ok", "summary": "You have no notes yet."}
    titles = [n["title"] for n in notes[:5]]
    return {
        "note_id": "", "content": "",
        "notes":   notes,
        "count":   len(notes),
        "status":  "ok",
        "summary": f"You have {len(notes)} note{'s' if len(notes) != 1 else ''}. "
                   f"Most recent: {', '.join(titles[:3])}.",
    }


def _search(notes_path: Path, query: str) -> dict:
    if not query.strip():
        return {"status": "error", "summary": "Query required for search.",
                "note_id": "", "content": "", "notes": [], "count": 0}

    q       = query.lower()
    results = []
    for f in notes_path.glob("*.txt"):
        content = f.read_text(encoding="utf-8", errors="replace")
        if q in f.stem.lower() or q in content.lower():
            # Find snippet around first match
            idx     = content.lower().find(q)
            start   = max(0, idx - 60)
            snippet = content[start:start + 160].replace("\n", " ")
            if start > 0:
                snippet = "…" + snippet
            results.append({
                "note_id":  f.stem,
                "title":    f.stem.replace("_", " "),
                "snippet":  snippet,
                "modified": datetime.datetime.fromtimestamp(
                    f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            })

    if not results:
        return {"note_id": "", "content": "", "notes": [], "count": 0,
                "status": "ok",
                "summary": f"No notes found matching '{query}'."}
    return {
        "note_id": "", "content": "",
        "notes":   results,
        "count":   len(results),
        "status":  "ok",
        "summary": f"Found {len(results)} note{'s' if len(results) != 1 else ''} "
                   f"matching '{query}': {results[0]['title']}.",
    }


def _delete(notes_path: Path, title: str) -> dict:
    if not title:
        return {"status": "error", "summary": "Title required for delete.",
                "note_id": "", "content": "", "notes": [], "count": 0}
    note_id  = _sanitise_title(title)
    filepath = notes_path / f"{note_id}.txt"
    if filepath.exists():
        filepath.unlink()
        return {"note_id": note_id, "content": "", "notes": [], "count": 0,
                "status": "deleted", "summary": f"Note '{title}' deleted."}
    return {"note_id": note_id, "content": "", "notes": [], "count": 0,
            "status": "not_found", "summary": f"No note '{title}' found."}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_dir(override: str) -> Path:
    d = Path(
        override
        or os.environ.get("HUGINN_NOTES_DIR", "")
        or _DEFAULT_DIR
    ).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sanitise_title(title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title.strip().lower())
    slug = re.sub(r"[\s-]+", "_", slug).strip("_")
    return slug[:64] or "untitled"


def _auto_title() -> str:
    return "note_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
