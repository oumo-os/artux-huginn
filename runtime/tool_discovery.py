"""
runtime/tool_discovery.py — Tool Discovery and Staging Pipeline.

Huginn supports a designated staging directory where operators (or other
systems) drop Python tool files. Logos scans this directory periodically,
reads each file's HUGINN_MANIFEST block, and stages it for user-confirmed
installation.

────────────────────────────────────────────────────────────────────────
Tool file format
────────────────────────────────────────────────────────────────────────

Every staged tool file must have a HUGINN_MANIFEST block in its module
docstring. The block is YAML between HUGINN_MANIFEST and END_MANIFEST:

    \"\"\"
    HUGINN_MANIFEST
    tool_id:            tool.smart_kettle.v1
    title:              Smart Kettle Control
    capability_summary: Control the smart kettle — boil, set temperature.
    polarity:           write
    permission_scope:   [kettle]
    inputs:
      action:        {type: string, enum: [boil, set_temp, cancel]}
      temperature_c: {type: integer, default: 100}
    outputs:
      status:        {type: string}
      current_temp:  {type: number}
    dependencies:
      - tplink-smarthome-api>=0.7
    perception_capable: false
    handler:            handle
    END_MANIFEST
    \"\"\"

    def handle(action: str, temperature_c: int = 100) -> dict:
        ...

Fields:
  tool_id            required  globally unique, e.g. tool.smart_kettle.v1
  title              required  short human name
  capability_summary required  one-sentence description for recall()
  polarity           required  "read" | "write"
  permission_scope   required  list of scope strings
  inputs             required  JSON Schema properties dict
  outputs            optional  JSON Schema properties dict
  dependencies       optional  list of pip install strings
  perception_capable optional  bool — if true, Sagax offers pipeline activation
  handler            optional  function name in the module (default: "handle")

────────────────────────────────────────────────────────────────────────
Discovery lifecycle
────────────────────────────────────────────────────────────────────────

1. Logos calls ToolDiscovery.scan() on each consolidation pass.
2. scan() finds .py files in STAGING_DIR not yet in the known set.
3. For each new file: parse manifest, write STM event, create HTM task
   (state="waiting", persistence="persist").
4. Exilis sees the STM event → "act" → Sagax wakes.
5. Sagax reads the pending HTM task, asks the user for confirmation.
6. User says yes → Sagax writes task note with affirmation.
7. Logos reads the affirmed task on next cycle → calls ToolManager.install().
8. If user says "urgent" → Sagax calls logos.request_early_cycle().

Once installed:
  - File is moved from staging/ to active/ directory.
  - LTM entry is written with the full descriptor.
  - Handler is loaded via importlib and registered in ToolManager.
  - HTM task is completed with install notes.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

MANIFEST_START = "HUGINN_MANIFEST"
MANIFEST_END   = "END_MANIFEST"

_REQUIRED_FIELDS = {"tool_id", "title", "capability_summary", "polarity",
                    "permission_scope", "inputs"}


@dataclass
class ToolManifest:
    """Parsed contents of a HUGINN_MANIFEST block."""
    tool_id:            str
    title:              str
    capability_summary: str
    polarity:           str
    permission_scope:   list[str]
    inputs:             dict[str, Any]
    outputs:            dict[str, Any]       = field(default_factory=dict)
    dependencies:       list[str]            = field(default_factory=list)
    perception_capable: bool                 = False
    handler:            str                  = "handle"

    # Set by the scanner — not in YAML
    source_path:        str                  = ""
    source_hash:        str                  = ""

    def to_ltm_dict(self) -> dict:
        return {
            "artifact_type":      "tool",
            "tool_id":            self.tool_id,
            "title":              self.title,
            "capability_summary": self.capability_summary,
            "polarity":           self.polarity,
            "permission_scope":   self.permission_scope,
            "inputs":             self.inputs,
            "outputs":            self.outputs,
            "dependencies":       self.dependencies,
            "perception_capable": self.perception_capable,
            "handler":            self.handler,
            "source_path":        self.source_path,
            "source_hash":        self.source_hash,
            "install_state":      "pending",
        }


@dataclass
class StagedTool:
    """State of a discovered-but-not-yet-installed tool."""
    manifest:    ToolManifest
    task_id:     str = ""          # HTM task ID created for this staging
    affirmed:    bool = False
    affirmed_at: str = ""
    urgent:      bool = False


# ---------------------------------------------------------------------------
# Manifest parser
# ---------------------------------------------------------------------------

def parse_manifest(source_code: str, source_path: str = "") -> Optional[ToolManifest]:
    """
    Extract and parse the HUGINN_MANIFEST block from a tool file's source.
    Returns None if no manifest is found or parsing fails.
    """
    # Find the manifest block
    start_idx = source_code.find(MANIFEST_START)
    end_idx   = source_code.find(MANIFEST_END, start_idx)
    if start_idx == -1 or end_idx == -1:
        return None

    yaml_text = source_code[start_idx + len(MANIFEST_START):end_idx].strip()

    try:
        if _HAVE_YAML:
            data = yaml.safe_load(yaml_text)
        else:
            data = _simple_yaml_parse(yaml_text)
    except Exception as e:
        raise ValueError(f"Manifest YAML parse error in {source_path}: {e}") from e

    if not isinstance(data, dict):
        return None

    # Validate required fields
    missing = _REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ValueError(
            f"Manifest in {source_path} is missing required fields: {missing}"
        )

    # Normalise permission_scope to list
    scope = data.get("permission_scope", [])
    if isinstance(scope, str):
        scope = [s.strip() for s in scope.split(",")]

    # Normalise dependencies
    deps = data.get("dependencies", [])
    if isinstance(deps, str):
        deps = [d.strip() for d in deps.strip().splitlines() if d.strip()]

    source_hash = hashlib.sha256(source_code.encode()).hexdigest()[:16]

    return ToolManifest(
        tool_id            = data["tool_id"],
        title              = data["title"],
        capability_summary = data["capability_summary"],
        polarity           = data.get("polarity", "write"),
        permission_scope   = scope,
        inputs             = data.get("inputs", {}),
        outputs            = data.get("outputs", {}),
        dependencies       = deps,
        perception_capable = bool(data.get("perception_capable", False)),
        handler            = data.get("handler", "handle"),
        source_path        = source_path,
        source_hash        = source_hash,
    )


# ---------------------------------------------------------------------------
# Tool Discovery
# ---------------------------------------------------------------------------

class ToolDiscovery:
    """
    Scans the staging directory for new tool files.
    Called by Logos on each consolidation pass.

    Parameters
    ----------
    staging_dir : str | Path
        Directory where new tool Python files are placed.
    active_dir : str | Path
        Directory where installed tool files are moved to.
    stm : STMStore
        For writing discovery events.
    htm : HTM
        For creating pending installation tasks.
    known_file : str | Path
        JSON file tracking already-seen source hashes (avoids re-discovery).
    """

    TASK_TAG = "tool_staging"

    def __init__(
        self,
        staging_dir: str,
        active_dir:  str,
        stm,
        htm,
        known_file:  str = "",
    ):
        self.staging_dir = Path(staging_dir)
        self.active_dir  = Path(active_dir)
        self.stm         = stm
        self.htm         = htm
        self._known_file = Path(known_file) if known_file else \
                           self.staging_dir / ".huginn_known.json"
        self._known: dict[str, str] = {}   # source_hash → tool_id
        self._load_known()

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.active_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main scan — called by Logos
    # ------------------------------------------------------------------

    def scan(self) -> list[StagedTool]:
        """
        Scan staging_dir for new or changed .py files.
        For each new file: parse manifest, write STM discovery event,
        create HTM pending-installation task.
        Returns the list of newly staged tools this pass.
        """
        newly_staged: list[StagedTool] = []

        for py_file in sorted(self.staging_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue

            source = py_file.read_text(encoding="utf-8")
            h      = hashlib.sha256(source.encode()).hexdigest()[:16]

            if h in self._known:
                continue   # already processed

            try:
                manifest = parse_manifest(source, str(py_file))
            except ValueError as e:
                # Bad manifest — write an internal error event and skip
                self.stm.record(
                    source="system", type="internal",
                    payload={"subtype":    "tool_discovery_error",
                             "file":       py_file.name,
                             "error":      str(e)},
                )
                self._known[h] = f"_error:{py_file.name}"
                self._save_known()
                continue

            if manifest is None:
                # No manifest — not a Huginn tool file, silently skip
                self._known[h] = f"_no_manifest:{py_file.name}"
                self._save_known()
                continue

            # Create HTM pending task
            task_id = self._create_staging_task(manifest)

            # Write STM discovery event (wakes Exilis → Sagax)
            self.stm.record(
                source="system", type="internal",
                payload={
                    "subtype":            "tool_discovered",
                    "tool_id":            manifest.tool_id,
                    "title":              manifest.title,
                    "capability_summary": manifest.capability_summary,
                    "polarity":           manifest.polarity,
                    "perception_capable": manifest.perception_capable,
                    "dependencies":       manifest.dependencies,
                    "task_id":            task_id,
                    "source_file":        py_file.name,
                    "source_hash":        h,
                },
            )

            staged = StagedTool(manifest=manifest, task_id=task_id)
            newly_staged.append(staged)

            self._known[h] = manifest.tool_id
            self._save_known()

        return newly_staged

    # ------------------------------------------------------------------
    # Affirmation check — called by Logos before install pass
    # ------------------------------------------------------------------

    def get_affirmed_tasks(self) -> list[StagedTool]:
        """
        Return staged tools whose HTM tasks have been affirmed by the user
        (task has a notebook entry with "user_affirmed: true").
        """
        affirmed = []
        for task in self.htm.query(
            state="waiting", tags_any=[self.TASK_TAG]
        ):
            for note in reversed(task.notebook):
                entry = note.get("entry", "")
                if "user_affirmed: true" in entry:
                    # Parse tool info from task progress field
                    try:
                        info = json.loads(task.progress or "{}")
                    except Exception:
                        info = {}
                    tool_id  = info.get("tool_id", "")
                    src_path = info.get("source_path", "")
                    if not src_path:
                        break

                    src = Path(src_path)
                    if not src.exists():
                        break

                    source   = src.read_text(encoding="utf-8")
                    manifest = parse_manifest(source, str(src))
                    if manifest is None:
                        break

                    urgent = "urgent: true" in entry
                    affirmed.append(StagedTool(
                        manifest    = manifest,
                        task_id     = task.task_id,
                        affirmed    = True,
                        affirmed_at = note.get("ts", ""),
                        urgent      = urgent,
                    ))
                    break
        return affirmed

    # ------------------------------------------------------------------
    # Move to active directory
    # ------------------------------------------------------------------

    def move_to_active(self, manifest: ToolManifest) -> Path:
        """
        Move a staged file to the active directory after successful install.
        Returns the new active path.
        """
        src  = Path(manifest.source_path)
        dest = self.active_dir / src.name
        shutil.move(str(src), str(dest))
        manifest.source_path = str(dest)
        return dest

    # ------------------------------------------------------------------
    # HTM task creation
    # ------------------------------------------------------------------

    def _create_staging_task(self, manifest: ToolManifest) -> str:
        task_id = self.htm.create(
            title        = f"Tool installation pending: {manifest.title}",
            initiated_by = "system",
            persistence  = "persist",
            tags         = [self.TASK_TAG, manifest.tool_id],
            progress     = json.dumps({
                "tool_id":     manifest.tool_id,
                "source_path": manifest.source_path,
                "source_hash": manifest.source_hash,
                "polarity":    manifest.polarity,
                "perception_capable": manifest.perception_capable,
                "deps":        manifest.dependencies,
            }),
        )
        self.htm.note(
            task_id,
            f"Discovered in staging directory. "
            f"Awaiting user confirmation via Sagax."
        )
        return task_id

    # ------------------------------------------------------------------
    # Known set persistence
    # ------------------------------------------------------------------

    def _load_known(self):
        if self._known_file.exists():
            try:
                self._known = json.loads(self._known_file.read_text())
            except Exception:
                self._known = {}

    def _save_known(self):
        self._known_file.write_text(json.dumps(self._known, indent=2))


# ---------------------------------------------------------------------------
# Simple YAML parser fallback (no PyYAML)
# ---------------------------------------------------------------------------

def _simple_yaml_parse(text: str) -> dict:
    """
    Minimal YAML parser for HUGINN_MANIFEST blocks.
    Handles: scalar values, inline lists ([a, b, c]), block lists (- item),
    and nested dicts with indentation.
    Sufficient for manifest parsing without requiring PyYAML.
    """
    result: dict = {}
    lines  = text.splitlines()
    i      = 0

    def _parse_value(raw: str) -> Any:
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("]"):
            return [v.strip().strip("'\"") for v in raw[1:-1].split(",") if v.strip()]
        if raw.lower() in ("true", "yes"):  return True
        if raw.lower() in ("false", "no"): return False
        try: return int(raw)
        except ValueError: pass
        try: return float(raw)
        except ValueError: pass
        return raw.strip("'\"")

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        indent = len(line) - len(stripped)
        if ":" in stripped:
            key, _, rest = stripped.partition(":")
            key  = key.strip()
            rest = rest.strip()

            if rest:
                # Nested dict on single line? e.g. {type: string}
                if rest.startswith("{") and rest.endswith("}"):
                    inner = {}
                    for part in rest[1:-1].split(","):
                        k2, _, v2 = part.partition(":")
                        inner[k2.strip()] = _parse_value(v2)
                    result[key] = inner
                else:
                    result[key] = _parse_value(rest)
                i += 1
            else:
                # Look-ahead: block list or nested dict
                children = []
                child_dict = {}
                i += 1
                while i < len(lines):
                    child_line = lines[i]
                    child_stripped = child_line.lstrip()
                    child_indent   = len(child_line) - len(child_stripped)
                    if not child_stripped or child_indent <= indent:
                        break
                    if child_stripped.startswith("- "):
                        children.append(_parse_value(child_stripped[2:]))
                    elif ":" in child_stripped:
                        k2, _, v2 = child_stripped.partition(":")
                        v2 = v2.strip()
                        if v2.startswith("{") and v2.endswith("}"):
                            inner = {}
                            for part in v2[1:-1].split(","):
                                k3, _, v3 = part.partition(":")
                                inner[k3.strip()] = _parse_value(v3)
                            child_dict[k2.strip()] = inner
                        else:
                            child_dict[k2.strip()] = _parse_value(v2)
                    i += 1
                result[key] = children if children else child_dict
        else:
            i += 1

    return result
