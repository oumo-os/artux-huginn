"""
HUGINN_MANIFEST
tool_id:            tool.config.write.v1
title:              Write system configuration
capability_summary: >
  Write or update a system configuration value stored in long-term memory.
  Use to set LLM backend, model names, API keys, Ollama host URL, or any
  other artux.config.* entry. Values are stored in Muninn LTM and recalled
  at next startup. Sensitive values (api_key) are stored in LTM only —
  they never appear in Sagax context after the write completes.
polarity:           write
permission_scope:   [system.config]
inputs:
  role:
    type: string
    description: "Which agent to configure: exilis | sagax | logos"
  key:
    type: string
    description: "Config field to set: backend | model | host | api_key | temperature | timeout"
  value:
    type: string
    description: "New value as a string. Numbers and booleans are coerced automatically."
outputs:
  status:
    type: string
  topic:
    type: string
  previous:
    type: string
dependencies: []
perception_capable: false
handler: handle
END_MANIFEST
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Config key → Muninn topic mapping (must match Logos._CONFIG_DEFAULTS keys)
# ---------------------------------------------------------------------------

_ROLE_TOPICS = {
    "exilis": "artux.config.llm.exilis.v1",
    "sagax":  "artux.config.llm.sagax.v1",
    "logos":  "artux.config.llm.logos.v1",
}

_VALID_KEYS = {"backend", "model", "host", "api_key", "temperature", "timeout"}

_SENSITIVE_KEYS = {"api_key"}


def _coerce(key: str, value: str) -> Any:
    """Coerce string value to appropriate Python type."""
    if key in ("temperature", "timeout"):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def handle(
    role:  str = "",
    key:   str = "",
    value: str = "",
    _muninn=None,   # injected by ToolManager at call time
) -> dict:
    """
    Write a single config field for a named agent role.

    The caller (Sagax) supplies role + key + value as strings.
    This tool reads the existing config entry, updates the field,
    and writes the updated entry back to Muninn LTM.

    The previous value is returned so Sagax can confirm the change
    to the user — but sensitive fields (api_key) return "***" as
    the previous value even if Sagax asks.
    """
    # --- Validate inputs -----------------------------------------------
    role  = role.strip().lower()
    key   = key.strip().lower()
    value = value.strip()

    if role not in _ROLE_TOPICS:
        return {
            "status": "error",
            "error":  f"Unknown role {role!r}. Valid: {sorted(_ROLE_TOPICS)}",
        }
    if key not in _VALID_KEYS:
        return {
            "status": "error",
            "error":  f"Unknown config key {key!r}. Valid: {sorted(_VALID_KEYS)}",
        }
    if not value:
        return {"status": "error", "error": "value must not be empty"}

    topic = _ROLE_TOPICS[role]

    # --- Require Muninn at call time ------------------------------------
    if _muninn is None:
        return {
            "status": "error",
            "error":  "Muninn not available — tool not properly wired",
        }

    # --- Read existing config entry ------------------------------------
    existing_cfg: dict = {}
    try:
        results = _muninn.recall(topic, top_k=1)
        if results:
            raw     = getattr(results[0], "entry", results[0])
            content = getattr(raw, "content", "") or ""
            if content:
                existing_cfg = json.loads(content)
    except Exception as e:
        return {"status": "error", "error": f"Recall failed: {e}"}

    # --- Capture previous value for confirmation -----------------------
    previous_raw = existing_cfg.get(key, "(not set)")
    previous_display = "***" if key in _SENSITIVE_KEYS else str(previous_raw)

    # --- Update field --------------------------------------------------
    new_cfg = dict(existing_cfg)
    new_cfg[key] = _coerce(key, value)

    # --- Write back to Muninn LTM -------------------------------------
    try:
        _muninn.store_ltm(
            content    = json.dumps(new_cfg),
            class_type = "config",
            topics     = [topic, "artux.config", "system"],
            confidence = 1.0,
        )
    except Exception as e:
        return {"status": "error", "error": f"Write failed: {e}"}

    display_value = "***" if key in _SENSITIVE_KEYS else value

    return {
        "status":   "ok",
        "topic":    topic,
        "role":     role,
        "key":      key,
        "value":    display_value,
        "previous": previous_display,
        "note": (
            "Config updated in LTM. Will take effect at next Artux startup "
            "or when the Orchestrator applies config mid-session."
        ),
    }
