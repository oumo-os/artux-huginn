"""
tool.llm.anthropic.v1 — Anthropic Claude provider for Huginn.

HUGINN_MANIFEST
tool_id:            tool.llm.anthropic.v1
title:              Anthropic Claude Provider
capability_summary: >
  Run inference against Anthropic's Claude models via the Anthropic API.
  Supports plain completion, streaming, JSON-mode output, and native
  tool calling. Compatible with all claude-* model families.
polarity:           read
permission_scope:   []
mode:               provider
direction:          ""
inputs:
  system:      {type: string}
  messages:    {type: array}
  model:       {type: string, description: "Claude model string. Reads {role}.model from HTM.states if omitted."}
  temperature: {type: number, default: 0.1}
  timeout:     {type: number, default: 60.0}
outputs:
  text:       {type: string}
  tool_calls: {type: array}
states:
  api_key:
    default: ""
    type: string
    description: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var if empty.
  model:
    default: "claude-haiku-4-5-20251001"
    type: string
    description: Default Claude model
  temperature:
    default: 0.1
    type: float
    description: Default generation temperature
  timeout:
    default: 60.0
    type: float
    description: Request timeout in seconds
dependencies:
  anthropic
END_MANIFEST
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Iterator, Optional

_client = None


def _get_client(api_key: str = ""):
    global _client
    if _client is None or (api_key and api_key != _client.api_key):
        import anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "Anthropic API key not set. "
                "Set ANTHROPIC_API_KEY env var or write to "
                "HTM.states: tool.llm.anthropic.v1.api_key"
            )
        _client = anthropic.Anthropic(api_key=key)
    return _client


def _resolve(kw: dict, key: str, default: Any = None) -> Any:
    if key in kw and kw[key] is not None and kw[key] != "":
        return kw[key]
    htm = kw.get("_htm")
    if htm is not None:
        for role in ("exilis", "sagax", "logos"):
            v = htm.states.get(f"{role}.{key}")
            if v is not None:
                return v
        v = htm.states.get(f"tool.llm.anthropic.v1.{key}")
        if v is not None:
            return v
    return default


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

def complete(
    system:      str,
    messages:    list,
    model:       str   = "",
    temperature: float = 0.1,
    timeout:     float = 60.0,
    **kw,
):
    from huginn.llm.client import LLMResponse
    api_key = _resolve(kw, "api_key", "")
    model   = model or _resolve(kw, "model", "claude-haiku-4-5-20251001")
    client  = _get_client(api_key)
    t0      = time.monotonic()

    resp = client.messages.create(
        model=model, max_tokens=4096,
        system=system, messages=messages,
    )
    text = next((b.text for b in resp.content if hasattr(b, "text")), "").strip()
    return LLMResponse(
        text=text, raw=resp,
        latency_ms=int((time.monotonic()-t0)*1000),
        model=model, backend="anthropic",
    )


def stream(
    system:      str,
    messages:    list,
    model:       str   = "",
    temperature: float = 0.1,
    **kw,
) -> Iterator:
    from huginn.llm.client import StreamChunk
    api_key = _resolve(kw, "api_key", "")
    model   = model or _resolve(kw, "model", "claude-haiku-4-5-20251001")
    client  = _get_client(api_key)

    with client.messages.stream(
        model=model, max_tokens=4096,
        system=system, messages=messages,
    ) as s:
        for text in s.text_stream:
            yield StreamChunk(delta=text, done=False)
        yield StreamChunk(delta="", done=True)


def complete_json(
    system:      str,
    user:        str,
    schema:      dict,
    model:       str   = "",
    temperature: float = 0.0,
    timeout:     float = 60.0,
    **kw,
) -> dict:
    api_key     = _resolve(kw, "api_key", "")
    model       = model or _resolve(kw, "model", "claude-haiku-4-5-20251001")
    client      = _get_client(api_key)
    schema_hint = json.dumps(schema, indent=2)
    full_sys    = (
        f"{system}\n\nRespond ONLY with a valid JSON object. "
        f"No markdown, no preamble.\nRequired shape:\n{schema_hint}"
    )
    current_user = user
    raw = ""
    for _ in range(3):
        resp = client.messages.create(
            model=model, max_tokens=1024,
            system=full_sys,
            messages=[{"role": "user", "content": current_user}],
        )
        raw    = next((b.text for b in resp.content if hasattr(b, "text")), "").strip()
        parsed = _safe_parse_json(raw)
        if parsed is not None:
            return parsed
        current_user = (
            f"{current_user}\n\n[Previous response was not valid JSON. "
            f"Return ONLY a JSON object matching the schema above.]"
        )
    raise ValueError(f"Anthropic failed to return valid JSON. Last: {raw[:200]!r}")


def complete_tools(
    system:      str,
    messages:    list,
    tools:       list,
    model:       str   = "",
    temperature: float = 0.1,
    timeout:     float = 60.0,
    **kw,
):
    from huginn.llm.client import LLMResponse
    api_key = _resolve(kw, "api_key", "")
    model   = model or _resolve(kw, "model", "claude-haiku-4-5-20251001")
    client  = _get_client(api_key)
    t0      = time.monotonic()

    ant_tools = [
        {
            "name":         t.get("function", t).get("name", ""),
            "description":  t.get("function", t).get("description", ""),
            "input_schema": t.get("function", t).get(
                "parameters", {"type": "object", "properties": {}}
            ),
        }
        for t in tools
    ]
    resp  = client.messages.create(
        model=model, max_tokens=4096,
        system=system, tools=ant_tools, messages=messages,
    )
    text, calls = "", []
    for block in resp.content:
        if hasattr(block, "text"):
            text += block.text
        elif block.type == "tool_use":
            calls.append({"id": block.id, "name": block.name,
                          "arguments": block.input})
    return LLMResponse(
        text=text.strip(), tool_calls=calls, raw=resp,
        latency_ms=int((time.monotonic()-t0)*1000),
        model=model, backend="anthropic",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_parse_json(raw: str):
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        r = json.loads(raw)
        return r if isinstance(r, dict) else None
    except Exception:
        return None
