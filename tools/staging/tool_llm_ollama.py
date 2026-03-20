"""
tool.llm.ollama.v1 — Ollama LLM provider for Huginn.

HUGINN_MANIFEST
tool_id:            tool.llm.ollama.v1
title:              Ollama LLM Provider
capability_summary: >
  Run inference against any locally-served Ollama model.
  Supports plain completion, streaming, JSON-mode, and tool calling
  (native + JSON fallback). Works with llama3, qwen, mistral, phi,
  gemma, and any other model available via `ollama pull`.
polarity:           read
permission_scope:   []
mode:               provider
direction:          ""
inputs:
  system:      {type: string}
  messages:    {type: array}
  model:       {type: string, description: "Ollama model name. Reads {role}.model from HTM.states if omitted."}
  temperature: {type: number, default: 0.1}
  timeout:     {type: number, default: 60.0}
outputs:
  text:       {type: string}
  tool_calls: {type: array}
states:
  host:
    default: "http://localhost:11434"
    type: string
    description: Ollama server base URL
  model:
    default: "llama3.2"
    type: string
    description: Default model name
  temperature:
    default: 0.1
    type: float
    description: Default generation temperature
  timeout:
    default: 60.0
    type: float
    description: Request timeout in seconds
dependencies:
  openai
END_MANIFEST
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Iterator, Optional


# ---------------------------------------------------------------------------
# Module-level client — created once on first call
# ---------------------------------------------------------------------------

_client = None
_tool_calling_works: Optional[bool] = None


def _get_client(host: str):
    """Lazy-init the OpenAI-compat client pointed at Ollama."""
    global _client
    # Re-init if host changed (e.g. after a state_set)
    current_base = getattr(getattr(_client, "_base_url", None), "_uri", None)
    target_base  = f"{host.rstrip('/')}/v1"
    if _client is None or str(current_base) != target_base:
        from openai import OpenAI
        _client = OpenAI(base_url=target_base, api_key="local")
    return _client


def _resolve(kw: dict, key: str, default: Any = None) -> Any:
    """Read from kwargs, falling back through HTM.states to hard default."""
    if key in kw and kw[key] is not None and kw[key] != "":
        return kw[key]
    htm = kw.get("_htm")
    if htm is not None:
        # States are stored under the calling role's namespace; try all three
        for role in ("exilis", "sagax", "logos"):
            v = htm.states.get(f"{role}.{key}")
            if v is not None:
                return v
        # Also try the tool's own namespace
        v = htm.states.get(f"tool.llm.ollama.v1.{key}")
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
    host   = _resolve(kw, "host", "http://localhost:11434")
    model  = model  or _resolve(kw, "model",       "llama3.2")
    temp   = temperature if temperature else float(_resolve(kw, "temperature", 0.1))
    tout   = timeout     if timeout     else float(_resolve(kw, "timeout",     60.0))
    client = _get_client(host)
    t0     = time.monotonic()

    resp = client.chat.completions.create(
        model       = model,
        messages    = [{"role": "system", "content": system}] + messages,
        temperature = temp,
        timeout     = tout,
    )
    text = (resp.choices[0].message.content or "").strip()
    return LLMResponse(
        text=text, raw=resp,
        latency_ms=int((time.monotonic()-t0)*1000),
        model=model, backend="ollama",
    )


def stream(
    system:      str,
    messages:    list,
    model:       str   = "",
    temperature: float = 0.1,
    **kw,
) -> Iterator:
    from huginn.llm.client import StreamChunk
    host   = _resolve(kw, "host",        "http://localhost:11434")
    model  = model or _resolve(kw, "model", "llama3.2")
    temp   = temperature if temperature else float(_resolve(kw, "temperature", 0.1))
    tout   = float(_resolve(kw, "timeout", 60.0))
    client = _get_client(host)

    with client.chat.completions.create(
        model       = model,
        messages    = [{"role": "system", "content": system}] + messages,
        temperature = temp,
        timeout     = tout,
        stream      = True,
    ) as s:
        for chunk in s:
            delta = chunk.choices[0].delta.content or ""
            done  = chunk.choices[0].finish_reason is not None
            yield StreamChunk(delta=delta, done=done, raw=chunk)


def complete_json(
    system:      str,
    user:        str,
    schema:      dict,
    model:       str   = "",
    temperature: float = 0.0,
    timeout:     float = 60.0,
    **kw,
) -> dict:
    host   = _resolve(kw, "host",  "http://localhost:11434")
    model  = model or _resolve(kw, "model", "llama3.2")
    tout   = timeout if timeout else float(_resolve(kw, "timeout", 60.0))
    client = _get_client(host)

    schema_hint = json.dumps(schema, indent=2)
    full_sys    = (
        f"{system}\n\nRespond ONLY with a valid JSON object. "
        f"No markdown, no preamble.\nRequired shape:\n{schema_hint}"
    )
    current_user = user
    for attempt in range(3):
        kw_req = {"response_format": {"type": "json_object"}} if attempt == 0 else {}
        try:
            resp = client.chat.completions.create(
                model       = model,
                messages    = [{"role": "system", "content": full_sys},
                               {"role": "user",   "content": current_user}],
                temperature = 0.0,
                timeout     = tout,
                **kw_req,
            )
        except Exception:
            resp = client.chat.completions.create(
                model       = model,
                messages    = [{"role": "system", "content": full_sys},
                               {"role": "user",   "content": current_user}],
                temperature = 0.0,
                timeout     = tout,
            )
        raw    = (resp.choices[0].message.content or "").strip()
        parsed = _safe_parse_json(raw)
        if parsed is not None:
            return parsed
        current_user = (
            f"{current_user}\n\n[Previous response was not valid JSON. "
            f"Return ONLY a JSON object matching the schema above.]"
        )
    raise ValueError(f"Ollama failed to return valid JSON. Last: {raw[:200]!r}")


def complete_tools(
    system:      str,
    messages:    list,
    tools:       list,
    model:       str   = "",
    temperature: float = 0.1,
    timeout:     float = 60.0,
    **kw,
):
    from huginn.llm.client import LLMResponse, _extract_json_tool_calls, _tool_schema_summary, _FALLBACK_SUFFIX
    global _tool_calling_works
    host   = _resolve(kw, "host",  "http://localhost:11434")
    model  = model or _resolve(kw, "model", "llama3.2")
    tout   = timeout if timeout else float(_resolve(kw, "timeout", 60.0))
    client = _get_client(host)
    t0     = time.monotonic()
    full   = [{"role": "system", "content": system}] + messages

    if _tool_calling_works is not False:
        try:
            resp  = client.chat.completions.create(
                model=model, messages=full, tools=tools,
                temperature=temperature, timeout=tout,
            )
            msg   = resp.choices[0].message
            calls = msg.tool_calls or []
            if calls or _tool_calling_works is True:
                _tool_calling_works = True
                parsed = [
                    {"id": c.id, "name": c.function.name,
                     "arguments": json.loads(c.function.arguments)}
                    for c in calls
                ]
                return LLMResponse(
                    text=(msg.content or "").strip(), tool_calls=parsed, raw=resp,
                    latency_ms=int((time.monotonic()-t0)*1000),
                    model=model, backend="ollama",
                )
            _tool_calling_works = True
            return LLMResponse(text=(msg.content or "").strip(),
                               latency_ms=int((time.monotonic()-t0)*1000),
                               model=model, backend="ollama")
        except Exception:
            _tool_calling_works = False

    # JSON fallback
    sys2  = _tool_schema_summary(tools) + _FALLBACK_SUFFIX
    msgs2 = [{"role": "system", "content": sys2}] + full[1:]
    resp  = client.chat.completions.create(
        model=model, messages=msgs2, temperature=temperature, timeout=tout,
    )
    raw = (resp.choices[0].message.content or "").strip()
    text, calls = _extract_json_tool_calls(raw)
    return LLMResponse(text=text, tool_calls=calls, raw=resp,
                       latency_ms=int((time.monotonic()-t0)*1000),
                       model=model, backend="ollama")


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
