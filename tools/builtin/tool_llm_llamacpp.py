"""
tool.llm.llamacpp.v1 — llama.cpp LLM provider for Huginn.

Runs GGUF models in-process via llama-cpp-python. No external server.
This is the default local inference backend shipped with Huginn.

HUGINN_MANIFEST
tool_id:            tool.llm.llamacpp.v1
title:              llama.cpp LLM Provider
capability_summary: >
  Run inference against any GGUF model in-process via llama-cpp-python.
  No external server required. Place a .gguf model file in the models/
  directory and Huginn will use it automatically. Supports plain completion,
  streaming, JSON mode (grammar-constrained), and tool calling via JSON
  extraction fallback. Compatible with Llama 3, Qwen, Phi, Mistral, Gemma,
  and any other GGUF-format model.
polarity:           read
permission_scope:   []
mode:               provider
direction:          ""
inputs:
  system:      {type: string}
  messages:    {type: array}
  model:       {type: string, description: "Ignored — model is set via states."}
  temperature: {type: number, default: 0.1}
  timeout:     {type: number, default: 120.0}
outputs:
  text:       {type: string}
  tool_calls: {type: array}
states:
  model_path:
    default: ""
    type: string
    description: Absolute path to the .gguf model file
  n_ctx:
    default: 4096
    type: integer
    description: Context window size (tokens)
  n_gpu_layers:
    default: 0
    type: integer
    description: GPU layers to offload (0 = CPU only; -1 = all)
  n_threads:
    default: 0
    type: integer
    description: CPU threads (0 = auto-detect)
  temperature:
    default: 0.1
    type: float
    description: Default generation temperature
  max_tokens:
    default: 2048
    type: integer
    description: Maximum tokens to generate per call
dependencies: []
END_MANIFEST

Installation
------------
  pip install llama-cpp-python

  # CPU-only (default):
  pip install llama-cpp-python

  # With CUDA GPU offloading:
  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

  # With Metal (Apple Silicon):
  CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

Model directory
---------------
  Place any GGUF model file in <artux_db_dir>/models/
  Example: ~/.artux/models/llama-3.2-3b-instruct-q4_k_m.gguf

  Huginn auto-detects the first .gguf file at startup and sets
  tool.llm.llamacpp.v1.model_path in HTM.states automatically.

  To switch models at runtime (Sagax can do this):
    <task_update>{"action": "state_set",
      "key": "tool.llm.llamacpp.v1.model_path",
      "value": "/path/to/other_model.gguf"}</task_update>
  The next inference call loads the new model (cold swap, ~seconds).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Iterator, Optional

# ---------------------------------------------------------------------------
# Module-level model instance — shared across all calls in this process
# ---------------------------------------------------------------------------

_model = None
_model_path: str = ""
_model_params: dict = {}


def _get_model(htm=None) -> Any:
    """
    Lazy-load (or reload) the llama.cpp model.
    Reloads when model_path or key params have changed.
    """
    global _model, _model_path, _model_params

    # Resolve config from HTM.states.
    # Per-role model paths are stored as:
    #   tool.llm.llamacpp.v1.model_path.{role}  (set by _assign_gguf_models)
    # falling back to the shared:
    #   tool.llm.llamacpp.v1.model_path
    # The caller passes _role via kw if available (see _ProviderToolAdapter).
    mp        = ""
    n_ctx     = 4096
    n_gpu     = 0
    n_threads = 0
    if htm is not None:
        role      = kw.get("_role", "")
        role_key  = f"tool.llm.llamacpp.v1.model_path.{role}" if role else ""
        mp = (
            (htm.states.get(role_key) if role_key else None)
            or htm.states.get("tool.llm.llamacpp.v1.model_path", "")
            or ""
        )
        n_ctx     = int(htm.states.get("tool.llm.llamacpp.v1.n_ctx",         4096))
        n_gpu     = int(htm.states.get("tool.llm.llamacpp.v1.n_gpu_layers",  0))
        n_threads = int(htm.states.get("tool.llm.llamacpp.v1.n_threads",     0))

    if not mp:
        raise RuntimeError(
            "No GGUF model path configured.\n"
            "Place a .gguf file in <artux_db_dir>/models/ or set:\n"
            "  HTM.states['tool.llm.llamacpp.v1.model_path'] = '/path/to/model.gguf'"
        )

    current_params = {
        "n_ctx": n_ctx, "n_gpu_layers": n_gpu, "n_threads": n_threads
    }
    # Reload if model path or params changed
    if _model is None or mp != _model_path or current_params != _model_params:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed.\n"
                "pip install llama-cpp-python"
            )
        _model        = Llama(
            model_path   = mp,
            n_ctx        = n_ctx,
            n_gpu_layers = n_gpu,
            n_threads    = n_threads if n_threads > 0 else None,
            verbose      = False,
        )
        _model_path   = mp
        _model_params = current_params

    return _model


def _resolve(kw: dict, key: str, default: Any = None) -> Any:
    """Read from kwargs, then HTM.states, then default."""
    if key in kw and kw[key] is not None and kw[key] != "":
        return kw[key]
    htm = kw.get("_htm")
    if htm is not None:
        for role in ("exilis", "sagax", "logos"):
            v = htm.states.get(f"{role}.{key}")
            if v is not None:
                return v
        v = htm.states.get(f"tool.llm.llamacpp.v1.{key}")
        if v is not None:
            return v
    return default


def _build_prompt(system: str, messages: list) -> str:
    """
    Build a ChatML-style prompt from system + messages.
    Most GGUF instruction models use this format.
    """
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle multi-part content (tool results etc.)
            content = " ".join(
                c.get("text", c.get("content", "")) if isinstance(c, dict) else str(c)
                for c in content
            )
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

def complete(
    system:      str,
    messages:    list,
    model:       str   = "",
    temperature: float = 0.1,
    timeout:     float = 120.0,
    **kw,
):
    from huginn.llm.client import LLMResponse
    m      = _get_model(kw.get("_htm"))
    temp   = float(_resolve(kw, "temperature", temperature))
    maxtok = int(_resolve(kw, "max_tokens", 2048))
    prompt = _build_prompt(system, messages)
    t0     = time.monotonic()

    result = m(prompt, temperature=temp, max_tokens=maxtok, echo=False)
    text   = result["choices"][0]["text"].strip()

    return LLMResponse(
        text=text, raw=result,
        latency_ms=int((time.monotonic()-t0)*1000),
        model=_model_path, backend="llamacpp",
    )


def stream(
    system:      str,
    messages:    list,
    model:       str   = "",
    temperature: float = 0.1,
    **kw,
) -> Iterator:
    from huginn.llm.client import StreamChunk
    m      = _get_model(kw.get("_htm"))
    temp   = float(_resolve(kw, "temperature", temperature))
    maxtok = int(_resolve(kw, "max_tokens", 2048))
    prompt = _build_prompt(system, messages)

    for chunk in m(prompt, temperature=temp, max_tokens=maxtok,
                   echo=False, stream=True):
        delta = chunk["choices"][0].get("text", "")
        done  = chunk["choices"][0].get("finish_reason") is not None
        yield StreamChunk(delta=delta, done=done, raw=chunk)


def complete_json(
    system:      str,
    user:        str,
    schema:      dict,
    model:       str   = "",
    temperature: float = 0.0,
    timeout:     float = 120.0,
    **kw,
) -> dict:
    """
    JSON-constrained completion using llama.cpp grammar mode.
    Falls back to prompt-based extraction if grammar fails.
    """
    m         = _get_model(kw.get("_htm"))
    maxtok    = int(_resolve(kw, "max_tokens", 1024))
    schema_h  = json.dumps(schema, indent=2)
    full_sys  = (
        f"{system}\n\nRespond ONLY with a valid JSON object. "
        f"No markdown, no preamble.\nRequired shape:\n{schema_h}"
    )
    prompt    = _build_prompt(full_sys, [{"role": "user", "content": user}])

    # Try grammar-constrained JSON mode first
    try:
        from llama_cpp import LlamaGrammar
        grammar = LlamaGrammar.from_string(
            'root   ::= "{" ws members ws "}"\\n'
            'members::= member ("," ws member)*\\n'
            'member ::= string ws ":" ws value\\n'
            'ws     ::= [ \\t\\n]*\\n'
            'value  ::= string | number | "true" | "false" | "null" | object | array\\n'
            'string ::= "\\"" [^\\"]* "\\""\\n'
            'number ::= "-"? [0-9]+ ("." [0-9]+)?\\n'
            'object ::= "{" ws members ws "}"\\n'
            'array  ::= "[" ws (value ("," ws value)*)? ws "]"\\n'
        )
        result = m(prompt, temperature=0.0, max_tokens=maxtok,
                   echo=False, grammar=grammar)
        raw    = result["choices"][0]["text"].strip()
        parsed = _safe_parse_json(raw)
        if parsed is not None:
            return parsed
    except Exception:
        pass

    # Fallback: plain prompt + extraction
    current_user = user
    for _ in range(3):
        result = m(prompt, temperature=0.0, max_tokens=maxtok, echo=False)
        raw    = result["choices"][0]["text"].strip()
        parsed = _safe_parse_json(raw)
        if parsed is not None:
            return parsed
        current_user = (
            f"{current_user}\n\n[Previous response was not valid JSON. "
            "Return ONLY a JSON object matching the schema above.]"
        )
        prompt = _build_prompt(full_sys, [{"role": "user", "content": current_user}])

    raise ValueError(f"llama.cpp failed to return valid JSON. Last: {raw[:200]!r}")


def complete_tools(
    system:      str,
    messages:    list,
    tools:       list,
    model:       str   = "",
    temperature: float = 0.1,
    timeout:     float = 120.0,
    **kw,
):
    """
    Tool calling via JSON extraction fallback.
    GGUF models rarely support native tool call format reliably;
    prompt-based JSON extraction works universally.
    """
    from huginn.llm.client import (
        LLMResponse, _tool_schema_summary, _FALLBACK_SUFFIX,
        _extract_json_tool_calls,
    )
    m      = _get_model(kw.get("_htm"))
    temp   = float(_resolve(kw, "temperature", temperature))
    maxtok = int(_resolve(kw, "max_tokens", 2048))
    t0     = time.monotonic()

    tool_sys = _tool_schema_summary(tools) + _FALLBACK_SUFFIX
    prompt   = _build_prompt(
        tool_sys,
        [{"role": "system", "content": system}] + messages,
    )
    result = m(prompt, temperature=temp, max_tokens=maxtok, echo=False)
    raw    = result["choices"][0]["text"].strip()
    text, calls = _extract_json_tool_calls(raw)

    return LLMResponse(
        text=text, tool_calls=calls, raw=result,
        latency_ms=int((time.monotonic()-t0)*1000),
        model=_model_path, backend="llamacpp",
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_parse_json(raw: str) -> Optional[dict]:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        r = json.loads(raw)
        return r if isinstance(r, dict) else None
    except Exception:
        return None
