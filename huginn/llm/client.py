"""
llm/client.py — LLM router for Huginn agents.

Architecture
------------
LLMClient is no longer a backend-specific client. It is a thin router that:

  1. Reads  "{role}.provider"    from HTM.states at call time
             "{role}.model"       temperature, timeout, etc.
  2. Looks up the provider in LLMClient._PROVIDERS (class-level registry)
  3. Calls through to the provider's four standard methods

Provider tools register themselves at install time via:
    LLMClient.register_provider(tool_id, handlers_dict)

If no provider is registered (first boot, fresh install), the built-in
_BuiltinProvider is used — it wraps the original Ollama/Anthropic logic
and keeps the system working out of the box.

Provider contract
-----------------
Every provider (tool or builtin) must expose four callables:

    complete(system, messages, model, temperature, timeout, **kw)  -> LLMResponse
    stream(system, messages, model, temperature, **kw)             -> Iterator[StreamChunk]
    complete_json(system, user, schema, model, temperature, timeout, **kw) -> dict
    complete_tools(system, messages, tools, model, temperature, timeout, **kw) -> LLMResponse

Backward compatibility
----------------------
LLMClient still accepts the old constructor kwargs (backend, model, host,
api_key, temperature, timeout) — they are forwarded to _BuiltinProvider as
fallback values.  reconfigure() writes to HTM.states instead of re-initing
backend clients.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    text:       str
    tool_calls: list  = field(default_factory=list)
    raw:        Any   = None
    latency_ms: int   = 0
    model:      str   = ""
    backend:    str   = ""


@dataclass
class StreamChunk:
    delta: str
    done:  bool = False
    raw:   Any  = None


# ---------------------------------------------------------------------------
# _BuiltinProvider
# ---------------------------------------------------------------------------

class _BuiltinProvider:
    """
    Drop-in provider used when no provider tool is installed.
    Supports backends: ollama, anthropic, openai, lmstudio, llamacpp.
    """

    _OPENAI_COMPAT = frozenset({"ollama", "openai", "lmstudio", "llamacpp"})

    def __init__(
        self,
        backend: str   = "ollama",
        model:   str   = "llama3.2",
        host:    str   = "http://localhost:11434",
        api_key: str   = "",
        temp:    float = 0.1,
        timeout: float = 60.0,
    ):
        self.backend = backend.lower()
        self.model   = model
        self.host    = host
        self.api_key = api_key
        self.temp    = temp
        self.timeout = timeout
        self._client     = None
        self._ant_client = None
        self._tool_calling_works: Optional[bool] = None
        self._reinit()

    def _reinit(self):
        if self.backend in self._OPENAI_COMPAT:
            self._init_openai_compat()
        elif self.backend == "anthropic":
            self._init_anthropic()

    def _init_openai_compat(self):
        try:
            from openai import OpenAI
            if self.backend == "openai":
                self._client = OpenAI(api_key=self.api_key or None)
            else:
                self._client = OpenAI(
                    base_url=f"{self.host.rstrip('/')}/v1",
                    api_key=self.api_key or "local",
                )
        except ImportError:
            pass

    def _init_anthropic(self):
        try:
            import anthropic as _ant, os
            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            self._ant_client = _ant.Anthropic(api_key=key) if key else None
        except ImportError:
            pass

    def _require_openai(self):
        if self._client is None:
            self._init_openai_compat()
        if self._client is None:
            raise ImportError(
                f"openai package required for {self.backend!r} backend.\n"
                "pip install openai"
            )

    def _require_anthropic(self):
        if self._ant_client is None:
            self._init_anthropic()
        if self._ant_client is None:
            raise ImportError(
                "anthropic package required.\npip install anthropic\n"
                "Set ANTHROPIC_API_KEY environment variable."
            )

    def complete(self, system, messages, model="", temperature=0.1,
                 timeout=60.0, **_) -> LLMResponse:
        m  = model or self.model
        t0 = time.monotonic()
        if self.backend in self._OPENAI_COMPAT:
            self._require_openai()
            resp = self._client.chat.completions.create(
                model=m, temperature=temperature, timeout=timeout,
                messages=[{"role": "system", "content": system}] + messages,
            )
            text = (resp.choices[0].message.content or "").strip()
        else:
            self._require_anthropic()
            resp = self._ant_client.messages.create(
                model=m, max_tokens=4096, system=system, messages=messages,
            )
            text = next((b.text for b in resp.content if hasattr(b,"text")), "").strip()
        return LLMResponse(text=text, raw=resp,
                           latency_ms=int((time.monotonic()-t0)*1000),
                           model=m, backend=self.backend)

    def stream(self, system, messages, model="", temperature=0.1, **_):
        m = model or self.model
        if self.backend in self._OPENAI_COMPAT:
            self._require_openai()
            with self._client.chat.completions.create(
                model=m, temperature=temperature, timeout=self.timeout, stream=True,
                messages=[{"role": "system", "content": system}] + messages,
            ) as s:
                for chunk in s:
                    delta = chunk.choices[0].delta.content or ""
                    done  = chunk.choices[0].finish_reason is not None
                    yield StreamChunk(delta=delta, done=done, raw=chunk)
        else:
            self._require_anthropic()
            with self._ant_client.messages.stream(
                model=m, max_tokens=4096, system=system, messages=messages,
            ) as s:
                for text in s.text_stream:
                    yield StreamChunk(delta=text, done=False)
                yield StreamChunk(delta="", done=True)

    def complete_json(self, system, user, schema, model="",
                      temperature=0.0, timeout=60.0, **_) -> dict:
        m           = model or self.model
        schema_hint = json.dumps(schema, indent=2)
        full_sys    = (
            f"{system}\n\nRespond ONLY with a valid JSON object. "
            f"No markdown, no preamble.\nRequired shape:\n{schema_hint}"
        )
        for attempt in range(3):
            raw = self._raw_complete(full_sys,
                                     [{"role": "user", "content": user}],
                                     m, temperature, timeout, json_mode=(attempt==0))
            parsed = _safe_parse_json(raw)
            if parsed is not None:
                return parsed
            user = (f"{user}\n\n[Previous response was not valid JSON. "
                    f"Return ONLY a JSON object matching the schema above.]")
        raise ValueError(f"LLM failed to return valid JSON. Last: {raw[:200]!r}")

    def _raw_complete(self, system, messages, model, temp, timeout, json_mode=False):
        if self.backend in self._OPENAI_COMPAT:
            self._require_openai()
            kw = {"response_format": {"type": "json_object"}} if json_mode else {}
            try:
                resp = self._client.chat.completions.create(
                    model=model, temperature=temp, timeout=timeout,
                    messages=[{"role": "system", "content": system}] + messages,
                    **kw,
                )
            except Exception:
                resp = self._client.chat.completions.create(
                    model=model, temperature=temp, timeout=timeout,
                    messages=[{"role": "system", "content": system}] + messages,
                )
            return (resp.choices[0].message.content or "").strip()
        else:
            self._require_anthropic()
            resp = self._ant_client.messages.create(
                model=model, max_tokens=1024, system=system, messages=messages,
            )
            return next((b.text for b in resp.content if hasattr(b,"text")), "").strip()

    def complete_tools(self, system, messages, tools, model="",
                       temperature=0.1, timeout=60.0, **_) -> LLMResponse:
        m  = model or self.model
        t0 = time.monotonic()
        if self.backend == "anthropic":
            return self._anthropic_tools(system, messages, tools, m, temperature, t0)
        return self._openai_compat_tools(system, messages, tools, m, temperature, timeout, t0)

    def _anthropic_tools(self, system, messages, tools, model, temp, t0):
        self._require_anthropic()
        resp  = self._ant_client.messages.create(
            model=model, max_tokens=4096, system=system,
            tools=_openai_tools_to_anthropic(tools), messages=messages,
        )
        text, calls = "", []
        for block in resp.content:
            if hasattr(block, "text"):
                text += block.text
            elif block.type == "tool_use":
                calls.append({"id": block.id, "name": block.name,
                               "arguments": block.input})
        return LLMResponse(text=text.strip(), tool_calls=calls, raw=resp,
                           latency_ms=int((time.monotonic()-t0)*1000),
                           model=model, backend=self.backend)

    def _openai_compat_tools(self, system, messages, tools, model, temp, timeout, t0):
        self._require_openai()
        full = [{"role": "system", "content": system}] + messages
        if self._tool_calling_works is not False:
            try:
                resp  = self._client.chat.completions.create(
                    model=model, messages=full, tools=tools,
                    temperature=temp, timeout=timeout,
                )
                msg   = resp.choices[0].message
                calls = msg.tool_calls or []
                if calls or self._tool_calling_works is True:
                    self._tool_calling_works = True
                    parsed = [
                        {"id": c.id, "name": c.function.name,
                         "arguments": json.loads(c.function.arguments)}
                        for c in calls
                    ]
                    return LLMResponse(
                        text=(msg.content or "").strip(), tool_calls=parsed, raw=resp,
                        latency_ms=int((time.monotonic()-t0)*1000),
                        model=model, backend=self.backend)
                self._tool_calling_works = True
                return LLMResponse(text=(msg.content or "").strip(),
                                   latency_ms=int((time.monotonic()-t0)*1000),
                                   model=model, backend=self.backend)
            except Exception:
                self._tool_calling_works = False
        # JSON fallback
        sys2  = _tool_schema_summary(tools) + _FALLBACK_SUFFIX
        msgs2 = [{"role": "system", "content": sys2}] + full[1:]
        resp  = self._client.chat.completions.create(
            model=model, messages=msgs2, temperature=temp, timeout=timeout,
        )
        raw = (resp.choices[0].message.content or "").strip()
        text, calls = _extract_json_tool_calls(raw)
        return LLMResponse(text=text, tool_calls=calls, raw=resp,
                           latency_ms=int((time.monotonic()-t0)*1000),
                           model=model, backend=self.backend)


# ---------------------------------------------------------------------------
# _ProviderToolAdapter
# ---------------------------------------------------------------------------

class _ProviderToolAdapter:
    """Adapts a registered provider handler dict to the provider interface."""

    def __init__(self, handlers: dict, htm, role: str = ""):
        self._h    = handlers
        self._htm  = htm
        self._role = role

    def _call(self, method: str, **kwargs):
        fn = self._h.get(method)
        if fn is None:
            raise NotImplementedError(
                f"Provider does not implement '{method}'. "
                f"Available: {list(self._h)}"
            )
        import inspect
        sig = inspect.signature(fn)
        if "_htm" in sig.parameters:
            kwargs["_htm"] = self._htm
        if "_role" in sig.parameters:
            kwargs["_role"] = self._role
        return fn(**kwargs)

    def complete(self, **kw)       -> LLMResponse: return self._call("complete", **kw)
    def stream(self, **kw):                         return self._call("stream", **kw)
    def complete_json(self, **kw)  -> dict:         return self._call("complete_json", **kw)
    def complete_tools(self, **kw) -> LLMResponse: return self._call("complete_tools", **kw)


# ---------------------------------------------------------------------------
# LLMClient — router
# ---------------------------------------------------------------------------

_BACKEND_TO_PROVIDER = {
    "ollama":    "tool.llm.ollama.v1",
    "anthropic": "tool.llm.anthropic.v1",
    "openai":    "tool.llm.openai.v1",
    "lmstudio":  "tool.llm.lmstudio.v1",
    "llamacpp":  "tool.llm.llamacpp.v1",
}


class LLMClient:
    """
    Thin router. Reads active provider from HTM.states at every call.

    HTM.states keys:
      {role}.provider    → provider tool_id  (e.g. "tool.llm.ollama.v1")
      {role}.model       → model name
      {role}.temperature → float
      {role}.timeout     → float (seconds)

    Falls back to _BuiltinProvider when no provider tool is installed.
    """

    # Class-level registry: tool_id → handler dict
    _PROVIDERS: Dict[str, Dict] = {}

    @classmethod
    def register_provider(cls, tool_id: str, handlers: dict) -> None:
        cls._PROVIDERS[tool_id] = handlers

    @classmethod
    def unregister_provider(cls, tool_id: str) -> None:
        cls._PROVIDERS.pop(tool_id, None)

    @classmethod
    def available_providers(cls) -> list:
        return list(cls._PROVIDERS.keys())

    def __init__(
        self,
        role:             str   = "sagax",
        htm               = None,
        fallback_backend: str   = "ollama",
        fallback_model:   str   = "llama3.2",
        fallback_host:    str   = "http://localhost:11434",
        fallback_api_key: str   = "",
        fallback_temp:    float = 0.1,
        fallback_timeout: float = 60.0,
        # Legacy direct kwargs
        backend:          str   = "",
        model:            str   = "",
        host:             str   = "",
        api_key:          str   = "",
        temperature:      float = None,
        timeout:          float = None,
    ):
        self.role = role
        self.htm  = htm
        self._fb  = _BuiltinProvider(
            backend = backend  or fallback_backend,
            model   = model    or fallback_model,
            host    = host     or fallback_host,
            api_key = api_key  or fallback_api_key,
            temp    = temperature if temperature is not None else fallback_temp,
            timeout = timeout     if timeout     is not None else fallback_timeout,
        )

    def _resolve(self):
        if self.htm is None:
            return self._fb, self._fb.model, self._fb.temp, self._fb.timeout

        provider_id = self.htm.states.get(f"{self.role}.provider", "")
        model       = self.htm.states.get(f"{self.role}.model",       self._fb.model)
        temp        = float(self.htm.states.get(f"{self.role}.temperature", self._fb.temp))
        timeout     = float(self.htm.states.get(f"{self.role}.timeout",     self._fb.timeout))

        if provider_id and provider_id in self._PROVIDERS:
            return _ProviderToolAdapter(
                self._PROVIDERS[provider_id], self.htm, role=self.role
            ), model, temp, timeout

        self._fb.model = model
        return self._fb, model, temp, timeout

    # Public API

    def complete(self, system, user="", temperature=None, messages=None) -> LLMResponse:
        prov, model, temp, timeout = self._resolve()
        t    = temperature if temperature is not None else temp
        msgs = list(messages or [])
        if user:
            msgs.append({"role": "user", "content": user})
        return prov.complete(system=system, messages=msgs,
                             model=model, temperature=t, timeout=timeout)

    def complete_json(self, system, user, schema, temperature=None) -> dict:
        prov, model, temp, timeout = self._resolve()
        t = temperature if temperature is not None else temp
        return prov.complete_json(system=system, user=user, schema=schema,
                                  model=model, temperature=t, timeout=timeout)

    def complete_tools(self, system, messages, tools, temperature=None) -> LLMResponse:
        prov, model, temp, timeout = self._resolve()
        t = temperature if temperature is not None else temp
        return prov.complete_tools(system=system, messages=messages, tools=tools,
                                   model=model, temperature=t, timeout=timeout)

    def stream(self, system, messages, temperature=None):
        prov, model, temp, _ = self._resolve()
        t = temperature if temperature is not None else temp
        return prov.stream(system=system, messages=messages,
                           model=model, temperature=t)

    def ping(self) -> bool:
        try:
            return bool(self.complete("You are a test.", "Reply with: ok",
                                      temperature=0).text)
        except Exception:
            return False

    def reconfigure(self, backend="", model="", host="", api_key="",
                    temperature=None, timeout=None, **_):
        """Compat shim — writes to HTM.states or updates fallback provider."""
        if self.htm is not None:
            for k, v in [("model", model), ("temperature", temperature),
                         ("timeout", timeout)]:
                if v is not None and v != "":
                    self.htm.states.set(f"{self.role}.{k}", v, mark_dirty=False)
            if backend:
                pid = _BACKEND_TO_PROVIDER.get(
                    backend.lower(), f"tool.llm.{backend.lower()}.v1"
                )
                self.htm.states.set(f"{self.role}.provider", pid, mark_dirty=False)
        else:
            if model:       self._fb.model   = model
            if temperature: self._fb.temp    = temperature
            if timeout:     self._fb.timeout = timeout
            if backend or host or api_key:
                if backend: self._fb.backend = backend.lower()
                if host:    self._fb.host    = host
                if api_key: self._fb.api_key = api_key
                self._fb._reinit()

    @property
    def model(self):
        if self.htm:
            return self.htm.states.get(f"{self.role}.model", self._fb.model)
        return self._fb.model

    @property
    def temperature(self):
        if self.htm:
            return float(self.htm.states.get(f"{self.role}.temperature", self._fb.temp))
        return self._fb.temp

    @property
    def backend(self):
        if self.htm:
            pid = self.htm.states.get(f"{self.role}.provider", "")
            if pid:
                return pid
        return self._fb.backend

    def __repr__(self):
        return (f"LLMClient(role={self.role!r}, "
                f"provider={self.backend!r}, model={self.model!r})")


class LLMPool:
    """Holds named LLMClient instances. Backward compatible."""
    def __init__(self, clients: dict):
        self._clients = clients
    def get(self, name: str) -> LLMClient:
        if name not in self._clients:
            raise KeyError(f"No LLMClient named {name!r}. Available: {list(self._clients)}")
        return self._clients[name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FALLBACK_SUFFIX = (
    "\n\n── Tool Calling ──\n"
    "To call tools, output ONLY a JSON object or array:\n"
    '  Single:   {"tool": "<n>", "arguments": {<args>}}\n'
    "  Multiple: [{...}, {...}]\n"
    "After all tool calls are resolved, respond in plain text.\n"
    "If no tools needed, respond in plain text directly."
)


def _safe_parse_json(raw: str) -> Optional[dict]:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        r = json.loads(raw)
        return r if isinstance(r, dict) else None
    except Exception:
        return None


def _tool_schema_summary(tools: list) -> str:
    lines = ["Available tools:"]
    for t in tools:
        fn       = t.get("function", t)
        name     = fn.get("name", "?")
        desc     = fn.get("description", "")[:100]
        params   = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        param_str = ", ".join(
            f"{k}{'*' if k in required else '?'}" for k in params
        )
        lines.append(f"  {name}({param_str}) — {desc}")
    return "\n".join(lines)


def _extract_json_tool_calls(raw: str) -> tuple:
    spans = _find_json_spans(raw)
    calls, used = [], set()
    for start, end in spans:
        try:
            obj = json.loads(raw[start:end])
        except Exception:
            continue
        if isinstance(obj, dict) and "tool" in obj:
            calls.append({"id": f"json-{len(calls)}", "name": obj["tool"],
                          "arguments": obj.get("arguments", obj.get("args", {}))})
            used.update(range(start, end))
        elif isinstance(obj, list):
            before = len(calls)
            for item in obj:
                if isinstance(item, dict) and "tool" in item:
                    calls.append({"id": f"json-{len(calls)}", "name": item["tool"],
                                  "arguments": item.get("arguments", {})})
            if len(calls) > before:
                used.update(range(start, end))
    remaining = "".join(ch for i, ch in enumerate(raw) if i not in used).strip()
    return remaining, calls


def _find_json_spans(text: str) -> list:
    spans, i, n = [], 0, len(text)
    while i < n:
        if text[i] in ('{', '['):
            opener, closer = text[i], ('}' if text[i] == '{' else ']')
            depth = in_str = escape = 0
            start = i
            for j in range(i, n):
                ch = text[j]
                if escape:   escape = False; continue
                if ch == '\\' and in_str: escape = True; continue
                if ch == '"': in_str = not in_str; continue
                if in_str:   continue
                if ch == opener:   depth += 1
                elif ch == closer: depth -= 1
                if depth == 0:
                    spans.append((start, j + 1))
                    i = j + 1
                    break
            else:
                i += 1
        else:
            i += 1
    return spans


def _openai_tools_to_anthropic(tools: list) -> list:
    return [
        {"name": t.get("function", t).get("name", ""),
         "description": t.get("function", t).get("description", ""),
         "input_schema": t.get("function", t).get("parameters",
                               {"type": "object", "properties": {}})}
        for t in tools
    ]
