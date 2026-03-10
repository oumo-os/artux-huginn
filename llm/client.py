"""
llm/client.py — Unified LLM client for Huginn agents.

Supports two backends:
  - Ollama  (via OpenAI-compatible /v1/ endpoint, openai package)
  - Anthropic (claude-* models, anthropic package)

Provides:
  - complete()          plain text completion
  - complete_json()     structured JSON output with schema enforcement
  - complete_tools()    tool-calling with native + JSON fallback
  - stream()            token-by-token streaming (Sagax Narrator)

All methods accept a unified call signature. The backend difference is
entirely internal. Agents never import anthropic or openai directly.

JSON fallback:
  For Ollama models that don't emit native tool_calls (common for smaller
  models), this client falls back to asking the model to emit JSON inline
  and extracting calls via bracket-depth scanning. This is the same
  technique proven in the existing demo files.

Structured output (complete_json):
  Uses response_format={"type":"json_object"} where supported.
  Falls back to prompting + parse for backends that don't support it.
  Output is always validated against the provided schema dict.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Optional


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    text:       str
    tool_calls: list[dict] = field(default_factory=list)
    raw:        Any = None          # the original backend response object
    latency_ms: int = 0
    model:      str = ""
    backend:    str = ""


@dataclass
class StreamChunk:
    delta:   str
    done:    bool = False
    raw:     Any  = None


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Unified LLM client. Instantiate once; pass to all agents.

    Parameters
    ----------
    backend : str
        "ollama" | "anthropic"
    model : str
        Model name as understood by the backend.
        Ollama: "llama3.2", "qwen2.5:0.5b", etc.
        Anthropic: "claude-haiku-4-5-20251001", "claude-sonnet-4-6", etc.
    host : str
        Ollama base URL (ignored for Anthropic). Default "http://localhost:11434".
    api_key : str
        Anthropic API key (ignored for Ollama). Reads ANTHROPIC_API_KEY env var
        if not supplied.
    temperature : float
        Default temperature. Agents may override per-call.
    timeout : float
        Request timeout in seconds.
    """

    def __init__(
        self,
        backend:     str   = "ollama",
        model:       str   = "llama3.2",
        host:        str   = "http://localhost:11434",
        api_key:     str   = "",
        temperature: float = 0.1,
        timeout:     float = 60.0,
    ):
        self.backend     = backend.lower()
        self.model       = model
        self.temperature = temperature
        self.timeout     = timeout
        self._tool_calling_works: Optional[bool] = None  # Ollama: discovered at runtime

        if self.backend == "ollama":
            self._init_ollama(host)
        elif self.backend == "anthropic":
            self._init_anthropic(api_key)
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'ollama' or 'anthropic'.")

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_ollama(self, host: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for Ollama backend.\n"
                "pip install openai"
            )
        self._client = OpenAI(base_url=f"{host}/v1", api_key="ollama")

    def _init_anthropic(self, api_key: str):
        try:
            import anthropic as _ant
        except ImportError:
            raise ImportError(
                "anthropic package required for Anthropic backend.\n"
                "pip install anthropic"
            )
        import os
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. "
                "Export it or pass api_key= to LLMClient."
            )
        self._ant_client = _ant.Anthropic(api_key=key)

    # ------------------------------------------------------------------
    # complete() — plain text
    # ------------------------------------------------------------------

    def complete(
        self,
        system:      str,
        user:        str,
        temperature: Optional[float] = None,
        messages:    Optional[list[dict]] = None,
    ) -> LLMResponse:
        """
        Plain text completion.

        If `messages` is provided, it is used as the full conversation
        history (for multi-turn agents like Sagax). The `user` string
        is appended as the final user turn.
        """
        t0   = time.monotonic()
        temp = temperature if temperature is not None else self.temperature
        msgs = list(messages or [])
        if user:
            msgs.append({"role": "user", "content": user})

        if self.backend == "ollama":
            resp = self._client.chat.completions.create(
                model       = self.model,
                messages    = [{"role": "system", "content": system}] + msgs,
                temperature = temp,
                timeout     = self.timeout,
            )
            text = (resp.choices[0].message.content or "").strip()
            return LLMResponse(
                text       = text,
                raw        = resp,
                latency_ms = int((time.monotonic() - t0) * 1000),
                model      = self.model,
                backend    = self.backend,
            )

        else:  # anthropic
            resp = self._ant_client.messages.create(
                model      = self.model,
                max_tokens = 4096,
                system     = system,
                messages   = msgs,
            )
            text = next(
                (b.text for b in resp.content if hasattr(b, "text")), ""
            ).strip()
            return LLMResponse(
                text       = text,
                raw        = resp,
                latency_ms = int((time.monotonic() - t0) * 1000),
                model      = self.model,
                backend    = self.backend,
            )

    # ------------------------------------------------------------------
    # complete_json() — structured output
    # ------------------------------------------------------------------

    def complete_json(
        self,
        system:      str,
        user:        str,
        schema:      dict,
        temperature: Optional[float] = None,
    ) -> dict:
        """
        Return a dict conforming to `schema`.

        Tries response_format=json_object first (Ollama / OpenAI compatible).
        Falls back to prompting + parse for backends/models that don't support it.
        Raises ValueError if the model returns unparseable JSON after retries.
        """
        schema_hint = _schema_to_hint(schema)
        full_system = (
            f"{system}\n\n"
            f"Respond ONLY with a valid JSON object. Do not include markdown fences, "
            f"preamble, or explanation.\n"
            f"Required JSON shape:\n{schema_hint}"
        )
        temp = temperature if temperature is not None else 0.0

        for attempt in range(3):
            if self.backend == "ollama":
                try:
                    resp = self._client.chat.completions.create(
                        model           = self.model,
                        messages        = [
                            {"role": "system", "content": full_system},
                            {"role": "user",   "content": user},
                        ],
                        temperature     = temp,
                        timeout         = self.timeout,
                        response_format = {"type": "json_object"},
                    )
                    raw = (resp.choices[0].message.content or "").strip()
                except Exception:
                    # Model doesn't support response_format — plain call
                    resp = self._client.chat.completions.create(
                        model       = self.model,
                        messages    = [
                            {"role": "system", "content": full_system},
                            {"role": "user",   "content": user},
                        ],
                        temperature = temp,
                        timeout     = self.timeout,
                    )
                    raw = (resp.choices[0].message.content or "").strip()

            else:  # anthropic
                resp = self._ant_client.messages.create(
                    model      = self.model,
                    max_tokens = 1024,
                    system     = full_system,
                    messages   = [{"role": "user", "content": user}],
                )
                raw = next(
                    (b.text for b in resp.content if hasattr(b, "text")), ""
                ).strip()

            parsed = _safe_parse_json(raw)
            if parsed is not None:
                return parsed

            # Retry with explicit correction prompt
            user = (
                f"{user}\n\n[Previous response was not valid JSON. "
                f"Return ONLY a JSON object matching the schema above.]"
            )

        raise ValueError(
            f"LLM failed to return valid JSON after 3 attempts. "
            f"Last raw output: {raw[:200]!r}"
        )

    # ------------------------------------------------------------------
    # complete_tools() — tool calling (native + JSON fallback)
    # ------------------------------------------------------------------

    def complete_tools(
        self,
        system:      str,
        messages:    list[dict],
        tools:       list[dict],
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        One round of tool elicitation.

        Returns LLMResponse with .text and .tool_calls populated.
        tool_calls is a list of {"name": str, "arguments": dict, "id": str}.

        The caller drives the agentic loop: execute the calls, append
        results as tool_result messages, and call again.
        """
        t0   = time.monotonic()
        temp = temperature if temperature is not None else self.temperature

        if self.backend == "anthropic":
            return self._anthropic_tools(system, messages, tools, temp, t0)
        else:
            return self._ollama_tools(system, messages, tools, temp, t0)

    def _anthropic_tools(
        self, system, messages, tools, temp, t0
    ) -> LLMResponse:
        import anthropic as _ant
        ant_tools = _openai_tools_to_anthropic(tools)
        resp = self._ant_client.messages.create(
            model      = self.model,
            max_tokens = 4096,
            system     = system,
            tools      = ant_tools,
            messages   = messages,
        )

        text  = ""
        calls = []
        for block in resp.content:
            if hasattr(block, "text"):
                text += block.text
            elif block.type == "tool_use":
                calls.append({
                    "id":        block.id,
                    "name":      block.name,
                    "arguments": block.input,
                })

        return LLMResponse(
            text       = text.strip(),
            tool_calls = calls,
            raw        = resp,
            latency_ms = int((time.monotonic() - t0) * 1000),
            model      = self.model,
            backend    = self.backend,
        )

    def _ollama_tools(
        self, system, messages, tools, temp, t0
    ) -> LLMResponse:
        sys_msg   = {"role": "system", "content": system}
        full_msgs = [sys_msg] + messages

        # Try native tool calling if not known to fail
        if self._tool_calling_works is not False:
            try:
                resp = self._client.chat.completions.create(
                    model       = self.model,
                    messages    = full_msgs,
                    tools       = tools,
                    temperature = temp,
                    timeout     = self.timeout,
                )
                msg   = resp.choices[0].message
                calls = msg.tool_calls or []

                if calls or self._tool_calling_works is True:
                    self._tool_calling_works = True
                    parsed = [
                        {
                            "id":        c.id,
                            "name":      c.function.name,
                            "arguments": json.loads(c.function.arguments),
                        }
                        for c in calls
                    ]
                    return LLMResponse(
                        text       = (msg.content or "").strip(),
                        tool_calls = parsed,
                        raw        = resp,
                        latency_ms = int((time.monotonic() - t0) * 1000),
                        model      = self.model,
                        backend    = self.backend,
                    )
                self._tool_calling_works = True
                return LLMResponse(
                    text       = (msg.content or "").strip(),
                    latency_ms = int((time.monotonic() - t0) * 1000),
                    model      = self.model,
                    backend    = self.backend,
                )

            except Exception:
                self._tool_calling_works = False

        # JSON fallback
        return self._ollama_json_fallback(full_msgs, tools, temp, t0)

    _FALLBACK_SUFFIX = (
        "\n\n── Tool Calling ──\n"
        "To call tools, output ONLY a JSON object or array:\n"
        '  Single:   {"tool": "<name>", "arguments": {<args>}}\n'
        "  Multiple: [{...}, {...}]\n"
        "After all tool calls are resolved, respond in plain text.\n"
        "If no tools needed, respond in plain text directly."
    )

    def _ollama_json_fallback(
        self, full_msgs, tools, temp, t0
    ) -> LLMResponse:
        summary = _tool_schema_summary(tools)
        system  = summary + self._FALLBACK_SUFFIX
        # Replace system message
        msgs    = [{"role": "system", "content": system}] + full_msgs[1:]

        resp = self._client.chat.completions.create(
            model       = self.model,
            messages    = msgs,
            temperature = temp,
            timeout     = self.timeout,
        )
        raw = (resp.choices[0].message.content or "").strip()
        text, calls = _extract_json_tool_calls(raw)

        return LLMResponse(
            text       = text,
            tool_calls = calls,
            raw        = resp,
            latency_ms = int((time.monotonic() - t0) * 1000),
            model      = self.model,
            backend    = self.backend,
        )

    # ------------------------------------------------------------------
    # stream() — token streaming for Sagax Narrator
    # ------------------------------------------------------------------

    def stream(
        self,
        system:      str,
        messages:    list[dict],
        temperature: Optional[float] = None,
    ) -> Generator[StreamChunk, None, None]:
        """
        Yield StreamChunk tokens as they arrive.
        Used by Sagax to produce the Narrator token stream.
        The Orchestrator consumes this generator in real time.
        """
        temp = temperature if temperature is not None else self.temperature

        if self.backend == "ollama":
            with self._client.chat.completions.create(
                model       = self.model,
                messages    = [{"role": "system", "content": system}] + messages,
                temperature = temp,
                timeout     = self.timeout,
                stream      = True,
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    done  = chunk.choices[0].finish_reason is not None
                    yield StreamChunk(delta=delta, done=done, raw=chunk)

        else:  # anthropic
            with self._ant_client.messages.stream(
                model      = self.model,
                max_tokens = 4096,
                system     = system,
                messages   = messages,
            ) as stream:
                for text in stream.text_stream:
                    yield StreamChunk(delta=text, done=False)
                yield StreamChunk(delta="", done=True)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if the backend is reachable."""
        try:
            resp = self.complete(
                system="You are a test.",
                user="Reply with: ok",
                temperature=0,
            )
            return bool(resp.text)
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"LLMClient(backend={self.backend!r}, model={self.model!r})"


# ---------------------------------------------------------------------------
# Multi-model factory
# ---------------------------------------------------------------------------

class LLMPool:
    """
    Holds named model instances for agents that need different model sizes.

    Usage:
        pool = LLMPool.from_config(config)
        exilis_llm = pool.get("fast")
        sagax_llm  = pool.get("sagax")
        logos_llm  = pool.get("logos")
    """

    def __init__(self, clients: dict[str, LLMClient]):
        self._clients = clients

    def get(self, name: str) -> LLMClient:
        if name not in self._clients:
            available = list(self._clients)
            raise KeyError(
                f"No LLMClient named {name!r}. Available: {available}"
            )
        return self._clients[name]

    @classmethod
    def from_config(cls, config) -> "LLMPool":
        """
        Build a pool from a config object with attributes:
            LLM_BACKEND          "ollama" | "anthropic"
            OLLAMA_HOST          e.g. "http://localhost:11434"
            FAST_MODEL           small/fast model for Exilis + consN
            SAGAX_MODEL          medium model for Sagax
            LOGOS_MODEL          large model for Logos
            ANTHROPIC_API_KEY    (optional, else reads env var)
            TEMPERATURE          default temperature
            TIMEOUT_SECS         request timeout
        """
        def _make(model):
            return LLMClient(
                backend     = config.LLM_BACKEND,
                model       = model,
                host        = getattr(config, "OLLAMA_HOST", "http://localhost:11434"),
                api_key     = getattr(config, "ANTHROPIC_API_KEY", ""),
                temperature = getattr(config, "TEMPERATURE", 0.1),
                timeout     = getattr(config, "TIMEOUT_SECS", 60.0),
            )

        return cls({
            "fast":  _make(config.FAST_MODEL),
            "sagax": _make(config.SAGAX_MODEL),
            "logos": _make(config.LOGOS_MODEL),
        })


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _schema_to_hint(schema: dict) -> str:
    """Convert a JSON schema dict to a compact readable hint."""
    try:
        return json.dumps(schema, indent=2)
    except Exception:
        return str(schema)


def _safe_parse_json(raw: str) -> Optional[dict]:
    """Strip markdown fences and parse JSON. Returns None on failure."""
    # Remove ```json ... ``` fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
        return None
    except Exception:
        return None


def _tool_schema_summary(tools: list[dict]) -> str:
    """Compact tool reference for the JSON fallback prompt."""
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


def _extract_json_tool_calls(raw: str) -> tuple[str, list[dict]]:
    """
    Extract tool calls from raw LLM output using bracket-depth scanning.
    Handles nested JSON correctly. Returns (remaining_text, list_of_calls).

    Ported from the battle-tested version in demo_local.py / demo_moonshine.py.
    """
    spans = _find_json_spans(raw)
    calls = []
    used  = set()

    for start, end in spans:
        fragment = raw[start:end]
        try:
            obj = json.loads(fragment)
        except Exception:
            continue

        if isinstance(obj, dict) and "tool" in obj:
            calls.append({
                "id":        f"json-{len(calls)}",
                "name":      obj["tool"],
                "arguments": obj.get("arguments", obj.get("args", {})),
            })
            used.update(range(start, end))
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and "tool" in item:
                    calls.append({
                        "id":        f"json-{len(calls)}",
                        "name":      item["tool"],
                        "arguments": item.get("arguments", item.get("args", {})),
                    })
            if calls:
                used.update(range(start, end))

    # Remaining text (strip consumed JSON spans)
    remaining = "".join(
        ch for i, ch in enumerate(raw) if i not in used
    ).strip()

    return remaining, calls


def _find_json_spans(text: str) -> list[tuple[int, int]]:
    """
    Find all top-level JSON object/array spans in `text`
    using bracket-depth scanning. Handles nested structures.
    """
    spans = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] in ('{', '['):
            opener  = text[i]
            closer  = '}' if opener == '{' else ']'
            depth   = 0
            in_str  = False
            escape  = False
            start   = i
            for j in range(i, n):
                ch = text[j]
                if escape:
                    escape = False
                    continue
                if ch == '\\' and in_str:
                    escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        spans.append((start, j + 1))
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1
    return spans


def _openai_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool schemas to Anthropic format."""
    result = []
    for t in tools:
        fn = t.get("function", t)
        result.append({
            "name":         fn.get("name", ""),
            "description":  fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result
