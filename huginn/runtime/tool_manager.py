"""
runtime/tool_manager.py — Two-tier ToolManager for Huginn.

Tier 1 — Memory tools (Muninn intrinsics):
    recall, record_stm, consolidate_ltm, create_entity, observe_entity,
    resolve_entity, get_stm_window, record_source, update_source_description

    These are dispatched directly to Muninn's ToolExecutor, which already
    implements them against the MemoryAgent. Schemas come from Muninn's
    get_tools() and are static.

    recall() accepts structured RecallQuery parameters from Sagax:
      operator, subject, topics, semantic_query, time_range, include_scars
    All parameters are optional — Sagax fills in whichever it has reasoned
    about. Topics give surgical exact-match precision; semantic_query gives
    embedding fallback. Both can be combined.

    Huginn's own programmatic recall() calls (config, startup procedure,
    tool discovery) use RecallQuery directly with topics= for surgical
    precision, falling back to a plain string if memory_module is absent.

Tier 2 — World tools (operator-registered):
    tool.set_ceiling_lights.v1, tool.play_music.v1, tool.read_calendar.v1 …

    Descriptors (schema + metadata) are stored as LTM artifacts with
    class_type="tool". Python handlers are registered at startup by the
    operator. Sagax discovers them via recall() — no hardcoded registry.

Tier 3 — Dynamically installed tools (via staging pipeline):
    Discovered by Logos via ToolDiscovery.scan(), confirmed by Sagax, then
    installed by install_tool(). Installation:
      1. pip-installs declared dependencies
      2. importlib-loads the module from the active directory
      3. Resolves the handler callable by manifest.handler name
      4. Writes the full LTM descriptor artifact
      5. Registers in self._world for immediate dispatch

The ToolManager:
    - Serves combined tool schemas to Sagax (memory + discovered world tools)
    - Dispatches execute() to the correct tier
    - Caches recently-used world tool schemas in HTM ASC (hot_tools)
    - Annotates every result with polarity so the Orchestrator's aug_call
      gate can reject write-polarity tools in <aug_call> blocks

Polarity convention:
    "read"  → safe for <aug_call> (recall, get_stm_window, sensor reads)
    "write" → must use <tool_call> (STM writes, actuators, calendar writes)
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Tool descriptor
# ---------------------------------------------------------------------------

@dataclass
class ToolDescriptor:
    """
    Runtime representation of a registered tool.
    Populated from LTM artifact content (world tools) or hardcoded
    (memory tools).
    """
    tool_id:             str
    title:               str
    capability_summary:  str
    polarity:            str                     # "read" | "write"
    permission_scope:    list[str]               # scopes required
    inputs:              dict[str, Any]          # JSON Schema properties
    outputs:             dict[str, Any]
    required:            list[str] = field(default_factory=list)
    tier:                str = "world"           # "memory" | "world"
    source_path:         str = ""               # absolute path to .py file (staged tools)
    handler:             Optional[Callable] = field(default=None, repr=False)
    # Service tool fields (populated from manifest)
    mode:                str = "callable"        # "callable" | "service"
    direction:           str = ""               # "input" | "output" | "io" | ""
    subscriptions:       list = field(default_factory=list)

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name":        self.tool_id,
                "description": f"{self.title}. {self.capability_summary}",
                "parameters": {
                    "type":       "object",
                    "properties": self.inputs,
                    "required":   self.required,
                },
            },
        }

    def to_anthropic_schema(self) -> dict:
        return {
            "name":        self.tool_id,
            "description": f"{self.title}. {self.capability_summary}",
            "input_schema": {
                "type":       "object",
                "properties": self.inputs,
                "required":   self.required,
            },
        }


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    tool_id:    str
    success:    bool
    output:     Any                    # str for memory tools, dict for world
    error:      str = ""
    polarity:   str = "write"
    duration_ms: int = 0


class InstallError(Exception):
    """Raised by ToolManager.install_tool() on unrecoverable install failure."""


# ---------------------------------------------------------------------------
# Memory tool polarity map
# ---------------------------------------------------------------------------

_MEMORY_POLARITY = {
    "recall":                   "read",
    "resolve_entity":           "read",
    "get_stm_window":           "read",
    "get_instructions":         "read",   # Huginn-native: fetches LTM instruction artifact
    "htm_state_get":            "read",   # Huginn-native: reads from HTM.states
    "record_stm":               "write",
    "consolidate_ltm":          "write",
    "create_entity":            "write",
    "observe_entity":           "write",
    "record_source":            "write",
    "update_source_description": "write",
}

# Tools handled natively by ToolManager (not dispatched to Muninn ToolExecutor)
_NATIVE_TOOLS = {"get_instructions", "htm_state_get"}


# ---------------------------------------------------------------------------
# ToolManager
# ---------------------------------------------------------------------------

class ToolManager:
    """
    Unified tool dispatch layer for Huginn.

    Parameters
    ----------
    muninn : MemoryAgent
        Used for world-tool descriptor recall and memory tool execution.
    htm : HTM
        Hot Task Manager — hot_tools cache updated on each call.
    """

    def __init__(self, muninn, htm=None):
        self.muninn = muninn
        self.htm    = htm

        # Muninn's ToolExecutor handles all memory-tier calls.
        # Soft import: allows Huginn to run in test environments where
        # the full Muninn package is not installed.
        try:
            from memory_module.tools import ToolExecutor
            self._memory_executor = ToolExecutor(muninn)
        except ImportError:
            # Fallback stub for test environments / standalone use
            class _StubExecutor:
                def execute(self, tool_id, args):
                    return {"status": "stub", "tool_id": tool_id}
            self._memory_executor = _StubExecutor()

        # World tool registry: tool_id → ToolDescriptor (with handler)
        self._world: dict[str, ToolDescriptor] = {}

        # Schema cache: avoids repeated LTM lookups for known tools
        self._schema_cache: dict[str, ToolDescriptor] = {}

    # ------------------------------------------------------------------
    # Registration (world tools)
    # ------------------------------------------------------------------

    def register(
        self,
        tool_id: str,
        handler: Callable,
        descriptor: Optional[ToolDescriptor] = None,
    ):
        """
        Register a Python handler for a world tool.

        If `descriptor` is not supplied, the descriptor is looked up from
        LTM. If it doesn't exist in LTM either, a minimal stub is created.
        The descriptor must be stored in Muninn LTM separately (see README
        for the consolidate_ltm() call pattern).
        """
        if descriptor is None:
            descriptor = self._fetch_descriptor_from_ltm(tool_id)
        descriptor.handler = handler
        descriptor.tier    = "world"
        self._world[tool_id]         = descriptor
        self._schema_cache[tool_id]  = descriptor

    def get_descriptor(self, tool_id: str) -> Optional["ToolDescriptor"]:
        """Return a ToolDescriptor by tool_id, checking world and schema caches."""
        if tool_id in self._world:
            return self._world[tool_id]
        if tool_id in self._schema_cache:
            return self._schema_cache[tool_id]
        return None

    def register_from_ltm(self, tool_id: str, handler: Callable):
        """
        Convenience: load descriptor from LTM and register.
        Raises ValueError if the descriptor isn't in LTM.
        """
        descriptor = self._fetch_descriptor_from_ltm(tool_id)
        if descriptor.capability_summary == "(not found in LTM)":
            raise ValueError(
                f"No tool descriptor found in LTM for {tool_id!r}. "
                f"Register it first with muninn.consolidate_ltm(class_type='tool', ...)."
            )
        self.register(tool_id, handler, descriptor)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, tool_id: str, args: dict) -> ToolResult:
        """
        Dispatch a tool call and return a ToolResult.

        Memory tools (tier 1) → ToolExecutor → MemoryAgent
        World tools  (tier 2) → registered handler callable
        """
        t0 = time.monotonic()

        # Tier 1: memory tools
        if tool_id in _MEMORY_POLARITY:
            try:
                # Native Huginn tools bypass Muninn's ToolExecutor entirely
                if tool_id in _NATIVE_TOOLS:
                    output = self._execute_native(tool_id, args)
                    return ToolResult(
                        tool_id     = tool_id,
                        success     = True,
                        output      = output,
                        polarity    = "read",
                        duration_ms = int((time.monotonic() - t0) * 1000),
                    )
                output = self._memory_executor.execute(tool_id, args)
                return ToolResult(
                    tool_id    = tool_id,
                    success    = True,
                    output     = output,
                    polarity   = _MEMORY_POLARITY[tool_id],
                    duration_ms = int((time.monotonic() - t0) * 1000),
                )
            except Exception as e:
                return ToolResult(
                    tool_id    = tool_id,
                    success    = False,
                    output     = "",
                    error      = str(e),
                    polarity   = _MEMORY_POLARITY.get(tool_id, "write"),
                    duration_ms = int((time.monotonic() - t0) * 1000),
                )

        # Tier 2: world tools
        descriptor = self._world.get(tool_id)
        if descriptor is None:
            # Try to lazy-load from LTM (handler not registered — schema only)
            descriptor = self._fetch_descriptor_from_ltm(tool_id)
            if descriptor.handler is None:
                return ToolResult(
                    tool_id = tool_id,
                    success = False,
                    output  = {},
                    error   = f"tool_not_registered: {tool_id}",
                    polarity = descriptor.polarity,
                    duration_ms = int((time.monotonic() - t0) * 1000),
                )

        try:
            # Inject _muninn if the handler declares it as a parameter.
            # This lets tools like tool_config_write.py access LTM directly
            # without coupling to Huginn internals.
            import inspect
            call_args = dict(args)
            sig = inspect.signature(descriptor.handler)
            if "_muninn" in sig.parameters:
                call_args["_muninn"] = self.muninn
            result = descriptor.handler(**call_args)
            output = result if isinstance(result, dict) else {"output": result}
            self._update_hot_tools(tool_id, descriptor)
            return ToolResult(
                tool_id    = tool_id,
                success    = True,
                output     = output,
                polarity   = descriptor.polarity,
                duration_ms = int((time.monotonic() - t0) * 1000),
            )
        except Exception as e:
            return ToolResult(
                tool_id    = tool_id,
                success    = False,
                output     = {},
                error      = str(e),
                polarity   = descriptor.polarity,
                duration_ms = int((time.monotonic() - t0) * 1000),
            )

    # ------------------------------------------------------------------
    # Installation (Tier 3 — dynamic staging pipeline)
    # ------------------------------------------------------------------

    def install_tool(
        self,
        manifest:       "ToolManifest",
        stm,
        htm,
        activate_pipeline: bool = False,
    ) -> "ToolDescriptor":
        """
        Install a staged tool. Called by Logos after user affirmation.

        Steps:
          1. Install declared pip dependencies (if any)
          2. Load the Python module from manifest.source_path
          3. Resolve the handler callable
          4. Write LTM descriptor artifact to Muninn
          5. Register in self._world for immediate dispatch
          6. Optionally create an HTM system task to activate as pipeline
          7. Write an install-complete STM event

        Returns the ToolDescriptor for the installed tool.
        Raises InstallError on any failure — caller should log and not flush.
        """
        tool_id = manifest.tool_id

        # Step 1: pip dependencies
        if manifest.dependencies:
            dep_errors = self._install_dependencies(manifest.dependencies, stm)
            if dep_errors:
                stm.record(
                    source="system", type="internal",
                    payload={
                        "subtype":   "tool_install_dep_warning",
                        "tool_id":   tool_id,
                        "failures":  dep_errors,
                    },
                )
                # Non-fatal: attempt to load anyway (may already be installed)

        # Step 2: load module
        try:
            module = self._load_module(manifest.source_path, tool_id)
        except Exception as e:
            raise InstallError(f"Module load failed for {tool_id}: {e}") from e

        # Step 3: resolve handler
        # Provider tools expose complete/stream/etc rather than a single handler.
        # They have no callable registered in _world — routing goes through
        # LLMClient._PROVIDERS instead. Use a no-op sentinel.
        if manifest.mode == "provider":
            handler = None   # sentinel — not dispatched via Tool Manager
        else:
            handler_name = manifest.handler or "handle"
            handler = getattr(module, handler_name, None)
            if handler is None:
                fallback_name = tool_id.replace(".", "_").replace("-", "_")
                handler = getattr(module, fallback_name, None)
            if handler is None or not callable(handler):
                raise InstallError(
                    f"No callable '{handler_name}' found in module {manifest.source_path}. "
                    f"Add a 'def {handler_name}(...)' function or set 'handler:' in manifest."
                )

        # Step 4: write LTM descriptor
        ltm_dict = manifest.to_ltm_dict()
        ltm_dict["install_state"] = "installed"
        try:
            self.muninn.consolidate_ltm(
                narrative  = json.dumps(ltm_dict),
                class_type = "tool",
                topics     = ["tool", tool_id] + manifest.permission_scope,
                confidence = 1.0,
            )
        except Exception as e:
            raise InstallError(f"LTM write failed for {tool_id}: {e}") from e

        # Step 5: register in world tier
        descriptor = ToolDescriptor(
            tool_id            = tool_id,
            title              = manifest.title,
            capability_summary = manifest.capability_summary,
            polarity           = manifest.polarity,
            permission_scope   = manifest.permission_scope,
            inputs             = manifest.inputs,
            outputs            = manifest.outputs,
            tier               = "world",
            source_path        = str(manifest.source_path),
            handler            = handler,
            mode               = manifest.mode,
            direction          = manifest.direction,
            subscriptions      = manifest.subscriptions,
        )
        self._world[tool_id]        = descriptor
        self._schema_cache[tool_id] = descriptor

        # Step 5b: write manifest state defaults to HTM.states
        # Each state param is stored as "{tool_id}.{param_name}" = default_value
        if manifest.states and htm is not None:
            for param, spec in manifest.states.items():
                key     = f"{tool_id}.{param}"
                default = spec.get("default") if isinstance(spec, dict) else spec
                htm.states.set_default(key, default)
            stm.record(
                source="system", type="internal",
                payload={
                    "subtype":  "tool_states_initialised",
                    "tool_id":  tool_id,
                    "keys":     [f"{tool_id}.{p}" for p in manifest.states],
                },
            )

        # Step 6: optionally activate as pipeline
        if activate_pipeline and manifest.perception_capable and htm is not None:
            # Create a new system HTM pipeline task so the Orchestrator picks it up
            htm.create(
                title        = f"Perception: {manifest.title}",
                initiated_by = "system",
                persistence  = "persist",
                tags         = ["pipeline", "perception", tool_id],
                progress     = json.dumps({
                    "artifact_id":       tool_id,
                    "active_by_default": True,
                }),
            )
            stm.record(
                source="system", type="internal",
                payload={
                    "subtype": "pipeline_activated",
                    "tool_id": tool_id,
                    "title":   manifest.title,
                },
            )

        # Step 7: register as LLM provider if mode="provider"
        if manifest.mode == "provider":
            # The module must expose the four standard callables directly.
            # We build a handler dict from the module's top-level functions.
            provider_handlers = {}
            for fn_name in ("complete", "stream", "complete_json", "complete_tools"):
                fn = getattr(module, fn_name, None)
                if fn and callable(fn):
                    provider_handlers[fn_name] = fn
            if provider_handlers:
                try:
                    from huginn.llm.client import LLMClient as _LC
                    _LC.register_provider(tool_id, provider_handlers)
                    stm.record(
                        source="system", type="internal",
                        payload={"subtype": "provider_registered",
                                 "tool_id": tool_id,
                                 "methods": list(provider_handlers)},
                    )
                except Exception as e:
                    stm.record(
                        source="system", type="internal",
                        payload={"subtype": "provider_register_error",
                                 "tool_id": tool_id, "error": str(e)},
                    )

        # Step 8: install-complete STM event
        stm.record(
            source="system", type="internal",
            payload={
                "subtype":             "tool_installed",
                "tool_id":             tool_id,
                "title":               manifest.title,
                "mode":                manifest.mode,
                "perception_capable":  manifest.perception_capable,
                "pipeline_activated":  activate_pipeline and manifest.perception_capable,
            },
        )

        return descriptor

    def _execute_native(self, tool_id: str, args: dict) -> str:
        """
        Execute a Huginn-native tool that does not delegate to Muninn ToolExecutor.

        get_instructions  — recall an LTM instruction artifact by topic
        htm_state_get     — read one key or a namespace prefix from HTM.states
        """
        if tool_id == "get_instructions":
            return self._get_instructions(args.get("topic", ""))

        if tool_id == "htm_state_get":
            if self.htm is None:
                return "(HTM not available)"
            key    = args.get("key", "")
            prefix = args.get("prefix", "")
            if key:
                val = self.htm.states.get(key)
                return f"{key} = {val!r}" if val is not None else f"(not set: {key})"
            if prefix:
                snap = self.htm.states.list(prefix)
                if not snap:
                    return f"(no states with prefix '{prefix}')"
                return "\n".join(f"{k} = {v!r}" for k, v in sorted(snap.items()))
            return "(provide key= or prefix=)"

        return f"(unknown native tool: {tool_id})"

    def _get_instructions(self, topic: str) -> str:
        """
        Recall a Huginn instruction artifact from Muninn LTM.

        Instruction artifacts are stored as class_type="instruction" with
        topics=["instruction", "instruction.<topic>", "artux.instruction"].
        Logos writes them on first boot via _ensure_instruction_defaults().
        """
        if not topic:
            topics = ["htm_tasks", "skill_execution", "memory", "states",
                      "live_tools", "staging", "entities", "speech_step"]
            return (
                "Available instruction topics:\n"
                + "\n".join(f"  {t}" for t in topics)
                + "\n\nFetch one with: get_instructions(topic='<name>')"
            )

        try:
            try:
                from memory_module.recall import RecallQuery
                q = RecallQuery(
                    topics=[f"instruction.{topic}", "instruction"],
                    top_k=1,
                )
            except ImportError:
                q = f"instruction {topic}"

            results = self.muninn.recall(q, top_k=1)
            if not results:
                return (
                    f"No instructions found for topic '{topic}'.\n"
                    f"Run get_instructions() with no args to see available topics."
                )
            entry = getattr(results[0], "entry", results[0])
            content = getattr(entry, "content", str(entry))
            return content

        except Exception as e:
            return f"(error fetching instructions for '{topic}': {e})"

    def _install_dependencies(
        self, dependencies: list[str], stm=None
    ) -> list[str]:
        """
        pip-install a list of dependency strings.
        Returns a list of any that failed (non-fatal).
        """
        failures = []
        for dep in dependencies:
            dep = dep.strip()
            if not dep:
                continue
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", dep,
                     "--quiet", "--no-warn-script-location"],
                    capture_output = True,
                    text           = True,
                    timeout        = 120,
                )
                if result.returncode != 0:
                    failures.append(f"{dep}: {result.stderr.strip()[:120]}")
            except subprocess.TimeoutExpired:
                failures.append(f"{dep}: install timeout")
            except Exception as e:
                failures.append(f"{dep}: {e}")
        return failures

    @staticmethod
    def _load_module(source_path: str, tool_id: str):
        """
        Load a Python source file as a module using importlib.
        The module name is derived from tool_id (dots/dashes → underscores).
        Re-uses a cached module if the source hasn't changed.
        """
        module_name = f"huginn_tool_{tool_id.replace('.', '_').replace('-', '_')}"
        spec   = importlib.util.spec_from_file_location(module_name, source_path)
        module = importlib.util.module_from_spec(spec)

        # Add to sys.modules so relative imports work inside the tool
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    # ------------------------------------------------------------------
    # Schema serving (for Sagax's LLM tool-call context)
    # ------------------------------------------------------------------

    def get_schemas(
        self,
        format:          str = "openai",
        include_memory:  bool = True,
        include_world:   bool = True,
        from_hot_cache:  bool = True,
        top_k_from_ltm:  int = 0,
    ) -> list[dict]:
        """
        Return tool schemas ready to pass to an LLM API call.

        Parameters
        ----------
        format : "openai" | "anthropic"
        include_memory : bool
            Include Muninn memory tool schemas.
        include_world : bool
            Include registered world tool schemas.
        from_hot_cache : bool
            Include schemas from HTM ASC hot_tools (recently used tools).
        top_k_from_ltm : int
            If > 0, also recall top_k tool descriptors from LTM by recency.
            Use sparingly — recall is expensive.
        """
        from memory_module.tools import get_tools as _muninn_get_tools

        schemas = []

        if include_memory:
            if format == "openai":
                schemas += _muninn_get_tools(format="openai")
            else:
                schemas += _muninn_get_tools(format="anthropic")

        if include_world:
            for descriptor in self._world.values():
                if format == "openai":
                    schemas.append(descriptor.to_openai_schema())
                else:
                    schemas.append(descriptor.to_anthropic_schema())

        if from_hot_cache and self.htm is not None:
            hot_tool_ids = list(self.htm.asc.hot_tools.keys())
            for tid in hot_tool_ids:
                if tid in self._schema_cache and tid not in self._world:
                    d = self._schema_cache[tid]
                    if format == "openai":
                        schemas.append(d.to_openai_schema())
                    else:
                        schemas.append(d.to_anthropic_schema())

        if top_k_from_ltm > 0:
            ltm_schemas = self._recall_tool_schemas_from_ltm(
                top_k_from_ltm, format
            )
            # Deduplicate by tool_id
            existing_ids = {s.get("function", s).get("name", "")
                            for s in schemas}
            for s in ltm_schemas:
                tid = s.get("function", s).get("name", "")
                if tid not in existing_ids:
                    schemas.append(s)

        return schemas

    def get_schema_for(self, tool_id: str, format: str = "openai") -> Optional[dict]:
        """Return the schema for a single tool by ID."""
        # Memory tier
        if tool_id in _MEMORY_POLARITY:
            from memory_module.tools import get_tools as _muninn_get_tools
            for s in _muninn_get_tools(format=format):
                name = s.get("function", s).get("name", "")
                if name == tool_id:
                    return s
            return None

        # World tier (cache first, then LTM)
        descriptor = (
            self._world.get(tool_id)
            or self._schema_cache.get(tool_id)
            or self._fetch_descriptor_from_ltm(tool_id)
        )
        if format == "openai":
            return descriptor.to_openai_schema()
        return descriptor.to_anthropic_schema()

    def get_polarity(self, tool_id: str) -> str:
        """Return 'read' or 'write' for a tool. Defaults to 'write' if unknown."""
        if tool_id in _MEMORY_POLARITY:
            return _MEMORY_POLARITY[tool_id]
        d = self._world.get(tool_id) or self._schema_cache.get(tool_id)
        return d.polarity if d else "write"

    # ------------------------------------------------------------------
    # LTM descriptor recall
    # ------------------------------------------------------------------

    def _fetch_descriptor_from_ltm(self, tool_id: str) -> ToolDescriptor:
        """
        Look up a tool descriptor from Muninn LTM by tool_id.
        Returns a stub descriptor if not found.
        """
        if tool_id in self._schema_cache:
            return self._schema_cache[tool_id]

        try:
            try:
                from memory_module.recall import RecallQuery as _RQ
                _q = _RQ(semantic_query=f"tool {tool_id}",
                         topics=["tool", tool_id], top_k=5)
            except ImportError:
                _q = f"tool {tool_id}"
            results = self.muninn.recall(_q, top_k=5)
            for r in results:
                try:
                    data = json.loads(r.entry.content)
                    if data.get("tool_id") == tool_id:
                        desc = _ltm_content_to_descriptor(data)
                        self._schema_cache[tool_id] = desc
                        return desc
                except Exception:
                    pass
        except Exception:
            pass

        # Stub
        return ToolDescriptor(
            tool_id            = tool_id,
            title              = tool_id,
            capability_summary = "(not found in LTM)",
            polarity           = "write",
            permission_scope   = [],
            inputs             = {},
            outputs            = {},
        )

    def _recall_tool_schemas_from_ltm(
        self, top_k: int, format: str
    ) -> list[dict]:
        """Recall recent tool descriptors from LTM for the schema list."""
        try:
            try:
                from memory_module.recall import RecallQuery as _RQ
                _q = _RQ(semantic_query="tool capability artifact",
                         topics=["tool"], top_k=top_k)
            except ImportError:
                _q = "tool artifact descriptor"
            results = self.muninn.recall(_q, top_k=top_k)
            schemas = []
            for r in results:
                try:
                    data = json.loads(r.entry.content)
                    if data.get("artifact_type") != "tool":
                        continue
                    tid = data.get("tool_id", "")
                    if not tid:
                        continue
                    desc = _ltm_content_to_descriptor(data)
                    self._schema_cache[tid] = desc
                    if format == "openai":
                        schemas.append(desc.to_openai_schema())
                    else:
                        schemas.append(desc.to_anthropic_schema())
                except Exception:
                    pass
            return schemas
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Hot tools cache update
    # ------------------------------------------------------------------

    def _update_hot_tools(self, tool_id: str, descriptor: ToolDescriptor):
        """Update HTM ASC hot_tools after a successful call."""
        if self.htm is None:
            return
        try:
            self.htm.asc.update_tool_usage(tool_id)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers for building tool result messages (Anthropic / OpenAI format)
    # ------------------------------------------------------------------

    def format_result_anthropic(self, tool_id: str, call_id: str, result: ToolResult) -> dict:
        """Format a ToolResult as an Anthropic tool_result content block."""
        content = (
            result.output if isinstance(result.output, str)
            else json.dumps(result.output)
        )
        if not result.success:
            content = f"Error: {result.error}"
        return {
            "type":        "tool_result",
            "tool_use_id": call_id,
            "content":     content,
            "is_error":    not result.success,
        }

    def format_result_openai(self, tool_id: str, call_id: str, result: ToolResult) -> dict:
        """Format a ToolResult as an OpenAI tool-role message."""
        content = (
            result.output if isinstance(result.output, str)
            else json.dumps(result.output)
        )
        if not result.success:
            content = f"Error: {result.error}"
        return {
            "role":         "tool",
            "tool_call_id": call_id,
            "content":      content,
        }


# ---------------------------------------------------------------------------
# LTM artifact → ToolDescriptor conversion
# ---------------------------------------------------------------------------

def _ltm_content_to_descriptor(data: dict) -> ToolDescriptor:
    """Parse a stored tool artifact dict into a ToolDescriptor."""
    inputs   = data.get("inputs", {})
    required = [
        k for k, v in inputs.items()
        if v.get("required", False) or k not in (
            data.get("defaults", {})
        )
    ]
    return ToolDescriptor(
        tool_id            = data.get("tool_id", data.get("name", "")),
        title              = data.get("title", ""),
        capability_summary = data.get("capability_summary", data.get("description", "")),
        polarity           = data.get("polarity", "write"),
        permission_scope   = data.get("permission_scope", []),
        inputs             = inputs,
        outputs            = data.get("outputs", {}),
        required           = data.get("required", required),
        tier               = "world",
    )


# ---------------------------------------------------------------------------
# register_tool() convenience helper (used in README examples)
# ---------------------------------------------------------------------------

def register_tool(muninn, tool_manager: ToolManager, descriptor_dict: dict, handler: Callable):
    """
    Write a tool descriptor to Muninn LTM and register its handler.

    Parameters
    ----------
    muninn : MemoryAgent
    tool_manager : ToolManager
    descriptor_dict : dict
        Must include: tool_id, title, capability_summary, polarity,
        permission_scope, inputs, outputs.
    handler : Callable
        Python function implementing the tool.

    Example
    -------
        register_tool(muninn, huginn.tool_manager, {
            "tool_id":            "tool.set_ceiling_lights.v1",
            "title":              "Set ceiling lights",
            "capability_summary": "Control ceiling ambient lighting.",
            "polarity":           "write",
            "permission_scope":   ["lights.ceiling"],
            "inputs":  {"colour": {"type": "string"},
                        "brightness": {"type": "integer", "default": 80}},
            "outputs": {"status": {"type": "string"}},
        }, my_lights_fn)
    """
    # Store descriptor in LTM so future sessions can recall it
    full = {"artifact_type": "tool", **descriptor_dict}
    muninn.consolidate_ltm(
        narrative   = json.dumps(full),
        class_type  = "tool",
        topics      = ["tool", descriptor_dict.get("tool_id", "")] + descriptor_dict.get("permission_scope", []),
        confidence  = 1.0,
    )

    # Register handler with the ToolManager
    descriptor = _ltm_content_to_descriptor(full)
    tool_manager.register(descriptor_dict["tool_id"], handler, descriptor)
