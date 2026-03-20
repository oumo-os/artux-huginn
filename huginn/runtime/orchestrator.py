"""
runtime/orchestrator.py — The Orchestrator: execution runtime and routing bridge.

No cognitive logic lives here. The Orchestrator:
  - Owns the session (entity, grants)
  - Runs the Perception Manager (pipeline supervision)
  - Runs the Exilis poll loop
  - Routes Sagax Narrator token stream to TTS / Tool Manager / HTM / UI
  - Enforces the permission gate on <tool_call> blocks
  - Dispatches <aug_call> inline (read-only, parallel, with timeouts)
  - Issues the two-stage nudge on urgent Exilis signals
  - Ticks the HTM scheduler at 1 Hz
  - Writes workbook entries on every block close

Token stream state machine (Orchestrator.md §3):
  IDLE → CAPTURING_THINKING | CAPTURING_CONTEMPLATION | STREAMING_SPEECH
       | BUFFERING_TOOL_CALL | BUFFERING_AUG_CALL | BUFFERING_TASK_UPDATE
       | BUFFERING_PROJECTION

This file wires together all Huginn components. Instantiate Orchestrator
last, passing in all other components.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .stm import STMStore, STMEvent
from .htm import HTM
from .perception import PerceptionManager, SessionContext


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class Session:
    session_id:            str = ""
    entity_id:             str = ""
    state:                 str = "active"   # active | suspended | ended
    permission_scope:      list[str] = field(default_factory=list)
    denied:                list[str] = field(default_factory=list)
    confirmation_required: list[str] = field(default_factory=list)

    def as_context(self) -> SessionContext:
        return SessionContext(
            session_id       = self.session_id,
            entity_id        = self.entity_id,
            permission_scope = self.permission_scope,
            denied           = self.denied,
        )

    def allows(self, scope: str) -> bool:
        if scope in self.denied:
            return False
        return scope in self.permission_scope

    def needs_confirmation(self, scope: str) -> bool:
        return scope in self.confirmation_required


# ---------------------------------------------------------------------------
# Narrator state machine states
# ---------------------------------------------------------------------------

class NarratorState:
    IDLE                    = "IDLE"
    CAPTURING_THINKING      = "CAPTURING_THINKING"   # catches <thinking> AND <think>
    CAPTURING_CONTEMPLATION = "CAPTURING_CONTEMPLATION"
    STREAMING_SPEECH        = "STREAMING_SPEECH"
    BUFFERING_TOOL_CALL     = "BUFFERING_TOOL_CALL"
    BUFFERING_AUG_CALL      = "BUFFERING_AUG_CALL"
    BUFFERING_TASK_UPDATE   = "BUFFERING_TASK_UPDATE"
    BUFFERING_PROJECTION    = "BUFFERING_PROJECTION"
    STREAMING_SPEECH_STEP   = "STREAMING_SPEECH_STEP"   # speech_step: streaming + waiting for user


# Block open/close patterns
_BLOCK_OPEN  = re.compile(r"<(thinking|think|contemplation|speech|speech_step|tool_call|aug_call|task_update|projection)(\s[^>]*)?>") 
_BLOCK_CLOSE = re.compile(r"</(thinking|think|contemplation|speech|speech_step|tool_call|aug_call|task_update|projection)>")
_TARGET_ATTR = re.compile(r'target="([^"]+)"')
_TIMEOUT_ATTR = re.compile(r'timeout_ms="(\d+)"')
_VAR_ATTR    = re.compile(r'var="([^"]+)"')


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Execution runtime. Owns session, coordinates all agents.

    Parameters
    ----------
    stm : STMStore
    htm : HTM
    perception : PerceptionManager
    exilis : Exilis
    sagax : Sagax
    logos : Logos
    tool_manager : ToolManager (or callable dict)
    on_tts_token : callable(str)     — forward speech tokens to TTS
    on_ui_projection : callable(dict) — forward projection to UI
    on_confirmation_required : callable(tool, session) → bool
        Called when a tool needs user confirmation. Should return True
        if the user confirmed.
    """

    DEFAULT_AUG_TIMEOUT_MS = 500
    SCHEDULER_TICK_HZ      = 1.0

    def __init__(
        self,
        stm:                       STMStore,
        htm:                       HTM,
        perception:                PerceptionManager,
        exilis,
        sagax,
        logos,
        tool_manager:              "ToolManager",
        on_tts_token:              Optional[Callable]  = None,
        on_ui_projection:          Optional[Callable]  = None,
        on_confirmation_required:  Optional[Callable]  = None,
        actuation_bus              = None,
        actuation_manager          = None,
    ):
        self.stm                      = stm
        self.htm                      = htm
        self.perception               = perception
        self.exilis                   = exilis
        self.sagax                    = sagax
        self.logos                    = logos
        self.tool_manager             = tool_manager
        self.on_tts_token             = on_tts_token
        self.on_ui_projection         = on_ui_projection
        self.on_confirmation_required = on_confirmation_required
        self.actuation_bus            = actuation_bus
        self.actuation_manager        = actuation_manager

        self.session = Session()

        # Narrator state machine
        self._narrator_state      = NarratorState.IDLE
        self._block_buffer        = ""
        self._speech_target       = ""
        self._aug_timeout_ms      = self.DEFAULT_AUG_TIMEOUT_MS
        # speech_step state
        self._speech_step_var:    str            = ""
        self._speech_step_event:  threading.Event = threading.Event()
        self._speech_step_pending: bool           = False
        # Speech chunker state
        self._chunk_buf:          str   = ""   # accumulates tokens between chunk boundaries
        self._chunk_token_count:  int   = 0

        # Sagax wake queue fed by Exilis
        self._sagax_wake_pending = False

        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="OrchestratorWorker")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, session: Session = None):
        if session:
            self.session = session
        self._running = True
        # Recall system config from Muninn LTM and apply to all LLM clients.
        # This runs synchronously before any agent starts so every agent
        # has correct backend/model/key config from its first cycle.
        self._apply_system_config(self._recall_system_config())
        # Wire Exilis callbacks
        self.exilis.on_act    = self._on_exilis_act
        self.exilis.on_urgent = self._on_exilis_urgent
        # Wire Sagax Narrator
        self.sagax.on_narrator_token = self.on_narrator_token
        # Register internal tools that Sagax can call via <tool_call>
        self._register_internal_tools()
        # Start all agents
        self.exilis.start()
        self.sagax.start()
        self.logos.start()
        # Start any HTM-registered live actuation tools
        if self.actuation_manager is not None:
            self.actuation_manager.start_from_htm(self.tool_manager)
        # Start scheduler tick
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True, name="SchedulerThread"
        )
        self._scheduler_thread.start()

    # ------------------------------------------------------------------
    # System config — recall from Muninn LTM, apply to all LLM clients
    # ------------------------------------------------------------------

    # Topic keys stored in Muninn — distinctive enough for exact structured match
    _CFG_TOPICS = {
        "exilis":  "artux.config.llm.exilis.v1",
        "sagax":   "artux.config.llm.sagax.v1",
        "logos":   "artux.config.llm.logos.v1",
    }

    def _recall_system_config(self) -> dict:
        """
        Recall all artux.config.* LTM entries from Muninn.

        Returns a dict keyed by role ("exilis", "sagax", "logos") mapping
        to a parsed config dict.  Missing roles fall back to an empty dict
        (the LLMClient stays at its fallback values from build_huginn).

        Keys never surface in Sagax context — this path is Orchestrator-only.
        """
        import json
        muninn = self.tool_manager.muninn
        configs = {}
        for role, topic in self._CFG_TOPICS.items():
            try:
                try:
                    from memory_module.recall import RecallQuery as _RQ
                    _q = _RQ(topics=[topic], top_k=1)
                except ImportError:
                    _q = topic
                results = muninn.recall(_q, top_k=1)
                if results:
                    entry = results[0]
                    # RecallResult wraps the LTMEntry — access .entry or directly
                    raw = getattr(entry, "entry", entry)
                    content = getattr(raw, "content", "")
                    if content:
                        configs[role] = json.loads(content)
            except Exception:
                pass   # non-fatal — fallback values remain
        return configs

    def _apply_system_config(self, configs: dict):
        """
        Apply recalled config dicts to each agent's LLMClient.

        Each agent exposes a .llm attribute holding its LLMClient.
        LLMClient.reconfigure() handles partial dicts safely.
        """
        role_agents = {
            "exilis": getattr(self.exilis, "llm", None),
            "sagax":  getattr(self.sagax,  "llm", None),
            "logos":  getattr(self.logos,  "llm", None),
        }
        for role, client in role_agents.items():
            cfg = configs.get(role, {})

            # Populate HTM.states from config first — LLMClient reads from
            # states at call time, so no reconfigure() call is needed.
            # load_from_config does NOT mark keys dirty (they came from LTM).
            if cfg:
                self.htm.states.load_from_config(cfg, namespace=role)

            # reconfigure() is kept as a compat shim for any code that still
            # calls it directly. It now just mirrors writes into states.
            if cfg and client is not None and hasattr(client, "reconfigure"):
                try:
                    client.reconfigure(
                        backend     = cfg.get("backend", cfg.get("provider", "")),
                        model       = cfg.get("model", ""),
                        host        = cfg.get("host", ""),
                        api_key     = cfg.get("api_key", ""),
                        temperature = cfg.get("temperature"),
                        timeout     = cfg.get("timeout"),
                    )
                except Exception as e:
                    self.stm.record(
                        source="system", type="internal",
                        payload={
                            "subtype": "config_apply_error",
                            "role":    role,
                            "error":   str(e),
                        },
                    )

    def _register_internal_tools(self):
        """
        Register Orchestrator-internal tools that Sagax can call via
        <tool_call> blocks. These never go through the permission gate
        (they are system calls, not world-facing actuators).
        """
        from .tool_manager import ToolDescriptor

        # request_early_logos_cycle — triggers an immediate Logos pass
        # Used by Sagax when user says "urgent" during tool install confirmation
        self.tool_manager.register(
            tool_id    = "request_early_logos_cycle",
            handler    = self._handle_early_logos_cycle,
            descriptor = ToolDescriptor(
                tool_id            = "request_early_logos_cycle",
                title              = "Request early Logos maintenance cycle",
                capability_summary = (
                    "Ask Logos to run an immediate maintenance cycle, "
                    "installing any affirmed staged tools without waiting "
                    "for the normal interval. Use only when user says urgent."
                ),
                polarity           = "write",
                permission_scope   = [],
                inputs             = {},
                outputs            = {"status": {"type": "string"}},
                tier               = "world",
            ),
        )

        self._register_am_tools()

    def _handle_early_logos_cycle(self) -> dict:
        """Handler for the request_early_logos_cycle internal tool."""
        try:
            self.logos.request_early_cycle()
            return {"status": "ok", "message": "Early Logos cycle requested."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _register_am_tools(self):
        """Register actuation start/stop control tools."""
        if self.actuation_manager is None:
            return
        from .tool_manager import ToolDescriptor

        def _start(tool_id: str = "") -> dict:
            if not tool_id:
                return {"status": "error", "error": "tool_id required"}
            desc = self.tool_manager.get_descriptor(tool_id)
            if desc is None:
                return {"status": "error", "error": f"Unknown tool: {tool_id}"}
            ok = self.actuation_manager.start_tool(
                tool_id       = tool_id,
                source_path   = desc.source_path,
                subscriptions = desc.subscriptions,
                title         = desc.title,
            )
            if ok:
                # Ensure HTM live_tool task exists
                existing = self.htm.query(tags_all=["live_tool", tool_id])
                if not existing:
                    self.htm.create(
                        title        = f"Live tool: {desc.title}",
                        initiated_by = "sagax",
                        persistence  = "persist",
                        tags         = ["live_tool", "actuation", tool_id],
                    )
            return {"status": "started" if ok else "failed", "tool_id": tool_id}

        def _stop(tool_id: str = "") -> dict:
            if not tool_id:
                return {"status": "error", "error": "tool_id required"}
            ok = self.actuation_manager.stop_tool(tool_id)
            return {"status": "stopped" if ok else "not_running", "tool_id": tool_id}

        def _list() -> dict:
            return {
                "running": self.actuation_manager.running_tool_ids(),
                "states":  self.htm.states.summary(),
            }

        for tid, fn, summary, inputs in [
            ("tool.actuation.start", _start,
             "Start a live service tool as a daemon. Use for TTS, ASR, display tools.",
             {"tool_id": {"type": "string", "description": "Tool ID to start"}}),
            ("tool.actuation.stop",  _stop,
             "Stop a running live service tool.",
             {"tool_id": {"type": "string", "description": "Tool ID to stop"}}),
            ("tool.actuation.list",  _list,
             "List all running live tools and their current states.",
             {}),
        ]:
            self.tool_manager.register(
                tool_id    = tid,
                handler    = fn,
                descriptor = ToolDescriptor(
                    tool_id            = tid,
                    title              = tid.split(".")[-1].capitalize() + " live tool",
                    capability_summary = summary,
                    polarity           = "write",
                    permission_scope   = [],
                    inputs             = inputs,
                    outputs            = {"status": {"type": "string"}},
                    tier               = "world",
                ),
            )

    def request_early_logos_cycle(self):
        """Public proxy — called directly if needed outside a tool_call context."""
        self.logos.request_early_cycle()

    def stop(self):
        self._running = False
        self.exilis.stop()
        self.sagax.stop()
        self.logos.stop()
        self._executor.shutdown(wait=False)

    def new_session(
        self,
        entity_id:            str,
        permission_scope:     list[str],
        denied:               list[str] = None,
        confirmation_required: list[str] = None,
    ) -> str:
        import uuid
        sid = f"sess-{uuid.uuid4().hex[:8]}"
        self.session = Session(
            session_id            = sid,
            entity_id             = entity_id,
            state                 = "active",
            permission_scope      = permission_scope,
            denied                = denied or [],
            confirmation_required = confirmation_required or [],
        )
        self.htm.new_session(sid)
        self.sagax.entity_id       = entity_id
        self.sagax.permission_scope = permission_scope
        return sid

    def end_session(self):
        self.session.state = "ended"
        self.stm.record(
            source="system", type="internal",
            payload={"subtype": "end_of_session",
                     "session_id": self.session.session_id},
        )
        self.logos.session_end_flush(self.session.session_id)

    # ------------------------------------------------------------------
    # Exilis signal handlers
    # ------------------------------------------------------------------

    def _on_exilis_act(self):
        """Queue a normal Sagax wake."""
        self.sagax.wake()

    def _on_exilis_urgent(self, event: STMEvent):
        """Two-stage nudge (Orchestrator.md §4)."""
        # Stage 1: immediate halt
        if self._narrator_state == NarratorState.STREAMING_SPEECH:
            # Mark in-progress speech event as suspended
            self.stm.record(
                source="system", type="output",
                payload={"subtype": "speech", "status": "suspended",
                         "reason": "urgent_interrupt"},
            )
            if self.on_tts_token:
                self.on_tts_token("\x00")   # sentinel: halt TTS

        elif self._narrator_state == NarratorState.BUFFERING_TOOL_CALL:
            self._block_buffer = ""
            self.htm.asc.workbook_write("tool_call_aborted", {"reason": "nudge"})

        elif self._narrator_state == NarratorState.BUFFERING_AUG_CALL:
            # Cancel pending aug dispatches — result will be nudge_interrupt
            self._block_buffer = ""

        self._narrator_state = NarratorState.IDLE

        # Stage 2: deliver context and wake Sagax with priority
        self.sagax.wake(signal=type("WakeSignal", (), {
            "priority": "urgent", "event": event
        })())

    # ------------------------------------------------------------------
    # Narrator token stream handler (called per-token from Sagax)
    # ------------------------------------------------------------------

    def on_narrator_token(self, token: str):
        """
        Process one token from the Sagax Narrator stream.
        Drives the tag-state machine and routes content to consumers.
        """
        # Feed to state machine character by character for simplicity
        # In production: buffer and scan for tag boundaries
        self._block_buffer += token

        # Check for block open
        if self._narrator_state == NarratorState.IDLE:
            m = _BLOCK_OPEN.search(self._block_buffer)
            if m:
                tag   = m.group(1)
                attrs = m.group(2) or ""
                self._block_buffer = self._block_buffer[m.end():]
                self._on_block_open(tag, attrs)
            return

        # Check for block close
        m = _BLOCK_CLOSE.search(self._block_buffer)
        if m:
            tag     = m.group(1)
            content = self._block_buffer[:m.start()]
            self._block_buffer = self._block_buffer[m.end():]
            self._on_block_close(tag, content)
            return

        # While streaming speech or speech_step: chunk + forward to TTS and bus
        if self._narrator_state in (NarratorState.STREAMING_SPEECH,
                                    NarratorState.STREAMING_SPEECH_STEP):
            if self.on_tts_token:
                self.on_tts_token(token)    # raw token stream for legacy callers
            self._feed_speech_chunk(token)  # chunk-level bus publish

    def _on_block_open(self, tag: str, attrs: str):
        state_map = {
            "thinking":       NarratorState.CAPTURING_THINKING,
            "think":          NarratorState.CAPTURING_THINKING,   # model alias
            "contemplation":  NarratorState.CAPTURING_CONTEMPLATION,
            "speech":         NarratorState.STREAMING_SPEECH,
            "speech_step":    NarratorState.STREAMING_SPEECH_STEP,
            "tool_call":      NarratorState.BUFFERING_TOOL_CALL,
            "aug_call":       NarratorState.BUFFERING_AUG_CALL,
            "task_update":    NarratorState.BUFFERING_TASK_UPDATE,
            "projection":     NarratorState.BUFFERING_PROJECTION,
        }
        self._narrator_state = state_map.get(tag, NarratorState.IDLE)

        if tag in ("speech", "speech_step"):
            m = _TARGET_ATTR.search(attrs)
            self._speech_target = m.group(1) if m else self.session.entity_id

        if tag == "speech_step":
            m = _VAR_ATTR.search(attrs)
            self._speech_step_var = m.group(1) if m else "speech_step_result"

        if tag == "aug_call":
            m = _TIMEOUT_ATTR.search(attrs)
            self._aug_timeout_ms = int(m.group(1)) if m else self.DEFAULT_AUG_TIMEOUT_MS

    def _on_block_close(self, tag: str, content: str):
        if tag == "thinking":
            # Debug log only — never stored
            self._narrator_state = NarratorState.IDLE
            return

        if tag == "contemplation":
            if self.actuation_bus:
                self.actuation_bus.publish_dict(
                    type="output", target="contemplation", complete="full",
                    text=content.strip(), session_id=self.session.session_id,
                )
            self.stm.record(
                source="system", type="output",
                payload={"subtype": "contemplation", "text": content.strip()},
            )
            self.htm.asc.workbook_write("contemplation", content.strip())

        elif tag == "speech":
            self._publish_speech_full(content.strip(), self._speech_target)
            self.stm.record(
                source="system", type="output",
                payload={"subtype": "speech", "text": content.strip(),
                         "target": self._speech_target, "status": "complete"},
            )
            self.htm.asc.workbook_write("speech", content.strip())

        elif tag == "speech_step":
            self._handle_speech_step(content)

        elif tag == "tool_call":
            self._handle_tool_call(content)

        elif tag == "aug_call":
            self._handle_aug_call(content)

        elif tag == "task_update":
            self._handle_task_update(content)

        elif tag == "projection":
            self._handle_projection(content)

        self._narrator_state = NarratorState.IDLE

    # ------------------------------------------------------------------
    # <tool_call> dispatch
    # ------------------------------------------------------------------

    def _handle_tool_call(self, content: str):
        tools = _parse_tool_list(content)
        for tool in tools:
            tool_id = tool.get("name", "")
            scope   = tool.get("permission_scope", [])

            # Permission gate
            denied_scope = next(
                (s for s in scope
                 if s in self.session.denied
                 or s not in self.session.permission_scope),
                None,
            )
            if denied_scope:
                self.stm.record(
                    source="system", type="tool_result",
                    payload={"tool_id": tool_id, "success": False,
                             "result": {"status": "denied",
                                        "reason": f"scope_not_granted: {denied_scope}"}},
                )
                self.htm.asc.workbook_write("tool_call_denied", tool)
                continue

            # Confirmation gate
            needs_conf = any(s in self.session.confirmation_required for s in scope)
            if needs_conf and self.on_confirmation_required:
                confirmed = self.on_confirmation_required(tool, self.session)
                if not confirmed:
                    self.stm.record(
                        source="system", type="tool_result",
                        payload={"tool_id": tool_id, "success": False,
                                 "result": {"status": "confirmation_declined"}},
                    )
                    continue

            # Dispatch asynchronously
            request_id = f"req-{tool_id}-{int(time.time_ns())}"
            self.htm.asc.workbook_write("tool_call", tool)

            # Create/update HTM task record for this tool call
            task_id = self.htm.create(
                title        = f"Tool: {tool_id}",
                initiated_by = "sagax",
                persistence  = "volatile",
                tags         = ["tool_call", tool_id],
                session_id   = self.session.session_id,
            )

            def _dispatch(t=tool, rid=request_id, tid=task_id):
                result = self.tool_manager.execute(t["name"], t.get("args", {}))
                self.perception.write_tool_result_event(
                    tool_id    = t["name"],
                    request_id = rid,
                    result     = result.output if hasattr(result, "output")
                                 else (result if isinstance(result, dict) else {"output": result}),
                    success    = result.success if hasattr(result, "success") else True,
                    duration_ms = result.duration_ms if hasattr(result, "duration_ms") else 0,
                )
                self.htm.complete(tid, output={
                    "result":  result.output if hasattr(result, "output") else result,
                    "success": result.success if hasattr(result, "success") else True,
                })

            self._executor.submit(_dispatch)

    # ------------------------------------------------------------------
    # <aug_call> dispatch (inline, read-only, blocking)
    # ------------------------------------------------------------------

    def _handle_aug_call(self, content: str):
        tools   = _parse_tool_list(content)
        results = {}

        futures = {}
        for tool in tools:
            name    = tool.get("name", "")
            args    = tool.get("args", {})
            t_ms    = tool.get("timeout_ms", self._aug_timeout_ms)

            # Reject any write-polarity tools in aug_call
            polarity = self.tool_manager.get_polarity(name)
            if polarity != "read":
                results[name] = {"status": "error",
                                 "reason": "aug_call_write_tool_rejected"}
                continue

            future = self._executor.submit(
                self.tool_manager.execute, name, args
            )
            futures[name] = (future, t_ms)

        for name, (future, t_ms) in futures.items():
            try:
                result = future.result(timeout=t_ms / 1000.0)
                results[name] = result if isinstance(result, dict) else {"output": result}
            except FuturesTimeout:
                results[name] = {"status": "timeout", "tool": name}
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}

        # Inject <aug_result> back into Sagax
        self.sagax.inject_aug_result(results)
        self.htm.asc.workbook_write("aug_call", {"tools": tools, "results": results})

    # ------------------------------------------------------------------
    # <task_update> write
    # ------------------------------------------------------------------

    def _handle_task_update(self, content: str):
        try:
            update = json.loads(content.strip())
        except Exception:
            return

        action  = update.get("action", "update")   # create | update | complete | park
        task_id = update.get("task_id", "")

        if action == "create":
            task_id = self.htm.create(
                title        = update.get("title", "Unnamed task"),
                initiated_by = "sagax",
                persistence  = update.get("persistence", "volatile"),
                tags         = update.get("tags", []),
                session_id   = self.session.session_id,
                remind_at    = update.get("remind_at"),
                expiry_at    = update.get("expiry_at"),
            )
        elif action == "update" and task_id:
            self.htm.update(
                task_id,
                state     = update.get("state"),
                progress  = update.get("progress"),
                resume_at = update.get("resume_at"),
                note      = update.get("note", ""),
            )
        elif action == "complete" and task_id:
            self.htm.complete(
                task_id,
                output     = update.get("output", {}),
                confidence = update.get("confidence", 1.0),
                note       = update.get("note", ""),
            )
        elif action == "park" and task_id:
            self.sagax.park_current_task(
                task_id,
                resume_at = update.get("resume_at", ""),
                note      = update.get("note", "Paused by Sagax."),
            )

        elif action == "state_set":
            key   = update.get("key", "")
            value = update.get("value")
            if key:
                self.htm.states.set(key, value)
                self.stm.record(
                    source="system", type="internal",
                    payload={"subtype": "state_changed",
                             "key": key, "value": value},
                )

        elif action == "state_get":
            # Result injected back as aug_result — caller must use aug_call
            # for inline reads. For fire-and-forget reads this is a no-op.
            key    = update.get("key", "")
            prefix = update.get("prefix", "")
            if key:
                val = self.htm.states.get(key)
                self.sagax.inject_aug_result({key: val})
            elif prefix:
                vals = self.htm.states.list(prefix)
                self.sagax.inject_aug_result(vals)

        elif action == "state_delete":
            key = update.get("key", "")
            if key:
                self.htm.states.delete(key)

        self.htm.asc.workbook_write("task_update", update)

    # ------------------------------------------------------------------
    # <projection> dispatch
    # ------------------------------------------------------------------

    def _handle_projection(self, content: str):
        try:
            data = json.loads(content.strip())
        except Exception:
            data = {"text": content.strip()}

        if self.on_ui_projection:
            self.on_ui_projection(data)

        self.stm.record(
            source="system", type="output",
            payload={"subtype": "projection", "data": data},
        )
        self.htm.asc.workbook_write("projection", data)

    # ------------------------------------------------------------------
    # Speech chunker — phrase-boundary publishing to ActuationBus
    # ------------------------------------------------------------------

    # Minimum tokens before a clause boundary triggers a chunk
    _CHUNK_MIN_TOKENS = 6
    # Characters that end a sentence (flush immediately)
    _SENTENCE_END = frozenset(".?!…")
    # Characters that end a clause (flush after min tokens)
    _CLAUSE_END   = frozenset(",;:")

    def _feed_speech_chunk(self, token: str) -> None:
        """
        Accumulate speech tokens and publish chunk events to the ActuationBus
        at natural phrase boundaries.

        partial — individual tokens (very high frequency, low latency)
        chunk   — phrase boundary flush (the main TTS synthesis unit)
        full    — written to STM at </speech> block close
        """
        if not self.actuation_bus:
            return

        self._chunk_buf         += token
        self._chunk_token_count += 1

        # Publish partial token (for avatar lip sync etc.)
        self.actuation_bus.publish_dict(
            type       = "output",
            target     = "speech",
            complete   = "partial",
            text       = token,
            session_id = self.session.session_id,
        )

        # Detect chunk boundary
        stripped = token.strip()
        is_sentence = any(c in stripped for c in self._SENTENCE_END)
        is_clause   = (any(c in stripped for c in self._CLAUSE_END)
                       and self._chunk_token_count >= self._CHUNK_MIN_TOKENS)

        if is_sentence or is_clause:
            self._flush_speech_chunk()

    def _flush_speech_chunk(self) -> None:
        """Publish the accumulated buffer as a chunk event and reset."""
        if not self._chunk_buf.strip() or not self.actuation_bus:
            return
        self.actuation_bus.publish_dict(
            type       = "output",
            target     = "speech",
            complete   = "chunk",
            text       = self._chunk_buf,
            session_id = self.session.session_id,
        )
        self._chunk_buf         = ""
        self._chunk_token_count = 0

    def _publish_speech_full(self, text: str, target: str) -> None:
        """
        Publish the complete speech block as a 'full' event to the bus
        and flush any remaining chunk buffer.
        Called at </speech> and </speech_step> close.
        """
        # Flush any remaining partial chunk first
        if self._chunk_buf.strip():
            self._flush_speech_chunk()

        if self.actuation_bus:
            self.actuation_bus.publish_dict(
                type       = "output",
                target     = "speech",
                complete   = "full",
                text       = text,
                session_id = self.session.session_id,
                entity_id  = target,
            )

    # ------------------------------------------------------------------
    # <speech_step> — conversational skill suspension
    # ------------------------------------------------------------------

    def _handle_speech_step(self, content: str):
        """
        Handle a closed </speech_step> block.

        The speech has already been streamed token-by-token to TTS
        (STREAMING_SPEECH_STEP state).  Now:
          1. Record the utterance in STM.
          2. Mark a pending speech_step and clear the threading.Event.
          3. Call sagax.pause_for_speech_step() — blocks Sagax generation
             until receive_speech_step_response() is called with the reply.
        """
        var = self._speech_step_var or "speech_step_result"
        self.stm.record(
            source="system", type="output",
            payload={
                "subtype": "speech_step",
                "text":    content.strip(),
                "var":     var,
                "target":  self._speech_target,
                "status":  "awaiting_response",
            },
        )
        self.htm.asc.workbook_write("speech_step", {
            "text": content.strip(), "var": var,
        })
        self._speech_step_pending = True
        self._speech_step_event.clear()
        self.sagax.pause_for_speech_step(
            var        = var,
            step_event = self._speech_step_event,
        )

    def receive_speech_step_response(self, value: str):
        """
        Deliver the user's response to a pending speech_step.

        Binds the value in Vigil.hot_parameters, injects
        <speech_step_result> into Sagax's stream, and signals Sagax
        to resume generation.
        """
        if not self._speech_step_pending:
            return
        var = self._speech_step_var or "speech_step_result"
        self.htm.asc.bind_parameter(var, value)
        self.stm.record(
            source="user", type="speech_step_response",
            payload={"var": var, "value": value},
        )
        self.sagax.inject_speech_step_result(var, value)
        self._speech_step_pending = False
        self._speech_step_var     = ""
        self._speech_step_event.set()

    def chat(self, user_input: str) -> None:
        """
        Public: deliver a user utterance to Artux.

        Routes to receive_speech_step_response() if a speech_step is
        pending; otherwise records in STM and wakes Sagax.
        """
        if self._speech_step_pending:
            self.receive_speech_step_response(user_input)
            return
        self.stm.record(
            source="user", type="audio",
            payload={"text": user_input},
        )
        self.sagax.wake()

    # ------------------------------------------------------------------
    # HTM scheduler tick
    # ------------------------------------------------------------------

    def _scheduler_loop(self):
        while self._running:
            time.sleep(1.0 / self.SCHEDULER_TICK_HZ)
            if not self._running:
                break
            try:
                changed = self.htm.scheduler_tick()
                for task_id, new_state in changed:
                    if new_state == "due":
                        # Wake Sagax to handle the due task
                        self.sagax.wake()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Hot cache updater (called after tool results, recall results)
    # ------------------------------------------------------------------

    def update_hot_cache_from_result(
        self, tool_id: str, result: dict
    ):
        """
        After any tool result, update relevant ASC surfaces.
        Called by the Tool Manager / Perception Manager.
        """
        # Update hot_tools
        self.htm.asc.update_tool_usage(tool_id)
        # If result has entity resolution, update hot_entities
        if "entity_id" in result and result.get("signature_resolved"):
            entity_id = result["entity_id"]
            if entity_id not in self.htm.asc.hot_entities:
                self.htm.asc.update_entity(
                    type("HotEntity", (), {
                        "entity_id": entity_id,
                        "name": result.get("entity_name", ""),
                        "status": "confirmed",
                        "confidence": result.get("sig_match_confidence", 1.0),
                        "voiceprint": None, "faceprint": None,
                        "name_claims": [],
                        "associations": {},
                        "last_addressed": _utcnow(),
                    })()
                )
        # If result has recall results, cache them
        if "recall_results" in result:
            self.htm.asc.add_recall(
                query        = result.get("query", ""),
                results      = result["recall_results"],
                query_topics = result.get("topics", []),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_tool_descriptor(self, tool_name: str) -> Optional[dict]:
        """Fetch tool descriptor from Muninn for polarity check."""
        try:
            try:
                from memory_module.recall import RecallQuery as _RQ
                _q = _RQ(semantic_query=f"tool {tool_name}", topics=["tool"], top_k=1)
            except ImportError:
                _q = f"tool {tool_name}"
            results = self.tool_manager.muninn.recall(_q, top_k=1)
            if results:
                content = results[0].entry.content
                return json.loads(content) if isinstance(content, str) else {}
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# ToolManager import (used by build_huginn factory)
# ---------------------------------------------------------------------------

from .tool_manager import ToolManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tool_list(content: str) -> list[dict]:
    """Parse one or more JSON tool call objects from a block's content."""
    content = content.strip()
    results = []
    # Each line may be a separate JSON object
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                results.append(obj)
            elif isinstance(obj, list):
                results.extend(o for o in obj if isinstance(o, dict))
        except Exception:
            pass
    return results


def _utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
