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
    CAPTURING_THINKING      = "CAPTURING_THINKING"
    CAPTURING_CONTEMPLATION = "CAPTURING_CONTEMPLATION"
    STREAMING_SPEECH        = "STREAMING_SPEECH"
    BUFFERING_TOOL_CALL     = "BUFFERING_TOOL_CALL"
    BUFFERING_AUG_CALL      = "BUFFERING_AUG_CALL"
    BUFFERING_TASK_UPDATE   = "BUFFERING_TASK_UPDATE"
    BUFFERING_PROJECTION    = "BUFFERING_PROJECTION"


# Block open/close patterns
_BLOCK_OPEN  = re.compile(r"<(thinking|contemplation|speech|tool_call|aug_call|task_update|projection)(\s[^>]*)?>")
_BLOCK_CLOSE = re.compile(r"</(thinking|contemplation|speech|tool_call|aug_call|task_update|projection)>")
_TARGET_ATTR = re.compile(r'target="([^"]+)"')
_TIMEOUT_ATTR = re.compile(r'timeout_ms="(\d+)"')


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

        self.session = Session()

        # Narrator state machine
        self._narrator_state  = NarratorState.IDLE
        self._block_buffer    = ""
        self._speech_target   = ""
        self._aug_timeout_ms  = self.DEFAULT_AUG_TIMEOUT_MS

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
        # Wire Exilis callbacks
        self.exilis.on_act    = self._on_exilis_act
        self.exilis.on_urgent = self._on_exilis_urgent
        # Wire Sagax Narrator
        self.sagax.on_narrator_token = self.on_narrator_token
        # Start all agents
        self.exilis.start()
        self.sagax.start()
        self.logos.start()
        # Start scheduler tick
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True, name="SchedulerThread"
        )
        self._scheduler_thread.start()

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

        # While streaming speech: forward tokens live to TTS
        if self._narrator_state == NarratorState.STREAMING_SPEECH:
            if self.on_tts_token:
                self.on_tts_token(token)

    def _on_block_open(self, tag: str, attrs: str):
        state_map = {
            "thinking":       NarratorState.CAPTURING_THINKING,
            "contemplation":  NarratorState.CAPTURING_CONTEMPLATION,
            "speech":         NarratorState.STREAMING_SPEECH,
            "tool_call":      NarratorState.BUFFERING_TOOL_CALL,
            "aug_call":       NarratorState.BUFFERING_AUG_CALL,
            "task_update":    NarratorState.BUFFERING_TASK_UPDATE,
            "projection":     NarratorState.BUFFERING_PROJECTION,
        }
        self._narrator_state = state_map.get(tag, NarratorState.IDLE)

        if tag == "speech":
            m = _TARGET_ATTR.search(attrs)
            self._speech_target = m.group(1) if m else self.session.entity_id

        if tag == "aug_call":
            m = _TIMEOUT_ATTR.search(attrs)
            self._aug_timeout_ms = int(m.group(1)) if m else self.DEFAULT_AUG_TIMEOUT_MS

    def _on_block_close(self, tag: str, content: str):
        if tag == "thinking":
            # Debug log only — never stored
            self._narrator_state = NarratorState.IDLE
            return

        if tag == "contemplation":
            self.stm.record(
                source="system", type="output",
                payload={"subtype": "contemplation", "text": content.strip()},
            )
            self.htm.asc.workbook_write("contemplation", content.strip())

        elif tag == "speech":
            self.stm.record(
                source="system", type="output",
                payload={"subtype": "speech", "text": content.strip(),
                         "target": self._speech_target, "status": "complete"},
            )
            self.htm.asc.workbook_write("speech", content.strip())

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
                    result     = result if isinstance(result, dict) else {"output": result},
                    success    = True,
                )
                self.htm.complete(tid, output={"result": result})

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
            descriptor = self._get_tool_descriptor(name)
            if descriptor and descriptor.get("polarity", "read") != "read":
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
            results = self.tool_manager.muninn.recall(
                f"tool {tool_name}", top_k=1
            )
            if results:
                content = results[0].entry.content
                return json.loads(content) if isinstance(content, str) else {}
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# ToolManager stub — replace with real implementation
# ---------------------------------------------------------------------------

class ToolManager:
    """
    Dispatches tool calls to registered Python handlers.
    Tool descriptors are stored in Muninn LTM; this registry maps
    tool_id → Python callable for runtime execution.
    """

    def __init__(self, muninn):
        self.muninn   = muninn
        self._handlers: dict[str, Callable] = {}

    def register(self, tool_id: str, handler: Callable):
        self._handlers[tool_id] = handler

    def execute(self, tool_id: str, args: dict) -> Any:
        handler = self._handlers.get(tool_id)
        if handler is None:
            return {"status": "error", "reason": f"tool_not_registered: {tool_id}"}
        try:
            return handler(**args)
        except Exception as e:
            return {"status": "error", "reason": str(e)}


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
