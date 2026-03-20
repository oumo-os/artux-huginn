"""
runtime/actuation_manager.py — Actuation Manager (AM).

Mirror of the PerceptionManager. Where PM routes world → STM,
AM routes STM output events → live actuator tools.

Architecture
------------

    Orchestrator publishes output events to ActuationBus
                         │
                   ActuationBus
                    (in-process)
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         Kokoro TTS  Desktop UI  Subtitle
          (live tool)  (live tool) (live tool)
          mode:service mode:service mode:service
          direction:output         direction:output

Each live output tool is represented by:
  1. An HTM Task — state tracking, Sagax management interface.
     tags: ["live_tool", "actuation", tool_id]
  2. An ActuationBus subscription — event routing.
  3. A daemon thread — consumes from its queue, calls tool.handle(event).

The HTM.states surface holds live configuration for each tool:
  "{tool_id}.enabled", "{tool_id}.speed", etc.
These are read by the tool's daemon thread on each event.

Sagax controls live tools via normal <task_update> and <tool_call>:
  Start:   <tool_call>{"name": "tool.actuation.start", "args": {"tool_id": "..."}}
  Stop:    <tool_call>{"name": "tool.actuation.stop",  "args": {"tool_id": "..."}}
  Config:  <task_update action="state_set" key="kokoro_tts.speed" value="1.4"/>

Orchestrator.start() auto-starts any tool with a live_tool HTM task
(state: "active", tags_all: ["live_tool", "actuation"]).
"""

from __future__ import annotations

import importlib.util
import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .actuation_bus import ActuationBus, ActuationEvent
from .htm import HTM
from .stm import STMStore


@dataclass
class LiveTool:
    """Runtime record for one active live tool."""
    tool_id:    str
    title:      str
    module:     Any                          # loaded Python module
    start_fn:   Optional[Callable] = None   # module.start(config) or None
    stop_fn:    Optional[Callable] = None   # module.stop()
    handle_fn:  Optional[Callable] = None   # module.handle(event_dict)
    thread:     Optional[threading.Thread] = None
    event_q:    Optional[queue.Queue]       = None
    running:    bool = False
    task_id:    str  = ""


class ActuationManager:
    """
    Manages live output tools: starts them, routes events, stops them.

    The AM is driven by HTM tasks (tags: ["live_tool", "actuation"]) and
    the ActuationBus. It does not poll STM directly.
    """

    def __init__(
        self,
        bus:  ActuationBus,
        htm:  HTM,
        stm:  STMStore,
    ):
        self.bus       = bus
        self.htm       = htm
        self.stm       = stm
        self._tools:   dict[str, LiveTool] = {}   # tool_id → LiveTool
        self._lock     = threading.Lock()

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    def start_tool(
        self,
        tool_id:       str,
        source_path:   str,
        subscriptions: list[dict],
        task_id:       str = "",
        title:         str = "",
    ) -> bool:
        """
        Start a live tool daemon.

        Loads the module from source_path, calls module.start(config) with
        current HTM.states, subscribes to the ActuationBus, and launches
        the dispatch thread.

        Returns True on success, False on failure (error logged to STM).
        """
        with self._lock:
            if tool_id in self._tools and self._tools[tool_id].running:
                return True   # already running

        try:
            # Load module
            spec   = importlib.util.spec_from_file_location(tool_id, source_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            self._record_error(tool_id, "module_load_failed", str(e))
            return False

        # Build config from HTM.states (namespace = tool_id)
        config = self.htm.states.list(prefix=f"{tool_id}.")
        # Strip the namespace prefix for the tool's convenience
        config = {k[len(tool_id)+1:]: v for k, v in config.items()}

        # Resolve handlers (all optional)
        start_fn  = getattr(module, "start",  None)
        stop_fn   = getattr(module, "stop",   None)
        handle_fn = getattr(module, "handle", None)

        # Call start()
        if start_fn:
            try:
                start_fn(config)
            except Exception as e:
                self._record_error(tool_id, "start_fn_failed", str(e))
                return False

        # Subscribe to ActuationBus
        event_q = None
        if handle_fn and subscriptions:
            for i, filt in enumerate(subscriptions):
                sub_id  = f"{tool_id}_{i}" if len(subscriptions) > 1 else tool_id
                event_q = self.bus.subscribe(sub_id, filt)

        # Create LiveTool record
        lt = LiveTool(
            tool_id   = tool_id,
            title     = title or tool_id,
            module    = module,
            start_fn  = start_fn,
            stop_fn   = stop_fn,
            handle_fn = handle_fn,
            event_q   = event_q,
            task_id   = task_id,
            running   = True,
        )

        # Launch dispatch thread (only if tool has a queue to consume)
        if handle_fn and event_q is not None:
            t = threading.Thread(
                target  = self._dispatch_loop,
                args    = (lt,),
                daemon  = True,
                name    = f"AM-{tool_id}",
            )
            lt.thread = t
            t.start()

        with self._lock:
            self._tools[tool_id] = lt

        self.stm.record(
            source="system", type="internal",
            payload={"subtype": "actuation_tool_started",
                     "tool_id": tool_id, "subs": subscriptions},
        )
        return True

    def stop_tool(self, tool_id: str) -> bool:
        """Stop a running live tool."""
        with self._lock:
            lt = self._tools.get(tool_id)
        if lt is None or not lt.running:
            return False

        lt.running = False

        # Unsubscribe from bus
        self.bus.unsubscribe(tool_id)
        for i in range(10):
            self.bus.unsubscribe(f"{tool_id}_{i}")

        # Call stop()
        if lt.stop_fn:
            try:
                lt.stop_fn()
            except Exception as e:
                self._record_error(tool_id, "stop_fn_failed", str(e))

        # Signal dispatch thread to exit
        if lt.event_q:
            lt.event_q.put(None)   # sentinel

        self.stm.record(
            source="system", type="internal",
            payload={"subtype": "actuation_tool_stopped", "tool_id": tool_id},
        )
        return True

    def stop_all(self) -> None:
        with self._lock:
            ids = list(self._tools.keys())
        for tid in ids:
            self.stop_tool(tid)

    # ----------------------------------------------------------------
    # Dispatch loop (runs in each tool's daemon thread)
    # ----------------------------------------------------------------

    def _dispatch_loop(self, lt: LiveTool) -> None:
        """
        Consume events from the tool's queue and call handle_fn.

        Before each handle() call, the tool can read current config from
        HTM.states (e.g. check kokoro_tts.speed for the latest value).
        The tool is responsible for reading its own state.
        """
        while lt.running:
            try:
                event = lt.event_q.get(timeout=1.0)
            except queue.Empty:
                continue

            if event is None:   # stop sentinel
                break

            try:
                lt.handle_fn(event.__dict__ if hasattr(event, "__dict__") else event)
            except Exception as e:
                self._record_error(lt.tool_id, "handle_fn_error", str(e))

    # ----------------------------------------------------------------
    # Orchestrator hook: start any HTM-active live tools at boot
    # ----------------------------------------------------------------

    def start_from_htm(self, tool_manager) -> int:
        """
        Start all live tools that have an active HTM task.
        Called by Orchestrator.start() after all components are running.
        Returns number of tools started.
        """
        tasks = self.htm.query(
            state    = "active",
            tags_all = ["live_tool", "actuation"],
        )
        started = 0
        for task in tasks:
            # Extract tool_id from task tags
            tool_id = next(
                (t for t in task.tags
                 if t not in ("live_tool", "actuation", "perception", "pipeline")),
                None,
            )
            if not tool_id:
                continue

            # Get descriptor to find source_path and subscriptions
            descriptor = tool_manager.get_descriptor(tool_id)
            if descriptor is None:
                continue

            source_path   = getattr(descriptor, "source_path", "")
            subscriptions = getattr(descriptor, "subscriptions", [])

            if not source_path:
                continue

            ok = self.start_tool(
                tool_id       = tool_id,
                source_path   = source_path,
                subscriptions = subscriptions,
                task_id       = task.task_id,
                title         = task.title,
            )
            if ok:
                started += 1

        return started

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _record_error(self, tool_id: str, subtype: str, error: str) -> None:
        try:
            self.stm.record(
                source="system", type="internal",
                payload={"subtype": subtype, "tool_id": tool_id, "error": error},
            )
        except Exception:
            pass

    def running_tool_ids(self) -> list[str]:
        with self._lock:
            return [tid for tid, lt in self._tools.items() if lt.running]

    def stats(self) -> dict:
        with self._lock:
            return {
                tid: {"running": lt.running, "task_id": lt.task_id}
                for tid, lt in self._tools.items()
            }
