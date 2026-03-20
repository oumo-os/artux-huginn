"""
runtime/actuation_bus.py — In-process pub/sub for output events.

Why a bus and not direct STM polling?
--------------------------------------
STM polling at 5 ms works for perception (events are rare) but is too
slow for chunked TTS — we need <50 ms from chunk to TTS queue. Writing
every partial/chunk token to SQLite and polling it back would also
pollute Logos's consolidation batch with low-value noise.

The ActuationBus is a lightweight in-process channel:
  - Orchestrator publishes output events (type, target, complete, text)
  - ActuationManager subscribes on behalf of each live output tool
  - Each subscriber has a filter dict specifying which events it wants
  - Matching events are pushed into per-subscriber asyncio-free Queues

STM still receives "full" events at block close (as before).
Logos never sees partial or chunk events.

Output event schema published to the bus
-----------------------------------------
{
  "type":     "output",
  "target":   "speech" | "display" | "contemplation",
  "complete": "partial" | "chunk" | "full",
  "text":     "<content>",
  "ts":       "<iso timestamp>",
  "session_id": "<session_id>",
}

Subscription filter
--------------------
A subscriber registers a filter dict. An event matches if ALL specified
filter keys match the event (missing keys in filter = wildcard):

  {"type": "output", "target": "speech", "complete": "chunk"}
    → matches speech chunk events only

  {"type": "output", "target": "speech"}
    → matches all speech events (chunk, partial, full)

  {}
    → matches everything (use sparingly)
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ActuationEvent:
    """A single event published to the ActuationBus."""
    type:       str                    # always "output"
    target:     str                    # speech | display | contemplation
    complete:   str                    # partial | chunk | full
    text:       str
    ts:         str  = field(default_factory=_utcnow)
    session_id: str  = ""
    meta:       dict = field(default_factory=dict)

    def matches(self, filt: dict) -> bool:
        """Return True if this event satisfies a filter dict."""
        for k, v in filt.items():
            if getattr(self, k, None) != v:
                return False
        return True


class ActuationBus:
    """
    In-process publish/subscribe bus for actuation events.

    Thread-safe. Subscribers receive events in their own Queue.
    No SQLite involved.
    """

    def __init__(self):
        self._lock:  threading.Lock                              = threading.Lock()
        self._subs:  dict[str, tuple[dict, queue.Queue]]         = {}
        # sub_id → (filter, queue)

    # ----------------------------------------------------------------
    # Subscriber management
    # ----------------------------------------------------------------

    def subscribe(
        self,
        subscriber_id: str,
        filt:          dict,
        maxsize:       int = 512,
    ) -> queue.Queue:
        """
        Register a subscriber. Returns the Queue it will receive events on.

        subscriber_id: unique string (typically the tool_id)
        filt:          filter dict — events matching this dict are delivered
        maxsize:       max queue depth; oldest events dropped when full
        """
        q = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subs[subscriber_id] = (filt, q)
        return q

    def unsubscribe(self, subscriber_id: str) -> None:
        with self._lock:
            self._subs.pop(subscriber_id, None)

    def update_filter(self, subscriber_id: str, filt: dict) -> None:
        with self._lock:
            if subscriber_id in self._subs:
                _, q = self._subs[subscriber_id]
                self._subs[subscriber_id] = (filt, q)

    # ----------------------------------------------------------------
    # Publisher
    # ----------------------------------------------------------------

    def publish(self, event: ActuationEvent) -> int:
        """
        Publish an event to all matching subscribers.
        Returns the number of subscribers that received it.
        Non-blocking: if a subscriber's queue is full, the event is dropped
        for that subscriber (never blocks the Orchestrator).
        """
        delivered = 0
        with self._lock:
            subs_snapshot = list(self._subs.values())

        for filt, q in subs_snapshot:
            if event.matches(filt):
                try:
                    q.put_nowait(event)
                    delivered += 1
                except queue.Full:
                    pass   # subscriber too slow — drop, never block
        return delivered

    def publish_dict(
        self,
        type:       str = "output",
        target:     str = "",
        complete:   str = "full",
        text:       str = "",
        session_id: str = "",
        **meta,
    ) -> int:
        """Convenience wrapper that constructs ActuationEvent and publishes."""
        return self.publish(ActuationEvent(
            type=type, target=target, complete=complete,
            text=text, session_id=session_id, meta=meta,
        ))

    # ----------------------------------------------------------------
    # Introspection
    # ----------------------------------------------------------------

    def subscriber_ids(self) -> list[str]:
        with self._lock:
            return list(self._subs.keys())

    def queue_depth(self, subscriber_id: str) -> Optional[int]:
        with self._lock:
            entry = self._subs.get(subscriber_id)
        return entry[1].qsize() if entry else None

    def stats(self) -> dict:
        with self._lock:
            return {
                sid: {"filter": f, "depth": q.qsize()}
                for sid, (f, q) in self._subs.items()
            }
