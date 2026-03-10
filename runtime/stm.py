"""
runtime/stm.py — STM management layer for Huginn.

Wraps Muninn's MemoryAgent public API to add:
  - Structured STMEvent envelope (JSON stored as STMSegment.content)
  - consN rolling narrative object (stored as a sentinel STMSegment)
  - logos_watermark flush cursor (stored as a sentinel STMSegment)
  - get_stm_window()   → Sagax/Exilis working context
  - get_raw_events()   → Logos full-fidelity consolidation path
  - flush_up_to()      → Logos-only flush after verified LTM writes
  - update_cons_n()    → Sagax-triggered rolling summarisation

Design invariants (from CognitiveModule.md §2.2):
  - events[] is ground truth; never altered by consN updates
  - consN.last_event_id is Sagax's read bookmark only
  - logos_watermark is Logos' independent flush cursor
  - flush_up_to() called by Logos only; Sagax never flushes
  - consN updated by Sagax only; Exilis reads but never writes it
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

def _ts_id() -> str:
    """Monotonic event ID: sortable, unique at millisecond scale."""
    ns = time.time_ns()
    return f"t{ns}"


# ---------------------------------------------------------------------------
# STMEvent — the canonical event envelope
# ---------------------------------------------------------------------------

@dataclass
class STMEvent:
    """
    A single structured event in the STM event log.

    Stored in Muninn as STMSegment.content = event.to_json().
    The id field maps to the segment's logical ordering key.
    """
    id:         str
    ts:         str
    source:     str                      # user | system | tool | sensor | log
    type:       str                      # speech | tool_call | tool_result |
                                         # task_update | sensor | output | internal
    payload:    dict[str, Any]
    confidence: float = 1.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, s: str) -> "STMEvent":
        d = json.loads(s)
        return cls(**d)

    @classmethod
    def make(
        cls,
        source:     str,
        type:       str,
        payload:    dict[str, Any],
        confidence: float = 1.0,
    ) -> "STMEvent":
        return cls(
            id=_ts_id(), ts=_utcnow(),
            source=source, type=type,
            payload=payload, confidence=confidence,
        )


# ---------------------------------------------------------------------------
# ConsN — rolling narrative summary
# ---------------------------------------------------------------------------

@dataclass
class ConsN:
    """
    The single rolling narrative summary. Owned and updated by Sagax only.
    Exilis reads it. Logos ignores it entirely.
    """
    summary_id:          str
    last_event_id:       str
    summary_text:        str
    version:             int
    created_at:          str
    topics:              list[str] = field(default_factory=list)
    confidence:          float = 1.0
    event_count_folded:  int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "ConsN":
        return cls(**json.loads(s))

    @classmethod
    def empty(cls) -> "ConsN":
        return cls(
            summary_id="consN_v0", last_event_id="",
            summary_text="", version=0, created_at=_utcnow(),
        )


# ---------------------------------------------------------------------------
# STMStore — main interface
# ---------------------------------------------------------------------------

# Prefixes used to tag sentinel STMSegments in Muninn
_CONS_N_PREFIX    = "__HUGINN_CONS_N__:"
_WATERMARK_PREFIX = "__HUGINN_WATERMARK__:"

# consN trigger thresholds (aligned with CognitiveModule.md §3.2)
CONS_N_MIN_NEW = 8
CONS_N_MAX_NEW = 20

# ---------------------------------------------------------------------------
# Huginn-private SQLite helper
# ---------------------------------------------------------------------------
# Muninn's STMManager.record() takes a plain string — no is_compression kwarg.
# We keep our sentinels (consN, watermark) in a separate Huginn metadata table
# in the same DB, avoiding any collision with Muninn's STM storage format.

def _ensure_huginn_tables(db_path: str):
    """Create the huginn_meta table if it doesn't exist."""
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS huginn_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS huginn_events (
                id         TEXT PRIMARY KEY,
                ts         TEXT NOT NULL,
                source     TEXT NOT NULL,
                type       TEXT NOT NULL,
                payload    TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0
            )
        """)
        conn.commit()


def _meta_get(db_path: str, key: str) -> Optional[str]:
    import sqlite3
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT value FROM huginn_meta WHERE key = ?", (key,)
            ).fetchone()
            return row[0] if row else None
    except Exception:
        return None


def _meta_set(db_path: str, key: str, value: str):
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO huginn_meta (key, value) VALUES (?, ?)",
            (key, value)
        )
        conn.commit()


class STMStore:
    """
    Manages the STM event log, consN, and logos_watermark
    on top of Muninn's MemoryAgent.

    Storage: events are stored as STMSegments with JSON content.
    Two special sentinel segments (is_compression=True) track
    the consN object and the logos_watermark independently.

    Parameters
    ----------
    muninn : MemoryAgent
        The shared Muninn memory agent instance.
    summarise_fn : callable, optional
        fn(existing_narrative: str, events: list[STMEvent]) -> str
        Injected by Sagax (an LLM call). Falls back to a plain join.
    """

    def __init__(
        self,
        muninn,
        summarise_fn: Optional[Callable] = None,
        db_path: str = "",
    ):
        self._muninn   = muninn
        self._summarise = summarise_fn or _default_summarise

        # Resolve db path: use muninn's db path if available, else separate file
        self._db_path = (
            db_path
            or getattr(getattr(muninn, "db", None), "path", None)
            or getattr(getattr(muninn, "db", None), "db_path", None)
            or "huginn.db"
        )
        _ensure_huginn_tables(self._db_path)

        # In-memory caches — rebuilt on first access
        self._events:    list[STMEvent] = []
        self._cons_n:    Optional[ConsN] = None
        self._watermark: str = ""
        self._loaded:    bool = False

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def _load(self):
        """Reconstruct in-memory state from huginn_events and huginn_meta tables."""
        import sqlite3
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, ts, source, type, payload, confidence "
                "FROM huginn_events ORDER BY id"
            ).fetchall()

        events = []
        for row in rows:
            try:
                events.append(STMEvent(
                    id=row[0], ts=row[1], source=row[2], type=row[3],
                    payload=json.loads(row[4]), confidence=row[5],
                ))
            except Exception:
                pass
        self._events = events

        # consN from meta
        cons_n_raw = _meta_get(self._db_path, "cons_n")
        if cons_n_raw:
            try:
                self._cons_n = ConsN.from_json(cons_n_raw)
            except Exception:
                self._cons_n = None

        # watermark from meta
        self._watermark = _meta_get(self._db_path, "logos_watermark") or ""

    # ------------------------------------------------------------------
    # Write API (called by Perception Manager only for percepts;
    #            Sagax for its own output events)
    # ------------------------------------------------------------------

    def record(
        self,
        source:     str,
        type:       str,
        payload:    dict[str, Any],
        confidence: float = 1.0,
    ) -> STMEvent:
        """
        Create and persist a new STM event.
        Written to huginn_events table (not Muninn's STMManager).
        Returns the event with its assigned id.
        """
        self._ensure_loaded()
        event = STMEvent.make(source=source, type=type,
                              payload=payload, confidence=confidence)
        self._write_event(event)
        self._events.append(event)
        return event

    def record_event(self, event: STMEvent) -> STMEvent:
        """Persist a pre-constructed STMEvent."""
        self._ensure_loaded()
        self._write_event(event)
        self._events.append(event)
        return event

    def _write_event(self, event: STMEvent):
        import sqlite3
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO huginn_events "
                "(id, ts, source, type, payload, confidence) VALUES (?,?,?,?,?,?)",
                (event.id, event.ts, event.source, event.type,
                 json.dumps(event.payload), event.confidence)
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_stm_window(self) -> dict[str, Any]:
        """
        Return the Sagax/Exilis working context:
          cons_n_text    — lossy rolling narrative (empty on cold start)
          new_events     — full-fidelity events after consN.last_event_id
          cold_start     — True if no consN exists yet

        This is what Sagax reads on each wake-up.
        Exilis reads this same window for triage context.
        """
        self._ensure_loaded()
        cons_n = self._cons_n

        if cons_n is None or not cons_n.last_event_id:
            return {
                "cons_n_text":    "",
                "cons_n_version": 0,
                "cons_n_topics":  [],
                "new_events":     list(self._events),
                "cold_start":     True,
            }

        anchor = self._index_of(cons_n.last_event_id)
        new_events = (
            self._events[anchor + 1:] if anchor >= 0 else list(self._events)
        )
        return {
            "cons_n_text":    cons_n.summary_text,
            "cons_n_version": cons_n.version,
            "cons_n_topics":  cons_n.topics,
            "new_events":     new_events,
            "cold_start":     False,
        }

    def get_raw_events(
        self,
        after_id: str = "",
        limit:    int = 100,
    ) -> list[STMEvent]:
        """
        Full-fidelity raw events for Logos consolidation.
        Returns events with id > after_id, up to limit.
        Stops a few events short of tLast to avoid racing Perception Manager.
        Logos passes logos_watermark as after_id.
        """
        self._ensure_loaded()
        events = self._events

        if after_id:
            anchor = self._index_of(after_id)
            events = events[anchor + 1:] if anchor >= 0 else list(events)

        # Leave 3-event safety margin at the head (still being written)
        safe = max(0, len(events) - 3)
        return events[:safe][:limit]

    def get_events_after(self, last_id: str) -> list[STMEvent]:
        """
        All events with id > last_id, including the freshest ones.
        Used by Exilis — no safety margin needed (it reads but never flushes).
        """
        self._ensure_loaded()
        if not last_id:
            return list(self._events)
        anchor = self._index_of(last_id)
        return self._events[anchor + 1:] if anchor >= 0 else list(self._events)

    def get_logos_watermark(self) -> str:
        """Return the id of the last event Logos has consolidated and flushed."""
        self._ensure_loaded()
        return self._watermark

    def get_cons_n(self) -> Optional[ConsN]:
        self._ensure_loaded()
        return self._cons_n

    def event_count_after_cons_n(self) -> int:
        """Number of raw events in the new-event window."""
        self._ensure_loaded()
        if not self._cons_n or not self._cons_n.last_event_id:
            return len(self._events)
        anchor = self._index_of(self._cons_n.last_event_id)
        return max(0, len(self._events) - (anchor + 1))

    def should_update_cons_n(self) -> bool:
        return self.event_count_after_cons_n() >= CONS_N_MAX_NEW

    # ------------------------------------------------------------------
    # consN update — called by Sagax only
    # ------------------------------------------------------------------

    def update_cons_n(
        self,
        force:       bool = False,
        topic_hints: list[str] = None,
    ) -> Optional[ConsN]:
        """
        Update the rolling narrative summary.

        Folds the first half of the new-event window into consN.
        Never modifies self._events.
        Returns the new ConsN, or None if not enough new events.

        Parameters
        ----------
        force : bool
            Skip the MIN_NEW_EVENTS guard. Use at session end or
            when Sagax judges the arc is complete.
        topic_hints : list[str]
            Topics Sagax already identified; merged into consN.meta.
        """
        self._ensure_loaded()
        window = self.get_stm_window()
        new_events = window["new_events"]

        if not force and len(new_events) < CONS_N_MIN_NEW:
            return None

        current = self._cons_n or ConsN.empty()

        # Fold first half; keep second half as live window for Sagax
        fold_count = max(1, len(new_events) // 2) if len(new_events) > 1 \
                     else len(new_events)
        to_fold    = new_events[:fold_count]
        if not to_fold:
            return None

        new_text   = self._summarise(current.summary_text, to_fold)
        topics     = list(set(
            (topic_hints or []) + _extract_topics(to_fold) + current.topics
        ))[:20]

        new_cons_n = ConsN(
            summary_id         = f"consN_v{current.version + 1}",
            last_event_id      = to_fold[-1].id,
            summary_text       = new_text,
            version            = current.version + 1,
            created_at         = _utcnow(),
            topics             = topics,
            confidence         = min((e.confidence for e in to_fold), default=1.0),
            event_count_folded = current.event_count_folded + fold_count,
        )

        self._persist_cons_n(new_cons_n)
        self._cons_n = new_cons_n
        return new_cons_n

    # ------------------------------------------------------------------
    # Logos flush — called by Logos only
    # ------------------------------------------------------------------

    def flush_up_to(self, event_id: str) -> int:
        """
        Delete raw events with id <= event_id from huginn_events.
        Advances logos_watermark.
        Called by Logos after verified LTM writes.
        Returns the number of events flushed.
        """
        self._ensure_loaded()
        anchor = self._index_of(event_id)
        if anchor < 0:
            return 0

        to_flush = self._events[:anchor + 1]
        keep     = self._events[anchor + 1:]

        import sqlite3
        flush_ids = [e.id for e in to_flush]
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                "DELETE FROM huginn_events WHERE id = ?",
                [(eid,) for eid in flush_ids]
            )
            conn.commit()

        self._events    = keep
        self._watermark = event_id
        self._persist_watermark()
        return len(to_flush)

    # ------------------------------------------------------------------
    # Sentinel persistence
    # ------------------------------------------------------------------

    def _persist_cons_n(self, cons_n: ConsN):
        """Write consN to huginn_meta table."""
        _meta_set(self._db_path, "cons_n", cons_n.to_json())

    def _persist_watermark(self):
        """Write logos_watermark to huginn_meta table."""
        _meta_set(self._db_path, "logos_watermark", self._watermark)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_of(self, event_id: str) -> int:
        for i, e in enumerate(self._events):
            if e.id == event_id:
                return i
        return -1

    def stats(self) -> dict[str, Any]:
        self._ensure_loaded()
        return {
            "event_count":       len(self._events),
            "first_event_id":    self._events[0].id  if self._events else None,
            "last_event_id":     self._events[-1].id if self._events else None,
            "cons_n_version":    self._cons_n.version if self._cons_n else None,
            "cons_n_last_event": self._cons_n.last_event_id if self._cons_n else None,
            "logos_watermark":   self._watermark,
            "new_event_window":  self.event_count_after_cons_n(),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_summarise(existing: str, events: list[STMEvent]) -> str:
    """Fallback: plain join. Replace with LLM call in production."""
    parts = [existing] if existing else []
    for e in events:
        text = (e.payload.get("text")
                or e.payload.get("description")
                or e.payload.get("content")
                or str(e.payload))
        parts.append(f"[{e.source}:{e.type}] {text}")
    return " | ".join(parts)


def _extract_topics(events: list[STMEvent]) -> list[str]:
    topics: set[str] = set()
    for e in events:
        for key in ("topics", "intent_hint", "tags"):
            val = e.payload.get(key)
            if isinstance(val, list):
                topics.update(str(v) for v in val)
            elif isinstance(val, str) and val:
                topics.add(val)
    return list(topics)
