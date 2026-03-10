"""
runtime/htm.py — Hot Task Manager (HTM)

Dual-surface store (CognitiveModule.md §8):

  Surface 1 — ActiveSessionCache (ASC)
    Ephemeral, per-session.  Queried on demand by Sagax via asc.get(surface).
    Never pushed wholesale at wake-up.

  Surface 2 — Tasks
    Durable.  Survives STM flush, consN GC, and session restart.
    SQLite-backed (or in-memory for tests).

HTM is NOT queryable via recall(), not part of the STM event stream,
and not a push mechanism.  Agents read it at cycle start.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

def _uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id:      str
    title:        str
    initiated_by: str                    # sagax | logos | system
    state:        str   = "active"       # active|waiting|paused|due|completed|expired|cancelled
    progress:     str   = ""
    resume_at:    str   = ""             # semantic pointer: "step_4_wall_lights"
    remind_at:    Optional[str] = None   # ISO datetime or None
    expiry_at:    Optional[str] = None
    notebook:     list  = field(default_factory=list)
    output:       dict  = field(default_factory=dict)
    tags:         list  = field(default_factory=list)
    persistence:  str   = "volatile"     # volatile | persist | audit_only
    session_id:   str   = ""
    created_at:   str   = field(default_factory=_utcnow)
    updated_at:   str   = field(default_factory=_utcnow)

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, s: str) -> "Task":
        return cls(**json.loads(s))

    def add_note(self, entry: str):
        self.notebook.append({"ts": _utcnow(), "entry": entry})
        self.updated_at = _utcnow()


# ---------------------------------------------------------------------------
# HotEntity — ASC entity cache entry
# ---------------------------------------------------------------------------

@dataclass
class HotEntity:
    entity_id:    str
    name:         str
    status:       str   = "confirmed"   # confirmed | implied | unresolved
    voiceprint:   Optional[list] = None
    faceprint:    Optional[list] = None
    name_claims:  list  = field(default_factory=list)
    associations: dict  = field(default_factory=dict)
    last_addressed: str = field(default_factory=_utcnow)
    confidence:   float = 1.0


# ---------------------------------------------------------------------------
# ActiveSessionCache (ASC)
# ---------------------------------------------------------------------------

class ActiveSessionCache:
    """
    Per-session ephemeral cache.
    Sagax queries specific surfaces on demand via asc.get(surface).
    Orchestrator writes to it; Logos flushes it at session end.

    ASC GC policy (consN session boundary):
      - hot_tools, hot_topics, hot_recalls, hot_state: prune stale entries
      - hot_entities[unresolved/implied]: NEVER pruned
      - hot_entities[confirmed]: prunable if not in new consN context
    """

    def __init__(self, session_id: str = ""):
        self.session_id     = session_id
        self.workbook:      list[dict] = []
        self.hot_entities:  dict[str, HotEntity] = {}
        self.hot_tools:     dict[str, dict] = {}
        self.hot_topics:    dict[str, dict] = {}
        self.hot_recalls:   list[dict] = []
        self.hot_state:     dict[str, Any] = {}
        self._segment       = 0
        self._archive:      list[dict] = []  # cold storage for Logos

    # --- Sagax read ---

    def get(self, surface: str) -> Any:
        """Query a single ASC surface. Returns a snapshot copy."""
        surfaces = {
            "workbook":     lambda: list(self.workbook),
            "hot_entities": lambda: dict(self.hot_entities),
            "hot_tools":    lambda: dict(self.hot_tools),
            "hot_topics":   lambda: dict(self.hot_topics),
            "hot_recalls":  lambda: list(self.hot_recalls),
            "hot_state":    lambda: dict(self.hot_state),
        }
        if surface not in surfaces:
            raise ValueError(f"Unknown ASC surface: {surface!r}")
        return surfaces[surface]()

    # --- Orchestrator writes ---

    def workbook_write(self, block_type: str, content: Any, result: Any = None):
        self.workbook.append({
            "ts":         _utcnow(),
            "block_type": block_type,
            "content":    content,
            "result":     result,
            "session_id": self.session_id,
            "segment":    self._segment,
        })

    def update_entity(self, entity: HotEntity):
        self.hot_entities[entity.entity_id] = entity

    def add_implied_entity(
        self,
        voiceprint: Optional[list] = None,
        faceprint:  Optional[list] = None,
        confidence: float = 0.0,
    ) -> str:
        implied_id = f"implied-{_uid()[:8]}"
        self.hot_entities[implied_id] = HotEntity(
            entity_id  = implied_id,
            name       = "Unknown",
            status     = "unresolved",
            voiceprint = voiceprint,
            faceprint  = faceprint,
            confidence = confidence,
        )
        return implied_id

    def resolve_implied(self, implied_id: str, entity_id: str, name: str):
        if implied_id in self.hot_entities:
            entry = self.hot_entities.pop(implied_id)
            entry.entity_id = entity_id
            entry.name      = name
            entry.status    = "confirmed"
            self.hot_entities[entity_id] = entry

    def update_tool_usage(self, tool_id: str):
        if tool_id not in self.hot_tools:
            self.hot_tools[tool_id] = {"tool_id": tool_id, "use_count": 0}
        self.hot_tools[tool_id]["last_used"]   = _utcnow()
        self.hot_tools[tool_id]["use_count"]  += 1

    def add_recall(self, query: str, results: list, query_topics: list = None):
        self.hot_recalls.append({
            "query":        query,
            "results":      results,
            "ts":           _utcnow(),
            "query_topics": query_topics or [],
        })
        if len(self.hot_recalls) > 20:
            self.hot_recalls = self.hot_recalls[-20:]

    def touch_entity(self, entity_id: str):
        if entity_id in self.hot_entities:
            self.hot_entities[entity_id].last_addressed = _utcnow()

    def set_state(self, key: str, value: Any):
        self.hot_state[key] = value

    # --- Orchestrator GC (consN session boundary) ---

    def gc(
        self,
        new_consN_topics:   list[str],
        new_consN_entities: list[str],
        active_task_tools:  list[str] = None,
    ):
        """
        Garbage collect stale surfaces on consN session boundary.
        hot_entities[unresolved/implied] are NEVER pruned.
        Archives current workbook segment to cold storage.
        """
        active_task_tools = active_task_tools or []
        topic_set  = set(new_consN_topics)
        entity_set = set(new_consN_entities)

        self.hot_topics  = {k: v for k, v in self.hot_topics.items() if k in topic_set}
        self.hot_recalls = [r for r in self.hot_recalls
                            if any(t in topic_set for t in r.get("query_topics", []))]
        self.hot_tools   = {k: v for k, v in self.hot_tools.items()
                            if k in active_task_tools}

        # Entities: never prune unresolved/implied
        for eid in list(self.hot_entities):
            e = self.hot_entities[eid]
            if e.status == "confirmed" and eid not in entity_set:
                del self.hot_entities[eid]

        # Archive workbook segment
        self._archive.append({
            "archive_id":  f"wb-{self.session_id}-seg{self._segment}",
            "session_id":  self.session_id,
            "segment":     self._segment,
            "archived_at": _utcnow(),
            "entries":     list(self.workbook),
        })
        self.workbook  = []
        self._segment += 1

    def get_archived_segments(self) -> list[dict]:
        return list(self._archive)

    # --- Logos flush (session end) ---

    def flush(self):
        """Full ASC flush. _archive is retained until Logos marks tasks consolidated."""
        self.workbook     = []
        self.hot_entities = {}
        self.hot_tools    = {}
        self.hot_topics   = {}
        self.hot_recalls  = []
        self.hot_state    = {}


# ---------------------------------------------------------------------------
# HTM — Hot Task Manager
# ---------------------------------------------------------------------------

class HTM:
    """
    Hot Task Manager.  Dual surface: ASC (ephemeral) + Tasks (durable).

    Usage:
        htm = HTM()                   # in-memory Tasks (tests)
        htm = HTM(db_path="artux.db") # persistent Tasks
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path
        self._conn:   Optional[sqlite3.Connection] = None
        self.asc = ActiveSessionCache()

        if db_path:
            self._init_db(db_path)

    def _init_db(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS htm_tasks (
                task_id      TEXT PRIMARY KEY,
                data         TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                state        TEXT NOT NULL,
                initiated_by TEXT NOT NULL,
                persistence  TEXT NOT NULL,
                session_id   TEXT NOT NULL DEFAULT ''
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_state ON htm_tasks(state)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_iby   ON htm_tasks(initiated_by)")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Task persistence helpers
    # ------------------------------------------------------------------

    @property
    def _mem(self) -> dict[str, Task]:
        if not hasattr(self, "_mem_store"):
            self._mem_store: dict[str, Task] = {}
        return self._mem_store

    def _save(self, task: Task):
        if self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO htm_tasks VALUES (?,?,?,?,?,?,?,?)",
                (task.task_id, task.to_json(), task.created_at, task.updated_at,
                 task.state, task.initiated_by, task.persistence, task.session_id)
            )
            self._conn.commit()
        else:
            self._mem[task.task_id] = task

    def _load(self, initiated_by=None, state=None, session_id=None,
              task_id=None, tags_any=None, tags_all=None):
        if self._conn:
            clauses, params = [], []
            if initiated_by: clauses.append("initiated_by=?"); params.append(initiated_by)
            if state:
                states = state.split("|")
                clauses.append(f"state IN ({','.join('?'*len(states))})"); params.extend(states)
            if session_id: clauses.append("session_id=?"); params.append(session_id)
            if task_id:    clauses.append("task_id=?");    params.append(task_id)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            rows  = self._conn.execute(
                f"SELECT data FROM htm_tasks {where} ORDER BY created_at", params
            ).fetchall()
            result = [Task.from_json(r[0]) for r in rows]
        else:
            result = list(self._mem.values())
            if initiated_by: result = [t for t in result if t.initiated_by == initiated_by]
            if state:
                states = state.split("|")
                result = [t for t in result if t.state in states]
            if session_id: result = [t for t in result if t.session_id == session_id]
            if task_id:    result = [t for t in result if t.task_id == task_id]
            result = sorted(result, key=lambda t: t.created_at)

        # tags_any / tags_all applied in Python (tags stored in JSON, not indexed)
        if tags_any:
            result = [t for t in result if any(tag in t.tags for tag in tags_any)]
        if tags_all:
            result = [t for t in result if all(tag in t.tags for tag in tags_all)]
        return result

    # ------------------------------------------------------------------
    # Public task API
    # ------------------------------------------------------------------

    def create(
        self,
        title:        str,
        initiated_by: str = "sagax",
        persistence:  str = "volatile",
        tags:         list = None,
        session_id:   str = "",
        remind_at:    Optional[str] = None,
        expiry_at:    Optional[str] = None,
    ) -> str:
        task = Task(
            task_id=_uid(), title=title, initiated_by=initiated_by,
            persistence=persistence, tags=tags or [], session_id=session_id,
            remind_at=remind_at, expiry_at=expiry_at,
        )
        self._save(task)
        return task.task_id

    def note(self, task_id: str, entry: str) -> bool:
        tasks = self._load(task_id=task_id)
        if not tasks: return False
        tasks[0].add_note(entry)
        self._save(tasks[0])
        return True

    def update(
        self,
        task_id:   str,
        state:     Optional[str] = None,
        progress:  Optional[str] = None,
        resume_at: Optional[str] = None,
        remind_at: Optional[str] = None,
        note:      Optional[str] = None,
    ) -> bool:
        tasks = self._load(task_id=task_id)
        if not tasks: return False
        t = tasks[0]
        if state     is not None: t.state     = state
        if progress  is not None: t.progress  = progress
        if resume_at is not None: t.resume_at = resume_at
        if remind_at is not None: t.remind_at = remind_at
        t.updated_at = _utcnow()
        if note: t.add_note(note)
        self._save(t)
        return True

    def complete(self, task_id: str, output: dict = None, confidence: float = 1.0,
                 note: str = "") -> bool:
        tasks = self._load(task_id=task_id)
        if not tasks: return False
        t = tasks[0]
        t.state     = "completed"
        t.output    = {"confidence": confidence, **(output or {})}
        t.updated_at = _utcnow()
        if note: t.add_note(note)
        self._save(t)
        return True

    def query(
        self,
        initiated_by: Optional[str] = None,
        state:        Optional[str] = None,
        session_id:   Optional[str] = None,
        task_id:      Optional[str] = None,
        tags_any:     Optional[list] = None,
        tags_all:     Optional[list] = None,
    ) -> list[Task]:
        """
        Filter tasks by any combination of criteria.

        state: pipe-separated OR  e.g. 'active|due|paused'
        tags_any: task must have AT LEAST ONE of these tags
        tags_all: task must have ALL of these tags
        """
        return self._load(initiated_by, state, session_id, task_id,
                          tags_any=tags_any, tags_all=tags_all)

    def mark_consolidated(self, task_id: str) -> bool:
        return self.update(task_id, note="[logos] consolidated")

    # ------------------------------------------------------------------
    # Scheduler tick (called by Orchestrator at ~1 Hz)
    # ------------------------------------------------------------------

    def scheduler_tick(self) -> list[tuple[str, str]]:
        """Advance task states. Returns list of (task_id, new_state) changes."""
        now, changed = _utcnow(), []
        for t in self.query(state="waiting"):
            if t.remind_at and t.remind_at <= now:
                self.update(t.task_id, state="due",
                            note=f"[scheduler] remind_at reached")
                changed.append((t.task_id, "due"))
        for t in self.query(state="active"):
            if t.expiry_at and t.expiry_at <= now:
                self.update(t.task_id, state="expired",
                            note=f"[scheduler] expired")
                changed.append((t.task_id, "expired"))
        return changed

    # ------------------------------------------------------------------
    # ASC shorthands
    # ------------------------------------------------------------------

    def get(self, surface: str) -> Any:
        """Shorthand for htm.asc.get(surface)."""
        return self.asc.get(surface)

    def new_session(self, session_id: str = ""):
        self.asc = ActiveSessionCache(session_id=session_id)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        all_tasks = self.query()
        by_state: dict[str, int] = {}
        for t in all_tasks:
            by_state[t.state] = by_state.get(t.state, 0) + 1
        return {
            "task_count":       len(all_tasks),
            "by_state":         by_state,
            "asc_entities":     len(self.asc.hot_entities),
            "asc_unresolved":   sum(1 for e in self.asc.hot_entities.values()
                                    if e.status in ("unresolved", "implied")),
            "workbook_entries": len(self.asc.workbook),
        }
