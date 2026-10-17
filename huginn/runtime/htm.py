"""
runtime/htm.py — Hot Task Manager (HTM)

Dual-surface store (CognitiveModule.md §8):

  Surface 1 — ActiveSessionCache (ASC)
    Ephemeral, per-session.  Queried on demand by Sagax via asc.get(surface).
    Never pushed wholesale at wake-up.  Thread-safe.

  Surface 2 — Tasks
    Durable.  Survives STM flush, consN GC, and session restart.
    SQLite-backed (or in-memory for tests).

HTM is NOT queryable via recall(), not part of the STM event stream,
and not a push mechanism.  Agents read it at cycle start.
"""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _synchronized(method):
    """
    Serialize a store method with the instance's `_lock` (RLock).

    Locks are method-scoped and never held across calls into other
    components — the acyclic order is ASC → ToolManager → HTM → STM.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


def _uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

_TASK_STATES = frozenset({
    "active", "waiting", "paused", "due", "completed",
    "retired", "expired", "cancelled",
})


@dataclass
class Task:
    task_id:      str
    title:        str
    initiated_by: str                    # sagax | logos | system
    state:        str   = "active"       # active|waiting|paused|due|completed|retired|expired|cancelled
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

    def add_note(self, entry: str, note_type: str = "note"):
        self.notebook.append({
            "ts":    _utcnow(),
            "type":  note_type,
            "entry": entry,
        })
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
    grants:       dict  = field(default_factory=dict)
    # grants shape: {permission_scope: [...], denied: [...], confirmation_required: [...]}
    # Populated from entity LTM record when entity is confirmed.
    # Empty dict = guest access only. Logos is the only writer of
    # the source grants in Muninn; this is a read-through cache.


# ---------------------------------------------------------------------------
# HotTool — ASC hot capabilities entry
# ---------------------------------------------------------------------------

_HOT_TOOLS_MAX = 20   # LRU cap


@dataclass
class HotTool:
    """
    Cached tool descriptor in the Vigil hot surface.
    Populated by Orchestrator on every successful tool dispatch.
    Sagax reads hot_tools in context — recently used tools appear inline
    without a Muninn recall call.

    reflex_pattern  — regex written by Logos when a skill becomes active.
                      Exilis tests this against speech before LLM triage.
    reflex_skill_id — the skill this reflex belongs to.
    """
    tool_id:            str
    title:              str
    capability_summary: str
    polarity:           str  = "read"
    permission_scope:   list = field(default_factory=list)
    mode:               str  = "callable"
    last_used:          str  = field(default_factory=_utcnow)
    use_count:          int  = 0
    fail_count:         int  = 0
    last_error:         str  = ""
    last_failed_at:     str  = ""
    reflex_pattern:     str  = ""
    reflex_skill_id:    str  = ""

    @property
    def health(self) -> float:
        """0.0 (always failing) … 1.0 (never failed)."""
        total = self.use_count + self.fail_count
        if total == 0:
            return 0.5
        return 1.0 - (self.fail_count / total)

    # Backward-compat dict-style access (some callers/tests index HotTool)
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class ToolFailureHistory:
    """
    Durable failure history for a tool, keyed by tool_id in the ASC's
    `tool_failures` surface. NOT subject to hot_capabilities LRU eviction:
    a degraded tool that drops out of the hot cache keeps its diagnostics,
    and re-seeds the hot entry (fail_count / last_error) if it ever returns.
    """
    tool_id:         str
    fail_count:      int   = 0
    first_failed_at: str   = ""
    last_failed_at:  str   = ""
    last_error:      str   = ""
    last_success_at: str   = ""


# ---------------------------------------------------------------------------
# ActiveSessionCache (ASC)
# ---------------------------------------------------------------------------

class Vigil(ActiveSessionCache if False else object):
    pass  # forward-declare for type hints


class ActiveSessionCache:
    """
    Vigil — the active present layer.

    Per-session ephemeral operational state.  Sagax queries specific surfaces
    on demand via asc.get(surface).  Orchestrator writes to it; Logos flushes
    it at session end.

    Surfaces
    --------
    workbook        — session reasoning log: all Narrator block outputs in order
    hot_entities    — HotEntity records keyed by entity_id
    hot_capabilities — recently used/discovered tools: {tool_id: HotTool}
    hot_topics      — salient topics from consN context: {topic: {salience, ts}}
    hot_recalls     — recent recall() results (capped at 20)
    hot_parameters  — session-scoped variable bindings (skill input bindings,
                      entity recall hints): {var_name: value}
    hot_state       — arbitrary session state for agent coordination
    tool_failures   — durable failure history per tool (survives LRU eviction)

    GC policy (consN session boundary)
    -----------------------------------
    hot_topics, hot_recalls         — prune if topic not in new consN
    hot_capabilities                — prune if tool not in active tasks
    hot_parameters                  — NEVER auto-pruned (cleared by skill completion)
    hot_entities[confirmed]         — prune if entity not in new consN context
    hot_entities[unresolved/implied]— NEVER pruned
    tool_failures                   — NEVER pruned (durable registry)
    """

    def __init__(self, session_id: str = ""):
        self.session_id      = session_id
        # Every mutator and snapshot reader is serialized via _synchronized —
        # the ASC is hit from 6+ threads (workers, Sagax, Exilis, Logos, PM).
        self._lock           = threading.RLock()
        self.workbook:       list[dict]          = []
        self.hot_entities:   dict[str, HotEntity]= {}
        self.hot_capabilities: dict[str, HotTool] = {}  # tool_id → HotTool
        self.hot_topics:     dict[str, dict]     = {}
        self.hot_recalls:    list[dict]          = []
        self.hot_parameters: dict[str, Any]      = {}  # skill vars, recall hints
        self.hot_state:      dict[str, Any]      = {}
        self.tool_failures:  dict[str, ToolFailureHistory] = {}
        self._segment        = 0
        self._archive:       list[dict]          = []  # cold storage for Logos

    # ----------------------------------------------------------------
    # Sagax read interface
    # ----------------------------------------------------------------

    @_synchronized
    def get(self, surface: str) -> Any:
        """
        Query a single Vigil surface. Returns a snapshot copy.

        Surfaces: workbook | hot_entities | hot_capabilities | hot_topics |
                  hot_recalls | hot_parameters | hot_state | tool_failures
        """
        surfaces = {
            "workbook":          lambda: list(self.workbook),
            "hot_entities":      lambda: dict(self.hot_entities),
            "hot_capabilities":  lambda: dict(self.hot_capabilities),
            "hot_topics":        lambda: dict(self.hot_topics),
            "hot_recalls":       lambda: list(self.hot_recalls),
            "hot_parameters":    lambda: dict(self.hot_parameters),
            "hot_state":         lambda: dict(self.hot_state),
            "tool_failures":     lambda: dict(self.tool_failures),
        }
        if surface not in surfaces:
            raise ValueError(
                f"Unknown Vigil surface: {surface!r}. "
                f"Valid: {list(surfaces)}"
            )
        return surfaces[surface]()

    # ----------------------------------------------------------------
    # Orchestrator write interface
    # ----------------------------------------------------------------

    @_synchronized
    def workbook_write(self, block_type: str, content: Any, result: Any = None):
        self.workbook.append({
            "ts":         _utcnow(),
            "block_type": block_type,
            "content":    content,
            "result":     result,
            "session_id": self.session_id,
            "segment":    self._segment,
        })

    # hot_entities
    @_synchronized
    def update_entity(self, entity: HotEntity):
        self.hot_entities[entity.entity_id] = entity

    def activate_entity(
        self,
        entity_id:     str,
        resolver:      Optional[Callable[[], tuple[str, dict]]] = None,
        status:        str = "confirmed",
        confidence:    float = 1.0,
    ) -> "HotEntity":
        """
        Locked check-or-update for a confirmed entity — one critical section.

        Inserts a fresh HotEntity when missing (calling `resolver()` — which
        returns (name, grants) — inside the lock), or refreshes empty grants
        on an existing entry. Always touches last_addressed. Replaces the
        racy check-then-insert direct dict writes in perception/orchestrator.
        """
        with self._lock:
            hot = self.hot_entities.get(entity_id)
            if hot is None:
                name, grants = "", {}
                if resolver is not None:
                    try:
                        name, grants = resolver()
                    except Exception:
                        pass
                hot = HotEntity(
                    entity_id  = entity_id,
                    name       = name or entity_id,
                    status     = status,
                    confidence = confidence,
                    grants     = grants or {},
                )
                self.hot_entities[entity_id] = hot
            elif not hot.grants and resolver is not None:
                try:
                    _, grants = resolver()
                    if grants:
                        hot.grants = grants
                except Exception:
                    pass
            hot.last_addressed = _utcnow()
            return hot

    @_synchronized
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

    @_synchronized
    def resolve_implied(self, implied_id: str, entity_id: str, name: str):
        if implied_id in self.hot_entities:
            entry = self.hot_entities.pop(implied_id)
            entry.entity_id = entity_id
            entry.name      = name
            entry.status    = "confirmed"
            self.hot_entities[entity_id] = entry

    @_synchronized
    def touch_entity(self, entity_id: str):
        if entity_id in self.hot_entities:
            self.hot_entities[entity_id].last_addressed = _utcnow()

    # hot_capabilities  (replaces the old thin hot_tools)
    @_synchronized
    def update_tool_usage(
        self,
        tool_id:            str,
        title:              str  = "",
        capability_summary: str  = "",
        polarity:           str  = "read",
        permission_scope:   list = None,
        mode:               str  = "callable",
        reflex_pattern:     str  = "",
        reflex_skill_id:    str  = "",
    ) -> None:
        """Record/refresh a tool in the Vigil hot_capabilities LRU. Evicts oldest over cap."""
        existing = self.hot_capabilities.get(tool_id)
        # A fresh entry re-enters with its durable failure history intact —
        # eviction must never reset a degraded tool's diagnostics.
        hist = self.tool_failures.get(tool_id)
        self.hot_capabilities[tool_id] = HotTool(
            tool_id            = tool_id,
            title              = title or (existing.title if existing else tool_id),
            capability_summary = capability_summary or (existing.capability_summary if existing else ""),
            polarity           = polarity or (existing.polarity if existing else "read"),
            permission_scope   = permission_scope if permission_scope is not None else (existing.permission_scope if existing else []),
            mode               = mode or (existing.mode if existing else "callable"),
            last_used          = _utcnow(),
            use_count          = (existing.use_count if existing else 0) + 1,
            fail_count         = (existing.fail_count if existing
                                  else (hist.fail_count if hist else 0)),
            last_error         = (existing.last_error if existing
                                  else (hist.last_error if hist else "")),
            last_failed_at     = (existing.last_failed_at if existing
                                  else (hist.last_failed_at if hist else "")),
            reflex_pattern     = reflex_pattern or (existing.reflex_pattern if existing else ""),
            reflex_skill_id    = reflex_skill_id or (existing.reflex_skill_id if existing else ""),
        )
        if len(self.hot_capabilities) > _HOT_TOOLS_MAX:
            oldest = min(self.hot_capabilities, key=lambda k: self.hot_capabilities[k].last_used)
            del self.hot_capabilities[oldest]

    @_synchronized
    def record_tool_success(
        self,
        tool_id:            str,
        title:              str  = "",
        capability_summary: str  = "",
        polarity:           str  = "read",
        permission_scope:   list = None,
        mode:               str  = "callable",
    ) -> None:
        """Record a successful dispatch — stamps last_success_at on any failure history."""
        self.update_tool_usage(
            tool_id            = tool_id,
            title              = title,
            capability_summary = capability_summary,
            polarity           = polarity,
            permission_scope   = permission_scope,
            mode               = mode,
        )
        hist = self.tool_failures.get(tool_id)
        if hist is not None:
            hist.last_success_at = _utcnow()

    @_synchronized
    def record_tool_failure(self, tool_id: str, error: str = "") -> None:
        """Record a failed dispatch in hot_capabilities and the durable
        failure registry. The registry survives LRU eviction, so a degraded
        tool never loses its history (fail_count, first/last failure, error)."""
        now = _utcnow()
        hist = self.tool_failures.get(tool_id)
        if hist is None:
            self.tool_failures[tool_id] = ToolFailureHistory(
                tool_id         = tool_id,
                fail_count      = 1,
                first_failed_at = now,
                last_failed_at  = now,
                last_error      = error,
            )
            hist = self.tool_failures[tool_id]
        else:
            hist.fail_count    += 1
            hist.last_failed_at = now
            hist.last_error     = error

        existing = self.hot_capabilities.get(tool_id)
        if existing is None:
            # Evicted or never used: the fresh entry re-seeds the full history
            self.hot_capabilities[tool_id] = HotTool(
                tool_id            = tool_id,
                title              = tool_id,
                capability_summary = "",
                fail_count         = hist.fail_count,
                last_error         = error,
                last_failed_at     = now,
            )
            return
        existing.fail_count     += 1
        existing.last_error      = error
        existing.last_failed_at  = now

    @_synchronized
    def set_tool_reflex(self, tool_id: str, reflex_pattern: str, skill_id: str = "") -> None:
        """Attach a reflex pattern to a hot tool (written by Logos on skill activation)."""
        if tool_id in self.hot_capabilities:
            self.hot_capabilities[tool_id].reflex_pattern  = reflex_pattern
            self.hot_capabilities[tool_id].reflex_skill_id = skill_id

    @_synchronized
    def get_reflex_tools(self) -> list:
        """Return HotCapabilities with an active reflex pattern."""
        return [t for t in self.hot_capabilities.values() if t.reflex_pattern]

    # backward compat
    @_synchronized
    def update_capability(self, tool_id: str, descriptor_snapshot: dict = None):
        d = descriptor_snapshot or {}
        self.update_tool_usage(
            tool_id=tool_id, title=d.get("title",""),
            capability_summary=d.get("capability_summary",""),
            polarity=d.get("polarity","read"),
            permission_scope=d.get("permission_scope",[]),
            mode=d.get("mode","callable"),
        )

    # hot_topics
    @_synchronized
    def add_topic(self, topic: str, salience: float = 1.0):
        self.hot_topics[topic] = {
            "salience": salience,
            "ts":       _utcnow(),
        }

    # hot_recalls
    @_synchronized
    def add_recall(self, query: str, results: list, query_topics: list = None):
        self.hot_recalls.append({
            "query":        query,
            "results":      results,
            "ts":           _utcnow(),
            "query_topics": query_topics or [],
        })
        if len(self.hot_recalls) > 20:
            self.hot_recalls = self.hot_recalls[-20:]

    # hot_parameters  (skill variable bindings)
    @_synchronized
    def bind_parameter(self, name: str, value: Any):
        """Bind a named variable. Used by skill steps and recall hints."""
        self.hot_parameters[name] = value

    @_synchronized
    def get_parameter(self, name: str, default: Any = None) -> Any:
        return self.hot_parameters.get(name, default)

    @_synchronized
    def clear_parameter(self, name: str):
        self.hot_parameters.pop(name, None)

    # hot_state
    @_synchronized
    def set_state(self, key: str, value: Any):
        self.hot_state[key] = value

    # ----------------------------------------------------------------
    # GC (consN session boundary)
    # ----------------------------------------------------------------

    @_synchronized
    def gc(
        self,
        new_consN_topics:   list[str],
        new_consN_entities: list[str],
        active_task_tools:  list[str] = None,
    ):
        """
        Garbage collect stale Vigil surfaces on consN session boundary.

        hot_parameters and hot_entities[unresolved/implied] are NEVER pruned.
        Archives current workbook segment to cold storage for Logos.
        tool_failures is NEVER pruned (durable registry).
        """
        active_task_tools = active_task_tools or []
        topic_set  = set(new_consN_topics)
        entity_set = set(new_consN_entities)
        tool_set   = set(active_task_tools)

        # Prune stale topics
        self.hot_topics = {
            k: v for k, v in self.hot_topics.items() if k in topic_set
        }

        # Prune stale recalls (keep if any query_topic is still salient)
        self.hot_recalls = [
            r for r in self.hot_recalls
            if any(t in topic_set for t in r.get("query_topics", []))
        ]

        # Prune stale capabilities (keep if tool is in an active task)
        self.hot_capabilities = {
            k: v for k, v in self.hot_capabilities.items() if k in tool_set
        }

        # hot_parameters: NEVER auto-pruned — cleared by skill completion

        # Entities: never prune unresolved/implied; prune confirmed if not salient
        for eid in list(self.hot_entities):
            e = self.hot_entities[eid]
            if e.status == "confirmed" and eid not in entity_set:
                del self.hot_entities[eid]

        # Archive workbook segment to cold storage
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

    # ----------------------------------------------------------------
    # Logos flush (session end)
    # ----------------------------------------------------------------

    def flush(self):
        """
        Full Vigil flush at session end.
        _archive is retained until Logos marks tasks consolidated.
        hot_parameters cleared here — they are session-scoped.
        Durable surfaces (tool_failures, hot_capabilities) are preserved
        so degraded tools stay degraded and usage carries forward.
        """
        tool_failures    = self.tool_failures
        hot_capabilities = self.hot_capabilities
        self.workbook        = []
        self.hot_entities    = {}
        self.hot_capabilities= hot_capabilities
        self.hot_topics      = {}
        self.hot_recalls     = []
        self.hot_parameters  = {}
        self.hot_state       = {}
        self.tool_failures   = tool_failures

    # ----------------------------------------------------------------
    # Convenience: summary for Sagax context injection
    # ----------------------------------------------------------------

    def summary_for_sagax(self) -> dict:
        """
        Build a compact Vigil summary to prepend to Sagax's user prompt.
        Only includes non-empty surfaces. Capability lines carry the tool
        health diagnostics (fail_count / last_error) so Sagax can warn the
        user before proceeding with a degraded tool.
        """
        out = {}
        if self.hot_entities:
            out["entities"] = [
                {"id": e.entity_id, "name": e.name, "status": e.status}
                for e in self.hot_entities.values()
            ]
        if self.hot_capabilities or self.tool_failures:
            caps = [
                {"tool_id": k, "title": v.title,
                 "use_count": v.use_count,
                 "fail_count": v.fail_count,
                 "health": round(v.health, 2),
                 "last_error": v.last_error}
                for k, v in self.hot_capabilities.items()
            ]
            # Evicted-but-failed tools keep surfacing: the durable registry
            # outlives the LRU, so Sagax can still warn about a degraded tool
            # even after it dropped out of the hot cache.
            hot_ids = set(self.hot_capabilities)
            for k, h in self.tool_failures.items():
                if k not in hot_ids and h.fail_count >= 1:
                    caps.append({
                        "tool_id": k, "title": k,
                        "use_count": 0,
                        "fail_count": h.fail_count,
                        "health": round(1.0 - (h.fail_count / (1 + h.fail_count)), 2),
                        "last_error": h.last_error,
                    })
            out["capabilities"] = caps
        if self.hot_topics:
            out["topics"] = list(self.hot_topics.keys())
        if self.hot_parameters:
            out["parameters"] = dict(self.hot_parameters)
        return out


# ---------------------------------------------------------------------------
# HTMStates — live key-value state store
# ---------------------------------------------------------------------------

class HTMStates:
    """
    Flat key-value store for live operational state.

    Completely separate from Tasks — these have no lifecycle, no notebook,
    no completion state. They just hold the current value of runtime
    parameters and tool configuration.

    Keys are namespaced: "{tool_id}.{param}"  or  "{system}.{param}"
    e.g.  "kokoro_tts.speed"   "sagax.model"   "session.quiet_mode"

    Sagax reads and writes via <task_update action="state_set|state_get">.
    AM reads directly to configure live tools.
    Logos persists changed states to LTM config on session end.
    """

    def __init__(self, conn: Optional[sqlite3.Connection] = None,
                 lock: Optional[threading.RLock] = None):
        self._store:    dict[str, Any] = {}
        self._dirty:    set[str]       = set()   # keys changed since last flush
        self._defaults: dict[str, Any] = {}      # written at tool install time
        self._conn      = conn                  # optional durable htm_states table
        # Shares the HTM lock: states and tasks use the same SQLite connection.
        self._lock      = lock or threading.RLock()
        # Change listener: called as on_change(key, value) after every set().
        # Wired by frontends (e.g. web UI pushes {"type":"state"} over WS).
        # MUST be cheap and non-blocking — it runs with the states lock held.
        self.on_change: Optional[Callable[[str, Any], None]] = None

    # ----------------------------------------------------------------
    # Core API
    # ----------------------------------------------------------------

    @_synchronized
    def set(self, key: str, value: Any, mark_dirty: bool = True) -> None:
        """Set a state value. Marks key dirty (pending Logos flush) by default."""
        self._store[key] = value
        if mark_dirty:
            self._dirty.add(key)
        self._persist(key)
        if self.on_change is not None:
            try:
                self.on_change(key, value)
            except Exception:
                pass

    @_synchronized
    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    @_synchronized
    def set_absent(self, key: str, value: Any) -> None:
        """
        Set a value only if it is not already present.

        The durable htm_states snapshot (restored at boot via load_from_db)
        is authoritative over recalled LTM config — a live runtime change
        must never be clobbered by a stale config on the next boot. Persists
        only when it actually sets the value; never marks dirty (nothing to
        write back to LTM on session end).
        """
        if key in self._store:
            return
        self._store[key] = value
        self._persist(key)

    @_synchronized
    def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._dirty.discard(key)
        self._persist(key, delete=True)

    # ----------------------------------------------------------------
    # Durable persistence (htm_states table)
    # ----------------------------------------------------------------

    @_synchronized
    def _persist(self, key: str, delete: bool = False) -> None:
        """Write a state change to the htm_states table, if a DB is bound."""
        if self._conn is None:
            return
        try:
            if delete or key not in self._store:
                self._conn.execute(
                    "DELETE FROM htm_states WHERE key = ?", (key,)
                )
            else:
                self._conn.execute(
                    "INSERT OR REPLACE INTO htm_states (key, value, updated_at) "
                    "VALUES (?, ?, ?)",
                    (key, json.dumps(self._store[key]), _utcnow()),
                )
            self._conn.commit()
        except Exception:
            pass   # persistence is best-effort — never break the runtime

    @_synchronized
    def load_from_db(self) -> None:
        """Restore all persisted states from the htm_states table at boot."""
        if self._conn is None:
            return
        try:
            rows = self._conn.execute(
                "SELECT key, value FROM htm_states"
            ).fetchall()
            for key, raw in rows:
                try:
                    self._store[key] = json.loads(raw)
                except Exception:
                    self._store[key] = raw
        except Exception:
            pass

    @_synchronized
    def list(self, prefix: str = "") -> dict[str, Any]:
        """Return all keys (optionally filtered by prefix) as a snapshot dict."""
        if not prefix:
            return dict(self._store)
        return {k: v for k, v in self._store.items() if k.startswith(prefix)}

    @_synchronized
    def set_default(self, key: str, value: Any) -> None:
        """
        Write a default value. Only applied if the key is not already set.
        Called by ToolManager when a tool is installed/activated.
        """
        self._defaults[key] = value
        if key not in self._store:
            self._store[key] = value
            # Defaults are not dirty — they don't need to be written back to LTM

    @_synchronized
    def flush_dirty(self) -> dict[str, Any]:
        """
        Return all dirty key-value pairs and clear the dirty set.
        Called by Logos at session end to decide what to persist to LTM config.
        """
        changed = {k: self._store[k] for k in self._dirty if k in self._store}
        self._dirty.clear()
        return changed

    @_synchronized
    def dirty_keys(self) -> set[str]:
        return set(self._dirty)

    # ----------------------------------------------------------------
    # Sagax-facing summary (injected into context)
    # ----------------------------------------------------------------

    @_synchronized
    def summary(self, max_keys: int = 30) -> str:
        """
        Structured snapshot for Sagax context injection.

        Cognitive states (sagax.state, logos.state) always appear first
        so they are visible even if the snapshot is truncated.
        Groups remaining keys by namespace.
        """
        if not self._store:
            return "(no states)"

        lines = []

        # Always show cognitive states first
        for key in ("sagax.state", "logos.state"):
            val = self._store.get(key)
            if val is not None:
                lines.append(f"  {key} = {val!r}")

        # Group remaining keys by namespace
        groups: dict[str, list[str]] = {}
        skip = {"sagax.state", "logos.state"}
        for k, v in sorted(self._store.items()):
            if k in skip:
                continue
            ns = k.split(".")[0] if "." in k else "global"
            groups.setdefault(ns, []).append(f"{k}={v!r}")

        count = len(lines)
        for ns, entries in groups.items():
            if count >= max_keys:
                lines.append(f"  ... ({len(self._store) - count} more keys)")
                break
            line = "  " + " | ".join(entries[:6])
            if len(entries) > 6:
                line += f" ... (+{len(entries)-6})"
            lines.append(line)
            count += len(entries)

        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Bulk load (called by Orchestrator._apply_system_config)
    # ----------------------------------------------------------------

    @_synchronized
    def load_from_config(self, config: dict, namespace: str) -> None:
        """
        Populate states from a recalled config dict without marking dirty.
        namespace: e.g. "sagax" for config key "model" → state "sagax.model"
        """
        for k, v in config.items():
            if k in ("description",):   # skip non-parameter metadata fields
                continue
            full_key = f"{namespace}.{k}"
            if full_key in self._store:
                # The durable htm_states snapshot (written at runtime by
                # state_set) is fresher than the last Logos flush to LTM —
                # never let it clobber live runtime changes.
                continue
            self._store[full_key] = v
            # Not dirty — these came from LTM, no need to write back


# ---------------------------------------------------------------------------
# HTM — Hot Task Manager
# ---------------------------------------------------------------------------

class HTM:
    """
    Hot Task Manager.  Dual surface: ASC (ephemeral) + Tasks (durable).

    Usage:
        htm = HTM()                   # in-memory Tasks (tests)
        htm = HTM(db_path="artux.db") # persistent Tasks + states
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path
        self._conn:   Optional[sqlite3.Connection] = None
        # Serializes the task store AND the states store (they share _conn).
        # The ASC has its own lock. Order: ASC → HTM → STM (no cycles).
        self._lock    = threading.RLock()
        self.asc    = ActiveSessionCache()

        # State-change sink: "even state changes are events". Wired by the
        # runtime builder (e.g. to the Orchestrator) so every task transition
        # becomes observable STM history. Signature:
        #   on_state_change(task_id, from_state, to_state, task)
        self.on_state_change: Optional[Callable[[str, str, str, "Task"], None]] = None

        if db_path:
            self._init_db(db_path)
            self.states = HTMStates(conn=self._conn, lock=self._lock)
            self.states.load_from_db()
        else:
            self.states = HTMStates(lock=self._lock)

    def _init_db(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False,
                                     timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
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
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS htm_states (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_state ON htm_tasks(state)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_iby   ON htm_tasks(initiated_by)")
        # Volatile tasks are throwaway working records: in-memory only since
        # 1.7.2. Purge legacy volatile rows written by older versions.
        self._conn.execute("DELETE FROM htm_tasks WHERE persistence='volatile'")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Task persistence helpers
    # ------------------------------------------------------------------

    @property
    def _mem(self) -> dict[str, Task]:
        if not hasattr(self, "_mem_store"):
            self._mem_store: dict[str, Task] = {}
        return self._mem_store

    @_synchronized
    def _save(self, task: Task):
        # In-memory store is the live working set for every task.
        self._mem[task.task_id] = task
        # The DB is a mirror for tasks that must survive restarts.
        if self._conn and task.persistence != "volatile":
            self._conn.execute(
                "INSERT OR REPLACE INTO htm_tasks VALUES (?,?,?,?,?,?,?,?)",
                (task.task_id, task.to_json(), task.created_at, task.updated_at,
                 task.state, task.initiated_by, task.persistence, task.session_id)
            )
            self._conn.commit()

    @_synchronized
    def _load(self, initiated_by=None, state=None, session_id=None,
              task_id=None, tags_any=None, tags_all=None, updated_after=None,
              persistence=None):
        if self._conn:
            clauses, params = [], []
            if initiated_by: clauses.append("initiated_by=?"); params.append(initiated_by)
            if state:
                states = state.split("|")
                clauses.append(f"state IN ({','.join('?'*len(states))})"); params.extend(states)
            if session_id: clauses.append("session_id=?"); params.append(session_id)
            if task_id:    clauses.append("task_id=?");    params.append(task_id)
            if persistence: clauses.append("persistence=?"); params.append(persistence)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            rows  = self._conn.execute(
                f"SELECT data FROM htm_tasks {where} ORDER BY created_at", params
            ).fetchall()
            result = [Task.from_json(r[0]) for r in rows]
            # Merge in-memory volatile tasks (never mirrored to the DB).
            result += [t for t in self._mem.values() if t.persistence == "volatile"]
            # Re-apply filters to the merged list (SQL WHERE only covered
            # persisted rows).
            if initiated_by: result = [t for t in result if t.initiated_by == initiated_by]
            if state:
                states = state.split("|")
                result = [t for t in result if t.state in states]
            if session_id: result = [t for t in result if t.session_id == session_id]
            if task_id:    result = [t for t in result if t.task_id == task_id]
            if persistence: result = [t for t in result if t.persistence == persistence]
        else:
            result = list(self._mem.values())
            if initiated_by: result = [t for t in result if t.initiated_by == initiated_by]
            if state:
                states = state.split("|")
                result = [t for t in result if t.state in states]
            if session_id: result = [t for t in result if t.session_id == session_id]
            if task_id:    result = [t for t in result if t.task_id == task_id]
            if persistence: result = [t for t in result if t.persistence == persistence]
            result = sorted(result, key=lambda t: t.created_at)

        # tags_any / tags_all applied in Python (tags stored in JSON, not indexed)
        if tags_any:
            result = [t for t in result if any(tag in t.tags for tag in tags_any)]
        if tags_all:
            result = [t for t in result if all(tag in t.tags for tag in tags_all)]
        if updated_after:
            result = [t for t in result if t.updated_at >= updated_after]
        return result

    # ------------------------------------------------------------------
    # Public task API
    # ------------------------------------------------------------------

    @_synchronized
    def create(
        self,
        title:        str,
        initiated_by: str = "sagax",
        persistence:  str = "persist",
        tags:         list = None,
        session_id:   str = "",
        remind_at:    Optional[str] = None,
        expiry_at:    Optional[str] = None,
        progress:     str = "",
    ) -> str:
        """
        Create a task. Persisted by default — timers, reminders, background
        processes and long-horizon tasks must survive restarts. Pass
        persistence="volatile" for throwaway work items.
        """
        task = Task(
            task_id=_uid(), title=title, initiated_by=initiated_by,
            persistence=persistence, tags=tags or [], session_id=session_id,
            remind_at=remind_at, expiry_at=expiry_at, progress=progress,
        )
        self._save(task)
        return task.task_id

    @_synchronized
    def note(self, task_id: str, entry: str, note_type: str = "note") -> bool:
        tasks = self._load(task_id=task_id)
        if not tasks: return False
        tasks[0].add_note(entry, note_type=note_type)
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
        output:    Optional[dict] = None,
    ) -> bool:
        # The state-change sink fires AFTER the lock is released — the HTM
        # lock must never be held across a call into another component.
        transition = None
        with self._lock:
            tasks = self._load(task_id=task_id)
            if not tasks: return False
            t = tasks[0]
            from_state = t.state
            if state is not None:
                if state not in _TASK_STATES:
                    t.add_note(f"[htm] invalid state rejected: {state!r}")
                    self._save(t)
                    return False
                t.state = state
            if progress  is not None: t.progress  = progress
            if resume_at is not None: t.resume_at = resume_at
            if remind_at is not None: t.remind_at = remind_at
            if output    is not None: t.output    = {**t.output, **output}
            t.updated_at = _utcnow()
            if note: t.add_note(note)
            self._save(t)
            if t.state != from_state:
                transition = (t, from_state)
        if transition is not None:
            self._emit_state_change(*transition)
        return True

    def complete(self, task_id: str, output: dict = None, confidence: float = 1.0,
                 note: str = "") -> bool:
        transition = None
        with self._lock:
            tasks = self._load(task_id=task_id)
            if not tasks: return False
            t = tasks[0]
            from_state = t.state
            t.state     = "completed"
            t.output    = {"confidence": confidence, **(output or {})}
            t.updated_at = _utcnow()
            if note: t.add_note(note)
            self._save(t)
            transition = (t, from_state)
        if transition is not None:
            self._emit_state_change(*transition)
        return True

    def retire(self, task_id: str, note: str = "") -> bool:
        """
        Retire an acknowledged task (state -> 'retired').

        Tasks are attention state: Sagax is the sole retirer, called once a
        task's result has been communicated, acknowledged, or promoted to LTM.
        Legal from any state. Retired tasks leave the attention surface and
        are eventually purged like other terminal states.
        """
        transition = None
        with self._lock:
            tasks = self._load(task_id=task_id)
            if not tasks: return False
            t = tasks[0]
            from_state = t.state
            if from_state == "retired":
                return True
            t.state = "retired"
            t.updated_at = _utcnow()
            if note: t.add_note(note)
            self._save(t)
            transition = (t, from_state)
        if transition is not None:
            self._emit_state_change(*transition)
        return True

    def _emit_state_change(self, task: "Task", from_state: str) -> None:
        """Fire the state-change sink — even state changes are events."""
        if self.on_state_change is None:
            return
        try:
            self.on_state_change(task.task_id, from_state, task.state, task)
        except Exception:
            pass

    def query(
        self,
        initiated_by: Optional[str] = None,
        state:        Optional[str] = None,
        session_id:   Optional[str] = None,
        task_id:      Optional[str] = None,
        tags_any:     Optional[list] = None,
        tags_all:     Optional[list] = None,
        updated_after: Optional[str] = None,
        persistence:  Optional[str] = None,
    ) -> list[Task]:
        """
        Filter tasks by any combination of criteria.

        state: pipe-separated OR  e.g. 'active|due|paused'
        tags_any: task must have AT LEAST ONE of these tags
        tags_all: task must have ALL of these tags
        updated_after: only tasks updated at/after this ISO timestamp
        persistence: 'volatile' | 'persist' | 'audit_only'
        """
        return self._load(initiated_by, state, session_id, task_id,
                          tags_any=tags_any, tags_all=tags_all,
                          updated_after=updated_after,
                          persistence=persistence)

    def mark_consolidated(self, task_id: str) -> bool:
        return self.update(task_id, note="[logos] consolidated")

    def notebook_entries(
        self,
        task_id:   str,
        note_type: str = "",
        limit:     int = 50,
    ) -> list[dict]:
        """
        Return notebook entries for a task, optionally filtered by type.

        Used by Logos when scanning for synthesis evidence — it queries
        note_type="evidence" rather than scanning all notes for a marker string.
        """
        tasks = self._load(task_id=task_id)
        if not tasks:
            return []
        entries = tasks[0].notebook
        if note_type:
            entries = [n for n in entries if n.get("type") == note_type]
        return entries[-limit:]

    # ------------------------------------------------------------------
    # Scheduler tick (called by Orchestrator at ~1 Hz)
    # ------------------------------------------------------------------

    @_synchronized
    def scheduler_tick(self) -> list[tuple[str, str]]:
        """Advance task states. Returns list of (task_id, new_state) changes."""
        now, changed = _utcnow(), []
        for t in self.query(state="waiting"):
            if t.remind_at and t.remind_at <= now:
                self.update(t.task_id, state="due",
                            note=f"[scheduler] remind_at reached")
                changed.append((t.task_id, "due"))
        # expiry_at is a deadline, distinct from reminders: honored for every
        # non-terminal state so a parked or due task can never be immortal.
        for t in self.query(state="active|waiting|paused|due"):
            if t.expiry_at and t.expiry_at <= now:
                self.update(t.task_id, state="expired",
                            note=f"[scheduler] expired")
                changed.append((t.task_id, "expired"))
        self._purge_stale(now)
        return changed

    @_synchronized
    def _purge_stale(self, now: str = "", max_age_days: int = 30) -> int:
        """
        Delete terminal tasks (completed/cancelled/expired/retired) older than
        max_age_days so a persistent htm.db doesn't grow unbounded.
        Runs at most once per hour (guarded by a monotonic clock).
        """
        import time as _time
        last = getattr(self, "_last_purge", 0.0)
        now_m = _time.monotonic()
        if now_m - last < 3600.0:
            return 0
        self._last_purge = now_m
        if self._conn is None:
            return 0
        try:
            from datetime import timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
            cur = self._conn.execute(
                "DELETE FROM htm_tasks WHERE state IN "
                "('completed','cancelled','expired','retired') AND updated_at < ?",
                (cutoff,),
            )
            self._conn.commit()
            return cur.rowcount or 0
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # ASC shorthands
    # ------------------------------------------------------------------

    def get(self, surface: str) -> Any:
        """Shorthand for htm.asc.get(surface)."""
        return self.asc.get(surface)

    def new_session(self, session_id: str = ""):
        old_asc = self.asc
        self.asc = ActiveSessionCache(session_id=session_id)
        # Preserve durable surfaces: tool_failures (cumulative failure
        # registry) and hot_capabilities (usage counts / health scores)
        # survive session boundaries so degraded tools stay degraded and
        # usage statistics carry forward.
        self.asc.tool_failures   = old_asc.tool_failures
        self.asc.hot_capabilities = old_asc.hot_capabilities

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @_synchronized
    def stats(self) -> dict:
        all_tasks = self.query()
        by_state: dict[str, int] = {}
        for t in all_tasks:
            by_state[t.state] = by_state.get(t.state, 0) + 1
        asc_entities = self.asc.get("hot_entities")
        return {
            "task_count":       len(all_tasks),
            "by_state":         by_state,
            "asc_entities":     len(asc_entities),
            "asc_unresolved":   sum(1 for e in asc_entities.values()
                                    if e.status in ("unresolved", "implied")),
            "workbook_entries": len(self.asc.get("workbook")),
        }
