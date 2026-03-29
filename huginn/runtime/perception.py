"""
runtime/perception.py — Perception Manager (Orchestrator submodule).

Responsibilities (CognitiveModule_Addendum.md §2):
  - Run all active perception pipelines (discovered as HTM tasks
    with initiated_by="system", state="active")
  - Each pipeline is a skill artifact with class_type="pipeline":
      tool chain with fully prefilled args, no LLM insertion between steps
  - Resolve biometric signatures against the Muninn entity registry
  - Write canonical STMEvents to STMStore
  - Notify the Exilis callback on each write
  - NO LLM. Zero cognitive logic.

Pipeline execution:
  Each pipeline step calls its tool handler with the args from the artifact,
  augmented by the previous step's output (output chaining).
  The Orchestrator supervises failure via HTM notebook entries.

Signature resolution:
  Any tool output with a "signature" field is intercepted here.
  The Perception Manager resolves it against Muninn, updates the
  Orchestrator session, and registers implied entities in ASC.
  This is the ONLY place in the system where signature resolution occurs.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .stm import STMStore, STMEvent
from .htm import HTM, HotEntity


# ---------------------------------------------------------------------------
# Tool handler registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Maps tool_id → Python callable.

    Operators register tool handlers here at startup. The Perception
    Manager (and Tool Manager) look up handlers by tool_id.
    """

    def __init__(self):
        self._handlers: dict[str, Callable] = {}

    def register(self, tool_id: str, handler: Callable):
        self._handlers[tool_id] = handler

    def get(self, tool_id: str) -> Optional[Callable]:
        return self._handlers.get(tool_id)

    def has(self, tool_id: str) -> bool:
        return tool_id in self._handlers


# ---------------------------------------------------------------------------
# Pipeline step result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    success:    bool
    output:     dict[str, Any]
    error:      str = ""
    duration_ms: int = 0


# ---------------------------------------------------------------------------
# Session context (minimal view needed by Perception Manager)
# ---------------------------------------------------------------------------

@dataclass
class SessionContext:
    session_id:       str = ""
    entity_id:        str = ""
    permission_scope: list[str] = field(default_factory=list)
    denied:           list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Perception Manager
# ---------------------------------------------------------------------------

SIG_THRESHOLD = 0.88   # minimum cosine similarity for entity match


class PerceptionManager:
    """
    Runs active perception pipelines and writes events to STM.

    Parameters
    ----------
    stm : STMStore
        Shared STM store.
    htm : HTM
        Hot Task Manager — pipelines are system HTM tasks.
    muninn : MemoryAgent
        For entity registry lookups (signature resolution).
    tools : ToolRegistry
        Tool handler registry.
    session : SessionContext
        Current session (mutable; updated when entity is resolved).
    on_event_written : callable
        Called after every STM write — wakes Exilis.
    sig_threshold : float
        Minimum cosine similarity for voiceprint/faceprint match.
    """

    def __init__(
        self,
        stm:               STMStore,
        htm:               HTM,
        muninn,
        tools:             ToolRegistry,
        session:           SessionContext,
        on_event_written:  Callable,
        sig_threshold:     float = SIG_THRESHOLD,
    ):
        self.stm              = stm
        self.htm              = htm
        self.muninn           = muninn
        self.tools            = tools
        self.session          = session
        self.on_event_written = on_event_written
        self.sig_threshold    = sig_threshold

    # ------------------------------------------------------------------
    # Primary entry point — called from Orchestrator tick
    # ------------------------------------------------------------------

    def run_active_pipelines(self):
        """
        Run one step of every active system pipeline.
        Called each Orchestrator tick (1 ms loop).

        Each pipeline task stores its current step pointer in
        task.progress. The pipeline artifact defines the full step list.
        """
        for task in self.htm.query(initiated_by="system", state="active",
                                   tags_any=["pipeline"]):
            pipeline_artifact = self._recall_pipeline(task)
            if pipeline_artifact is None:
                self.htm.note(task.task_id,
                              "[perception] Pipeline artifact not found in LTM.")
                continue
            self._execute_pipeline(task, pipeline_artifact)

    def _recall_pipeline(self, task) -> Optional[dict]:
        """Fetch the pipeline artifact from Muninn LTM."""
        try:
            try:
                from memory_module.recall import RecallQuery as _RQ
                _q = _RQ(semantic_query=task.title,
                         topics=["pipeline", "perception"], top_k=1)
            except ImportError:
                _q = task.title
            results = self.muninn.recall(_q, top_k=1)
            if results:
                content = results[0].entry.content
                return json.loads(content) if isinstance(content, str) else content
        except Exception:
            pass
        return None

    def _execute_pipeline(self, task, artifact: dict):
        """
        Execute one pipeline cycle (all steps, in order, output-chained).
        Writes a canonical STMEvent on completion.
        Logs failures to HTM notebook.
        """
        steps = artifact.get("steps", [])
        if not steps:
            return

        payload = {}  # accumulates across steps

        for step in sorted(steps, key=lambda s: s.get("order", 0)):
            tool_id = step.get("artifact_id", "")
            args    = dict(step.get("args", {}))  # prefilled — no LLM insertion

            # Merge previous step output into args (output chaining)
            args.update(payload)

            handler = self.tools.get(tool_id)
            if handler is None:
                self.htm.note(task.task_id,
                              f"[perception] Tool not registered: {tool_id}")
                return

            import time
            t0 = time.monotonic()
            try:
                result  = handler(**args)
                elapsed = int((time.monotonic() - t0) * 1000)
                payload.update(result if isinstance(result, dict) else {"output": result})
            except Exception as e:
                elapsed = int((time.monotonic() - t0) * 1000)
                raise PipelineStepError(
                    f"Step {step.get('order')} ({tool_id}) failed: {e}",
                    tool_id=tool_id,
                ) from e

        # Resolve signatures before writing to STM
        if "signature" in payload:
            payload = self._resolve_signature(payload)

        # Write canonical STMEvent
        event = self.stm.record(
            source     = artifact.get("source_type", "sensor"),
            type       = artifact.get("event_type", "sensor"),
            payload    = payload,
            confidence = payload.get("confidence", 1.0),
        )

        # Update pipeline task progress
        self.htm.update(
            task.task_id,
            progress=f"Last event: {event.id}",
        )

        # Wake Exilis
        self.on_event_written()

    # ------------------------------------------------------------------
    # Direct event injection (for tool results routed back through
    # the Orchestrator — not perception pipelines)
    # ------------------------------------------------------------------

    def write_tool_result_event(
        self,
        tool_id:     str,
        request_id:  str,
        result:      dict,
        success:     bool = True,
        duration_ms: int  = 0,
    ) -> STMEvent:
        """
        Write a tool execution result to STM.
        Called by Orchestrator Tool Manager after any tool completes.
        """
        payload = {
            "tool_id":     tool_id,
            "request_id":  request_id,
            "result":      result,
            "success":     success,
            "duration_ms": duration_ms,
        }
        if "signature" in result:
            payload["signature"] = result.pop("signature")
            payload = self._resolve_signature(payload)

        event = self.stm.record(
            source="tool", type="tool_result",
            payload=payload, confidence=1.0,
        )
        self.on_event_written()
        return event

    def write_speech_event(
        self,
        text:       str,
        entity_id:  str = "",
        confidence: float = 1.0,
        extra:      dict = None,
    ) -> STMEvent:
        """
        Write a transcribed speech event to STM.
        Called by the ASR pipeline tool (voice_to_stm).
        """
        payload = {
            "text":      text,
            "entity_id": entity_id,
            **(extra or {}),
        }
        # entity_id may be empty if signature resolution hasn't run yet;
        # the pipeline will have already resolved it via _resolve_signature
        event = self.stm.record(
            source="user", type="speech",
            payload=payload, confidence=confidence,
        )
        self.on_event_written()
        return event

    def write_sensor_event(
        self,
        event_name: str,
        data:       dict,
        confidence: float = 1.0,
    ) -> STMEvent:
        """Write a normalised sensor event to STM."""
        payload = {"event": event_name, **data}
        event   = self.stm.record(
            source="sensor", type="sensor",
            payload=payload, confidence=confidence,
        )
        self.on_event_written()
        return event

    # ------------------------------------------------------------------
    # Signature resolution
    # ------------------------------------------------------------------

    def _resolve_signature(self, payload: dict) -> dict:
        """
        Match a biometric embedding against the Muninn entity registry.
        Enriches payload with entity_id on match; registers implied
        entity in ASC on non-match.

        This is the ONLY place in the system where signature resolution occurs.
        """
        sig        = payload.pop("signature", {})
        embedding  = sig.get("embedding", [])
        kind       = sig.get("kind", "voiceprint")
        sig_conf   = sig.get("confidence", 0.0)

        if not embedding:
            payload["signature_resolved"] = False
            return payload

        # Query Muninn for entities that carry a signature of this kind.
        #
        # Muninn treats signatures as entity associations — entities whose
        # content JSON includes a "signatures" list are the candidates.
        # We retrieve them via topic-targeted recall ("signature", kind)
        # and then apply cosine similarity locally.
        #
        # Topic tagging convention: entities with voiceprints are stored
        # with topics=["signature", "voiceprint"] by whichever code
        # registered them (Sagax via create_entity, or operator at setup).
        matched_id  = None
        best_score  = 0.0

        try:
            try:
                from memory_module.recall import RecallQuery as _RQ
                _q = _RQ(topics=["signature", kind], top_k=20)
            except ImportError:
                _q = f"signature {kind}"

            candidates = self.muninn.recall(_q, top_k=20)
            for result in candidates:
                entity = getattr(result, "entry", result)
                stored = self._get_stored_embedding(entity, kind)
                if stored:
                    sim = _cosine_similarity(embedding, stored)
                    if sim > best_score:
                        best_score = sim
                        matched_id = getattr(entity, "id", None)
        except Exception:
            pass

        # Fallback: text-clue entity resolution if recall found nothing
        if not matched_id or best_score < self.sig_threshold:
            try:
                fallback = self.muninn.resolve_entity(
                    clues=f"person with {kind}", top_k=10
                )
                for entity, score in fallback:
                    stored = self._get_stored_embedding(entity, kind)
                    if stored:
                        sim = _cosine_similarity(embedding, stored)
                        if sim > best_score:
                            best_score = sim
                            matched_id = entity.id
            except Exception:
                pass

        if matched_id and best_score >= self.sig_threshold:
            payload["entity_id"]            = matched_id
            payload["signature_resolved"]   = True
            payload["sig_match_confidence"] = round(best_score, 4)
            self._open_or_update_session(matched_id)
            self._update_asc_entity(matched_id, sig_conf)
        else:
            implied_id = self.htm.asc.add_implied_entity(
                voiceprint = embedding if kind == "voiceprint" else None,
                faceprint  = embedding if kind == "faceprint"  else None,
                confidence = sig_conf,
            )
            payload["entity_id"]          = implied_id
            payload["signature_resolved"] = False
            payload["implied"]            = True

        return payload

    def _get_stored_embedding(self, entity, kind: str) -> Optional[list[float]]:
        """Extract a stored signature embedding from an Entity record."""
        try:
            data = json.loads(entity.content) if isinstance(entity.content, str) else {}
            for s in data.get("signatures", []):
                if s.get("kind") == kind:
                    return s.get("embedding")
        except Exception:
            pass
        return None

    def _open_or_update_session(self, entity_id: str):
        """
        Update the session when an entity is confirmed.
        Loads grants from the entity's LTM record.
        """
        if self.session.entity_id == entity_id:
            return  # already this entity
        try:
            results = self.muninn.resolve_entity(entity_id, top_k=1)
            if not results:
                return
            entity = results[0][0]
            data   = json.loads(entity.content) if isinstance(entity.content, str) else {}
            grants = data.get("grants", {})
            self.session.entity_id        = entity_id
            self.session.permission_scope = grants.get("permission_scope", [])
            self.session.denied           = grants.get("denied", [])
        except Exception:
            pass

    def _update_asc_entity(self, entity_id: str, confidence: float):
        """Mark a confirmed entity as active in ASC.hot_entities."""
        if entity_id not in self.htm.asc.hot_entities:
            name = ""
            try:
                results = self.muninn.resolve_entity(entity_id, top_k=1)
                if results:
                    name = results[0][0].name
            except Exception:
                pass
            self.htm.asc.hot_entities[entity_id] = HotEntity(
                entity_id  = entity_id,
                name       = name,
                status     = "confirmed",
                confidence = confidence,
            )
        self.htm.asc.touch_entity(entity_id)


# ---------------------------------------------------------------------------
# Pipeline step helpers
# ---------------------------------------------------------------------------

class PipelineStepError(Exception):
    def __init__(self, message: str, tool_id: str = ""):
        super().__init__(message)
        self.tool_id = tool_id


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
