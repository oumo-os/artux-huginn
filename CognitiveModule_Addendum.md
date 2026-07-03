# Huginn — Architecture Addendum

**Status:** Superseded by CognitiveModule.md v2.0  
**Retained for:** Historical reference — documents the P-1/P-3/P-4 design decisions that shaped the final architecture.

This addendum was written to correct misspecifications in CognitiveModule.md v0.4:

1. **Exilis does not ingest** — the Perception Manager (Orchestrator submodule) owns all I/O. Exilis only reads STM and signals the Orchestrator.
2. **All cognitive decisions are LLM calls** — no hardcoded classifiers or heuristics.
3. **P-1 CLOSED** — perception pipelines are `class_type: "pipeline"` skill artifacts.
4. **P-3 CLOSED** — Exilis shares Sagax's model and consN for triage coherence.
5. **P-4 CLOSED** — active perception pipelines are HTM tasks (`initiated_by: "system"`, `state: "active"`).

All five corrections are now incorporated into CognitiveModule.md v2.0. The Exilis loop design has since evolved further — from a 5ms fixed poll to a continuous loop gated on `events_pending()` (a single EXISTS query with near-zero latency). See `§3` of CognitiveModule.md v2.0.

The `_resolve_signature` path described here (embedding similarity via a separate `resolve_entity_by_embedding` method) was revised: Huginn now queries Muninn via `RecallQuery(topics=["signature", kind])` followed by cosine similarity locally, falling back to text-clue entity resolution. Muninn was not modified — signatures are stored as entity content fields using the existing relationship model.
