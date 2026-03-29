# Orchestrator — Routing Bridge Specification

**Repository:** `oumo-os/artux-huginn`  
**Document status:** Living design spec — v2.0  
**Companion:** CognitiveModule.md v2.0

---

## 0 — Purpose

The Orchestrator is the execution runtime between Sagax and everything else. It has no cognitive logic — it routes, enforces, and manages lifecycle. Eight internal responsibilities:

1. **Token stream router** — tag-state machine on Sagax Narrator output
2. **Speech chunker** — publishes `partial`/`chunk`/`full` events to ActuationBus
3. **Workbook writer** — full session ledger in ASC
4. **Two-stage nudge** — Stage 1 halts TTS and closes open block as `full+interrupted`; Stage 2 delivers urgent context to Sagax
5. **Permission gate** — validates every `<tool_call>` against session grants
6. **`<aug_call>` handler** — parallel dispatch with per-tool timeouts; pauses Sagax generation
7. **HTM scheduler** — `waiting→due`, `active→expired` at ~1 Hz
8. **`on_consn_updated`** — triggered by Sagax after consN update; runs ASC GC with extracted topics/entities; archives workbook segment

---

## 1 — Position in Stack

```
Exilis ──────────────────────────────────────────→ STM
   │ triage signal                                   ↑
   ↓                                                 │ output events
Orchestrator ←── Sagax Narrator token stream ────────┘
   │
   ├── Speech chunker → ActuationBus (partial/chunk/full)
   ├── Tool Manager   (tool_call dispatch)
   ├── HTM            (task_update writes, states, scheduler)
   └── ASC            (workbook, hot cache updates)
```

---

## 2 — Token Stream Router

```
IDLE
  → <thinking>/<think>  → CAPTURING_THINKING
  → <contemplation>     → CAPTURING_CONTEMPLATION
  → <speech>            → STREAMING_SPEECH
  → <speech_step>       → STREAMING_SPEECH_STEP
  → <tool_call>         → BUFFERING_TOOL_CALL
  → <aug_call>          → BUFFERING_AUG_CALL
  → <task_update>       → BUFFERING_TASK_UPDATE
  → <projection>        → BUFFERING_PROJECTION

CAPTURING_THINKING / CAPTURING_THINK:
  → on close: discard — no STM, no workbook, no TTS

STREAMING_SPEECH / STREAMING_SPEECH_STEP:
  → per token: on_tts_token(token) + _feed_speech_chunk(token)
    _feed_speech_chunk:
      publishes partial to bus
      publishes chunk to bus at phrase boundary (punctuation after min_tokens)
  → on close: _publish_speech_full() + record STM output/speech + workbook

BUFFERING_TOOL_CALL:
  → on close: permission_check → Tool Manager dispatch → create/update HTM task

BUFFERING_AUG_CALL:
  → on close: validate all polarity:read → dispatch parallel with timeouts
              inject <aug_result> → resume Sagax generation

BUFFERING_TASK_UPDATE:
  → on close: parse action (create|update|complete|park|state_set|state_get|state_delete)
              write to HTM Tasks or HTM.States

BUFFERING_PROJECTION:
  → on close: dispatch to UI + record STM output/projection
```

---

## 3 — Two-Stage Nudge (v2)

When Exilis raises `urgent`:

**Stage 1 — immediate halt (< 50ms):**

| Narrator state | Action |
|---|---|
| `STREAMING_SPEECH` | Flush chunk buffer → ActuationBus `full+interrupted`. Write STM `output/speech status:full interrupted:true partial_text:"..."`. TTS sentinel. |
| `STREAMING_SPEECH_STEP` | Same as above. Clear `_speech_step_pending`, unblock waiting thread. |
| `BUFFERING_TOOL_CALL` | Discard buffer. Log `tool_call_aborted` to workbook. |
| `BUFFERING_AUG_CALL` | Cancel pending futures. Inject `{"status":"nudge_interrupt"}` as aug_result. |
| `IDLE` | No action needed. |

All states → `IDLE`. Buffers zeroed.

**Stage 2 — context delivery (< 50ms after Stage 1):**
Deliver urgent event to Sagax's input queue. Wake with priority flag.

**No `suspended` status exists in v2.** Everything in STM is `full`.

---

## 4 — ActuationBus Integration

The Orchestrator publishes to the ActuationBus on all speech-related events:

```python
# Per token (STREAMING_SPEECH or STREAMING_SPEECH_STEP):
bus.publish_dict(type="output", target="speech", complete="partial",
                 text=token, session_id=session_id)

# At phrase boundary (punctuation after _CHUNK_MIN_TOKENS):
bus.publish_dict(type="output", target="speech", complete="chunk",
                 text=accumulated_chunk, session_id=session_id)

# At </speech> close (or nudge interrupt):
bus.publish_dict(type="output", target="speech", complete="full",
                 text=full_text, interrupted=interrupted, session_id=session_id)

# At </contemplation> close:
bus.publish_dict(type="output", target="contemplation", complete="full", ...)
```

Subscribers register filter dicts against these events. The bus is non-blocking — full subscriber queues drop silently rather than blocking the Orchestrator.

---

## 5 — `on_consn_updated(new_summary_text)`

Called by Sagax after every successful consN update. The Orchestrator:

1. Extracts topic words and named entities from `new_summary_text` (simple regex)
2. Calls `htm.asc.gc(topics, entities)` — prunes stale hot topics, recalls, capabilities
   - `hot_parameters` and `hot_entities[unresolved]` are **never pruned**
3. Archives current workbook segment (marks consN session boundary)
4. Writes `internal/consn_updated` event to STM

This closes a v1 gap where ASC context accumulated indefinitely.

---

## 6 — `<task_update>` State Operations

In addition to task lifecycle (`create|update|complete|park`), `<task_update>` supports state operations:

```json
{"action": "state_set",    "key": "sagax.model",  "value": "phi4"}
{"action": "state_get",    "key": "sagax.model"}
{"action": "state_get",    "prefix": "sagax"}
{"action": "state_delete", "key": "session.quiet_mode"}
```

`state_set` writes to `HTM.states` and records an internal STM event. Logos flushes dirty states to LTM at session end. `state_get` with `prefix` returns the whole namespace.

---

## 7 — Permission Gate

Every `<tool_call>` block is validated before Tool Manager dispatch:

```python
for scope in tool.permission_scope:
    if scope in session.grants.denied:          → inject tool_result denied
    if scope not in session.grants.permission:  → inject tool_result denied
    if scope in session.grants.confirmation:    → emit speech confirmation; hold tool

# <aug_call> blocks bypass the gate — polarity:read only
# Attempting polarity:write in an <aug_call> → error result injected
```

New-skill confirmation: skills with `requires_confirmation_for_n_runs > 0` require explicit user confirmation before dispatch. Counter decrements on confirmed runs.

---

## 8 — HTM Scheduler

Runs at ~1Hz in a daemon thread:

```python
for task in htm.query(state="waiting"):
    if task.remind_at and task.remind_at <= now:
        htm.update(task_id, state="due")
        if session.active: orchestrator.nudge(reason="task_due", task_id=task_id)

for task in htm.query(state="active"):
    if task.expiry_at and task.expiry_at <= now:
        htm.update(task_id, state="expired")
        stm.record(internal, subtype="task_expired", task_id=...)
```

Every `<tool_call>` block creates or updates a minimal HTM task record (Logos reads these for skill synthesis traces).

---

## 9 — Session Model

```json
{
  "session_id":  "sess-2026-03-09-morning",
  "entity_id":   "entity-john-001",
  "started_at":  "2026-03-09T07:00:00Z",
  "state":       "active | suspended | ended",
  "grants": {
    "permission_scope":       ["microphone", "lights", "kettle"],
    "denied":                 ["camera", "email.send"],
    "confirmation_required":  ["calendar.write"]
  }
}
```

Session grants are loaded from the entity's LTM record when the Perception Manager confirms identity via signature matching.

---

## 10 — Observability

The Orchestrator emits `orchestrator_health` to STM on each consN session boundary:

```json
{
  "type":             "orchestrator_health",
  "session_id":       "...",
  "consN_version":    7,
  "tool_calls":       12,
  "aug_calls":        4,
  "aug_timeouts":     1,
  "nudges_issued":    1,
  "permission_denials": 0,
  "asc_gc_pruned": {"hot_topics": 3, "hot_recalls": 2},
  "asc_retained":  {"hot_entities_unresolved": 1}
}
```

---

## 11 — Open Items

| # | Question | Status |
|---|---|---|
| 1 | `<aug_call>` shared wall-clock budget | Deferred v0.6 |
| 2 | Multi-tool `<tool_call>` partial failure policy | Intent: continue; each tool gets own result |
| 3 | Session grant mid-session update (e.g., camera approved verbally) | Open |
| 4 | Orchestrator sandboxing (separate process from Sagax) | Architecture decision |
