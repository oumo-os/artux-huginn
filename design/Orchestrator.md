# Orchestrator — Routing Bridge Specification

**Repository:** `oumo-os/artux-huginn`  
**Document status:** Living design spec — v1.0  
**Companion document:** `design/CognitiveModule.md` v1.0

---

## 0 — Purpose and Scope

The Orchestrator is the execution runtime sitting between Sagax and the rest of the system. It has **no cognitive logic**. It does not plan, reason, or make decisions about what to do. It routes signals, enforces contracts, and manages the lifecycle of a session.

This document defines:
- The Orchestrator's eight internal surfaces
- Session model and permission gate
- Token stream routing (full block-by-block specification)
- Two-stage nudge protocol
- `<aug_call>` dispatch and timeout handling
- ASC garbage collection
- Workbook lifecycle
- HTM scheduler

---

## 1 — Position in the Stack

```
Exilis ──────────────────────────────────→ STM (Muninn)
   │                                          ↑
   │ triage signal (ignore/act/urgent)        │ result events
   ↓                                          │
Orchestrator ←── Sagax Narrator token stream ─┘
   │
   ├── Tool Manager (tool_call dispatch)
   ├── TTS (speech streaming)
   ├── UI (projection dispatch)
   ├── HTM (task_update writes, scheduler)
   └── ASC (workbook writes, hot cache updates)
```

**Data flows:**
- Exilis → Orchestrator: triage signals, `urgent` interrupts
- Sagax → Orchestrator: structured token stream (Narrator output)
- Orchestrator → Tool Manager: validated `<tool_call>` blocks
- Orchestrator → TTS: live-streamed `<speech>` tokens
- Orchestrator → STM: STM events on block close (via `record_stm`)
- Orchestrator → ASC: workbook entries, hot cache updates, GC
- Orchestrator → HTM: task record writes, scheduler ticks

---

## 2 — Session Model

### 2.1 — Session Object

The Orchestrator owns and maintains the active session object:

```json
{
  "session_id":  "sess-2026-03-09-morning",
  "entity_id":   "entity-john-001",
  "started_at":  "2026-03-09T07:00:00Z",
  "state":       "active | suspended | ended",
  "grants": {
    "permission_scope":       ["microphone", "kettle", "lights", "calendar.read"],
    "denied":                 ["camera", "email.send"],
    "confirmation_required":  ["calendar.write", "file.delete", "kettle.set_temp_above_90"]
  }
}
```

### 2.2 — Session Lifecycle

```
session.create(entity_id, grants)  → session_id; initialize ASC; open workbook
session.suspend(reason)            → state = "suspended"; flush in-flight dispatches
session.resume(session_id)         → state = "active"; Sagax reads HTM for resume points
session.end(session_id)            → state = "ended"; archive workbook; notify Logos for ASC flush
```

Session end triggers:
- Current workbook archived to cold storage.
- `end_of_session` event written to STM.
- Logos notified (via HTM task or direct signal) to perform final ASC flush.

### 2.3 — Permission Gate

The Orchestrator validates every `<tool_call>` against the active session grants **before** dispatching to the Tool Manager.

```python
def permission_check(tool_call_block, session):
    for tool in tool_call_block.tools:
        scope = tool.get("permission_scope", [])
        for s in scope:
            if s in session.grants.denied:
                return deny(tool, reason="explicitly_denied")
            if s not in session.grants.permission_scope:
                return deny(tool, reason="scope_not_granted")
            if s in session.grants.confirmation_required:
                return require_confirmation(tool, session)
    return allow()
```

**On `deny`:** Inject a `tool_result` event with `status:"denied"`, `reason`. Sagax reads this on next wake and reasons about the failure.

**On `require_confirmation`:** Emit a `<speech>` confirmation request to the user. That tool is held; other tools in the same block that don't require confirmation proceed normally.

**`<aug_call>` blocks bypass the permission gate** — they are `polarity:"read"` only. The `polarity` field on the tool descriptor is the enforcement point. If a tool with `polarity:"write"` appears in an `<aug_call>`, the Orchestrator rejects the block and injects an error result.

---

## 3 — Token Stream Routing

The Orchestrator operates a tag-state machine on Sagax's Narrator output. It processes the stream in real time, routing each block as it closes.

### 3.1 — State Machine

```
IDLE
  → on "<thinking>" open:      → CAPTURING_THINKING
  → on "<contemplation>" open: → CAPTURING_CONTEMPLATION
  → on "<speech" open:         → STREAMING_SPEECH
  → on "<tool_call>" open:     → BUFFERING_TOOL_CALL
  → on "<aug_call" open:       → BUFFERING_AUG_CALL
  → on "<task_update" open:    → BUFFERING_TASK_UPDATE
  → on "<projection>" open:    → BUFFERING_PROJECTION

CAPTURING_THINKING
  → on close:   write to debug log only; IDLE

CAPTURING_CONTEMPLATION
  → on close:   record_stm(output/contemplation);
                workbook_write(block);
                IDLE

STREAMING_SPEECH
  → while open: stream tokens to TTS live
  → on close:   record_stm(output/speech, status:"complete");
                workbook_write(block);
                IDLE
  → on NUDGE:   write status:"suspended", resume_point to in-progress speech event;
                halt TTS; IDLE

BUFFERING_TOOL_CALL
  → on close:   result = permission_check(block, session)
                if allow:
                    dispatch_to_tool_manager(block)
                    create_or_update_htm_task(block)
                    workbook_write(block)
                if deny:
                    inject_tool_result_denied(block)
                    workbook_write(block)
                IDLE

BUFFERING_AUG_CALL
  → on close:   validate_all_read_polarity(block)
                dispatch_parallel_with_timeouts(block)
                inject_aug_result(results)
                workbook_write(block)
                IDLE

BUFFERING_TASK_UPDATE
  → on close:   htm_write(block)
                workbook_write(block)
                IDLE

BUFFERING_PROJECTION
  → on close:   dispatch_to_ui(block)
                record_stm(output/projection)
                workbook_write(block)
                IDLE
```

### 3.2 — Workbook Write Contract

On every block close (except `<thinking>`), the Orchestrator writes the full payload to `ASC.workbook`:

```json
{
  "ts":         "2026-03-09T07:15:00Z",
  "block_type": "tool_call | aug_call | contemplation | speech | task_update | projection",
  "content":    { /* full raw block content */ },
  "result":     { /* dispatch result if applicable; null for contemplation/speech */ },
  "session_id": "sess-2026-03-09-morning"
}
```

`<thinking>` blocks are **never** written to the workbook.

---

## 4 — Two-Stage Nudge Protocol

When Exilis raises an `urgent` triage signal mid-Sagax generation:

### Stage 1 — Immediate halt (target < 50 ms)

1. Halt TTS output.
2. If current state is `STREAMING_SPEECH`: write `status:"suspended"` and `resume_point` to the in-progress STM speech event.
3. If current state is `BUFFERING_TOOL_CALL`: discard buffer; write `tool_call_aborted` to workbook.
4. If current state is `BUFFERING_AUG_CALL`: cancel pending aug dispatches; inject `{"status": "nudge_interrupt"}` as `<aug_result>`.
5. Transition to `IDLE`. Freeze further dispatch.

### Stage 2 — Context delivery (target < 50 ms after Stage 1)

1. Deliver the urgent STM event to Sagax's input queue.
2. Wake Sagax with priority flag.
3. Sagax reads the new event and decides whether to resume, pivot, or park the task.

### Nudge sources

- Exilis `urgent` triage (departure, emergency sensor, urgent user speech)
- HTM scheduler tick raising a `due` task requiring immediate Sagax attention
- Operator-injected interrupt

---

## 5 — `<aug_call>` Dispatch

### 5.1 — Dispatch Flow

```python
def handle_aug_call(block):
    # 1. Validate read polarity
    for tool in block.tools:
        descriptor = recall_tool_descriptor(tool.name)
        if descriptor.polarity != "read":
            inject_aug_result({
                "status": "error",
                "reason": "aug_call_write_tool_rejected",
                "tool":   tool.name
            })
            return

    # 2. Resolve per-tool timeouts
    block_timeout = block.attributes.get("timeout_ms", 500)
    for tool in block.tools:
        tool.effective_timeout = tool.args.pop("timeout_ms", block_timeout)

    # 3. Dispatch all in parallel; Sagax generation paused here
    futures = [dispatch_async(tool) for tool in block.tools]

    # 4. Collect results (each independently timed out)
    results = {}
    for tool, future in zip(block.tools, futures):
        try:
            results[tool.name] = future.result(timeout=tool.effective_timeout / 1000)
        except TimeoutError:
            results[tool.name] = {"status": "timeout", "tool": tool.name}
        except Exception as e:
            results[tool.name] = {"status": "error", "tool": tool.name, "reason": str(e)}

    # 5. Inject <aug_result> inline; resume Sagax generation
    inject_aug_result(results)

    # 6. Write to workbook
    workbook_write({"block_type": "aug_call", "tools": [t.name for t in block.tools], "results": results})
```

### 5.2 — Timeout Semantics

| Scenario | Behaviour |
|---|---|
| Per-tool `timeout_ms` declared | Overrides block-level default for that tool |
| Block-level `timeout_ms` only | All tools use the block-level value |
| No timeout declared anywhere | Default: 500 ms |
| Individual tool times out | `{"status": "timeout", "tool": "n"}` in `<aug_result>`; others proceed normally |
| All tools time out | Full `<aug_result>` is all timeouts; Sagax generation resumes; Sagax handles gracefully |
| Nudge arrives mid-aug_call | Cancel all pending futures; inject `{"status": "nudge_interrupt"}`; proceed to Stage 2 |

### 5.3 — Sagax Generation Pause

Sagax generation is halted at `<aug_call>` close. The partial token stream is buffered. The Orchestrator injects `<aug_result>` into Sagax's input as inline context. Sagax then continues generating from the buffered position, incorporating the results before producing its next token.

---

## 6 — ASC Garbage Collection

### 6.1 — Trigger

ASC GC is called by `orchestrator.on_consN_update(new_summary_text)`, invoked from the `update_consN` flow in `runtime/stm.py`.

### 6.2 — GC Algorithm

```python
def asc_gc(new_consN_text):
    new_topics   = extract_topics(new_consN_text)
    new_entities = extract_entity_refs(new_consN_text)
    active_tools = get_active_task_tool_hints()

    # hot_topics: keep only topics in new consN context
    asc.hot_topics = [t for t in asc.hot_topics if t in new_topics]

    # hot_recalls: keep if query topics overlap new context
    asc.hot_recalls = [r for r in asc.hot_recalls
                       if any(t in new_topics for t in r.query_topics)]

    # hot_tools: keep if referenced in new consN or in active HTM tasks
    asc.hot_tools = [t for t in asc.hot_tools
                     if t.tool_id in active_tools or t.referenced_in(new_consN_text)]

    # hot_state: prune stale tool-specific state not in active tasks
    asc.hot_state = prune_stale_tool_state(asc.hot_state, active_tools)

    # hot_entities[confirmed]: prune if not in new consN entity refs
    # hot_entities[unresolved/implied]: NEVER pruned — always retained
    for entity in list(asc.hot_entities.values()):
        if entity.status == "confirmed" and entity.entity_id not in new_entities:
            del asc.hot_entities[entity.entity_id]

    # Archive workbook segment; open new one
    archive_workbook_segment(session_id=current_session_id)
    asc.workbook = new_workbook_segment()
```

### 6.3 — Workbook Archival

On each consN session boundary, the current workbook segment is archived:

```json
{
  "archive_id":    "wb-sess-2026-03-09-morning-seg3",
  "session_id":    "sess-2026-03-09-morning",
  "segment":       3,
  "consN_version": 7,
  "archived_at":   "2026-03-09T09:30:00Z",
  "entries":       [ /* all workbook entries for this segment */ ]
}
```

**Retention policy:** Cold-archived workbook segments are retained until **both** of the following are true:
1. Logos has examined the segment and called `htm.mark_consolidated()` on all relevant tasks.
2. The configurable retention period has elapsed (default: 30 days).

After both conditions, the segment is eligible for permanent deletion. Operators may override the retention period per deployment.

---

## 7 — HTM Scheduler

### 7.1 — Tick Loop

The Orchestrator runs a scheduler tick at 1 Hz minimum:

```python
def scheduler_tick():
    now = utcnow()

    for task in htm.query(state="waiting"):
        if task.remind_at and task.remind_at <= now:
            htm.update(task.task_id, state="due")
            if session.state == "active":
                nudge(reason="task_due", task_id=task.task_id)

    for task in htm.query(state="active"):
        if task.expiry_at and task.expiry_at <= now:
            htm.update(task.task_id, state="expired")
            record_stm({
                "type":    "internal",
                "payload": {"subtype": "task_expired", "task_id": task.task_id}
            })
```

### 7.2 — Minimal Task Record on `<tool_call>`

Every `<tool_call>` block creates or updates a minimal HTM task record for Logos' synthesis trace:

```python
def on_tool_call_close(block, session):
    active = htm.query(initiated_by="sagax", state="active",
                       session_id=session.session_id)
    if active:
        htm.note(active[0].task_id,
                 f"tool_call dispatched: {[t.name for t in block.tools]}")
    else:
        htm.create(
            title=f"tool_call: {[t.name for t in block.tools]}",
            initiated_by="sagax",
            persistence="volatile",
            tags=["tool_call_trace"]
        )
```

Explicit `<task_update>` blocks from Sagax override and enrich these minimal records.

---

## 8 — Hot Cache Updates

| Event | ASC Update |
|---|---|
| Entity resolved via signature router | `hot_entities[entity_id]` updated: status="confirmed" |
| Exilis flags `signature_unresolved` | `hot_entities["implied-N"]` created: embedding + confidence |
| `<tool_call>` dispatched | `hot_tools[tool_id].last_used` updated |
| `recall()` result in `<aug_call>` | `hot_recalls` entry added: `{query, top_k_results, ts}` |
| `<speech>` to entity | `hot_entities[entity_id].last_addressed` updated |

---

## 9 — Tool Manager Interface

The Orchestrator dispatches to the Tool Manager via a structured request:

```json
{
  "request_id":       "orch-req-001",
  "tool_id":          "tool.set_ceiling_lights.v1",
  "inputs":           { "colour": "warm_red", "brightness": 60 },
  "modality":         "request",
  "session_id":       "sess-2026-03-09-morning",
  "permission_scope": ["lights.ceiling"],
  "task_id":          "task-mood-001"
}
```

All results — regardless of modality — return to **Exilis** as STM events, not directly to Sagax. The Orchestrator does not hold results; Exilis records them and they enter the new-event window Sagax reads on its next wake.

**Modality handling:**

| Modality | Orchestrator behaviour |
|---|---|
| `request` | Dispatch; Exilis records result on return |
| `job` | Dispatch; Orchestrator polls; result emitted as STM event when complete |
| `subscribe` | Open subscription; each callback pushes event to Exilis queue |
| `stream` | Open bidirectional channel; stream events routed through Exilis |

---

## 10 — Observability

The Orchestrator emits an `orchestrator_health` event to STM on each consN session boundary:

```json
{
  "type":             "orchestrator_health",
  "session_id":       "sess-2026-03-09-morning",
  "consN_version":    7,
  "ts":               "2026-03-09T09:30:00Z",
  "tool_calls":       12,
  "aug_calls":        4,
  "aug_timeouts":     1,
  "nudges_issued":    1,
  "permission_denials": 0,
  "asc_gc_pruned": {
    "hot_topics": 3,
    "hot_recalls": 2,
    "hot_tools": 0,
    "hot_entities_confirmed": 0
  },
  "asc_retained": {
    "hot_entities_unresolved": 1
  }
}
```

---

## 11 — Open Questions

| # | Question | Status |
|---|---|---|
| 1 | **Workbook retention period** — default 30 days; should it be per-deployment or per-session configurable? | Open |
| 2 | **Multi-tool `<tool_call>` partial failure** — if one tool in a multi-tool block fails, abort the block or continue with remaining tools? | Open (current intent: continue; each tool gets its own result event) |
| 3 | **`<aug_call>` shared wall-clock budget** — one total timeout across all tools in the block. Deferred; per-tool override covers most practical cases. | v0.6 |
| 4 | **Orchestrator sandboxing** — separate process from Sagax to enforce no-direct-LTM-write constraint? | Architecture decision |
| 5 | **Session grant updates mid-session** — can grants be expanded verbally (e.g. user approves camera access)? Or must a new session be created? | Open |
