# Huginn — Cognitive Module Specification

**Repository:** `oumo-os/artux-huginn`  
**Document status:** Living design spec — v1.0  
**Supersedes:** CognitiveModule.md v0.8

---

## 0 — Glossary

| Term | Definition |
|---|---|
| **Event** | An atomic, timestamped, structured record appended to STM. Maps to `STMSegment` in `models.py`; treated as synonymous throughout this spec. |
| **consN** | The single rolling, *deliberately lossy* narrative summary in STM. Gives Exilis and Sagax contextual grounding for reasoning. Never used by Logos. Exactly one consN exists per STM at any time. Updating consN marks a session boundary and triggers ASC garbage collection. |
| **consN.last_event_id** | Sagax's bookmark: the last event id folded into its working narrative. Not a Logos consolidation boundary. |
| **Sagax new-event window** | Raw events with id *after* `consN.last_event_id`. Sagax reads consN + this window as working context. |
| **Logos consolidation cursor** | `stm.logos_watermark` — Logos' independent pointer to the last STM event it has verified and flushed. |
| **Artifact** | Any LTM-resident object retrievable by `recall()`: LTM entries, tool descriptors, skills, procedures, entity ledgers, concept clusters. |
| **Tool descriptor** | An LTM artifact describing a callable capability: name, inputs, outputs, cost, latency, polarity, modality, invocation schema, and permission scope. |
| **Polarity** | Whether a tool is read-only (`read`) or can effect changes in the world (`write`). |
| **Modality** | How a tool communicates: `request` (synchronous call/return), `job` (async, poll-able), `subscribe` (push events until unsubscribed), `stream` (bidirectional channel). |
| **Skill** | An LTM artifact synthesised by Logos from repeated successful execution traces. Expresses a reusable, rationale-bearing *guidance sequence* — a list of natural-language and tool-implied steps that Sagax interprets, not executes mechanically. |
| **Procedure** | A temporal workflow orchestrating multiple skills, possibly with schedules, conditions, and branching. |
| **Orchestrator** | The routing and execution runtime. No cognitive logic. Routes Sagax token stream; manages permissions, session, HTM scheduler, nudge, aug_call dispatch, and hot cache updates. Defined fully in `design/Orchestrator.md`; this spec covers its interface contract only. |
| **Muninn** | The Memory Module — SQLite-backed store implementing all memory APIs. Corresponds to the existing `memory_module/` package. |
| **HTM** | Hot Task Manager. Dual-surface: `ActiveSessionCache` (ephemeral, per-session) + `Tasks` (durable, survives STM flush). See §8. |
| **ASC** | `HTM.ActiveSessionCache`. Sagax queries it on demand — not sent wholesale at agent wake-up. |
| **Workbook** | The complete per-session stream mirror written by the Orchestrator: every token block, full tool payloads, recall result sets, aug call exchanges. Lives in ASC. Sagax uses it for failure-mode resume. Logos uses it as a debug supplement for skill synthesis. |
| **Narrator** | Sagax's output interface — not a separate agent. Produces a single structured token stream. The Orchestrator routes each block in real time. |
| **Logos** | Background agent. Sole author of durable LTM. Consolidates STM, synthesises skills, maintains memory hygiene. Also performs final ASC flush after session end. |
| **Sagax** | Interactive planning and reasoning agent. Reads STM + LTM + ASC on demand. Writes STM only. Produces output via Narrator token stream. |
| **Exilis** | Fast attention gate. Ingests, normalises, classifies, runs triage loop, performs signature resolution. Never actuates, never writes LTM. |
| **Signature** | A biometric or device embedding (voiceprint, faceprint, device ID) emitted by perception tools in their result payload. Used to identify entities across modalities. |
| **Implied entity** | An entity inferred by Sagax from cross-modal evidence (voice + face + name claim + context) but not yet confirmed. Lives in `ASC.hot_entities` until Logos resolves or archives it. |

---

## 1 — Executive Summary

Artux is a four-component cognitive stack designed so no single component couples fast-path latency to slow-path intelligence.

| Component | Speed | Responsibility |
|---|---|---|
| **Exilis** | < 100 ms | Attention gate: ingest, normalise, triage (`ignore` / `act` / `urgent`), signature resolution. Records all events to STM. Never actuates. |
| **Orchestrator** | < 50 ms | Routing bridge. No cognitive logic. Routes Exilis signals → Sagax; routes Sagax token stream → TTS / Tool Manager / UI / HTM / Workbook. Eight internal surfaces (§4.4). |
| **Sagax + Narrator** | 1–30 s | Event-driven reasoning and streaming output via structured token stream. Reads ASC on demand. Uses HTM Tasks to park and resume multi-step work. |
| **Logos** | Background | STM→LTM consolidation, skill synthesis, memory hygiene, ASC flush after session end. |

**Four non-negotiable design constraints:**

1. **Tools are memory artifacts.** Tool descriptors live in Muninn. Sagax discovers capabilities via `recall()`, not a hard-coded registry.
2. **consN is always singular, rolling, and lossy — by design.** Raw STM events are **never removed by consN updates**; they persist until Logos flushes them after verified LTM consolidation.
3. **Logos is the sole author of durable LTM.** Sagax reasons in the hot window; Logos decides what earns permanent storage.
4. **Skills are guidance, not scripts.** A skill is a rationale-bearing step sequence that Sagax interprets. Sagax fills in arguments, handles interactive steps, and tracks progress via HTM Tasks.

---

## 2 — Data Model

### 2.1 — Event / STMSegment

"Event" is the conceptual term. `STMSegment` is the concrete type in `models.py`. Do not create a new model.

Required fields on every event:

```python
{
  "id":         "t2026-03-08T12:34:56Z-0001",  # stable across replay
  "ts":         "2026-03-08T12:34:56Z",
  "source":     "user | system | tool | sensor | log",
  "type":       "speech | tool_call | tool_result | task_update | sensor | output | internal",
  "payload":    {},     # structured JSON; Exilis defines shape per type
  "confidence": 0.92    # Exilis-assigned; propagated to downstream LTM
}
```

Output event subtypes (written to STM by Orchestrator on Sagax block close):

```python
# type = "output", payload.subtype one of:
"contemplation"   # in-role world reasoning; Logos consolidates
"speech"          # speech to a target entity; status: "complete" | "suspended"
"projection"      # UI surface update
```

> **`<thinking>` is never an STM event.** It is LLM scratchpad — debug log only. May break the fourth wall (reference own architecture, constraints). Never reaches STM, never reaches Logos. See §9 for full token grammar.

> **`<tool_call>` and `<aug_call>` are not STM output events.** They are dispatched via Orchestrator → Tool Manager. Their results surface as `tool_result` events recorded by Exilis.

> **Backward compatibility:** `STMSegment.is_compression` maps to `type = "internal"`, `payload.subtype = "consN"`. Both representations must be handled during the transition period.

### 2.2 — STM State Object

```json
{
  "tFirst": "t0001",
  "tLast":  "t0123",

  "events": [
    /* All events t0001–t0123. Full fidelity. Append-only.
       Never modified or removed by Sagax or consN updates.
       Flushed only by Logos after verified LTM consolidation. */
  ],

  "consN": {
    "summary_id":    "consN_v7",
    "last_event_id": "t0050",
    "summary_text":  "Narrative covering t0001–t0050. Deliberately lossy.",
    "version":       7,
    "created_at":    "2026-03-08T11:00:00Z",
    "meta": {
      "topics":             ["movie", "mood_lighting", "John"],
      "confidence":         0.92,
      "event_count_folded": 50
    }
  },

  "logos_watermark": "t0045"
  /* Last event Logos has consolidated and flushed.
     Events with id <= logos_watermark have been deleted.
     Fully independent of consN.last_event_id. */
}
```

**Invariants:**
- `events` is ground truth. Never altered by consN updates.
- `consN.last_event_id` is Sagax's read bookmark only.
- `logos_watermark` is Logos' independent flush cursor.
- After flush, events ≤ `logos_watermark` no longer exist in `events`.
- `consN` is a single object. Each update replaces the previous version entirely — no `consN1 + consN2`.
- `consN.last_event_id` and `logos_watermark` may diverge significantly. Neither is an error.

### 2.3 — LTM Partition Map

| Partition | `class_type` | Owner | Notes |
|---|---|---|---|
| Episodic entries | `"observation"` or `"event"` | Logos | Consolidated sequences with evidence pointers |
| Semantic assertions | `"assertion"` | Logos | High-confidence durable facts |
| Decisions | `"decision"` | Logos | Recorded choices with rationale |
| Procedures | `"procedure"` | Logos | Temporal workflow templates |
| Skills | `"skill"` | Logos | Synthesised from execution traces; interpreted by Sagax |
| Tool descriptors | `"tool"` | Logos / operator | Callable capability descriptors with polarity + modality |
| Concept clusters | `"concept_cluster"` | Logos | Topic groupings with evidence refs |

Entities and SourceRefs use the existing `Entity` and `SourceRef` models from `models.py` unchanged, extended with a `signatures` field (§2.5).

### 2.4 — Tool Descriptor: Polarity and Modality Fields

Every tool descriptor carries two additional fields:

```json
{
  "polarity":           "read | write",
  "modality":           "request | job | subscribe | stream",
  "signature_emission": true
}
```

- **`polarity: "read"`** — retrieves information without side effects. Eligible for `<aug_call>`.
- **`polarity: "write"`** — effects changes in the world. Requires `<tool_call>` and permission gate.
- **`modality`** governs how the Orchestrator dispatches and how results return.
- **`signature_emission: true`** — result payload will include a `signature` field; Orchestrator routes it to Exilis for entity resolution.

### 2.5 — Signature Field (Tool Result Payload)

Perception tools emit a standard `signature` block in their result payload:

```json
"signature": {
  "kind":       "voiceprint | faceprint | device_id",
  "embedding":  [0.21, -0.44, 0.87, "..."],
  "confidence": 0.91
}
```

The Orchestrator detects this field and hands the embedding to Exilis' signature resolution loop (§4.1).

Entity records gain a `signatures` field:

```json
{
  "entity_id": "entity-john-001",
  "name":      "John",
  "signatures": [
    { "kind": "voiceprint", "embedding": [...], "registered_at": "...", "confidence": 0.94 }
  ]
}
```

`resolve_entity()` gains an `embedding_similarity` path alongside its existing text-clues path. If similarity exceeds threshold, the event is enriched with `entity_id`. If not, `signature_unresolved: true` is flagged for Sagax inference.

### 2.6 — Session Object (Orchestrator-owned)

```json
{
  "session_id":   "sess-2026-03-09-morning",
  "entity_id":    "entity-john-001",
  "started_at":   "2026-03-09T07:00:00Z",
  "grants": {
    "permission_scope":       ["microphone", "kettle", "calendar.read"],
    "denied":                 ["camera", "email.send"],
    "confirmation_required":  ["calendar.write", "file.delete"]
  }
}
```

---

## 3 — consN Update Algorithm

### 3.1 — Purpose and Intentional Lossiness

consN is a contextual aid for Sagax and Exilis — not a record. Its job is to let Sagax reason about older events without re-reading every raw event on each turn. This compression is **deliberately lossy**: exact timestamps collapse to relative references, repetition merges, fine detail smooths. Each successive update compounds this fuzziness.

**consN is never used by Logos.** Logos always reads raw events via `get_raw_events()`.

### 3.2 — consN Update as Session Boundary

A consN update is not merely a compression pass — it is a **session boundary event**. When Sagax updates consN, the Orchestrator also:

1. Triggers **ASC garbage collection**: hot topics, recalls, and tools not referenced in the new consN context are pruned. `hot_entities` with `signature_unresolved: true` are **always retained**.
2. Archives the current workbook to cold audit storage for Logos examination.
3. Begins a fresh workbook for the new session segment.

Confirmed entities in `hot_entities` may optionally be evicted if no longer referenced. Unresolved/implied entities always persist until Logos resolves or archives them.

### 3.3 — Pseudocode

```python
MIN_NEW_EVENTS = 8

def update_consN(current_consN, stm_events):
    anchor_idx = index_of(stm_events, current_consN.last_event_id)
    new_events  = stm_events[anchor_idx + 1:]

    if len(new_events) < MIN_NEW_EVENTS:
        return current_consN  # not enough new material

    fold_count  = len(new_events) // 2
    to_fold     = new_events[:fold_count]

    new_summary_text  = summarise(current_consN.summary_text, to_fold)
    new_last_event_id = to_fold[-1].id

    # Trigger ASC GC via Orchestrator — separate call, not part of this function
    orchestrator.on_consN_update(new_summary_text)

    return {
        "summary_id":         f"consN_v{current_consN.version + 1}",
        "last_event_id":      new_last_event_id,
        "summary_text":       new_summary_text,
        "version":            current_consN.version + 1,
        "created_at":         utcnow(),
        "meta": {
            "topics":             extract_topics(to_fold),
            "confidence":         min_confidence(to_fold),
            "event_count_folded": current_consN.meta.event_count_folded + fold_count,
        }
    }
    # stm_events is UNCHANGED. Events are never removed here.
```

### 3.4 — Cold-Start Rule

```python
COLD_START_MIN = 8

def maybe_init_consN(stm_events):
    if len(stm_events) < COLD_START_MIN:
        return None  # Sagax reads all events raw
    return update_consN(empty_consN(), stm_events)
```

### 3.5 — Trigger Conditions

consN updates are triggered by **Sagax** when:
1. Raw events after `consN.last_event_id` exceeds `MAX_NEW_EVENTS` (default: 20).
2. An `"end_of_session"` event arrives.
3. Sagax judges a planning arc is complete and its narrative context has grown stale.

Logos never triggers consN updates.

### 3.6 — The `summarise()` Contract

```python
def summarise(existing_narrative: str, new_events: list[Event]) -> str:
    """
    Must:
    - Be strictly shorter than existing_narrative + concatenated new event texts.
    - Preserve named entities, causal relationships, and outcomes.
    - Drop filler, exact timestamps (fold to relative), repeated phrasing.
    - Be idempotent: same inputs yield semantically equivalent output.
    - Not fabricate information absent from the inputs.
    """
```

Concrete implementation: LLM call (small model, temperature 0). Caller wraps with retry + fallback (join events with newline if LLM fails).

---

## 4 — Agent Responsibilities & Contracts

### 4.1 — Exilis (Attention Gate)

**Model:** Small (< 1B parameters). Target latency < 100 ms per event.

**Triage loop:**

```
normalise percept
→ record_stm(event)
→ read context: consN.summary_text + new-event window
   (new-event window includes Sagax output events — enables backchannel detection)
→ triage: ignore | act | urgent
→ [if event.payload.signature exists] run signature resolution loop
```

**Triage output:**
- `ignore` — routine background event; no Sagax wake required
- `act` — Sagax should process when next available
- `urgent` — interrupt Sagax immediately via Orchestrator nudge

**Context-aware triage:** The same raw event may yield different triage outcomes depending on consN context. Example: "ahaaa" after a Sagax sentence = `ignore`. "Could you summarise it" mid-stream = `urgent`.

**Backchannel detection:** Exilis reads Sagax output events (stored as `type:"output"` in the new-event window) to distinguish listener backchannels from genuine interruptions. Sagax's speech is already in STM — no separate channel required.

**Signature resolution loop:**

```python
def resolve_signature(event):
    if "signature" not in event.payload:
        return event

    result = resolve_entity(
        embedding=event.payload.signature.embedding,
        kind=event.payload.signature.kind,
        threshold=0.88
    )

    if result.match:
        event.payload["entity_id"] = result.entity_id
    else:
        event.payload["signature_unresolved"] = True
        # Event still recorded — Sagax will infer cross-modally

    return event
```

**Hard prohibitions:**
- Never actuates. Never calls the Orchestrator for tool dispatch.
- Never writes LTM.
- Never modifies event content after `record_stm` is called.

### 4.2 — Sagax + Narrator (Interactive Reasoning Agent)

**Model:** Medium-large (3–13B parameters). Target: first response ≤ 5 s, complex plans ≤ 30 s.

**Sagax is event-driven.** It sleeps between Exilis signals. On wake-up, it reads HTM Tasks for any active or due tasks before processing the triggering event.

**Narrator is Sagax's output interface** — not a separate agent. It produces a single structured token stream. The Orchestrator routes each block in real time (§9).

**Responsibilities:**
- Read STM as: `consN.summary_text` + raw events after `consN.last_event_id`.
- Query ASC **on demand** for warm context (§8.1). ASC is not pushed to Sagax at wake-up.
- Execute the recall-driven planning loop (§6).
- Interpret skills as guidance sequences — reason through steps, fill in arguments, handle interactive beats (§7.2).
- Create and update HTM Tasks to track multi-step work across interruptions.
- Use the workbook (via ASC query) for failure-mode resume — avoid redoing work already confirmed complete.
- When `signature_unresolved: true` appears on an event, reason cross-modally (voice + face + name claims + prior recall) to build implied entity associations in `ASC.hot_entities`.
- Trigger consN updates when the new-event window grows stale or a planning arc completes.
- Record all output as STM events (`type:"output"`, subtype: `contemplation` | `speech` | `projection`).

**ASC query pattern — Sagax asks for what it needs:**
- "Who is currently active?" → `asc.get("hot_entities")`
- "What tools am I using this session?" → `asc.get("hot_tools")`
- "What did I recall earlier about Christmas lighting?" → `asc.get("hot_recalls")`
- "Where did I get to in the movie mood task?" → `asc.get("workbook")` (failure resume)

LTM `recall()` is used as fallback or supplement when ASC does not have the answer.

**Hard prohibitions:**
- Never calls `consolidate_ltm` for systemic LTM writes. (Narrow exception: a single, clearly bounded decision or assertion Sagax is confident is durable.)
- Never synthesises or modifies skills/procedures.
- Never flushes STM events.
- Never flushes `hot_entities` unresolved/implied entries.

### 4.3 — Logos (Background Agent)

**Model:** Large (9–30B parameters). Runs asynchronously, typically every few minutes or on event-count trigger.

**Logos cycle start:** reads HTM Tasks for any in-progress Logos tasks; resumes from task notebook. Only then begins the consolidation pass on new raw events.

**Responsibilities:**

- **Consolidation:** Read raw events via `get_raw_events(after_id=logos_watermark)`. Never uses `consN.summary_text` as input. Reads full-fidelity events including `contemplation` entries (richer episodic context than raw tool calls alone).

- **Skill synthesis:** Read completed Sagax task notebooks (`persistence:"persist"`) as structured procedural traces. Combined with raw event sequences, this gives Logos both the execution record and Sagax's running commentary. Synthesis thresholds: ≥3 successful executions, ≥0.85 structural similarity, 0 failures in last 5, spread ≥2 days.

- **Workbook as debug supplement:** Logos may examine cold-archived workbooks (moved there on consN session boundary) for richer context — full tool payloads, recall result sets, aug call results. Supplements the raw event log; does not replace it.

- **Memory hygiene:** Cluster LTM fragments into `concept_cluster` artifacts, merge near-duplicate descriptors, flag stale artifacts for archival, run `run_decay()` and `run_maintenance()`.

- **Source management:** `record_source` and `update_source_description` for non-text evidence.

- **ASC flush:** After session end (detected from STM `end_of_session` event or HTM state), Logos performs final ASC flush: resolves or archives remaining implied entities, cleans all ASC surfaces. Logos is the only component authorised to flush `hot_entities`.

- **STM flush:** After verified LTM consolidation, advance `logos_watermark` and delete raw events ≤ that marker.

**Failure semantics:**
- Transactional loop — single pass failure does not abort the whole run.
- After 3 consecutive failures on the same sequence: archive with reason `"logos_failure"`, continue.
- Emits `logos_health` event after each pass (§12).

### 4.4 — Orchestrator (Routing Bridge)

The Orchestrator is not a cognitive agent. It has no reasoning capability. It routes signals and enforces contracts. Eight internal surfaces:

**1. Token stream parser + router.** Tag-state machine on Sagax's Narrator output. On each block open/close, routes to the appropriate surface (§9).

**2. Workbook writer.** On every block close, writes the full payload to `ASC.workbook`. Complete session ledger: every token block, full tool payloads, recall result sets, aug call results.

**3. Two-stage nudge.** Stage 1 (< 50 ms): halt TTS, write `status:"suspended"` to the current open speech STM event, freeze tool dispatch. Stage 2: deliver the context update (new `urgent` event from Exilis) to Sagax's input queue.

**4. HTM scheduler tick.** On each tick: advance `waiting → due` and `active → expired` based on current time. Tick rate: 1 Hz minimum.

**5. Permission gate.** Validates every `<tool_call>` against the active session grants before Tool Manager dispatch. Denied → emit failure result event. Confirmation-required → emit `<speech>` confirmation request to user before proceeding.

**6. `<aug_call>` handler.** Buffer, dispatch synchronously (all tools in block in parallel), inject `<aug_result>` inline into Sagax's generation, enforce per-tool and block-level timeout budget. **Sagax generation is paused mid-stream** waiting for `<aug_result>` injection before continuing.

**7. Signature router.** Pulls `signature` fields from tool results. Hands to Exilis resolution loop before the result event is recorded.

**8. Hot cache updater.** On resolved entity, tool use, or recall result: updates the relevant ASC surface fields (`hot_entities`, `hot_tools`, `hot_recalls`).

---

## 5 — Memory APIs (Muninn Surface)

| Operation | Caller | Notes |
|---|---|---|
| `record_stm(content)` | Exilis, Sagax | Write any event to STM. |
| `get_stm_window()` | Sagax | Returns `consN.summary_text` + raw events after `consN.last_event_id`. |
| `get_raw_events(after_id, limit)` | Logos | Full-fidelity events for consolidation. |
| `get_logos_watermark()` | Logos | Logos' read cursor. |
| `recall(query, top_k)` | Sagax | Primary capability discovery. Returns `list[RecallResult]`. |
| `create_entity(name, description, topics)` | Sagax | First encounter with a significant person or object. |
| `observe_entity(entity_id, observation, authority, memory_ref)` | Sagax | New fact about known entity. |
| `resolve_entity(clues, embedding, kind, top_k, threshold)` | Exilis, Sagax | Text-clues path or embedding-similarity path. Returns match + entity_id if found. |
| `consolidate_ltm(narrative, class_type, topics, confidence)` | Logos (primary); Sagax (narrow exception) | Durable LTM write. |
| `record_source(ltm_entry_id, location, type, description, meta)` | Logos | Register external evidence. |
| `update_source_description(source_id, new_description)` | Logos, Sagax | Enrich source after VLM/ASR re-examination. |
| `run_decay()` | Logos | Time-based confidence decay pass. |
| `run_maintenance()` | Logos | Archive weak entries; purge stale scars. |

**HTM API surface:**

| Operation | Caller | Notes |
|---|---|---|
| `htm.create(title, initiated_by, persistence, tags)` | Sagax, Logos | Create a new task. Returns `task_id`. |
| `htm.note(task_id, entry)` | Sagax, Logos | Append a notebook entry. |
| `htm.update(task_id, state, progress, resume_at, remind_at)` | Sagax, Logos, Orchestrator | Update task state. |
| `htm.complete(task_id, output, confidence)` | Sagax, Logos | Mark task done; write output. |
| `htm.query(initiated_by, state)` | Sagax, Logos | Read tasks at cycle start. |
| `htm.mark_consolidated(task_id)` | Logos | Mark task as examined by Logos; eligible for archival. |

**ASC API surface (Orchestrator-owned; Sagax queries):**

| Operation | Caller | Notes |
|---|---|---|
| `asc.get(surface)` | Sagax | Query one surface: `hot_entities`, `hot_tools`, `hot_recalls`, `hot_topics`, `workbook`, `hot_state`. |
| `asc.gc(new_consN_topics)` | Orchestrator | Garbage collect stale surfaces on consN session boundary. Never touches `hot_entities[unresolved]`. |
| `asc.flush(session_id)` | Logos | Full ASC flush after session end. |

**Performance contracts:**
- `recall()` ≤ 500 ms for `top_k ≤ 10` on warm cache.
- `record_stm()` non-blocking from Exilis' perspective (async queue acceptable).
- `asc.get()` ≤ 20 ms (in-memory).

---

## 6 — Sagax Planning Loop

### 6.1 — Core Principle

Sagax *reasons first, then searches*. It does not scan a registry. For each required step it articulates what needs to happen in natural language, then asks Muninn whether it knows how.

When Sagax discovers a skill, it does not mechanically execute the steps. It reads the skill's guidance sequence, reasons about it (typically in `<contemplation>`), decides how to proceed, fills in arguments, handles interactive beats, and tracks progress via an HTM Task.

### 6.2 — Skill Execution Example

Given `skill.set_movie_mood`:

```
steps:
  1. Find out the genre/title of the movie — ask the user if unknown
  2. Identify fitting palette and theme (recall or ask user)
  3. set_ceiling_lights
  4. set_wall_lights
  5. close_blinds
  6. start_popcorn_machine (confirm with user first)
```

Sagax on reading this:

```xml
<contemplation>
  I see skill.set_movie_mood. Step 1 requires the movie title — I don't know it yet.
  I'll create a task to track progress then ask John.
</contemplation>
<task_update task_id="new" title="Set movie mood — John" state="active"
             progress="step 1: need movie title" resume_at="step_1_response"/>
<speech target="entity-john-001">
  What movie are you watching tonight, John?
</speech>
```

On next Exilis trigger (John says "Home Alone"):

```xml
<contemplation>
  John said Home Alone — Christmas theme. Step 2: recall preferred Christmas
  lighting themes.
</contemplation>
<aug_call timeout_ms="400">
  {"name": "recall", "args": {"query": "Christmas lighting palette preferred", "top_k": 5}}
  {"name": "asc.get", "args": {"surface": "hot_entities"}}
</aug_call>
<aug_result>
  {"recall": [...warm red, gold, candlelight...], "asc.get": {"john": {...}}}
</aug_result>
<contemplation>
  Recall returned warm red + gold palette. John confirmed active. Proceeding steps 3–5.
  Step 6 (popcorn) — interactive, I'll ask after lights are set.
</contemplation>
<task_update task_id="task-mood-001" progress="steps 3–5: setting lights and blinds"
             resume_at="step_6_popcorn"/>
<tool_call>
  {"name": "set_ceiling_lights", "args": {"colour": "warm_red", "brightness": 60}}
  {"name": "set_wall_lights",    "args": {"colour": "gold", "brightness": 40}}
  {"name": "close_blinds",       "args": {}}
</tool_call>
```

When the skill is fully executed: `htm.complete(task_id)`. Logos finds the task notebook + raw events and has a clean structured trace for synthesis — without Sagax reconstructing the sequence from scratch.

### 6.3 — Planning Pseudocode

```python
MAX_PLAN_STEPS  = 20
MAX_RECALL_ITER = 3
STUCK_THRESHOLD = 2

def sagax_plan_and_execute(goal, context):
    # Check HTM for active/due tasks at cycle start
    active_tasks = htm.query(initiated_by="sagax", state="active|due")
    if active_tasks:
        return resume_from_task(active_tasks[0])

    plan        = []
    world_state = get_current_state()
    stuck_count = 0
    step_count  = 0

    while not goal_satisfied(goal, world_state):
        if step_count >= MAX_PLAN_STEPS:
            record_stm(f"Planning aborted: exceeded {MAX_PLAN_STEPS} steps")
            return failure("max_steps_exceeded", plan)

        step = reason_next_step(goal, world_state, plan, context)
        if step is None:
            return failure("cannot_decompose", plan)

        candidates = []
        for attempt in range(MAX_RECALL_ITER):
            q = broaden_query(step.description, attempt)
            candidates = recall(q, top_k=5 + attempt * 5)
            if candidates:
                break

        if not candidates:
            stuck_count += 1
            if stuck_count >= STUCK_THRESHOLD:
                return failure("no_capability_found", plan)
            continue

        stuck_count = 0
        chosen = rank_and_select(candidates, context, world_state)

        # Skills are interpreted, not mechanically executed
        if chosen.artifact_type == "skill":
            return execute_skill_guided(chosen, goal, context)

        unmet = unmet_preconditions(chosen, world_state)
        if unmet:
            prereqs = synthesize_prereqs(unmet, world_state, context)
            if not prereqs:
                return failure("unsatisfied_preconditions", plan)
            plan.extend(prereqs)

        plan.append(chosen)
        step_count += 1
        exec_result = orchestrator.execute(chosen)
        record_stm(exec_result.to_event())
        world_state = update_world_state(exec_result)

        if exec_result.failed:
            fallback = select_fallback(candidates, chosen, exec_result)
            if fallback:
                exec_result = orchestrator.execute(fallback)
                record_stm(exec_result.to_event())
                world_state = update_world_state(exec_result)
                if exec_result.failed:
                    return failure("fallback_failed", plan)
            else:
                return failure("execution_failed", plan)

    record_stm(f"Goal satisfied: {goal} — {len(plan)} steps")
    return success(plan)
```

### 6.4 — Candidate Ranking Policy

Priority order (highest first):
1. **Procedure** — most specific match, already-tested workflow.
2. **Skill** — tested guidance sequence; Sagax interprets.
3. **Tool** — atomic capability; Sagax composes the sequence itself.
4. **LTM assertion with capability hint** — lower confidence; use with caution.

Within a tier, rank by: `confidence × semantic_score × (1 / latency_estimate)`.

### 6.5 — Workbook-Based Failure Resume

When Sagax wakes after a failure or interruption, before replanning:

```python
def attempt_workbook_resume(task_id):
    task     = htm.query(task_id=task_id)[0]
    workbook = asc.get("workbook")

    last_confirmed = find_last_success(task.notebook, workbook)
    if last_confirmed:
        return resume_from(last_confirmed.next_step, task)
    else:
        return replan(task.goal)
```

This is the primary reason Sagax queries the workbook — avoiding expensive replanning and re-execution of already-completed steps.

---

## 7 — Artifact Schemas

### 7.1 — Tool Descriptor

```json
{
  "artifact_type":       "tool",
  "tool_id":             "tool.set_ceiling_lights.v1",
  "title":               "Set ceiling lights",
  "description":         "Sets the colour and brightness of ceiling smart bulbs.",
  "capability_summary":  "Control ceiling ambient lighting. Use for mood, wakeup, movie scenes.",
  "polarity":            "write",
  "modality":            "request",
  "signature_emission":  false,
  "inputs": {
    "colour":     { "type": "string" },
    "brightness": { "type": "int", "min": 0, "max": 100, "default": 80 }
  },
  "outputs": {
    "status": { "type": "string", "enum": ["ok", "error"] }
  },
  "preconditions":        ["lights.ceiling.reachable"],
  "postconditions":       ["lights.ceiling.set"],
  "permission_scope":     ["lights.ceiling"],
  "latency_estimate_ms":  300,
  "cost":                 0.0,
  "confidence":           1.0,
  "version":              1,
  "registered_at":        "2026-03-08T09:00:00Z"
}
```

### 7.2 — Skill

Skills contain a **guidance sequence** — a rationale-bearing ordered list that Sagax interprets. Steps may be natural-language directives, tool-implied steps, or interactive beats. Sagax fills in arguments and decides how to handle each step.

```json
{
  "artifact_type": "skill",
  "skill_id":      "skill.set_movie_mood.v1",
  "title":         "Set movie mood",
  "description":   "Configure ambient environment to match the tone of a movie being watched.",
  "capability_summary": "Sets lighting, blinds, and accessories to match movie genre/theme.",
  "polarity":      "write",
  "modality":      "request",
  "inputs": {
    "entity_id": { "type": "string", "description": "The person watching" }
  },
  "outputs": {
    "result": { "type": "string", "enum": ["mood_set", "partial", "failed"] }
  },
  "rationale": "Determine the movie for context, derive a palette, apply it to all relevant surfaces.",
  "steps": [
    {
      "order":       1,
      "guidance":    "Find out the genre or title of the movie — ask the user if unknown.",
      "interactive": true,
      "tool_hint":   null
    },
    {
      "order":       2,
      "guidance":    "Identify fitting colour palette and theme. Recall user preferences or derive from genre.",
      "interactive": false,
      "tool_hint":   "recall | ask_user"
    },
    {
      "order":       3,
      "guidance":    "Set ceiling lights to primary theme colour.",
      "interactive": false,
      "tool_hint":   "tool.set_ceiling_lights"
    },
    {
      "order":       4,
      "guidance":    "Set wall lights to accent colour.",
      "interactive": false,
      "tool_hint":   "tool.set_wall_lights"
    },
    {
      "order":       5,
      "guidance":    "Close blinds if it is daytime.",
      "interactive": false,
      "tool_hint":   "tool.close_blinds"
    },
    {
      "order":       6,
      "guidance":    "Start popcorn machine if user has confirmed snack preference.",
      "interactive": true,
      "tool_hint":   "tool.start_popcorn_machine"
    }
  ],
  "preconditions":  ["lights.reachable"],
  "postconditions": ["environment.mood_set"],
  "evidence": [
    "exec_trace_2026-03-01T19:00:00Z",
    "exec_trace_2026-03-03T20:15:00Z",
    "exec_trace_2026-03-06T21:00:00Z"
  ],
  "confidence":                    0.90,
  "version":                       1,
  "synthesised_at":                "2026-03-08T06:00:00Z",
  "human_verified":                false,
  "requires_confirmation_for_n_runs": 2
}
```

**Interactive steps** (`interactive: true`) require Sagax to pause and gather input before proceeding — ask the user, recall a preference, or observe context. This does not mean the Orchestrator auto-injects a confirmation. It means Sagax's reasoning must include a response-gathering beat.

### 7.3 — Skill Synthesis Thresholds (Logos Policy)

Logos promotes a trace cluster to a skill when **all** of the following hold:

| Criterion | Default threshold | Rationale |
|---|---|---|
| Successful executions | ≥ 3 | Avoids promoting one-off lucky plans |
| Structural similarity | ≥ 0.85 cosine across step sequences | Ensures one coherent pattern |
| No failure in window | 0 failures in last 5 executions | Prevents promoting unreliable sequences |
| Time span | ≥ 2 different days | Avoids session artifacts |
| Human opt-in (configurable) | Optional confirmation gate | Safety for high-risk capabilities |

Confidence written as: `successes / total_executions`, clamped to `[0.5, 0.97]`.

Synthesis decisions are logged as `logos_synthesis_candidate` events:

```json
{
  "type":           "logos_synthesis_candidate",
  "candidate_id":   "cand-movie-mood-0003",
  "title":          "Set movie mood",
  "evidence_count": 3,
  "similarity":     0.91,
  "decision":       "promoted | rejected | deferred",
  "reason":         "threshold met after 3rd successful trace"
}
```

### 7.4 — Skill Versioning Policy

A new version (`v2`, `v3`, …) is created — never replacing `v1` — when:
- A step's tool hint is replaced by a different artifact.
- A precondition or postcondition is added or removed.
- Structural similarity between the new trace cluster and existing skill drops below 0.85.

Minor improvements (`capability_summary`, `confidence`, `latency_estimate_ms`) are written **in place** — no version bump.

Deprecated versions are archived, not deleted. Findable by `recall()` for audit; ranked below active versions.

### 7.5 — Procedure

```json
{
  "artifact_type": "procedure",
  "procedure_id":  "procedure.morning_wakeup.v1",
  "title":         "Morning routine",
  "description":   "Timed sequence of actions to prepare the environment at wake-up.",
  "schedule_hint": "user.pref.morning_time",
  "steps": [
    { "time_offset_s": 0,   "artifact_id": "skill.set_alarm",      "args": { "time": "{{user_pref.wake_time}}" } },
    { "time_offset_s": 0,   "artifact_id": "skill.turn_on_lights", "args": {} },
    { "time_offset_s": 60,  "artifact_id": "skill.start_kettle",   "args": {} },
    { "time_offset_s": 240, "artifact_id": "skill.open_windows",   "args": {} }
  ],
  "inputs":        ["user_pref"],
  "preconditions": ["home.occupied"],
  "evidence":      ["ltm_entry_370", "ltm_entry_412"],
  "confidence":    0.88,
  "version":       1
}
```

---

## 8 — HTM (Hot Task Manager)

HTM is a dual-surface store. It is not queryable via `recall()`, not part of the STM event stream, and not a push mechanism. Agents read it at cycle start.

### 8.1 — Surface 1: ActiveSessionCache (ASC)

Ephemeral. Per-session. Flushed by Logos after session end. **Not pushed to Sagax at wake-up — queried on demand via `asc.get(surface)`.**

```
HTM.ActiveSessionCache
│
├── workbook        Complete stream mirror: every token block, full tool payloads,
│                   recall result sets, aug call exchanges.
│                   Primary value: Sagax failure-mode resume — avoids re-executing
│                   already-confirmed steps.
│                   Secondary: Logos debug supplement for skill synthesis.
│                   Archived to cold storage on each consN session boundary.
│
├── hot_entities    Currently referenced entities (entity_id + summary)
│                   + unresolved/implied entities (signature embeddings, name
│                     claims, cross-modal correlations built by Sagax).
│                   NEVER pruned by ASC GC.
│                   Flushed only by Logos after session end.
│
├── hot_tools       Tools/skills invoked or recalled this session.
│                   Eligible for ASC GC.
│
├── hot_topics      Active topic threads from current session.
│                   Eligible for ASC GC.
│
├── hot_recalls     {query, top_k_results} pairs recently returned.
│                   Eligible for ASC GC.
│
└── hot_state       System + tool runtime parameters.
                    Eligible for ASC GC.
```

**ASC GC policy (triggered on consN update):**
- `hot_tools`, `hot_topics`, `hot_recalls`, `hot_state`: prune entries not referenced in the new consN context.
- `hot_entities[unresolved/implied]`: **always retained**. These persist until Logos resolves or archives them after session end — like that lingering sense of recognition from the stranger who waved at you at the bus station.
- `hot_entities[confirmed]`: may be pruned if not referenced in new consN context. Logos can re-surface from LTM if needed later.

### 8.2 — Surface 2: Tasks (Durable)

Survives STM flush. Survives consN GC. Survives session restart.

```json
{
  "task_id":      "task-uuid-001",
  "title":        "Set movie mood — John",
  "initiated_by": "sagax | logos | system",
  "state":        "active | waiting | paused | due | completed | expired | cancelled",
  "progress":     "Step 3 complete: ceiling set to warm red. Next: wall lights.",
  "resume_at":    "step_4_wall_lights",
  "remind_at":    null,
  "expiry_at":    "2026-03-09T23:59:00Z",
  "notebook": [
    { "ts": "...", "entry": "Started. Goal: set mood for Home Alone." },
    { "ts": "...", "entry": "Asked John about movie. Response: Home Alone." },
    { "ts": "...", "entry": "Recalled Christmas theme: warm red + gold." },
    { "ts": "...", "entry": "Step 3 done: ceiling_lights ok." },
    { "ts": "...", "entry": "Interrupted: John left. Resuming at step 4 on return." }
  ],
  "output": {
    "summary":    "Ceiling set. Wall lights and blinds pending.",
    "confidence": 0.80
  },
  "tags":        ["lighting", "john", "movie"],
  "persistence": "volatile | persist | audit_only"
}
```

**`persistence` values:**
- `volatile` — discard after session end.
- `persist` — retain; Logos reads notebook for skill synthesis.
- `audit_only` — retain for operator review; not used for synthesis.

**All `<tool_call>` blocks create or update a minimal HTM task record.** This gives Logos structured execution traces without inferring from raw events alone.

### 8.3 — What HTM Is Not

- Not queryable via `recall()`.
- `<task_update>` writes directly to HTM — no STM event is produced.
- Not a push mechanism. Sagax checks at wake-up; Logos checks at cycle start.

---

## 9 — Narrator Token Grammar

Sagax produces a single structured token stream. The Orchestrator operates a tag-state machine on it, routing each block as it closes.

### 9.1 — Block Reference

| Block | On open | On close | STM event | Workbook | Notes |
|---|---|---|---|---|---|
| `<thinking>` | capture | debug log only | **Never** | No | LLM scratchpad. May break fourth wall. Never in memory. |
| `<contemplation>` | capture | write to STM (`output/contemplation`) | ✓ | ✓ | In-role world reasoning. Logos consolidates alongside raw events. |
| `<speech target="id">` | stream to TTS live | write to STM (`output/speech`) | ✓ | ✓ | `status:"complete"` or `"suspended"`. |
| `<tool_call>` | buffer | dispatch to Tool Manager via Orchestrator permission gate | No | ✓ | Supports multiple tools in one block. Creates/updates HTM task record. |
| `<aug_call timeout_ms="N">` | buffer | dispatch synchronously (all tools in parallel) | No | ✓ | Read-only tools (`polarity:"read"`) only. Sagax generation **pauses** until `<aug_result>` injected. |
| `<aug_result>` | — | injected by Orchestrator | No | ✓ | Contains results for all tools in the corresponding `<aug_call>`. |
| `<task_update>` | buffer | write directly to HTM | No | ✓ | No STM event produced. |
| `<projection>` | buffer | dispatch to UI | ✓ | ✓ | |

### 9.2 — Multi-Tool Blocks

Both `<tool_call>` and `<aug_call>` support multiple tools in one block. The Orchestrator dispatches them in parallel.

```xml
<!-- aug_call: block-level default timeout; all tools dispatched in parallel -->
<aug_call timeout_ms="500">
  {"name": "recall",         "args": {"query": "Christmas lighting themes", "top_k": 5}}
  {"name": "resolve_entity", "args": {"clues": "male voice, said his name is John"}}
</aug_call>

<!-- per-tool timeout override: overrides block default for that tool only -->
<aug_call timeout_ms="400">
  {"name": "recall",       "args": {"query": "movie mood presets"},   "timeout_ms": 300}
  {"name": "check_sensor", "args": {"sensor": "ambient_light"},       "timeout_ms": 100}
</aug_call>

<!-- tool_call: multi-tool write dispatch; no timeout — async via Tool Manager -->
<tool_call>
  {"name": "set_ceiling_lights", "args": {"colour": "warm_red", "brightness": 60}}
  {"name": "set_wall_lights",    "args": {"colour": "gold",     "brightness": 40}}
  {"name": "close_blinds",       "args": {}}
</tool_call>
```

**Timeout rules:**
- Per-tool `timeout_ms` overrides the block-level default for that specific tool.
- Block-level `timeout_ms` applies to any tool without its own override.
- On timeout: Orchestrator injects `{"status": "timeout", "tool": "<name>"}` in `<aug_result>`. Sagax handles the missing result in reasoning without aborting.
- `<tool_call>` blocks do not use timeout — they are dispatched to the Tool Manager's async job queue and results surface as STM events.

### 9.3 — `<thinking>` vs `<contemplation>` — Hard Distinction

| | `<thinking>` | `<contemplation>` |
|---|---|---|
| **Subject** | The model thinking about itself | The agent thinking about the world |
| **Scope** | May reference own architecture, constraints, system prompt, token budget | Observations, inferences, interpretations as Sagax in-role |
| **STM** | Never | Always written as `output/contemplation` |
| **Workbook** | No | Yes |
| **Logos** | Invisible | Consolidates into episodic LTM |
| **Purpose** | Debug scratchpad | Epistemic record of Sagax's reasoning state |

### 9.4 — STM Is a Narrative Ledger, Not a Technical Log

STM holds what Sagax perceived and concluded — not the machinery. Full tool payloads, recall result sets, and aug call exchanges live in the workbook. STM carries `output/contemplation` and `output/speech` events that form the auditable narrative of what Sagax understood and communicated.

---

## 10 — Consolidation & Flush Rules

### 10.1 — What Logos Consolidates

Logos reads `get_raw_events(after_id=logos_watermark, limit=N)`. It stops a few events short of `tLast` to avoid racing Exilis.

Logos never reads `consN.summary_text` as consolidation input.

For each batch, Logos produces:
- One episodic LTM entry per meaningful arc.
- Entity observations for named or affected entities.
- Semantic assertions for facts with confidence ≥ 0.85.
- Skill/procedure candidates if trace cluster meets thresholds (§7.3).
- Source attachments for non-text evidence.

`contemplation` events are consolidated alongside raw events — they provide richer episodic narrative context than raw tool calls alone.

### 10.2 — Safe Flush Protocol

```
1. Fetch batch: get_raw_events(after_id=logos_watermark, limit=N)
   Let batch_end_id = id of last event in batch.

2. Write all LTM artifacts derived from this batch.

3. Verify each write succeeded (Muninn returns created entry IDs).

4. Store evidence pointers in every LTM entry:
       {"stm_event_ids": ["t0001", ..., "t0050"]}

5. Only after step 4: stm.flush_up_to(batch_end_id)
   Muninn deletes raw events with id <= batch_end_id.

6. logos_watermark = batch_end_id.

7. Emit "logos_consolidation_complete" event to STM.
```

Partial failure at steps 2–4 must not proceed to step 5. The batch stays until Logos retries.

### 10.3 — Pointer Independence

| Pointer | Owner | Meaning |
|---|---|---|
| `consN.last_event_id` | Sagax | "I've incorporated events up to here into my working narrative" |
| `logos_watermark` | Logos | "I've consolidated events up to here and flushed them" |

These may diverge significantly. Neither direction is an error.

---

## 11 — Failure & Safety Behaviors

### 11.1 — Sagax Failure Modes

| Failure | Behaviour |
|---|---|
| No artifact found after `MAX_RECALL_ITER` | Record to STM; ask user or admit inability |
| Execution failed, no fallback | Record failure event; escalate to user |
| Unsatisfied preconditions, cannot satisfy | Record to STM; explain blocking condition |
| Max steps exceeded | Abort; record partial trace for Logos; ask user to break goal smaller |
| Stuck (`STUCK_THRESHOLD` consecutive no-progress) | Record to STM; escalate |
| Interrupted mid-skill | Write `task_update` state:"paused" with `resume_at`; write `status:"suspended"` on open speech events |

### 11.2 — Safe Actuation Policy

The Orchestrator (not Sagax) enforces:

1. **Permission scope check:** Every `<tool_call>` must declare a `permission_scope` matching the active session grants.
2. **New-skill confirmation:** Any skill with `requires_confirmation_for_n_runs > 0` requires explicit user confirmation before dispatch. Counter decrements on confirmed runs.
3. **Destructive action gate:** Any tool with `"destructive": true` requires confirmation regardless of `requires_confirmation_for_n_runs`.

### 11.3 — Skill Deprecation

Logos flags a skill for deprecation when:
- Failure rate in last 10 executions exceeds 40%.
- A structural change in a dependent tool makes the step sequence invalid.
- A human operator explicitly marks it deprecated.

Deprecated skills are archived, not deleted. Ranked below active skills by `recall()`. Include deprecation notice in `capability_summary`.

### 11.4 — Audit Trail

Every actuation produces a `tool_call` STM event via Exilis, including `request_id`, `artifact_id`, `inputs`, `outputs`, `duration_ms`, and `success`. Logos collects these for skill evidence and safety review. The workbook retains full tool payloads for the session debug record.

---

## 12 — Observability

Every Logos pass emits a `logos_health` event to STM:

```json
{
  "type":          "logos_health",
  "pass_id":       "logos-pass-0042",
  "started_at":    "2026-03-08T06:00:00Z",
  "duration_ms":   4210,
  "consolidated":  { "episodic": 3, "assertions": 7, "entity_updates": 2 },
  "synthesised":   { "skills": 1, "procedures": 0 },
  "skipped":       0,
  "errors":        0,
  "stm_flushed":   45,
  "stm_remaining": 8,
  "asc_flushed":   false
}
```

Sagax can surface Logos health on request. A dashboard or monitoring tool subscribes to `logos_health` events from Muninn.

---

## 13 — Implementation Roadmap

| # | File | Action | Notes |
|---|---|---|---|
| 1 | `design/CognitiveModule.md` | ✓ This document (v1.0) | |
| 2 | `design/Orchestrator.md` | Create | Full routing spec, session grants, permission model, ASC GC, nudge, aug_call handling, workbook retention policy |
| 3 | `specs/tool_descriptor_schema.json` | Create | Formal JSON Schema for §7.1, including polarity + modality |
| 4 | `specs/skill_schema.json` | Create | Formal JSON Schema for §7.2, including guidance steps + interactive flag |
| 5 | `specs/procedure_schema.json` | Create | Formal JSON Schema for §7.5 |
| 6 | `runtime/stm.py` | Create | `append_event`, `get_window`, `update_consN`, `flush_up_to`, `get_raw_events`, `logos_watermark` |
| 7 | `runtime/htm.py` | Create | Dual-surface HTM: ASC (dict-backed, per-session) + Tasks (SQLite-backed, durable). Full API from §5. |
| 8 | `agents/exilis.py` | Create | Triage loop, signature resolution, backchannel detection |
| 9 | `agents/sagax.py` | Create | Planning loop, Narrator token stream, HTM integration, ASC on-demand query, workbook resume |
| 10 | `agents/logos.py` | Create | Consolidation daemon, skill synthesis from task notebooks + raw events, ASC flush |
| 11 | `tests/test_cons_n.py` | Create | Trace-driven: cold-start, update, flush invariants, ASC GC trigger |
| 12 | `tests/test_sagax_skill.py` | Create | skill.set_movie_mood guided execution: ask → recall → tool calls → task complete |
| 13 | `tests/test_logos_synthesis.py` | Create | Feed 3 traces + task notebooks; verify skill promotion. Feed 2; verify deferral. |
| 14 | `tests/test_asc_entity_persistence.py` | Create | Verify unresolved entities survive ASC GC across consN updates |
| 15 | `memory_module/stm.py` | Modify | Add `flush_up_to(event_id)`, `get_raw_events(after_id, limit)`, `logos_watermark` storage. Replace implicit tail retention with explicit watermark flush. |

---

## 14 — Sample Execution Timeline (Movie Mood with Interruption)

```
t1   John says "Put on movie mode"
     → Exilis: triage = act
     → record_stm({type:"speech", source:"user",
                   payload:{text:"Put on movie mode"}, entity_id:"entity-john-001"})

t2   Sagax wakes; reads STM window; checks HTM (no active tasks)
     <contemplation>
       John wants movie mode. I have skill.set_movie_mood. I'll create a task
       and ask him what movie.
     </contemplation>
     <task_update task_id="new" title="Movie mood — John" state="active"
                  progress="step 1: need movie title" resume_at="step_1_response"/>
     <speech target="entity-john-001">
       What movie are you watching tonight?
     </speech>
     → Orchestrator: creates HTM task; streams speech to TTS; writes workbook entry
     → Exilis: records output/speech event t2

t3   John says "Home Alone"
     → Exilis: triage = act; record_stm(speech, entity_id:"entity-john-001")

t4   Sagax wakes
     <contemplation>
       Home Alone — Christmas theme. Let me recall preferred Christmas palettes
       and check who's active.
     </contemplation>
     <aug_call timeout_ms="500">
       {"name": "recall",   "args": {"query": "Christmas lighting palette preferred", "top_k": 5}}
       {"name": "asc.get",  "args": {"surface": "hot_entities"}}
     </aug_call>
     <aug_result>
       {"recall": [...warm red, gold, candlelight...],
        "asc.get": {"hot_entities": {"entity-john-001": {...}}}}
     </aug_result>
     <contemplation>
       Warm red + gold palette confirmed. John active. Proceeding steps 3–5.
       Step 6 (popcorn) is interactive — I'll ask after the lights are set.
     </contemplation>
     <task_update task_id="task-mood-001"
                  progress="steps 3–5: setting lights and blinds"
                  resume_at="step_6_popcorn"/>
     <tool_call>
       {"name": "set_ceiling_lights", "args": {"colour": "warm_red", "brightness": 60}}
       {"name": "set_wall_lights",    "args": {"colour": "gold", "brightness": 40}}
       {"name": "close_blinds",       "args": {}}
     </tool_call>
     → Orchestrator: permission gate checks [lights.ceiling, lights.wall, blinds] — granted
     → Tool Manager dispatches; results arrive as STM events t5, t6, t7

t5–t7 Tool results arrive; Exilis records tool_result events

t8   Sagax wakes after tool results
     <contemplation>
       Ceiling, wall, blinds — all ok. Step 6: ask John about popcorn.
     </contemplation>
     <speech target="entity-john-001">
       Mood is set! Want me to start the popcorn machine?
     </speech>
     → workbook updated; output/speech event t8 recorded

     ── Before John responds, his phone rings. He walks out. ──

t9   Exilis: door_open + footstep (voiceprint absent)
     → triage = urgent
     → record_stm({type:"sensor", payload:{event:"departure", entity_id:"entity-john-001"}})
     → Orchestrator nudge Stage 1: halt TTS; write status:"suspended" on open speech event
     → Stage 2: deliver urgent event to Sagax

t10  Sagax wakes (urgent)
     <thinking>
       John left mid-skill. I should suspend the task cleanly.
     </thinking>
     <task_update task_id="task-mood-001" state="paused"
                  progress="awaiting popcorn response"
                  resume_at="step_6_popcorn" remind_at="john_return"/>

     ── John returns 10 minutes later. ──

t11  Exilis: door_open + voiceprint match → entity-john-001 → triage = act

t12  Sagax wakes; reads HTM — task-mood-001 state:"paused", remind_at:"john_return"
     <contemplation>
       John is back. Task was paused at step 6. Workbook shows steps 3–5 confirmed
       complete. Resuming at the popcorn question — no need to redo lighting.
     </contemplation>
     <speech target="entity-john-001">
       Welcome back! Still want popcorn?
     </speech>

t13  John says "Yes please"
     → Sagax:
     <tool_call>{"name": "start_popcorn_machine", "args": {}}</tool_call>
     <task_update task_id="task-mood-001" state="completed"
                  output='{"result":"mood_set"}' confidence="0.95"/>
     → Exilis records tool_result event t14

     ── Task complete. Logos eligible for synthesis pass. ──

L1   Logos wakes
     → reads HTM: task-mood-001, persistence:"persist"
     → reads task notebook (6 entries: step-by-step + interruption + resume)
     → reads raw events t1–t14 in full fidelity
     → reads cold-archived workbook for full tool payloads + aug_call recall results
     → produces episodic LTM: "John requested movie mood for Home Alone; Christmas
       palette applied; interrupted by departure; resumed on return; popcorn started."
     → entity observations: john.movie_preference, john.snack_preference
     → skill synthesis check: 1 execution — below threshold (need 3)
     → emits logos_synthesis_candidate: {decision:"deferred", reason:"1 of 3 required"}
     → flush_up_to(t14); logos_watermark = t14
     → logos_health event emitted
```

---

## 15 — Open Questions & Deferred Decisions

| # | Question | Status | Deferred to |
|---|---|---|---|
| 1 | **Orchestrator spec** — full routing, session grants, permission model, sandboxing, ASC GC impl, workbook retention policy | **LOAD-BEARING** | `design/Orchestrator.md` |
| 2 | **Multi-agent STM write conflicts** — concurrent Sagax instances | Open | v0.5 |
| 3 | **Async Muninn** — `aiosqlite` for Exilis throughput | Benchmark first | v0.5 |
| 4 | **Vector index backend** — ChromaDB or FAISS at scale | Open | v0.5 |
| 5 | **Logos scheduling** — fixed interval vs. event-triggered | Default: 5-min + size trigger | Configurable |
| 6 | **Skill A/B testing** — run v1 + v2 in parallel before deprecating | Open | v0.6 |
| 7 | **LTM entry deduplication** — merge vs. accumulate on repeated facts | Open | v0.5 |
| 8 | **STMSegment migration** — `is_compression` → `type:"internal"` reconciliation | Open | First Exilis PR |
| 9 | **Output event granularity** — streaming tokens vs. complete response in STM speech event | Open | v0.5 |
| 10 | **Speaker identification** | **CLOSED** — signature entity model (§2.5) | |
| 11 | **`<aug_call>` pause semantics** | **CONFIRMED** — Sagax generation pauses mid-stream waiting for `<aug_result>` (§9.1) | |
| 12 | **HTM naming** — possible expansion of "HTM" acronym to "Hot State Manager" | Deferred | Naming pass |
| 13 | **Logos-initiated skills** — should Logos propose a skill to Sagax mid-session, not just post-flush? | Open | v0.6 |
| 14 | **Workbook retention policy** — how long cold-archived workbooks are kept before permanent deletion | Open | `design/Orchestrator.md` |

---

## 16 — Closing Design Notes

**Memory and tools are one ecosystem.** Tool descriptors, skills, and procedures live in LTM. Recall surfaces the right capability for the moment. Logos retires the ones that stop working.

**Skills are guidance, not scripts.** The value of a skill is not the exact tool call sequence — it is the rationale, the ordering, the interactive beats, and the argument hints. Sagax interprets; the task notebook records; Logos synthesises. This is what makes the system robust to capability changes, argument variation, and user interaction mid-skill.

**consN serves Sagax, not Logos.** Intentional lossiness is a feature — it keeps Sagax's context window bounded while preserving enough narrative for correct triage and planning. Logos never touches it.

**The workbook is the complete session ledger.** STM is the narrative ledger. The workbook is the technical ledger. Together they give Sagax the context to resume without redoing work, and Logos the evidence to synthesise correctly.

**Unresolved entities are first-class citizens.** The bus station stranger — the voice half-recognised, the face that seemed familiar — does not get discarded when the session moves on. It persists in `hot_entities` as an implied entity, accumulates cross-modal evidence, and waits for Logos to confirm or archive it. This is what makes identity robust across partial observations.

**Logos earns trust by evidence, not declaration.** No capability is promoted to a skill until it has proven itself across multiple independent executions. Human confirmation gates on new skills ensure the system can be audited before acting autonomously on learned patterns.

**Failures are first-class events.** Every failure is a structured STM event. Logos learns from failures just as from successes. Operators have a complete audit trail through the combination of STM events, task notebooks, and cold-archived workbooks.
