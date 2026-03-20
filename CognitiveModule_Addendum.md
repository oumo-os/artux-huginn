# Huginn — Architecture Clarification

**Status:** Addendum to `CognitiveModule.md` v0.4 and `Orchestrator.md` v1.0  
**Supersedes:** §1 (Exilis role summary), §4.1 (Exilis responsibilities), and the implicit assumption that Exilis is an ingestion agent.  
**Does not change:** consN model, Logos, Sagax planning loop, memory APIs, HTM, artifact schemas.

---

## Why This Document Exists

The previous spec described Exilis as performing "ingest, normalise, classify, record_stm" in the executive summary. This was imprecise and led to an implementation where Exilis owned raw percepts, drove I/O pipelines, and contained hardcoded classification logic. That is the antithesis of the design.

This addendum corrects the record and closes four open design questions:

1. **Where perception data comes from** — the Perception Manager, a submodule of the Orchestrator.
2. **What Exilis actually does** — reads STM events already written there; uses a small LLM to triage; produces a single signal. Nothing more.
3. **That all cognitive decisions are LLM-driven** — no hardcoded regex, no rule engines, no heuristics. Every classification, every triage decision, every narrative update is an LLM call.
4. **[P-1 CLOSED] Perception pipelines are skills** — `class_type: "pipeline"`, no LLM argument insertion between steps, default args prefilled in the artifact. Modifications tracked in LTM meta (Logos) and ASC/HTM (Sagax).
5. **[P-3 CLOSED] Exilis shares Sagax's model and consN** — Exilis sees the world through Sagax's eyes. consN is owned and updated by Sagax only; Exilis reads it as context. The consN `summarise()` call was always Sagax's responsibility.
6. **[P-4 CLOSED] Perception pipelines are HTM tasks** — active perception pipelines are ongoing HTM tasks (`initiated_by: "system"`, `state: "active"`, `persistence: "persist"`). The Orchestrator queries HTM to know what to run. Failures are logged to the task notebook. Sagax and Logos supervise naturally via the same HTM interface used for all other tasks. No separate pipeline state or supervision mechanism.

---

## 1 — Corrected Data Flow

```
                    ┌─────────────────────────────────────────────────────┐
                    │                   ORCHESTRATOR                       │
                    │                                                       │
                    │  ┌──────────────────────────────────────────────┐   │
                    │  │         Perception Manager (submodule)        │   │
                    │  │                                               │   │
                    │  │  perception pipelines (tool chains):          │   │
                    │  │   microphone → ASR → embed → [sig_resolve]   │   │
                    │  │   camera     → VLM → embed → [sig_resolve]   │   │
                    │  │   sensor     → normalise                      │   │
                    │  │   tool_result → [sig_resolve] → format        │   │
                    │  │                         │                     │   │
                    │  │               writes structured event         │   │
                    │  └─────────────────────────┼─────────────────────┘  │
                    │                            │                         │
                    │              ┌─────────────▼──────────┐             │
                    │              │     STM (Muninn)        │             │
                    │              │  events[]  consN  wmark │             │
                    │              └─────────────┬──────────┘             │
                    │                            │ notify: new event        │
                    │              ┌─────────────▼──────────┐             │
                    │              │         Exilis          │             │
                    │              │  (small LLM, <100ms)   │             │
                    │              │  reads: consN + window  │             │
                    │              │  output: ignore/act/    │             │
                    │              │          urgent         │             │
                    │              └─────────────┬──────────┘             │
                    │                            │ triage signal            │
                    │              ┌─────────────▼──────────┐             │
                    │              │  Routing / Wake logic   │             │
                    │              │  (no cognitive logic)   │             │
                    │              └─────────────┬──────────┘             │
                    └────────────────────────────┼────────────────────────┘
                                                 │
                                       ┌─────────▼──────────┐
                                       │        Sagax        │
                                       │ (medium LLM 1-30s)  │
                                       └─────────────────────┘
```

The Perception Manager and Exilis are **both** inside the Orchestrator's operational scope. Exilis does not own any I/O. It owns one decision: *does Sagax need to wake up?*

---

## 2 — Perception Manager (Orchestrator Submodule)

### 2.1 — Role

The Perception Manager is responsible for:
- Querying HTM for active perception tasks (`initiated_by: "system"`, `state: "active"`) and running their pipelines
- Formatting raw modality outputs into canonical STM events
- Triggering signature resolution for events that carry biometric data
- Writing the finalised, structured event to `stm.events`
- Notifying Exilis that a new event is ready
- On pipeline failure: restarting the pipeline, logging to the task notebook, and writing an `internal` error event to STM

The Perception Manager has **no language model**. It executes tool pipelines and formats their output.

### 2.2 — Perception Pipelines Are Skills

A perception pipeline is an LTM artifact with `class_type: "pipeline"`. It uses the same schema as a skill (§7.2 in CognitiveModule.md) with two key differences:

1. **No LLM argument insertion between steps.** The pipeline runs tool-to-tool with no reasoning layer in between. Each tool's output is passed directly as input to the next tool in the chain.
2. **Default arguments are prefilled in the artifact itself.** There are no interactive beats, no `tool_hint: "ask_user"`. The pipeline runs fully automatically.

```json
{
  "artifact_type":  "pipeline",
  "skill_id":       "pipeline.voice_to_stm.v1",
  "title":          "Voice → STM",
  "description":    "Captures audio from microphone, transcribes, resolves speaker identity, writes to STM.",
  "class_type":     "pipeline",
  "steps": [
    { "order": 1, "artifact_id": "tool.audio_capture.v1",
      "args": { "sample_rate": 16000, "vad_threshold": 0.5 } },
    { "order": 2, "artifact_id": "tool.asr_transcribe.v1",
      "args": { "model": "whisper-base", "language": "auto" } },
    { "order": 3, "artifact_id": "tool.stm_write_speech.v1",
      "args": {} }
  ],
  "permission_scope": ["microphone"],
  "confidence":    1.0,
  "version":       1
}
```

**Modifications** to a running pipeline's parameters (e.g., changing ASR model, adjusting VAD threshold) are made in two ways:
- **Durable changes** (survive session restart): Logos writes to the LTM artifact's `meta` field.
- **Session-scoped changes**: Sagax writes to the pipeline's HTM task notebook or the ASC hot state. These take effect immediately but are not persisted beyond the session.

This means the same interface Logos and Sagax use for all other tasks applies to perception pipelines — no special API.

### 2.3 — Pipelines Run as HTM Tasks

Every active perception pipeline corresponds to a durable HTM task:

```json
{
  "task_id":      "task-pipeline-voice-001",
  "title":        "pipeline.voice_to_stm.v1",
  "initiated_by": "system",
  "state":        "active",
  "persistence":  "persist",
  "resume_at":    "",
  "notebook": [
    {"ts": "...", "entry": "[start] Pipeline activated by Sagax on John's arrival"},
    {"ts": "...", "entry": "[error] ASR tool timed out — restarted (attempt 2)"},
    {"ts": "...", "entry": "[ok] Running normally"}
  ],
  "tags": ["perception", "voice", "pipeline"]
}
```

The Orchestrator's Perception Manager loop is simply:

```python
def perception_tick():
    active_pipelines = htm.query(initiated_by="system", state="active",
                                 tags_include=["pipeline"])
    for task in active_pipelines:
        pipeline = recall(task.title)           # fetch artifact from LTM
        try:
            run_pipeline(pipeline, task)        # execute tool chain, write STM event
        except Exception as e:
            htm.note(task.task_id, f"[error] {e} — restarting")
            stm.record("system", "internal",
                       {"subtype": "pipeline_error", "pipeline": task.title, "error": str(e)})
            restart_pipeline(task)
```

**Starting a pipeline:** Sagax emits a `<tool_call>` to `tool.start_pipeline` with `pipeline_id` as argument. The Orchestrator creates the HTM task and begins execution.

**Stopping a pipeline:** Sagax emits a `<tool_call>` to `tool.stop_pipeline`. The Orchestrator sets the HTM task state to `"cancelled"`.

**Supervising pipelines:** Sagax and Logos read the HTM task notebook naturally — the same interface used for all other tasks. No separate log or monitoring surface.

### 2.4 — Canonical STM Event Format

Every pipeline terminates in a structured event written to `stm.events`. All events share this envelope:

```json
{
  "id":         "t2026-03-10T07:14:32Z_000041",
  "ts":         "2026-03-10T07:14:32Z",
  "source":     "user | system | tool | sensor | log",
  "type":       "speech | tool_call | tool_result | task_update | sensor | output | internal",
  "payload":    { },
  "confidence": 0.91
}
```

**Speech event payload:**
```json
{
  "text":               "Put on movie mode.",
  "entity_id":          "entity-john-001",
  "signature_resolved": true,
  "voiceprint_match":   0.94,
  "language":           "en",
  "duration_ms":        1240
}
```

**Observation event payload (VLM):**
```json
{
  "description":       "John and an unrecognised person in the living room.",
  "entities_detected": ["entity-john-001"],
  "implied_entities":  ["implied-a3f9"],
  "faceprints":        [{"entity_id": "entity-john-001", "score": 0.96},
                        {"entity_id": "implied-a3f9",    "score": null}]
}
```

**Sensor event payload:**
```json
{
  "event":      "arrival",
  "location":   "front_door",
  "modalities": ["motion", "door_contact"]
}
```

### 2.5 — Signature Resolution (Perception Manager Responsibility)

When a perception tool emits a biometric embedding (voiceprint, faceprint), the Perception Manager resolves it against the entity registry before writing the event to STM. This is not Exilis's responsibility.

```
Resolution flow:
  embedding received from ASR / VLM tool
  → query Muninn: resolve_entity(embedding, kind="voiceprint", threshold=0.88)
  → MATCH:    enrich event with entity_id, sig_match_confidence
               open/update session grants for this entity
  → NO MATCH: generate implied entity ID, store embedding in ASC.hot_entities
               set payload.entity_id = implied-{uuid}, payload.signature_resolved = false
  → write enriched event to STM
```

The Orchestrator session object is updated when an entity is resolved. An **implied entity** is held in `ASC.hot_entities` with `status: "unresolved"` and `grants: guest` until Logos performs a post-session resolution pass.

### 2.6 — Tool Adapter Contract

Every perception tool must produce output conforming to the Artux tool descriptor schema. Where a modality produces embeddings, the tool's output schema must declare:

```json
{
  "outputs": {
    "signature": {
      "kind":       "voiceprint | faceprint | device_id",
      "embedding":  ["float"],
      "confidence": "float"
    }
  }
}
```

The Perception Manager inspects tool output schemas at pipeline registration time and routes signature-bearing outputs through the resolution flow automatically.

---

## 3 — Exilis (Corrected Specification)

### 3.1 — What Exilis Is

Exilis is a **passive triage agent**. It does not ingest data. It does not write events. It does not resolve signatures. It does not own any I/O.

Exilis wakes when the Orchestrator notifies it that a new event has been written to STM. It reads the current working context (consN + new-event window), passes that to a small, fast LLM, and emits exactly one signal: `ignore`, `act`, or `urgent`. Then it sleeps.

**The entire Exilis implementation is one LLM call inside a notification handler.**

### 3.2 — Exilis Contract

```
TRIGGER:   Orchestrator notifies: "new STM event written"

INPUT:
  - consN.summary_text       (older context, lossy)
  - new_events[]             (full-fidelity events since consN.last_event_id)
  - The event that just triggered the notification (always last in new_events)
  - Active HTM task summaries (what Sagax is currently working on, if anything)

LLM CALL:
  Model:     Small/fast (≤ 1B params, target < 80 ms)
  Prompt:    See §3.3
  Output:    Structured JSON: { "triage": "ignore | act | urgent", "reason": "..." }

OUTPUT:
  → ignore:  No signal sent. Exilis sleeps.
  → act:     Orchestrator queues a wake signal for Sagax.
             Sagax reads context on its next cycle.
  → urgent:  Orchestrator issues a two-stage nudge (see Orchestrator.md §4).
             Sagax's current output is interrupted immediately.
```

**Hard prohibitions (unchanged from v0.4):**
- Never actuates
- Never writes to STM
- Never writes to LTM
- Never calls `recall()`
- Never modifies event content
- Never calls the Tool Manager

### 3.3 — Exilis Prompt

```
system: |
  You are Exilis, the attention gate for an ambient AI assistant.
  
  Your only job is to decide whether the AI needs to respond to what just happened.
  
  You are given:
    - CONTEXT: a lossy summary of recent history
    - NEW EVENTS: what has just occurred (full detail)
    - ACTIVE TASKS: what the AI is currently working on
  
  You must output exactly one JSON object:
    {"triage": "ignore" | "act" | "urgent", "reason": "<one sentence>"}
  
  Definitions:
    ignore  → This event requires no response. The AI can continue what it is doing
              (or stay asleep). Examples: listener backchannels ("mm-hmm", "yeah"),
              ambient noise, duplicate sensor readings, the AI's own speech being
              detected by the microphone, system heartbeats.
  
    act     → This event warrants Sagax's attention on its next natural cycle.
              The AI is not mid-speech, or the event is not time-critical. 
              Examples: a new user request, a completed tool result, a sensor
              event that changes context.
  
    urgent  → This event requires immediate interruption of any current AI output.
              Use this sparingly. Examples: the user speaks with a new substantive
              request while the AI is mid-sentence, a departure/emergency sensor
              fires, the user says "stop" or "wait".
  
  When in doubt between ignore and act: choose act.
  When in doubt between act and urgent: choose act.
  Only choose urgent when there is a clear reason to interrupt immediately.

user: |
  CONTEXT (older history, lossy):
  {{ consN.summary_text or "No prior context (cold start)." }}
  
  ACTIVE TASKS:
  {{ active_task_summaries or "None." }}
  
  NEW EVENTS (chronological, full detail):
  {{ new_events_formatted }}
  
  The last event is the one that just triggered this check.
  Classify it.
```

### 3.4 — Why Not Rules?

Hardcoded triage rules fail in every ambient context:

- "mm-hmm" is a backchannel in English, an affirmative in some dialects, and a partial word in a longer utterance. A rule cannot know which.
- A sensor reading of `motion: detected` might mean John got up for water (ignore) or a stranger just entered the room (urgent), depending on whether John is home and whether the session is active.
- The AI's own speech arriving back through the microphone looks identical to a user utterance in raw audio. A rule checking "is Sagax currently speaking?" introduces a race condition with the STM write.

Exilis uses an LLM because triage is a *contextual judgement*, not a *pattern match*. The small model sees the same context Sagax would see and makes the same kind of inference — just faster and without reasoning about what to do next.

### 3.5 — LLM Model: Shared with Sagax (P-3 CLOSED)

Exilis uses **the same model as Sagax**, and reads **the same consN** that Sagax maintains.

This is the correct design for three reasons:

1. **Exilis sees the world through Sagax's eyes.** The triage judgment — "does Sagax need to wake up?" — requires exactly the same contextual understanding Sagax has. Using a different model with a different world-model would introduce divergence: Exilis might decide "ignore" based on a context Sagax would read as "act".

2. **consN is Sagax's narrative.** It was always Sagax's responsibility to update it via `summarise()`. Exilis reads it as a read-only input. The model that produces consN and the model that reads it should be the same — same tokeniser, same representation, same latency profile.

3. **Operational simplicity.** One model to serve, one temperature setting (`0`), one context format. The only runtime difference is that Exilis uses a tightly constrained structured-output prompt while Sagax runs its full Narrator stream.

| Property | Exilis call | Sagax call |
|---|---|---|
| Model | Same (shared) | Same |
| consN | Reads (never writes) | Reads + triggers updates |
| Temperature | 0 | 0.1–0.4 |
| Max tokens | ~50 (JSON label + reason) | Full Narrator stream |
| Context | consN + new-event window | consN + new-event window + HTM tasks |

**Recommended local model:** Qwen2.5-3B or Llama-3.2-3B via Ollama — small enough for < 80 ms Exilis calls, capable enough for Sagax's planning and Narrator output.

**Recommended API:** Claude Haiku for Exilis calls, Claude Sonnet for Sagax — but they share the same consN format, so context is always interpretable across both.

---

## 4 — LLM Assignment Across All Agents

Every cognitive decision in the system is an LLM call. There is no rule engine, no hardcoded classifier, no heuristic.

| Agent | Role | Model | Temperature | Notes |
|---|---|---|---|---|
| **Perception Manager** | No LLM | — | — | Pure tool pipeline execution |
| **Exilis** | Triage: ignore / act / urgent | **Shared with Sagax** | 0 | Structured JSON; reads Sagax's consN |
| **Sagax** | Planning, reasoning, Narrator output | 3–13B params | 0.1–0.4 | Full Narrator token stream |
| **consN `summarise()`** | Rolling narrative update | **Shared with Sagax** | 0 | Triggered by Sagax; same model, constrained output |
| **Logos** | Consolidation + skill synthesis | 9–30B params | 0 | Separate larger model; runs async |

**Key relationships:**
- Exilis + consN summariser share Sagax's model. Three prompt shapes, one serving model.
- Logos is deliberately separate — larger, async, its outputs must be high-fidelity enough for permanent LTM.

---

## 5 — Revised Executive Summary (replaces §1 in CognitiveModule.md)

| Layer | Speed | Responsibility |
|---|---|---|
| **Perception Manager** | Continuous | Runs active pipeline HTM tasks. Writes canonical events to STM. Resolves signatures. Restarts failed pipelines and logs to HTM notebook. No LLM. |
| **Exilis** | < 100 ms | Woken on each new STM event. Reads consN (Sagax's shared narrative) + new-event window. One LLM triage call (shared model, temperature 0). Emits `ignore / act / urgent`. Never actuates, never writes. |
| **Sagax** | 1–30 s | Woken by Orchestrator on `act` or `urgent`. Reads STM context + HTM tasks (including pipeline task notebooks). Reasons with medium LLM. Produces structured Narrator token stream. Starts/stops perception pipelines via tool calls. Updates consN when window grows large. |
| **Logos** | Minutes → days | Reads raw STM events. Large LLM consolidation. Writes durable LTM including pipeline artifact LTM meta modifications. Synthesises skills from execution traces. Flushes STM after verified writes. Performs post-session entity resolution. |

**Four non-negotiable design constraints:**

1. **All cognitive decisions are LLM calls.** No hardcoded classifiers, rule engines, or heuristics in any agent.
2. **Tools and pipelines are memory artifacts.** Tool descriptors and perception pipeline descriptors (`class_type: "pipeline"`) live in Muninn as `recall()`-able artifacts. Sagax discovers and manages them without a registry.
3. **consN is always singular, rolling, and lossy.** Owned and updated by Sagax. Read by Exilis as shared context. Never used by Logos.
4. **Logos is the sole author of durable LTM.** Sagax reasons and acts in the hot window; Logos decides what earns permanent storage, including durable pipeline parameter modifications.

---

## 6 — What This Means for Implementation

### What was wrong in the previous code

| File | Problem | Fix |
|---|---|---|
| `agents/exilis.py` | Exilis owned `ingest()`, ran I/O, called `self.stm.record()`, contained regex patterns | Delete. Exilis is a notification handler + one LLM call. ~80 lines total. |
| `agents/exilis.py` | `_resolve_signature()` was in Exilis | Move to Perception Manager |
| `runtime/stm.py` | `STMStore.record()` was called from Exilis | `record()` is called from the Perception Manager only |
| Nothing existed | Perception pipeline architecture was missing entirely | New: `runtime/perception.py` |
| Nothing existed | No LLM client abstraction | New: `llm/client.py` |

### Correct file layout

```
huginn/
├── agents/
│   ├── exilis.py       # Notification handler + one LLM triage call. ~80 lines.
│   ├── sagax.py        # Planning loop + Narrator token stream + consN updates.
│   └── logos.py        # Consolidation daemon + skill synthesis + pipeline LTM meta writes.
│
├── runtime/
│   ├── stm.py          # consN, logos_watermark, flush, raw event access.
│   ├── htm.py          # Hot Task Manager: ASC + Tasks (including pipeline tasks).
│   ├── perception.py   # Perception Manager: HTM-driven pipeline runner + sig resolution + STM write.
│   └── orchestrator.py # Routing bridge: token stream, permission gate, nudge, aug_call.
│
├── llm/
│   ├── client.py       # Unified LLM client (Ollama / Anthropic API). Shared by all agents.
│   └── prompts.py      # All system prompts as versioned, testable constants.
```

### The correct Exilis implementation shape

```python
class Exilis:
    """
    Woken by Orchestrator on each new STM event.
    Reads context. Makes one LLM call (shared model). Returns triage label. Sleeps.
    """

    def __init__(self, stm: STMStore, htm: HTM, llm: LLMClient):
        self.stm = stm
        self.htm = htm
        self.llm = llm   # same model instance as Sagax

    def on_new_event(self) -> TriageSignal:
        context      = self.stm.get_stm_window()         # consN + new_events
        active_tasks = self.htm.query(state="active|paused")

        result = self.llm.complete(
            system      = EXILIS_TRIAGE_PROMPT,           # see §3.3
            user        = format_context(context, active_tasks),
            schema      = TriageOutputSchema,             # {"triage": enum, "reason": str}
            temperature = 0,
            max_tokens  = 60,
        )

        return TriageSignal(label=result.triage, reason=result.reason)
```

### The correct Perception Manager implementation shape

```python
class PerceptionManager:
    """
    Runs active perception pipeline tasks from HTM.
    Writes canonical events to STM.
    Resolves signatures. Restarts failed pipelines.
    No LLM.
    """

    def __init__(self, stm: STMStore, htm: HTM, muninn, on_event_written: Callable):
        self.stm              = stm
        self.htm              = htm
        self.muninn           = muninn
        self.on_event_written = on_event_written   # → Exilis.on_new_event()

    def tick(self):
        """Called at Orchestrator's perception tick rate."""
        pipelines = self.htm.query(initiated_by="system", state="active")
        for task in pipelines:
            pipeline_artifact = self.muninn.recall(task.title, top_k=1)
            if not pipeline_artifact:
                continue
            try:
                payload = self._run_pipeline(pipeline_artifact, task)
                self.stm.record(
                    source=pipeline_artifact.source_type,
                    type=pipeline_artifact.event_type,
                    payload=payload,
                    confidence=payload.get("confidence", 1.0),
                )
                self.on_event_written()   # wake Exilis
            except Exception as e:
                self.htm.note(task.task_id, f"[error] {e} — restarting")
                self.stm.record("system", "internal", {
                    "subtype": "pipeline_error",
                    "pipeline": task.title,
                    "error": str(e),
                })
                self._restart(task)

    def _run_pipeline(self, pipeline, task) -> dict:
        result = {}
        for step in sorted(pipeline.steps, key=lambda s: s["order"]):
            tool   = self._load_tool(step["artifact_id"])
            args   = {**step.get("args", {}), **result}   # chain output → next input
            result = tool.run(args)
        if "signature" in result:
            result = self._resolve_signature(result, task)
        return result

    def _resolve_signature(self, payload: dict, task) -> dict:
        # entity lookup → session update → implied entity registration in ASC
        # This is the ONLY place signature resolution happens.
        ...
```

---

## 7 — P-1 CLOSED: Perception Pipelines Are Skills

### 7.1 — Schema

A perception pipeline reuses the existing skill schema with `class_type: "pipeline"`. The distinction is behavioural, not structural:

| Property | Skill | Pipeline |
|---|---|---|
| `class_type` | `"skill"` | `"pipeline"` |
| LLM argument insertion between steps | Yes — Sagax fills in args contextually | **No** — args fully prefilled in the artifact |
| Step execution | Sagax interprets, fills args, handles interactive beats | Orchestrator executes mechanically, no reasoning |
| Initiated by | Sagax (on demand) | Orchestrator (continuously while HTM task is active) |
| Synthesised by Logos | Yes | No — operator-authored; Logos may update `meta` only |

Because args are prefilled, a pipeline step looks like:

```json
{
  "order":       1,
  "artifact_id": "tool.audio_capture.v1",
  "args":        { "sample_rate": 16000, "channels": 1 }
}
```

No argument is left unset. If a tool needs a value that varies at runtime, it must come from the previous tool's output — not from LLM inference mid-pipeline.

### 7.2 — Parameter Tweaks and State

Pipeline parameters can change without a version bump via two surfaces:

**LTM meta (durable, Logos-mediated):** The pipeline artifact's `meta` field holds operator-configured defaults and Logos-observed adjustments. Logos notices the ASR tool underperforms in the kitchen → writes `noise_profile: "kitchen"` to the pipeline's LTM meta. Persists across sessions.

```json
{
  "artifact_id": "pipeline.voice_to_stm.v1",
  "class_type":  "pipeline",
  "meta": {
    "active_by_default":     true,
    "sample_rate_override":  16000,
    "noise_profile":         "kitchen",
    "logos_observed_at":     "2026-03-10T06:00:00Z"
  }
}
```

**HTM task notebook (session-scoped):** Sagax adjusts a pipeline mid-session (e.g., lowers mic sensitivity while John is on a call) → noted in the pipeline's HTM task notebook. Readable by Sagax via `htm.query()`. Logos examines it post-session and decides whether to promote the adjustment to LTM meta.

---

## 8 — P-3 CLOSED: Exilis Shares consN With Sagax

### 8.1 — Shared Context, Single Owner

Exilis sees the world through Sagax's eyes. There is exactly one `consN`; Sagax is its sole author. Exilis reads it as triage context — never updates it, never writes it, never influences what gets folded.

This is correct because Exilis's job is to answer: *"Given what Sagax knows, does Sagax need to wake up right now?"* Using the same consN means Exilis works with identical priors. A separate summary would mean Exilis could triage differently from how Sagax would reason — an incoherence.

The `summarise()` call that updates consN was always Sagax's responsibility. Nothing changes.

### 8.2 — Exilis Runs in a Tight Loop

Exilis processes all available new events as a **batch** — one LLM call per poll cycle, not one call per event. If five events arrive in 20 ms, Exilis sees all five in one `get_stm_window()` read and makes one triage decision.

```python
while orchestrator.running:
    new_events = stm.get_events_after(last_processed_id)
    if not new_events:
        sleep(0.005)   # 5 ms poll; replace with push notify when aiosqlite available
        continue

    context      = stm.get_stm_window()   # consN + new-event window
    active_tasks = htm.query(state="active|paused", initiated_by="sagax")

    result = llm.complete(
        system      = EXILIS_TRIAGE_PROMPT,
        user        = format_context(context, active_tasks, new_events),
        schema      = TriageOutputSchema,   # {"triage": enum, "reason": str}
        temperature = 0,
    )

    last_processed_id = new_events[-1].id

    if result.triage == "urgent":
        orchestrator.nudge(new_events[-1])
    elif result.triage == "act":
        orchestrator.queue_sagax_wake()
    # ignore → continue
```

### 8.3 — Latency Budget

| Step | Target |
|---|---|
| `stm.get_stm_window()` | < 5 ms |
| `htm.query(state="active\|paused")` | < 2 ms |
| LLM call (≤ 1B local model) | < 60 ms |
| Signal dispatch | < 2 ms |
| **Total** | **< 70 ms** |

Poll interval of 5 ms means Exilis adds at most 5 ms detection latency on top of the LLM call. Well within the 100 ms budget.

---

## 9 — P-4 CLOSED: Perception Pipelines Are HTM Tasks

### 9.1 — Pipelines As Ongoing Tasks

Every active perception pipeline is an HTM task with `initiated_by: "system"` and `persistence: "persist"`. This is the entire supervision model. No separate pipeline registry. No Orchestrator-internal pipeline state machine. No split between LTM and ASC for pipeline status.

```json
{
  "task_id":      "task-pipeline-voice-to-stm",
  "title":        "Perception: voice → STM",
  "initiated_by": "system",
  "state":        "active",
  "progress":     "Running. Last event: t2026-03-10T07:14:32Z",
  "persistence":  "persist",
  "tags":         ["pipeline", "perception", "audio"],
  "notebook": [
    {"ts": "...", "entry": "Started. Sample rate 16000 Hz."},
    {"ts": "...", "entry": "[failure] ASR timeout 3200ms. Retried. OK."},
    {"ts": "...", "entry": "[logos] Noise profile updated to 'kitchen' in LTM meta."}
  ]
}
```

### 9.2 — Orchestrator Supervision Loop

```python
while running:
    htm.scheduler_tick()   # waiting→due, active→expired

    for task in htm.query(initiated_by="system", state="active"):
        pipeline = recall_pipeline(task.task_id)
        try:
            run_pipeline_step(pipeline, task)
        except TransientError as e:
            htm.note(task.task_id, f"[failure] {e}. Retrying.")
            schedule_retry(task)
        except HardFailure as e:
            htm.update(task.task_id, state="paused",
                       note=f"[hard failure] {e}. Awaiting Sagax/operator.")
            stm.record(source="system", type="internal",
                       payload={"subtype": "pipeline_failure",
                                "task_id": task.task_id, "error": str(e)})

    handle_exilis_signals()
    sleep(0.001)
```

### 9.3 — Failure Handling

| Failure type | Response | HTM state |
|---|---|---|
| Transient (timeout, brief I/O error) | Retry with backoff; note in notebook | `active` |
| Repeated transient (3 retries) | Pause; write `internal` error event to STM | `paused` |
| Hard failure (hardware gone, tool missing) | Pause; write error event; Sagax/Logos investigate | `paused` |
| Sagax/operator decision | Explicit `htm.update(state="cancelled")` | `cancelled` |

Sagax discovers a paused pipeline via `htm.query(state="paused", tags=["pipeline"])` and reads the notebook to understand what failed. Logos examines `persist` pipeline tasks post-session and can promote notebook-observed parameter adjustments to LTM meta — closing the adaptation loop.

### 9.4 — Starting Pipelines

At Orchestrator startup, pipelines with `meta.active_by_default: true` are started:

```python
default_pipelines = muninn.recall("active_by_default pipeline", top_k=20)
for pipeline in default_pipelines:
    htm.create(
        title        = f"Perception: {pipeline.title}",
        initiated_by = "system",
        persistence  = "persist",
        tags         = ["pipeline", "perception"] + pipeline.tags,
    )
```

Sagax activates additional pipelines at runtime by creating HTM tasks (e.g., camera pipeline activated on confirmed entity with camera grants).

---

## 10 — Open Questions Remaining

| # | Question | Status |
|---|---|---|
| P-2 | Sagax pipeline activation — direct `htm.create()` or via `<tool_call>` to `tool.activate_pipeline`? | Open — leaning direct HTM write |
| Orch-4 | Orchestrator sandboxing | Deferred |
| Orch-5 | Session grant updates mid-session when new entity confirmed | Open |
| Orch-2 | Multi-tool `<tool_call>` partial failure | Open — intent: continue; each tool gets own result |
