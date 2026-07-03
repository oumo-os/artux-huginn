# Huginn — Cognitive Module Specification

**Repository:** `oumo-os/artux-huginn`  
**Document status:** Living design spec — v2.0  
**Supersedes:** CognitiveModule.md v1.0, CognitiveModule_Addendum.md

---

## Change Summary v1 → v2

| Area | v1 | v2 |
|---|---|---|
| Exilis loop | 5ms fixed poll interval | Continuous loop gated on `events_pending()` EXISTS query |
| LLM clients | Hard-coded backends in `client.py` | Provider tools — any GGUF/API backend installable via staging |
| Model assignment | Same model for all roles | Per-role models: exilis→tiny, sagax→medium, logos→large |
| HTM surfaces | Tasks + ASC | Tasks + ASC + **States** (live operational parameters) |
| System prompt | 256-line manual | 84-line operating contract + 8 LTM instruction artifacts |
| Skill synthesis | Hardcoded threshold counters | Logos observes friction patterns → HTM evaluation tasks → proposals |
| Actuation | `on_tts_token` callback only | **ActuationBus** — in-process pub/sub, partial/chunk/full events |
| Interrupted speech | `status: "suspended"` | `status: "full", interrupted: true` — everything in STM is full |
| Default stack | None — required external backends | `tools/builtin/` — llamacpp, Kokoro TTS, Moonshine ASR, text UI |
| ASC GC trigger | Never fired | Fires on every `consN` update via `on_consn_updated()` |
| Perception model | Fixed step-based pipelines per sensor | Capture/process split — PM fans out CaptureFrames to N processors |
| Model sizing | One recommendation for all hardware | Four computation profiles (micro/standard/workstation/server), see §4.3 |

---

## 0 — Glossary

| Term | Definition |
|---|---|
| **STMEvent** | An atomic, timestamped, structured record appended to STM. `source`, `type`, `payload`, `confidence`. All events in STM are `full` — no partial events stored. |
| **consN** | The single rolling, deliberately lossy narrative summary. Owned by Sagax; read by Exilis for triage coherence. Updated by Sagax via `update_cons_n()`; triggers ASC GC on each update. |
| **events_pending** | `STMStore.events_pending(since_id)` — single `SELECT EXISTS` query. Exilis's attention gate: no inference unless True. |
| **HTM.States** | Flat key-value store for live operational parameters. Namespace: `{tool_id}.{param}` or `{role}.{param}`. Persisted to LTM by Logos at session end. |
| **Provider tool** | An LLM inference backend installed as a normal tool. Exposes `complete`, `stream`, `complete_json`, `complete_tools`. `LLMClient` routes to the role's active provider via `HTM.states`. |
| **ActuationBus** | In-process pub/sub for output events. Non-blocking (full queues drop silently). Subscribers register filter dicts; the bus pushes matching events. |
| **Builtin tools** | Tools in `tools/builtin/` — ship with Huginn, auto-installed by Logos on first pass without staging confirmation. |
| **get_instructions** | Native `polarity:read` tool. `get_instructions(topic)` → RecallQuery → returns LTM instruction artifact content verbatim. Eligible for `<aug_call>`. |
| **Interrupted event** | When a nudge fires mid-speech, the open block is closed and written to STM as `status: "full", interrupted: true`. No `suspended` status exists in v2. |
| **Artifact** | Any LTM-resident object retrievable by `recall()`: tool descriptors, skills, procedures, entity ledgers, instruction manuals, config entries, concept clusters. |

---

## 1 — Executive Summary

| Component | Speed | Responsibility |
|---|---|---|
| **Perception Manager** | Continuous | Runs active pipeline HTM tasks. Writes canonical events to STM. Resolves signatures via entity relationship recall. No LLM. |
| **Exilis** | < 1ms gate + < 80ms inference | Tight continuous loop. `events_pending()` gate — no inference when quiet. When events present: one batched LLM triage call (shared model with Sagax, temperature 0). Emits `ignore / act / urgent`. Never writes. |
| **Sagax** | 1–30s | Planning loop, Narrator token stream, HTM task management, consN updates, `speech_step` suspension, provider via `HTM.states`. |
| **Orchestrator** | < 50ms | Tag-state machine router, permission gate, speech chunker, ActuationBus publisher, two-stage nudge, HTM scheduler, `on_consn_updated` → ASC GC. |
| **Logos** | Background | STM→LTM consolidation, observation-driven skill evaluation, tool install (staging + builtin), instruction/config/default-tool bootstrapping, state persistence. |

**Five non-negotiable design constraints:**

1. **All cognitive decisions are LLM calls.** No classifiers, no rule engines, no heuristics.
2. **Providers are tools.** LLM inference is not special infrastructure — it is a swappable, installable tool.
3. **Tools and instructions are memory artifacts.** Sagax discovers capabilities and operational guidance the same way it discovers knowledge — via `recall()`.
4. **consN is always singular, rolling, and lossy.** Raw STM events are never deleted by consN updates.
5. **Logos is the sole author of durable LTM.** Sagax reasons in the hot window; Logos decides what earns permanent storage.

---

## 2 — Data Model

### 2.1 — STMEvent

```python
{
  "id":         "t2026-03-08T12:34:56Z-0001",
  "ts":         "2026-03-08T12:34:56Z",
  "source":     "user | system | tool | sensor | log",
  "type":       "speech | tool_call | tool_result | task_update | sensor | output | internal",
  "payload":    {},
  "confidence": 0.92
}
```

**All STM events are `full`.** Partial and chunk events exist only on the ActuationBus. When a nudge interrupts a speech block, the Orchestrator closes it and writes `status: "full", interrupted: true, partial_text: "..."` to STM. The bus also receives a terminal `full` event with `interrupted: true`.

Output event subtypes written by Orchestrator:
```python
"output"  payload.subtype: "contemplation" | "speech" | "speech_step" | "projection"
"internal" payload.subtype: "consn_updated" | "logos_health" | "tool_installed" | ...
```

### 2.2 — STM State Object

```json
{
  "events":          [],
  "consN": {
    "summary_text":  "Narrative covering t0001–t0050.",
    "last_event_id": "t0050",
    "version":       7
  },
  "logos_watermark": "t0045"
}
```

`consN.last_event_id` and `logos_watermark` are independent pointers. Neither direction of divergence is an error.

### 2.3 — HTM Surfaces

```
HTM
├── Tasks          Durable. Lifecycle records with notebooks. persist | volatile | audit_only.
├── ASC            Ephemeral per-session.
│   ├── workbook       Complete session stream mirror.
│   ├── hot_entities   Confirmed + unresolved/implied. Never auto-pruned.
│   ├── hot_capabilities  Tool/skill snapshots. Pruned by ASC GC on consN update.
│   ├── hot_topics     Active topic threads. Pruned by ASC GC.
│   ├── hot_recalls    Recent recall result sets. Pruned by ASC GC.
│   ├── hot_parameters speech_step bindings. NEVER auto-pruned.
│   └── hot_state      Tool runtime parameters. Pruned by ASC GC.
└── States         Live key-value operational parameters.
                   Namespace: {tool_id}.{param} or {role}.{param}
                   Persisted to LTM by Logos at session end (dirty tracking).
                   Loaded from LTM by Orchestrator at boot.
```

### 2.4 — LTM Partition Map

| `class_type` | Owner | Notes |
|---|---|---|
| `"observation"` | Logos | Consolidated perception arcs |
| `"assertion"` | Logos | High-confidence durable facts |
| `"skill"` | Logos | Synthesised guidance sequences. `status: "proposed"` until confirmed. |
| `"tool"` | Logos / operator | Callable capability descriptors |
| `"pipeline"` | Logos / operator | Perception/actuation pipeline descriptors |
| `"provider"` | Logos / operator | LLM inference backend descriptors |
| `"instruction"` | Logos | Operational manuals for Sagax (immutable except on arch revision) |
| `"config"` | Logos | System/role LLM configuration |
| `"concept_cluster"` | Logos | Topic groupings |

---

## 3 — Exilis — Continuous Loop Design

```python
# Conceptual implementation
while running:
    if not stm.events_pending(last_processed_id):
        os.sched_yield()   # cooperative yield — near-zero latency
        continue           # no inference — quiet environment

    new_events = stm.get_events_after(last_processed_id)
    last_processed_id = new_events[-1].id

    context      = stm.get_stm_window()        # consN + new-event window
    active_tasks = htm.query(state="active|paused", initiated_by="sagax")

    signal = llm.complete_json(
        system      = EXILIS_TRIAGE_PROMPT,
        user        = format_context(context, active_tasks, new_events),
        schema      = {"triage": "string", "reason": "string"},
        temperature = 0,
    )

    if signal.triage == "urgent":  orchestrator.nudge(new_events[-1])
    elif signal.triage == "act":   orchestrator.queue_sagax_wake()
    # ignore → continue (the most common path)
```

**`events_pending(since_id)` is a single `SELECT EXISTS` query.** No rows loaded. Returns in < 1ms. Detection latency is bounded by this query, not a sleep interval.

**`idle_yield_s`** replaces `poll_interval_s`. Default 0.0 (cooperative multitasking via `os.sched_yield()`). Set > 0 on systems with thermal constraints.

---

## 4 — Provider Model

### 4.1 — LLMClient Router

```
LLMClient(role="sagax", htm=htm)
    ↓
_resolve() → reads from HTM.states:
    {role}.provider    → "tool.llm.llamacpp.v1"
    {role}.model       → "llama-3.2-3b-instruct-q4_k_m.gguf"  (for display)
    {role}.temperature → 0.2
    {role}.timeout     → 60.0
    ↓
_ProviderToolAdapter(handlers, htm, role="sagax")
    ↓
fn(system, messages, model, temperature, _htm=htm, _role="sagax")
```

Provider tools read their per-role model path from:
```
tool.llm.llamacpp.v1.model_path.{role}   (set by _assign_gguf_models)
tool.llm.llamacpp.v1.model_path          (shared fallback)
```

### 4.2 — Per-Role Model Assignment

`_assign_gguf_models(models_dir, htm)` is called by `build_huginn()` before any agent starts. Assignment priority:

1. `HTM.states` already set (LTM recall wins — operator config persisted)
2. `models/models.yaml` — explicit per-role filenames
3. Prefix-named files: `exilis_*.gguf`, `sagax_*.gguf`, `logos_*.gguf`
4. Size-ordered fallback: smallest→exilis, middle→sagax, largest→logos
5. Single file: shared across all roles (dev/test)

Typical production layout:
```
models/
  exilis_qwen2.5-0.5b-instruct-q8_0.gguf   # ~500 MB — fast triage
  sagax_llama-3.2-3b-instruct-q4_k_m.gguf  # ~2 GB   — reasoning
  logos_llama-3.1-8b-instruct-q4_k_m.gguf  # ~5 GB   — consolidation
```

### 4.3 — Computation Profiles

There is no single "recommended model set," and this section will go
stale — check current benchmarks before committing to a deployment. Model
choices below reflect the state of local/edge models as of mid-2026;
Artux runs on hardware ranging from a Raspberry Pi to a workstation with
a discrete GPU, and the model tier assigned to each role should match the
deployment target and the current model landscape, not a frozen default.

Four reference profiles. All model choices below are GGUF, run via
`tool.llm.llamacpp.v1`, and follow the standard `exilis_*`/`sagax_*`/`logos_*`
naming convention from §4.2.

| Profile | Target hardware | RAM | Exilis | Sagax | Logos |
|---|---|---|---|---|---|
| **micro** | Raspberry Pi 4/5, SBC, embedded | 4–8 GB | LFM2.5-230M | LFM2.5-1.2B-Instruct | shared with Sagax |
| **standard** | Mini PC, older laptop, NUC | 8–16 GB | LFM2.5-230M or Qwen3.5-0.8B | LFM2.5-8B-A1B (1.5B active) | Gemma4 12B |
| **workstation** | Desktop, GPU-equipped laptop | 16–32 GB + GPU | Qwen3.5-0.8B | Gemma4 26B (MoE, 3.8B active) | Qwen3.6-35B-A3B (3B active) |
| **server** | Dedicated inference box, multi-GPU | 32 GB+ + GPU(s) | Qwen3.5-0.8B | Qwen3.6-35B-A3B or Gemma4 26B | Larger local MoE or API provider |

**Why LFM2.5-230M for Exilis.** Exilis makes a single structured-output
call per triage pass — read consN plus new events, emit
`{triage, reason}`. It never reasons about open-ended tasks. LFM2.5-230M
is purpose-trained for exactly this shape of work (structured extraction,
tool-call classification) rather than general reasoning, and it
out-performs models several times its size on tool-use and data-extraction
benchmarks as a direct consequence of that specialisation. It runs under
375 MB in 4-bit and is fast enough to be genuinely free on Pi-class CPUs —
Exilis's own attention gate (`events_pending()`) means inference only
fires when there's something to classify, so the model needs to be cheap
per-call more than it needs to be fast in aggregate throughput terms.

Where broader world knowledge in the triage decision actually matters
(e.g. Exilis needs to recognise a wider range of topical cues to decide
urgent vs ignore), Qwen3.5-0.8B is the fallback — larger context, native
multimodal support, and stronger general knowledge at a modest size cost.

**Why LFM2.5 for micro-tier Sagax.** LFM2.5-1.2B-Instruct is a genuine
step up from prior sub-2B options for instruction-following and
tool/agentic use specifically — the profile Sagax needs, since Narrator
grammar adherence is fundamentally an instruction-following problem.
Under 1 GB RAM, real reasoning capability rather than pure classification.
This is the single largest improvement over the previous generation of
recommendations for constrained hardware.

**micro** — expect Sagax response latency in the few-second range on
CPU-only Pi-class hardware. `n_gpu_layers: 0` is the only viable setting.
Skill synthesis and deep consolidation will be slow — consider a longer
`logos_interval_s` (600–900s) to reduce contention with Sagax for CPU
cycles, since they cannot run simultaneously on limited cores without one
starving the other.

**standard** — LFM2.5-8B-A1B activates only 1.5B parameters per token
despite 8.3B total, making it fast on CPU-only hardware while offering
real reasoning depth and a 128K context window. This is a meaningfully
better Sagax choice than a dense 3–4B model at similar latency cost.

**workstation** — GPU offload becomes worthwhile. Set
`tool.llm.llamacpp.v1.n_gpu_layers` to a positive value (or `-1` for full
offload) once VRAM allows. Gemma4 26B activates only 3.8B parameters per
token (MoE) — competitive with much larger dense models on instruction
following while fitting realistic VRAM budgets with quantisation and
KV-cache tuning. Qwen3.6-35B-A3B is a similarly strong choice with only
3B active parameters and native long context.

**server** — at this tier, consider whether Logos should be a local GGUF
model at all. Its workload (consolidation, skill synthesis, deep
evaluation) benefits from larger context and stronger reasoning more than
from low latency. A hosted API provider (`tool.llm.anthropic.v1`,
installed via staging) for Logos specifically — while Exilis and Sagax
stay local — is a reasonable hybrid: privacy-sensitive real-time
interaction stays on-device, and only consolidation (which can tolerate
higher latency and, depending on the operator's threat model, be treated
as less sensitive since it's already-processed summaries rather than raw
perception) goes to a cloud model.

**Choosing quantisation.** Q4_K_M is the general-purpose default —
meaningfully smaller than Q5/Q6 with acceptable quality loss for triage
and reasoning tasks. Q8_0 is worth the extra size for Exilis specifically
when running a dense model, since its output is structured JSON
classification where precision at the decision boundary matters more than
raw model capacity — this matters less for LFM2.5-230M, which is small
enough to run at higher precision by default. Avoid Q2/Q3 quantisations
for Sagax — Narrator grammar adherence degrades noticeably below Q4.

**LFM2.5 tool-call format note.** LFM2.5 models default to Pythonic
function-call syntax between `<|tool_call_start|>`/`<|tool_call_end|>`
special tokens rather than JSON. `tool.llm.llamacpp.v1`'s `complete_tools`
uses a JSON-extraction fallback (`_extract_json_tool_calls`) that expects
JSON-shaped output — when running LFM2.5 as a provider, add an explicit
system-prompt instruction requesting JSON tool calls (LFM2.5 supports this
override natively) rather than relying on the Pythonic default.

**License note.** Check current license terms before committing to a
model for a commercial deployment — some model families (including recent
Liquid AI releases) use dual-use or revenue-gated commercial terms rather
than a permissive license like Apache 2.0. This changes release to
release; verify against the model card at deployment time, not against
this document.

**Mixed profiles are normal.** A Pi-based satellite node running only
Exilis + a lightweight Sagax, paired with a workstation-tier Logos
reachable over the network, is a valid deployment — this is the
multi-Muninn federation pattern referenced in the whitepaper (external
Muninn instances as tools), extended to inference: nothing prevents an
operator from pointing `logos.provider` at a `tool.llm.*` installed
against a remote endpoint rather than a local GGUF file.

### 4.4 — Switching Providers at Runtime

```xml
<!-- Switch Sagax to Anthropic Claude -->
<task_update>{"action":"state_set","key":"sagax.provider","value":"tool.llm.anthropic.v1"}</task_update>
<task_update>{"action":"state_set","key":"sagax.model","value":"claude-sonnet-4-6"}</task_update>

<!-- Revert to local -->
<task_update>{"action":"state_set","key":"sagax.provider","value":"tool.llm.llamacpp.v1"}</task_update>
<task_update>{"action":"state_set","key":"sagax.model","value":"llama-3.2-3b-instruct-q4_k_m.gguf"}</task_update>
```

Changes take effect on the next inference call. No restart.

---

## 4.5 — Perception Architecture: Capture / Process Split

Sensor tools split into two roles that never overlap:

**Capture tools** (`direction: capture`) own hardware exclusively. A
microphone capture tool knows nothing about ASR, speaker identity, or any
other consumer of audio — it just produces `CaptureFrame` objects and
exposes `get_queue()`. `sounddevice`, `opencv-python`, or any other
hardware dependency lives only in the capture tool, never in Huginn core.

**Process tools** (`direction: process`) receive `CaptureFrame` objects
and turn them into STM events. A process tool has no hardware dependency
— it consumes whatever PerceptionManager routes to it. Multiple process
tools can consume the same capture stream simultaneously with zero
duplication: Moonshine ASR and a voice-identity tool both declare
`consumes: [audio_chunk]` and both receive every microphone chunk from a
single capture loop.

```
tool.capture.microphone.v1  (emits: [audio_chunk])
         │
         ▼
  PerceptionManager._dispatch_loop
         │
    ┌────┴────┐
    ▼         ▼
tool.asr.moonshine.v1      tool.identity.voice.v1
(consumes: [audio_chunk])  (consumes: [audio_chunk])
    │                           │
    ▼                           ▼
STM: speech event          STM: voiceprint signature event
```

**CaptureFrame:**
```python
@dataclass
class CaptureFrame:
    kind: str        # "audio_chunk" | "video_frame" | "screen_frame"
    data: Any         # numpy array, bytes — processor-defined
    ts:   str         # ISO timestamp, auto-populated
    meta: dict        # sample_rate, resolution, device_id, etc.
```

**PerceptionManager fan-out:**
```python
pm.start_capture_fanout(tool_manager)
  → builds {kind: [push_fn, ...]} from all installed process tools
  → starts one dispatch thread per capture tool
  → each thread drains its queue and calls every matching push_fn
  → one processor's failure never blocks the others (caught + logged)
```

A processor's `push(frame, _stm=None, _htm=None)` is called directly by
the dispatch thread. Processors that need to avoid blocking the dispatch
loop (e.g. a slow model) should enqueue internally and process on their
own thread — this is what `tool.asr.moonshine.v1` and the identity
processors do.

**Why this split matters:** adding a new sensor consumer — a second ASR
model, an ambient sound classifier, a wake-word-adjacent function —
requires zero changes to PerceptionManager or the capture tool. Install a
new `direction: process, consumes: [audio_chunk]` tool and it starts
receiving frames on the next fan-out rebuild. The inverse also holds:
swapping the microphone capture implementation (e.g. moving from
`sounddevice` to a networked audio source) requires zero changes to any
processor — they only know about `CaptureFrame`, never about hardware.

---

## 5 — Narrator Token Grammar

| Block | On open | On close | STM event | Bus publish |
|---|---|---|---|---|
| `<thinking>` / `<think>` | capture | **discard** — never stored | ✗ | ✗ |
| `<contemplation>` | capture | write `output/contemplation` | ✓ | `full` to `target:contemplation` |
| `<speech target="id">` | stream tokens → TTS + bus `partial` and `chunk` | write `output/speech status:full` | ✓ | `full` to `target:speech` |
| `<speech_step var="x" target="id">` | stream + set pending | write `output/speech_step status:full` | ✓ | `full` |
| `<tool_call>` | buffer | dispatch via permission gate; create/update HTM task | ✗ | ✗ |
| `<aug_call timeout_ms="N">` | buffer | dispatch parallel (read-only only); pause generation | ✗ | ✗ |
| `<aug_result>` | — | injected by Orchestrator | ✗ | ✗ |
| `<task_update>` | buffer | write to HTM (tasks or states) | ✗ | ✗ |
| `<projection>` | buffer | dispatch to UI | ✓ | `full` to `target:display` |

**Interrupted blocks:** When a nudge fires, the Orchestrator closes the open block and writes `status: "full", interrupted: true, partial_text: "..."`. The ActuationBus receives a terminal `full` event with `interrupted: true`. No `suspended` status in STM.

### Speech output event completeness levels (ActuationBus only)

| Level | Trigger | Consumers |
|---|---|---|
| `partial` | Each token during streaming | Avatar lip sync |
| `chunk` | Phrase boundary (punctuation after min tokens) | TTS synthesis (Kokoro, etc.) |
| `full` | Block close or interrupt | STM, Logos, text UI, avatar reset |

---

## 6 — On-Demand Instruction System

The system prompt (`SAGAX_PLAN_v2`) is an operating contract: grammar table, six micro-examples, topic directory, seven hard rules. All detailed guidance is in Muninn LTM as instruction artifacts.

```xml
<aug_call timeout_ms="400">
{"name": "get_instructions", "args": {"topic": "TOPIC"}}
</aug_call>
```

| Topic | When to fetch |
|---|---|
| `htm_tasks` | Starting multi-step work, handling interruption |
| `skill_execution` | Skill recalled for the first time this session |
| `memory` | Complex recall strategy or entity operations |
| `states` | Changing model/provider or tool configuration |
| `live_tools` | Starting/stopping a TTS, ASR, or output daemon |
| `staging` | STAGING TOOLS shows pending items |
| `entities` | New person appears or identity uncertain |
| `speech_step` | Skill step requires user input mid-execution |

`get_instructions` is registered as `polarity: read` — fully eligible for `aug_call`. Logos writes instruction artifacts on first boot from `prompts.py` constants. Never overwritten if an operator updated them.

---

## 7 — Skill Synthesis — Observation-Driven

No hardcoded thresholds. Logos observes execution friction from workbook and task notebooks, creates HTM evaluation tasks, accumulates evidence, proposes skills when evidence is strong.

```
Logos observes: Sagax tried 5 tools to find kitchen light
     ↓
_identify_synthesis_candidates() → LLM sees task notebooks
     ↓
Creates HTM task: "Skill synthesis candidate: kitchen_lighting"
  tags: [synthesis_candidate, synthesis_candidate.kitchen_lighting]
     ↓
Evidence accumulates over sessions via notebook entries [evidence] ...
     ↓
_advance_evaluation_tasks() → LLM evaluates candidate
     ↓
decision: "propose" → _propose_skill()
  writes LTM entry with status: "proposed"
  creates skill_proposal confirmation task (same flow as tool staging)
     ↓
Sagax presents to user at natural pause → user confirms
     ↓
Skill active (status: "proposed" → confirmed)
```

The HTM evaluation task notebook IS the evidence record. Logos's observations over multiple sessions build it. Detail in Sagax's task notebooks directly improves synthesis quality.

---

## 8 — Default Tool Stack (tools/builtin/)

Auto-installed by `Logos._ensure_default_tools()` on first pass. No staging confirmation. Dependencies checked before install — missing packages are skipped gracefully.

| Tool | Direction | Description | Deps |
|---|---|---|---|
| `tool.llm.llamacpp.v1` | provider | In-process GGUF inference. Per-role model paths. JSON grammar mode. | `llama-cpp-python` |
| `tool.capture.microphone.v1` | capture | Mic capture only — emits `audio_chunk`. No processing. | `sounddevice numpy` |
| `tool.capture.camera.v1` | capture | Camera capture only — emits `video_frame`. No processing. | `opencv-python numpy` |
| `tool.asr.moonshine.v1` | process | Consumes `audio_chunk` → STM speech events. No microphone dependency. | `moonshine-onnx numpy` |
| `tool.tts.kokoro.v1` | output | Kokoro ONNX TTS daemon. Chunk synthesis, live speed/voice config. | `kokoro-onnx sounddevice` |
| `tool.ui.text.v1` | io | Terminal text I/O. stdin→STM, ActuationBus→stdout. Zero deps. | — |

Note: `tool.asr.moonshine.v1` has no hardware dependency — it processes
whatever `audio_chunk` frames `tool.capture.microphone.v1` produces. This
means a second process tool (e.g. `tool.identity.voice.v1` in staging) can
consume the exact same microphone stream with zero additional capture
overhead. See §4.5 for the full capture/process architecture.

Zero-dependency startup:
```
models/sagax_*.gguf    →  GGUF auto-detected, llamacpp configured
pip install kokoro-onnx moonshine-onnx sounddevice
huginn.start()         →  fully functional, no API keys, no external servers
```

---

## 9 — Consolidated File Layout

```
huginn/
├── __init__.py          build_huginn(), _assign_gguf_models(), HuginnInstance
├── agents/
│   ├── exilis.py        Continuous loop, events_pending gate, batched triage
│   ├── sagax.py         Planning loop, Narrator, speech_step, consN update → GC
│   └── logos.py         Consolidation, observation-driven synthesis,
│                          _ensure_{startup_procedure,system_config,
│                                   instruction_defaults,default_tools}
├── runtime/
│   ├── stm.py           STMStore with events_pending()
│   ├── htm.py           Tasks + ASC + States (HTMStates)
│   ├── actuation_bus.py In-process pub/sub, non-blocking
│   ├── actuation_manager.py Live tool daemon lifecycle
│   ├── perception.py    Pipeline runner, signature resolution via recall()
│   ├── orchestrator.py  Router, chunker, on_consn_updated, interrupted flag
│   ├── tool_manager.py  Two-tier dispatch, get_instructions, htm_state_get,
│   │                      provider registration, _execute_native
│   └── tool_discovery.py Manifest parser (mode/direction/states/provider)
└── llm/
    ├── client.py        LLMClient router, _BuiltinProvider, _ProviderToolAdapter
    │                      with _role injection
    └── prompts.py       SAGAX_PLAN_v2, LOGOS_*, EXILIS_*,
                           8x INSTRUCTION_*_v1 artifacts

tools/
├── builtin/             Ships with Huginn — auto-installed, no staging
│   ├── tool_llm_llamacpp.py       GGUF provider, per-role model paths
│   ├── tool_capture_microphone.py Mic capture (direction: capture)
│   ├── tool_capture_camera.py     Camera capture (direction: capture)
│   ├── tool_asr_moonshine.py      ASR processor (direction: process)
│   ├── tool_tts_kokoro.py         Kokoro TTS daemon (direction: output)
│   └── tool_ui_text.py            Terminal I/O
└── staging/             Operator-dropped — requires user confirmation
    ├── tool_llm_ollama.py         Ollama provider
    ├── tool_llm_anthropic.py      Anthropic Claude provider
    ├── tool_identity_voice.py     Speaker ID processor (ECAPA-TDNN)
    ├── tool_identity_face.py      Face ID processor (InsightFace ArcFace)
    └── tool_config_write.py       LTM config editor
```

---

## 10 — Open Items

| # | Item | Status |
|---|---|---|
| 1 | Async STM push notification (aiosqlite — replace Exilis poll with push) | Deferred v0.5 |
| 2 | Multi-agent STM write conflict handling | Deferred v0.5 |
| 3 | Avatar live tool integration (TCP socket, `tool.ui.avatar.v1`) | Deferred |
| 4 | Output event refactor — partial/chunk in STM (currently bus-only) | Deferred v0.5 |
| 5 | Multi-Muninn federation (home + work + mobile as external tools) | Architecture stable, no code change needed |
| 6 | Artux avatar web component + tkinter widget integration | Deferred |
