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

### 4.3 — Switching Providers at Runtime

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

| Tool | Description | Deps |
|---|---|---|
| `tool.llm.llamacpp.v1` | In-process GGUF inference. Per-role model paths. JSON grammar mode. | `llama-cpp-python` |
| `tool.tts.kokoro.v1` | Kokoro ONNX TTS daemon. Chunk synthesis, live speed/voice config. | `kokoro-onnx sounddevice` |
| `tool.asr.moonshine.v1` | Moonshine ONNX ASR daemon. Writes STM speech events identical to typed input. | `moonshine-onnx sounddevice` |
| `tool.ui.text.v1` | Terminal text I/O. stdin→STM, ActuationBus→stdout. Zero deps. | — |

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
│   ├── tool_llm_llamacpp.py    GGUF provider, per-role model paths
│   ├── tool_tts_kokoro.py      Kokoro TTS daemon
│   ├── tool_asr_moonshine.py   Moonshine ASR daemon
│   └── tool_ui_text.py         Terminal I/O
└── staging/             Operator-dropped — requires user confirmation
    ├── tool_llm_ollama.py       Ollama provider
    ├── tool_llm_anthropic.py    Anthropic Claude provider
    └── tool_config_write.py     LTM config editor
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
