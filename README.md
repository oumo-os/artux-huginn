# 🦅 Huginn — Cognitive Module

**Huginn thinks. [Muninn](https://github.com/oumo-os/artux-muninn) remembers.**

In Norse mythology, Odin's two ravens fly out each day and return at dusk.
**Muninn** brings back memory — everything seen and heard, preserved faithfully.
**Huginn** brings back thought — meaning, judgement, and the capacity to act.
Neither is useful without the other.

This repository is Huginn: the reasoning, planning, and learning layer of the
Artux cognitive stack. It reads from and writes to Muninn, but it never *is*
Muninn. The division is intentional and load-bearing.

---

## What Huginn Does

```
World → Perception Manager → STM (Muninn) → Exilis → Sagax → Orchestrator → World
             │                     ↑            ↓         ↓            ↓
        sig resolution         consN +      triage    Narrator    Tool Manager
        pipeline runner       HTM Tasks    signal     stream      / TTS / UI
                                  │
                             Logos ← raw events
                                ↓
                          LTM (Muninn)
               Skills · Procedures · Tools · Instructions
```

| Component | Speed | Role |
|---|---|---|
| **Perception Manager** | Continuous | Runs active perception pipeline tasks from HTM. Resolves biometric signatures. Writes canonical events to STM. No LLM. |
| **Exilis** | < 70 ms | Woken on each new STM event. One batched LLM triage call. Emits `ignore / act / urgent`. Never writes, never actuates. |
| **Sagax** | 1–30 s | Recall-driven reasoning. Reads STM + HTM tasks + live states. Produces structured Narrator token stream. |
| **Orchestrator** | < 50 ms | Routing bridge. Token stream router, permission gate, speech chunker, nudge, ActuationBus publisher. |
| **Logos** | Background | STM→LTM consolidation, skill synthesis, tool installation, instruction management. Sole LTM author. |

---

## How It Feels

### Interruption without losing your place

John asks for movie mood lighting. Sagax recalls `skill.set_movie_mood`,
creates an HTM task, asks what film, sets ceiling and wall lights to warm red and
gold. Before the popcorn step, John's phone rings and he walks out. Exilis
classifies the departure as `urgent`. The Orchestrator halts TTS, suspends
the open speech block, parks the task with `resume_at: step_6_popcorn`. Sagax
pivots cleanly. When John returns, his voiceprint is matched at the door,
the session grants reload, and Sagax reads the task notebook — steps 1–5
confirmed, resuming at popcorn. Not from the beginning. Not from a reconstruction.
From the exact step where it stopped.

### Identity as authority

Artux does not ask you to log in. When John speaks, the Perception Manager
matches his voiceprint against registered signatures and confirms his identity.
The Orchestrator loads his session grants — what he is allowed to do, what
requires confirmation, what is denied. An unrecognised voice becomes an implied
entity with guest access only. No personal calendar entries, no actuation of
personal devices, until identity is established from two or more independent
sources. Logos resolves implied entities post-session.

### Swapping the inference engine at runtime

```
<task_update>{"action": "state_set", "key": "sagax.provider",
              "value": "tool.llm.anthropic.v1"}</task_update>
<task_update>{"action": "state_set", "key": "sagax.model",
              "value": "claude-sonnet-4-6"}</task_update>
```

Two writes to HTM.states. The change takes effect on Sagax's next inference
call. No restart. No config file. The provider tool was installed through the
same staging workflow as a kettle controller or a TTS daemon.

### Picking up the manual mid-execution

Sagax encounters `skill.set_movie_mood` for the first time. Before interpreting
the guidance steps, it fetches the full execution manual inline:

```xml
<aug_call timeout_ms="400">
{"name": "get_instructions", "args": {"topic": "skill_execution"}}
</aug_call>
```

Generation pauses, the instruction artifact arrives from Muninn LTM, and Sagax
continues with full procedural context. The system prompt stayed lean; the
detail came on demand.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                              HUGINN                                │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                       ORCHESTRATOR                           │  │
│  │                                                              │  │
│  │  Perception Manager ──► STM ──► Exilis ──► Routing          │  │
│  │  (pipeline runner,      │       (triage     (act/urgent)    │  │
│  │   sig resolution)       │        < 70ms)         │          │  │
│  │                         │                         ▼         │  │
│  │  ActuationBus ◄─ Speech chunker              Sagax          │  │
│  │  partial/chunk/full     │                 (planning loop    │  │
│  │       │                 │                  Narrator stream) │  │
│  │       ▼                 │                        │          │  │
│  │  Live tool daemons      │                 Tool Manager      │  │
│  │  (TTS, ASR, avatar)     │                 permission gate   │  │
│  │                         │                                   │  │
│  │  HTM ┌─ Tasks (durable) │                                   │  │
│  │      ├─ ASC (session)   │                                   │  │
│  │      └─ States (live)   │                                   │  │
│  └──────────────────────── │ ─────────────────────────────────┘  │
│                             │                                      │
│                      ┌──────▼──────┐                              │
│                      │   MUNINN    │                              │
│                      │    STM      │                              │
│                      └──────┬──────┘                              │
│                             │ raw events                          │
│                      ┌──────▼──────┐                              │
│                      │    Logos    │  consolidation, synthesis     │
│                      └──────┬──────┘  tool install, instructions  │
│                             │                                      │
│                      ┌──────▼──────┐                              │
│                      │   MUNINN    │                              │
│                      │    LTM      │                              │
│                      │ tools·skills│                              │
│                      │ instructions│                              │
│                      │ entities    │                              │
│                      └─────────────┘                              │
└────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### All cognitive decisions are LLM calls

No hardcoded classifiers, rule engines, or heuristics anywhere in the stack.
Exilis triage, consN summarisation, Sagax planning, Logos consolidation, skill
synthesis, entity resolution — every judgement is an LLM call. Exilis uses a
fast small model (shared with Sagax for coherent priors). Logos uses the largest
available for high-fidelity LTM quality.

### LLM providers are tools

`LLMClient` is a thin router. It reads `{role}.provider` and `{role}.model`
from `HTM.states` at every call. Provider tools (`tool.llm.ollama.v1`,
`tool.llm.anthropic.v1`, …) register four callables at install time. Switching
backends is two `state_set` writes; the change takes effect immediately. New
providers — local or remote, open or proprietary — are installed through the
same staging workflow as any other tool. There is no hard-coded backend support.

### Tools are memory artifacts

Tool descriptors, skills, procedures, and instruction manuals all live in Muninn
LTM as `recall()`-able artifacts. Sagax discovers what it can do the same way
it discovers what it knows — by asking Muninn. There is no hard-coded capability
registry. The tool ontology is self-organising: Logos adds new capabilities as
they are installed, marks deprecated ones, and the next recall surfaces the
current picture automatically.

### HTM States as operational ground truth

HTM has three surfaces. Tasks track lifecycle and resumption. ASC holds the
session's warm context (entities, recent recalls, workbook). States holds live
operational parameters — model names, tool configuration, feature flags — as
plain key-value pairs with namespace conventions. States persists to LTM at
session end via Logos; recovers at boot via Orchestrator config recall. Sagax
reads and writes states directly through `<task_update>` blocks. No config
files. No environment variables for runtime behaviour.

### The system prompt is an operating contract

`SAGAX_PLAN_v2` is 84 lines. It contains a grammar table, six micro-examples
covering the constructs most likely to cause stumbles on first use, a topic
directory for the on-demand instruction system, and seven hard rules. All
walkthroughs, edge cases, and detailed examples live in eight LTM instruction
artifacts. Sagax fetches them inline via `get_instructions(topic)` when it
encounters an unfamiliar task type or needs precision. The system prompt
communicates what to do and where to look; Muninn holds how to do it.

### consN is lossy by design

Sagax maintains a single rolling narrative (`consN`) that compresses older
events as new ones arrive. It is intentionally lossy — exact timestamps
collapse, repetition merges, fine detail smooths. Exilis reads the same consN
for triage coherence. Logos never touches it; Logos reads raw events directly
via `get_raw_events()`. STM events are never deleted by consN compression —
they persist until Logos flushes them after verified LTM writes.

### Identity is authority

Permission grants are attached to entities, not sessions. When a known entity's
voiceprint or faceprint is matched, the Orchestrator opens a session with their
pre-configured grants. Unrecognised signatures become implied entities with guest
access. Logos performs post-session identity resolution. No login screens. No
passwords. Identity is biometric; authority follows identity.

### Logos earns skills through evidence

A skill is synthesised from repeated successful execution traces. Thresholds:
≥3 successful runs, ≥0.85 step similarity, 0 failures in the last 5, spread
across ≥2 days. New skills require explicit confirmation for the first 2 runs.
Skills are guidance sequences Sagax interprets — not scripts it executes
mechanically. The HTM task notebook is the structured trace Logos reads for
synthesis. Detail in notebook entries directly improves synthesis quality.

---

## Installation

```bash
# Memory backend — required
git clone https://github.com/oumo-os/artux-muninn
pip install -e ./artux-muninn

# Huginn
pip install -e ./artux-huginn

# Semantic recall quality (strongly recommended — falls back to TF-IDF without)
pip install sentence-transformers
```

A local LLM backend is required before any provider tool is installed.
The built-in `_BuiltinProvider` supports Ollama (default), Anthropic, OpenAI,
LM Studio, and llama.cpp as fallback until a provider tool takes over:

```bash
# Ollama (recommended for local-first setups)
ollama serve
ollama pull llama3.2

# Or set ANTHROPIC_API_KEY for Anthropic fallback
```

---

## Quick Start

### Minimal

```python
from huginn import build_huginn
from memory_module import MemoryAgent

muninn = MemoryAgent("artux.db")

huginn = build_huginn(
    muninn           = muninn,
    fallback_backend = "ollama",     # used until a provider tool is installed
    fallback_model   = "llama3.2",
    fallback_host    = "http://localhost:11434",
)

huginn.start()

# Sagax boots, recalls its startup procedure from LTM, and is ready.
# Logos writes LTM defaults on first pass (config, instructions, startup procedure).
reply = huginn.sagax.chat("What can you do?")
print(reply)
```

### With an active session

```python
huginn.orchestrator.new_session(
    entity_id        = "entity-john-001",
    permission_scope = ["lights", "kettle", "calendar.read"],
    denied           = ["camera"],
)
huginn.start()
```

### Switching providers at runtime

```python
# After tool.llm.anthropic.v1 is installed via staging:
huginn.htm.states.set("sagax.provider", "tool.llm.anthropic.v1")
huginn.htm.states.set("sagax.model",    "claude-sonnet-4-6")
# Takes effect on Sagax's next inference call. No restart.
```

### Registering an entity with grants

```python
import json

muninn.create_entity(
    name    = "John",
    content = json.dumps({
        "grants": {
            "permission_scope":      ["microphone", "lights", "kettle", "calendar.read"],
            "denied":                ["camera", "email.send"],
            "confirmation_required": ["calendar.write"],
        },
        "signatures": [
            {"kind": "voiceprint", "embedding": john_voiceprint_embedding},
        ],
    }),
)
```

### Dropping in a new tool

```python
# 1. Drop a .py file with a HUGINN_MANIFEST block into tools/staging/
# 2. Logos discovers it on its next pass and creates a staging HTM task
# 3. Sagax asks the user to confirm at the next natural pause
# 4. User confirms → Logos installs, writes LTM descriptor, activates

# For provider tools specifically:
# manifest declares mode: provider
# install_tool() calls LLMClient.register_provider() automatically
```

---

## Configuration

All runtime configuration lives in `HTM.states` (populated from Muninn LTM at boot).
`build_huginn()` only takes fallback values used before the first Logos pass.

| `build_huginn()` param | Default | Description |
|---|---|---|
| `fallback_backend` | `"ollama"` | Backend used before a provider tool is installed |
| `fallback_model` | `"llama3.2"` | Model used before LTM config is recalled |
| `fallback_host` | `"http://localhost:11434"` | Ollama host for fallback |
| `logos_interval_s` | `300` | Logos consolidation pass interval (seconds) |
| `staging_dir` | `"<db_dir>/tools/staging"` | Drop new tool `.py` files here |
| `active_dir` | `"<db_dir>/tools/active"` | Installed tool files moved here |

**Runtime parameters** (written to `HTM.states` by Logos on first pass, readable
and writable by Sagax at any time):

| Key | Example value | Description |
|---|---|---|
| `{role}.provider` | `"tool.llm.ollama.v1"` | Active LLM provider for exilis/sagax/logos |
| `{role}.model` | `"llama3.2"` | Model name passed to the provider |
| `{role}.temperature` | `0.2` | Generation temperature |
| `{role}.timeout` | `60.0` | Request timeout in seconds |
| `{tool_id}.{param}` | `"tool.tts.kokoro.v1.speed" = 1.4` | Tool-specific state |
| `session.{param}` | `"session.quiet_mode" = false` | Session-scoped overrides |

---

## File Layout

```
huginn/
├── __init__.py                   build_huginn() factory + HuginnInstance
├── agents/
│   ├── exilis.py                 Attention gate — 5ms poll, batched LLM triage
│   ├── sagax.py                  Planning agent — Narrator stream, HTM, consN
│   └── logos.py                  Consolidation daemon — LTM synthesis, skill promotion,
│                                   tool install, instruction defaults, config defaults
├── runtime/
│   ├── stm.py                    STMStore — STMEvent, consN, logos_watermark
│   ├── htm.py                    HTM — Tasks + ActiveSessionCache + States
│   ├── perception.py             Perception Manager — pipeline runner, sig resolution
│   ├── orchestrator.py           Routing bridge — token stream, permission gate,
│   │                               nudge, speech chunker, ActuationBus publisher
│   ├── actuation_bus.py          In-process pub/sub for output events
│   ├── actuation_manager.py      Live tool daemon lifecycle management
│   ├── tool_manager.py           Two-tier dispatch: memory + world tools.
│   │                               Native tools: get_instructions, htm_state_get
│   └── tool_discovery.py         Staging scanner, HUGINN_MANIFEST parser
└── llm/
    ├── client.py                 LLMClient router + _BuiltinProvider fallback
    └── prompts.py                SAGAX_PLAN_v2 (lean), LOGOS_*, EXILIS_*,
                                    8x INSTRUCTION_*_v1 artifact constants

tools/
└── staging/                      Drop new .py tool files here
    ├── tool_llm_ollama.py        Ollama LLM provider (mode: provider)
    ├── tool_llm_anthropic.py     Anthropic Claude provider (mode: provider)
    ├── tool_config_write.py      Write LTM config entries (Sagax-callable)
    └── ...                       Other starter tools

tests/
└── test_huginn.py                132 tests across 12 classes
```

---

## Narrator Token Grammar

Sagax produces a single structured token stream. The Orchestrator routes each block in real time.

```xml
<contemplation>           World reasoning — stored in STM, read by Logos</contemplation>
<speech target="id">      Spoken aloud — streamed live to TTS and ActuationBus</speech>
<speech_step var="x"
  target="id">            Say + wait — binds user response to x in HTM.states</speech_step>
<tool_call>               Write-capable actions — dispatched through permission gate</tool_call>
<aug_call timeout_ms="N"> Read-only inline lookups — generation pauses for results</aug_call>
<aug_result>              Injected by Orchestrator — Sagax never emits this</aug_result>
<task_update>             HTM task lifecycle and state operations</task_update>
<projection>              Structured data to UI</projection>
```

`<thinking>` and `<think>` are silently discarded — never stored, never streamed.

---

## On-Demand Instructions

Sagax fetches detailed operational manuals from Muninn LTM inline, without
expanding the system prompt:

```xml
<aug_call timeout_ms="400">
{"name": "get_instructions", "args": {"topic": "htm_tasks"}}
</aug_call>
```

Available topics: `htm_tasks` · `skill_execution` · `memory` · `states` ·
`live_tools` · `staging` · `entities` · `speech_step`

Logos writes these artifacts on first boot. Operators can update them by writing
new LTM entries with the same topic tags — Logos never overwrites custom content.

---

## Tool Manifest Format

```python
"""
HUGINN_MANIFEST
tool_id:            tool.my_tool.v1
title:              My Tool
capability_summary: What this tool does. Used by Sagax for capability discovery.
polarity:           read     # or write
permission_scope:   []       # e.g. [lights.ceiling, calendar.read]
mode:               callable # or service (daemon) or provider (LLM backend)
direction:          ""       # input | output | io | "" (for callable/provider)
inputs:
  param1: {type: string, description: "..."}
outputs:
  result: {type: string}
states:                      # service/provider tools — defaults written to HTM.states
  speed:
    default: 1.0
    type: float
    description: Playback speed
dependencies:
  some-package
END_MANIFEST
"""

def handle(param1: str, _muninn=None, _htm=None) -> dict:
    # _muninn and _htm are injected automatically if declared
    return {"result": "ok"}
```

**Mode values:**
- `callable` — standard tool, `handle()` called synchronously
- `service` — daemon thread, subscribes to ActuationBus, exposes `start/stop/handle`
- `provider` — LLM inference backend, exposes `complete/stream/complete_json/complete_tools`

---

## Relationship to Muninn

| | Muninn | Huginn |
|---|---|---|
| **Does** | Store, retrieve, decay, archive | Perceive, reason, plan, consolidate, learn |
| **Owns** | STM events, LTM entries, entities, sources | Agent logic, HTM (Tasks + ASC + States), Narrator, skill synthesis |
| **Writes LTM** | Never (passive store) | Yes — via Logos only |
| **Uses `recall()`** | Implements it | Calls it constantly |
| **Can run alone** | Yes (as a library) | No (requires Muninn) |

---

## Status

v2.0.0 — architecture stable, core implementation complete.

- [x] Architecture spec + addendum (`design/CognitiveModule.md` v1.0)
- [x] Orchestrator spec (`design/Orchestrator.md` v1.0)
- [x] `llm/client.py` — provider router + `_BuiltinProvider` fallback
- [x] `llm/prompts.py` — `SAGAX_PLAN_v2` (lean), all Logos prompts, 8 instruction artifacts
- [x] `runtime/stm.py` — STMEvent, consN, watermark
- [x] `runtime/htm.py` — Tasks + ActiveSessionCache + **States**
- [x] `runtime/actuation_bus.py` — in-process pub/sub (partial/chunk/full)
- [x] `runtime/actuation_manager.py` — live tool daemon lifecycle
- [x] `runtime/perception.py` — HTM-driven pipeline runner, signature resolution
- [x] `runtime/orchestrator.py` — token router, speech chunker, permission gate, nudge
- [x] `runtime/tool_manager.py` — two-tier dispatch, `get_instructions`, `htm_state_get`, provider registration
- [x] `runtime/tool_discovery.py` — staging scanner, manifest parser (mode/direction/states fields)
- [x] `agents/exilis.py` — batched triage
- [x] `agents/sagax.py` — planning loop, Narrator, consN, `speech_step` suspension
- [x] `agents/logos.py` — consolidation, skill synthesis, tool install, instruction/config defaults, state persistence
- [x] `huginn/__init__.py` — `build_huginn()` factory, bus/AM wired
- [x] `tools/staging/tool_llm_ollama.py` — Ollama provider tool
- [x] `tools/staging/tool_llm_anthropic.py` — Anthropic Claude provider tool
- [x] Test suite — 132 tests, 12 classes, all passing
- [ ] Async support (`aiosqlite` — push notification replaces Exilis poll loop)
- [ ] Multi-agent STM write conflict handling
- [ ] Output event refactor (partial/chunk/full in STM, not just ActuationBus)
- [ ] Artux avatar integration (React + tkinter widget, TCP state commands)

---

## Naming

In Norse cosmology, Odin sends both ravens out at dawn. Huginn flies to observe
and reason; Muninn flies to retain. Odin fears losing Muninn more —
*"Huginn I fear may not return, but I worry more for Muninn."*
This captures the dependency correctly: without memory, thought has no ground
to stand on.

---

## License

MIT
