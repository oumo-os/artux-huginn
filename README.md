# 🦅 Huginn — Cognitive Module

**Huginn thinks. [Muninn](https://github.com/oumo-os/artux-muninn) remembers.**

In Norse mythology, Odin's two ravens fly out each day and return at dusk. **Muninn** brings back memory — everything seen and heard, preserved faithfully. **Huginn** brings back thought — meaning, judgement, and the capacity to act. Neither is useful without the other.

This repository is Huginn: the reasoning, planning, and learning layer of the Artux cognitive stack. It reads from and writes to Muninn, but it never *is* Muninn. The division is intentional and load-bearing.

---

## What Huginn Does

Huginn is a five-component cognitive stack that turns a stream of perceptual events into purposeful, improving action:

```
World → Perception Manager → STM (Muninn) → Sagax → Orchestrator → World
               │                    ↑           ↓            ↓
         sig resolution          consN +    Narrator     Tool Manager
         pipeline runner        HTM Tasks   token stream   / TTS / UI
               │                    │
             Exilis               Logos
          (triage gate)         (async consolidation)
                                     │
                              staging/ scan
                              install affirmed
                                     ↓
                               LTM (Muninn)
                          Skills · Procedures · Tools
```

| Component | Speed | Role |
|---|---|---|
| **Perception Manager** | Continuous | Runs active perception pipelines (audio, camera, sensors). Resolves biometric signatures against the entity registry. Writes canonical events to STM. No LLM. |
| **Exilis** | < 70 ms | Reads the STM window (consN + new events). One small LLM call per poll cycle. Emits `ignore` / `act` / `urgent`. Nothing else. |
| **Sagax** | 1–30 s | Recall-driven reasoning. Reads STM context, queries the Hot Task Manager, searches Muninn for capabilities, produces structured output via the Narrator token stream. Updates consN. |
| **Orchestrator** | < 50 ms | Routing bridge. No cognitive logic. Runs perception pipelines, routes the Narrator stream to TTS, tools, UI, and HTM. Enforces session permissions. Issues nudges. |
| **Logos** | Minutes → days | Background consolidation. Reads raw STM events, writes durable LTM. Synthesises skills. Scans staging directory for new tools. Installs affirmed tool files. Owns STM flush. |

---

## How It Feels

### Handling an interruption without losing your place

John walks in and says *"Put on movie mode."* Sagax picks up `skill.set_movie_mood`, asks John what film he wants, and begins working through the guidance steps — setting ceiling lights to warm red, wall lights to gold, closing the blinds.

Halfway through, John asks: *"By the way, when is my meeting tomorrow?"*

Artux doesn't freeze, stall, or lose the lighting job. Instead:

- Exilis classifies the new question as `act` — a genuine request, not a backchannel.
- The Orchestrator nudges Sagax mid-stream: speech is suspended, the lighting task is parked in the **Hot Task Manager** with a notebook entry — *"Step 4 complete: ceiling and wall set. Paused at step 5 (blinds). Resume at close_blinds."*
- Sagax pivots, reads John's calendar, and answers the question.
- John says *"Thanks."* Sagax reads the HTM — lighting task is still waiting — and resumes: *"Back to your movie setup — closing the blinds now."*

### Knowing who you're talking to — and what they're allowed to do

Artux doesn't ask you to log in. When John walks into the room and speaks, the Perception Manager captures his voiceprint, runs it against the entity registry, and confirms his identity through signature matching. This confirmation flows into the Orchestrator as the active session's `entity_id`.

From that identity, Artux knows what John is allowed to do — his profile grants lighting, kettle, and calendar read access, denies the camera, and requires verbal confirmation for calendar writes. Unknown speakers are treated as guests with restricted access until Logos can confirm or archive their identity.

### Discovering and installing new tools

You drop a Python file into `tools/staging/`. On its next maintenance cycle (every 5 minutes by default), Logos scans the directory, reads the tool's manifest, and writes a discovery event to STM. Sagax notices it and, at the next natural pause in conversation, tells John:

*"I found a new tool in the staging directory — it's called Smart Kettle Control. It can boil water and set a specific temperature. It needs kettle permission and requires the `tplink-smarthome-api` package. Want me to install it? It's also capable of running as a background sensor pipeline — would you like it active by default?"*

John says *"Yes to both."* Sagax notes the affirmation in the HTM task. Logos installs it on the next cycle — pip-installing the dependency, loading the module, registering the handler, writing the LTM artifact, and optionally activating it as a perception pipeline.

If John says *"Actually, install it now,"* Sagax calls `request_early_logos_cycle` and Logos runs immediately.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                                HUGINN                                   │
│                                                                         │
│  sensors ──►  ┌────────────────────────────────────────────────────┐   │
│  audio   ──►  │                  ORCHESTRATOR                       │   │
│  camera  ──►  │                                                     │   │
│               │  ┌───────────────────────────────────────────────┐ │   │
│               │  │            Perception Manager                  │ │   │
│               │  │  pipeline runner · sig resolution · STM write  │ │   │
│               │  └──────────────────────────┬────────────────────┘ │   │
│               │                 events       │ Exilis woken         │   │
│               │                    ▼         │                      │   │
│               │  ┌─────────────────────┐     │                      │   │
│               │  │     MUNINN (STM)     │     │                      │   │
│               │  │  events[]   consN    │     │                      │   │
│               │  │  logos_watermark     │     │                      │   │
│               │  └──────────┬──────────┘     │                      │   │
│               │             │  window         ▼                      │   │
│               │             │     ┌──────────────────┐               │   │
│               │             └────►│     Exilis        │               │   │
│               │                   │  < 70 ms triage   │               │   │
│               │                   └────────┬──────────┘               │   │
│               │                            │ act / urgent              │   │
│               │                            ▼                           │   │
│               │                   ┌──────────────────┐  Narrator       │   │
│               │                   │      Sagax        │────────────────┤   │
│               │                   │   planning loop   │  stream        │   │
│               │                   └────────┬──────────┘                │   │
│               │                            │ asc.get()                 │   │
│               │                            ▼                           │   │
│               │              ┌─────────────────────────────────────┐  │   │
│               │              │      HTM (Hot Task Manager)          │  │   │
│               │              │   ActiveSessionCache  |  Tasks       │  │   │
│               │              └─────────────────────────────────────┘  │   │
│               │  ┌───────────────────────────────┐                    │   │
│               │  │            Logos               │◄── raw events      │   │
│               │  │  consolidation · skill synth   │                    │   │
│               │  │  staging scan · tool install   │                    │   │
│               │  └───────────────────────────────┘                    │   │
│               └────────────────────────────────────────────────────── ┘   │
│                                       │ consolidate_ltm()                  │
│                                       ▼                                    │
│                      ┌────────────────────────────────────┐               │
│                       │            MUNINN (LTM)             │               │
│                       │  tools · skills · entities · facts  │               │
│                       └────────────────────────────────────┘               │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Adding a New Tool

This is the most common task. There are two paths.

### Path A: Operator registration (startup)

Write the descriptor to Muninn LTM and register the Python handler at startup. Use this for tools that ship with the system or are installed by the operator before first boot.

```python
from huginn.runtime.tool_manager import register_tool

register_tool(muninn, huginn.tool_manager, {
    "tool_id":            "tool.set_ceiling_lights.v1",
    "title":              "Set ceiling lights",
    "capability_summary": "Control ceiling ambient lighting — mood, wakeup, movie scenes.",
    "polarity":           "write",
    "permission_scope":   ["lights.ceiling"],
    "inputs": {
        "colour":     {"type": "string",  "description": "CSS colour name or hex"},
        "brightness": {"type": "integer", "description": "0–100", "default": 80},
    },
    "outputs": {
        "status": {"type": "string"},
    },
}, my_lights_fn)
```

Sagax discovers it automatically via `recall()` — there is no tool registry to update.

### Path B: Drop-in staging (recommended for new tools at runtime)

Create a Python file with a `HUGINN_MANIFEST` block and drop it into the `tools/staging/` directory. Logos will discover it, surface it to the user through Sagax, and install it after confirmation — without any restart.

**File format:**

```python
"""
HUGINN_MANIFEST
tool_id:            tool.smart_kettle.v1
title:              Smart Kettle Control
capability_summary: Boil water or set a target temperature on the smart kettle.
polarity:           write
permission_scope:   [kettle]
inputs:
  action:        {type: string, enum: [boil, set_temp, cancel]}
  temperature_c: {type: integer, default: 100}
outputs:
  status:        {type: string}
  current_temp:  {type: number}
dependencies:
  - tplink-smarthome-api>=0.7
perception_capable: false
handler:            handle
END_MANIFEST
"""

def handle(action: str, temperature_c: int = 100) -> dict:
    """Called by ToolManager on every tool_call to this tool."""
    # your implementation here
    ...
    return {"status": "ok", "current_temp": 100.0}
```

**Manifest fields:**

| Field | Required | Description |
|---|---|---|
| `tool_id` | ✓ | Globally unique. Convention: `tool.<name>.v<N>` |
| `title` | ✓ | Short human name shown to the user |
| `capability_summary` | ✓ | One sentence. What Sagax uses to decide when to call this tool. Write it as a recall query answer — *"Use this to..."* |
| `polarity` | ✓ | `read` — safe for inline aug_call. `write` — must go through permission gate via tool_call. |
| `permission_scope` | ✓ | List of scope strings. The entity's grants must include all of these. |
| `inputs` | ✓ | JSON Schema properties dict. Keys match the handler's kwargs. |
| `outputs` | | JSON Schema properties dict for the return value. |
| `dependencies` | | `pip install` strings. Logos installs these before loading the module. |
| `perception_capable` | | If `true`, Sagax will ask whether to activate it as a background perception pipeline. |
| `handler` | | Name of the callable in the file. Defaults to `handle`. |

**What happens after you drop the file:**

1. Next Logos cycle (≤5 min), the file is scanned and a `tool_discovered` event is written to STM.
2. Exilis classifies it as `act`. Sagax wakes and reads the pending staging task.
3. At the next natural pause in conversation, Sagax describes the tool and asks the user for confirmation.
4. User says yes → Sagax writes `user_affirmed: true` to the HTM task notebook.
5. Next Logos cycle installs it: pip deps → module load → LTM artifact → handler registration.
6. Sagax responds: *"Noted, I'll install it on the next maintenance cycle — likely within the next 4 minutes, unless you need it urgently."*
7. If the user says *"urgent"* or *"now"*: Sagax emits `request_early_logos_cycle` and Logos runs immediately.

After installation, the file is moved from `tools/staging/` to `tools/active/` and Sagax can call the tool immediately.

### Perception pipelines

A tool that runs continuously (audio capture, camera feed, a background sensor) is a perception pipeline. It differs from a regular tool in one way: it runs as a system HTM task, and the Orchestrator executes it on every tick rather than waiting for Sagax to call it.

To register a pipeline explicitly at startup:

```python
import json
muninn.consolidate_ltm(
    narrative = json.dumps({
        "artifact_type": "skill",
        "class_type":    "pipeline",
        "title":         "Voice → STM",
        "source_type":   "user",
        "event_type":    "speech",
        "meta": {"active_by_default": True},
        "steps": [
            {"order": 1, "artifact_id": "tool.audio_capture.v1",
             "args": {"sample_rate": 16000, "channels": 1}},
            {"order": 2, "artifact_id": "tool.asr.v1",
             "args": {"model": "whisper-base", "language": "en"}},
        ],
    }),
    class_type = "skill",
)
huginn.tools.register("tool.audio_capture.v1", my_capture_fn)
huginn.tools.register("tool.asr.v1",           my_asr_fn)
```

If `meta.active_by_default` is `true`, the Orchestrator starts it as an HTM system task on boot. For staged tools with `perception_capable: true`, Sagax will ask the user during installation whether to enable it as a pipeline.

---

## Installation

```bash
pip install openai anthropic         # at least one LLM backend
# Ollama: https://ollama.com — ollama serve; ollama pull llama3.2
```

---

## Quick Start

### Factory setup

```python
from huginn import build_huginn
from memory_module import MemoryAgent

muninn = MemoryAgent("artux.db")

huginn = build_huginn(
    muninn      = muninn,
    llm_backend = "ollama",
    fast_model  = "qwen2.5:0.5b",   # Exilis + consN summarise
    sagax_model = "llama3.2",
    logos_model = "llama3.2",
    staging_dir = "./tools/staging", # optional — defaults to ./tools/staging
)

huginn.start()
reply = huginn.sagax.chat("What can you do?")
print(reply)
```

### With an active session

```python
huginn.orchestrator.new_session(
    entity_id        = "john",
    permission_scope = ["lights", "kettle", "calendar.read"],
    denied           = ["camera"],
)
huginn.start()
```

### Registering an entity with grants

```python
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

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `fast_model` | `"qwen2.5:0.5b"` | Exilis triage + consN summarise (≤1B recommended) |
| `sagax_model` | `"llama3.2"` | Sagax planning loop (3–13B recommended) |
| `logos_model` | `"llama3.2"` | Logos consolidation (larger = better LTM quality) |
| `logos_interval_s` | `300` | Logos consolidation pass interval (seconds) |
| `staging_dir` | `"<db_dir>/tools/staging"` | Drop new tool `.py` files here |
| `active_dir` | `"<db_dir>/tools/active"` | Installed tool files are moved here |
| `HUGINN_LOGOS_BATCH_SIZE` | `50` | Max raw events per Logos pass |
| `HUGINN_CONS_N_MIN_NEW` | `8` | New events before consN update becomes eligible |
| `HUGINN_CONS_N_MAX_NEW` | `20` | New events before consN update is forced |
| `HUGINN_SKILL_MIN_RUNS` | `3` | Minimum successful runs before skill promotion |
| `HUGINN_SKILL_MIN_DAYS` | `2` | Minimum days spread for skill promotion |
| `HUGINN_SIG_THRESHOLD` | `0.88` | Minimum cosine similarity for voiceprint/faceprint match |

---

## File Layout

```
huginn/
├── __init__.py                   build_huginn() factory + HuginnInstance
├── agents/
│   ├── exilis.py                 Attention gate — poll loop, one LLM triage call
│   ├── sagax.py                  Planning agent — Narrator stream, HTM, consN
│   └── logos.py                  Consolidation daemon — LTM, skill synthesis, tool install
├── runtime/
│   ├── stm.py                    STMStore — STMEvent, consN, watermark (huginn_* tables)
│   ├── htm.py                    HTM — ActiveSessionCache + Tasks (SQLite)
│   ├── perception.py             Perception Manager — pipeline runner, sig resolution
│   ├── orchestrator.py           Routing bridge — Narrator router, permission gate, nudge
│   ├── tool_manager.py           Two-tier tool dispatch — memory tools + world tools
│   └── tool_discovery.py         Staging scanner — manifest parser, install lifecycle
└── llm/
    ├── client.py                 Unified LLM client (Ollama + Anthropic)
    └── prompts.py                All system prompts as versioned constants

tools/
├── staging/                      Drop new tool .py files here
└── active/                       Installed tools live here

design/
├── CognitiveModule.md            Architecture spec v1.0
├── CognitiveModule_Addendum.md   Corrections — supersedes §1 and §4.1
└── Orchestrator.md               Orchestrator routing bridge spec v1.0
```

---

## Key Design Decisions

### All cognitive decisions are LLM calls

No hardcoded classifiers, rule engines, or heuristics. Every triage decision, every narrative update, every plan, every consolidation is an LLM call. The models are different sizes; the principle is not.

### Tools are memory artifacts

Tool descriptors live in Muninn LTM. Sagax discovers them via `recall()`. There is no hardcoded registry, no import list, no switch statement. A tool that has never been called has the same discoverability as one called a thousand times.

### Perception pipelines are skills — and HTM tasks

A pipeline is a `class_type: "pipeline"` skill artifact. Steps are executed mechanically by the Orchestrator with no LLM between them. Every active pipeline runs as a system HTM task, so Sagax and Logos see pipeline state through the same interface they use for everything else.

### consN is lossy by design

Sagax maintains one rolling narrative (`consN`) that compresses older events as new ones arrive. It is Sagax's private contextual shorthand — not a record. Logos never reads it; it reads raw events instead.

### Identity is authority

Identity is established through signature matching — voiceprint, faceprint, or device ID. Session grants are per-entity, enforced by the Orchestrator's permission gate. Sagax declares what scope a tool needs; the Orchestrator enforces it.

---

## Relationship to Muninn

| | Muninn | Huginn |
|---|---|---|
| **Does** | Store, retrieve, decay, archive | Perceive, reason, plan, consolidate, learn |
| **Owns** | STM events, LTM entries, entities, sources | Agent logic, HTM, Narrator, skill synthesis, tool install |
| **Writes LTM** | Never (passive store) | Yes — via Logos only |
| **Uses `recall()`** | Implements it | Calls it constantly |
| **Can run alone** | Yes (as a library) | No (requires Muninn) |

---

## Starter Tools

Ten tools ship ready to install via the staging pipeline. Drop the files into
`tools/staging/` and Huginn will discover and present them for confirmation.

### Perception pipelines

| File | Tool ID | What it does | Dependencies |
|---|---|---|---|
| `tool_voice_moonshine.py` | `tool.voice.moonshine.v1` | Microphone → Moonshine ONNX ASR. Offline, CPU-native, no PyTorch. Best for short commands. | `moonshine-onnx`, `sounddevice` |
| `tool_voice_whisper.py` | `tool.voice.whisper.v1` | Microphone → faster-whisper ASR. Higher accuracy, multilingual, GPU-capable. | `faster-whisper`, `sounddevice` |
| `tool_vision_smolvlm.py` | `tool.vision.smolvlm.v1` | Webcam → SmolVLM scene description. In-process, offline, saves frame to disk. | `transformers`, `torch`, `opencv-python`, `Pillow` |

### Output

| File | Tool ID | What it does | Dependencies |
|---|---|---|---|
| `tool_tts.py` | `tool.tts.v1` | Speak text aloud. Tries pyttsx3 → piper → espeak-ng in order. | `pyttsx3` (piper/espeak optional) |

### Information lookups (safe for `<aug_call>`)

| File | Tool ID | What it does | Dependencies |
|---|---|---|---|
| `tool_weather.py` | `tool.weather.v1` | Current weather + 3-day forecast via wttr.in. No API key. | `requests` |
| `tool_web_search.py` | `tool.web.search.v1` | DuckDuckGo search — instant answers + organic results. No API key. | `requests` |
| `tool_calendar_read.py` | `tool.calendar.read.v1` | Upcoming events from local `.ics` file or CalDAV URL. | `icalendar`, `requests` |
| `tool_system_status.py` | `tool.system.status.v1` | CPU, memory, disk, uptime snapshot. | `psutil` |

### Utility (write polarity)

| File | Tool ID | What it does | Dependencies |
|---|---|---|---|
| `tool_timer.py` | `tool.timer.v1` | Set/cancel/list countdown timers. Fires an STM event on expiry → wakes Sagax. | _(stdlib only)_ |
| `tool_notes.py` | `tool.notes.v1` | Append, read, list, search plain-text notes on disk. | _(stdlib only)_ |

All tools follow the `HUGINN_MANIFEST` format and can be used as templates
for custom tools. See [Adding a New Tool](#adding-a-new-tool) above.

---

## Status

Architecture stable. Core implementation complete. Starter tool set complete.

- [x] Architecture spec (`design/CognitiveModule.md` v1.0)
- [x] Orchestrator spec (`design/Orchestrator.md` v1.0)
- [x] Architecture addendum — P-1, P-3, P-4 resolved (`design/CognitiveModule_Addendum.md`)
- [x] `llm/client.py` — unified LLM client (Ollama + Anthropic, streaming, JSON, tool calling)
- [x] `llm/prompts.py` — all system prompts as versioned constants (includes staging confirmation protocol)
- [x] `runtime/stm.py` — STMEvent envelope, consN, watermark (huginn_events + huginn_meta tables)
- [x] `runtime/htm.py` — Hot Task Manager (ASC + Tasks, tags_any/tags_all filtering)
- [x] `runtime/perception.py` — pipeline runner, signature resolution, STM write
- [x] `runtime/orchestrator.py` — token stream router, permission gate, nudge, early-cycle proxy
- [x] `runtime/tool_manager.py` — two-tier dispatch (memory tools + world tools), dynamic install
- [x] `runtime/tool_discovery.py` — staging scanner, HUGINN_MANIFEST parser, install lifecycle
- [x] `agents/exilis.py` — 5 ms poll loop, batched LLM triage
- [x] `agents/sagax.py` — planning loop, Narrator stream, consN, staging confirmation dialogue
- [x] `agents/logos.py` — consolidation daemon, skill synthesis, staging scan + tool install, early cycle
- [x] `huginn/__init__.py` — `build_huginn()` factory with staging dirs wired
- [x] Starter tool set (10 tools in `tools/staging/`)
- [ ] **Test suite** — smoke test `build_huginn()`, unit tests for STM/HTM/ToolDiscovery, integration test Huginn ↔ Muninn end-to-end
- [ ] **Vigil layer** — reconceptualise HTM as a broader active-state layer with partitions: `hot_entities`, `hot_capabilities`, `hot_topics`, `hot_recalls`, `workbook`, `hot_parameters`, `tasks`
- [ ] **`speech_step` skills** — conversational skill execution: emit utterance, await response, bind to variable, suspend/resume task
- [ ] Async support (`aiosqlite` — push notification replaces Exilis poll loop)
- [ ] Multi-agent STM write conflict handling (v0.5)

---

## Naming

In Norse cosmology, Odin sends both ravens out at dawn. Huginn flies to observe and reason; Muninn flies to retain. Odin fears losing Muninn more — *"Huginn I fear may not return, but I worry more for Muninn."* This captures the dependency correctly: without memory, thought has no ground to stand on.

The module split reflects this. Huginn is the active, volatile, reasoning layer. Muninn is the persistent, careful record. Each has a clear job. Neither does the other's.

---

## License

MIT
