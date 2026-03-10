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
| **Logos** | Minutes → days | Background consolidation. Reads raw STM events, writes durable LTM. Synthesises skills from repeated successful execution traces. Owns STM flush. |

---

## How It Feels

Before describing the machinery, here is what Huginn makes possible.

### Handling an interruption without losing your place

John walks in and says *"Put on movie mode."* Sagax picks up `skill.set_movie_mood`, asks John what film he wants, and begins working through the guidance steps — setting ceiling lights to warm red, wall lights to gold, closing the blinds.

Halfway through, John asks: *"By the way, when is my meeting tomorrow?"*

Artux doesn't freeze, stall, or lose the lighting job. Instead:

- Exilis classifies the new question as `act` — a genuine request, not a backchannel.
- The Orchestrator nudges Sagax mid-stream: speech is suspended, the lighting task is parked in the **Hot Task Manager** with a notebook entry — *"Step 4 complete: ceiling and wall set. Paused at step 5 (blinds). Resume at close_blinds."*
- Sagax pivots, reads John's calendar, and answers the question.
- John says *"Thanks."* Sagax reads the HTM — lighting task is still waiting — and resumes: *"Back to your movie setup — closing the blinds now."*

The lighting task never restarted from scratch. The calendar question didn't require a new session or a context reset. Both threads ran cleanly, tracked in parallel through the task manager.

This is what the HTM enables: Sagax can hold multiple threads of work simultaneously, park them on interruption, and resume from exactly where it left off — even if STM has been flushed or a consN compression has run in between.

### Knowing who you're talking to — and what they're allowed to do

Artux doesn't ask you to log in. When John walks into the room and speaks, the Perception Manager captures his voiceprint, runs it against the entity registry, and confirms his identity through signature matching. This confirmation flows into the Orchestrator as the active session's `entity_id`.

From that identity, Artux knows what John is allowed to do. Session grants are attached to entities:

- John's profile gives Artux access to his calendar, the smart lighting, and the kettle.
- The camera is denied — John has explicitly opted out of visual perception.
- Home assistant commands require verbal confirmation before execution.

When a second person, Sam, enters the room, the Perception Manager captures an unrecognised voiceprint. Sam is an implied entity — Artux heard someone it doesn't know yet. It holds the voiceprint as a partial identity in the Hot Task Manager's active session cache, accumulates evidence (voice pattern, position, what Sam says about herself) and waits. If Sam introduces herself — *"Hi, I'm Sam"* — Sagax links the name claim to the voiceprint, updates the implied entity, and the next time Sam returns, the Perception Manager will resolve her immediately.

Until Sam is confirmed, Artux treats her as a guest with restricted access. It won't read out private calendar entries or actuate personal devices — it doesn't know who Sam is. Identity is authority. Authority is scope.

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

## Key Design Decisions

### All cognitive decisions are LLM calls

No hardcoded classifiers, rule engines, or heuristics anywhere in the stack. Every triage decision, every narrative update, every plan, every consolidation, every skill evaluation is an LLM call. Exilis uses a ≤1B model for speed. Sagax uses a 3–13B model. Logos uses the largest available. The models are different; the principle is not.

### Perception pipelines are skills — and HTM tasks

A perception pipeline is a `class_type: "pipeline"` skill artifact with fully prefilled arguments. There is no LLM insertion between pipeline steps; the Orchestrator executes tool chains mechanically. When a pipeline step needs a value that varies at runtime, it must come from the previous step's output — not from inference.

Every active pipeline runs as an HTM task with `initiated_by: "system"`. The Orchestrator supervises it by querying HTM. Failures are logged to the task notebook. Sagax and Logos see pipeline state through the same HTM interface they use for everything else — no separate supervision mechanism.

Pipeline parameters have two change surfaces: **LTM meta** (durable, Logos-managed — e.g. Logos notices ASR degrades in the kitchen and writes a noise profile) and **HTM task notebook** (session-scoped — Sagax adjusts mic sensitivity mid-session). Logos can promote notebook observations to LTM meta post-session, closing the adaptation loop.

### Exilis sees the world through Sagax's eyes

There is exactly one `consN` and Sagax is its sole author. Exilis reads the same rolling narrative for triage context. Using the same model and the same consN means Exilis triages as Sagax would — no incoherence between the attention gate and the reasoning agent.

Exilis batches all available new events into a single LLM call per 5 ms poll cycle. If five events arrive in 20 ms, Exilis makes one call covering all five. Target total latency: < 70 ms.

### HTM: multi-tasking without context collision

The **Hot Task Manager** is a dual-surface store that lets Sagax hold multiple threads of work simultaneously without them colliding in STM.

**Surface 1 — ActiveSessionCache (ASC):** Per-session, ephemeral. Holds the session workbook (complete stream mirror), active entities, recently-used tools, and active recall results. Sagax queries it on demand — it's not pushed wholesale at wake-up. The workbook lets Sagax resume a paused task without re-executing confirmed steps.

**Surface 2 — Tasks:** Durable. Survives STM flush, consN compression, and session restart. Each task has a notebook (running commentary), a `resume_at` pointer, and a `persistence` flag. When Sagax is interrupted mid-skill, it writes a `<task_update>` block, parks the task, and picks up the new request. When it returns, it reads the HTM, finds the parked task, checks the workbook for the last confirmed step, and continues from there — not from the beginning.

### consN is lossy by design

Sagax maintains a single rolling narrative (`consN`) that compresses older events as new ones arrive. It is intentionally lossy — approximate timestamps, merged repetition, smoothed detail. It is Sagax's private contextual shorthand, not a record of truth.

`consN` has three load-bearing properties:
1. **Single object.** There is never more than one. Each update replaces the previous version entirely.
2. **Triggers ASC garbage collection.** A consN update is a session boundary: stale topics, recalls, and tools are pruned from the cache. Unresolved implied entities are **never pruned** — they linger until Logos resolves them.
3. **Invisible to Logos.** Logos reads raw events. Building LTM from consN would compound lossiness across every consolidation pass.

### Identity is authority

Artux does not maintain a login system. Identity is established through **signature matching** — voiceprint, faceprint, or device ID — handled entirely by the Perception Manager.

When a known entity is confirmed, the Orchestrator opens a session with that entity's pre-configured grants:

```
permission_scope:       [microphone, lights, kettle, calendar.read]
denied:                 [camera, email.send]
confirmation_required:  [calendar.write, file.delete]
```

Tools that require scopes outside the active grants are blocked at the Orchestrator's permission gate before they ever reach the Tool Manager. Sagax does not manage permissions — it declares what scope a tool needs; the Orchestrator enforces it.

Unknown entities (unresolved voiceprints, unrecognised faces) are treated as guests with restricted access. Their implied identity accumulates evidence in the ASC until Logos can confirm or archive them. Privacy-sensitive tools — cameras, personal calendars, messaging — require a confirmed entity. Artux will not expose personal data to someone it hasn't identified.

### Skills are guidance, not scripts

When Sagax discovers `skill.set_movie_mood` via `recall()`, it doesn't execute the steps mechanically. It reads the guidance sequence, reasons about each step, fills in arguments, and handles interactive beats — asking John what movie, recalling his lighting preferences, deciding whether to prompt for popcorn.

Skills are synthesised by Logos from repeated successful Sagax execution traces. The threshold is ≥3 successful runs, ≥0.85 step similarity, 0 failures in the last 5, spread across ≥2 days. New skills require a configurable number of human-confirmed runs before they run unconfirmed.

Pipelines are the **exception** to this: they have no LLM interpretation and are operator-authored, not synthesised. They run exactly as written.

---

## Installation

```bash
pip install openai anthropic         # at least one LLM backend
# Ollama: https://ollama.com — ollama serve; ollama pull llama3.2
```

---

## Quick Start

### Factory setup (recommended)

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
)

huginn.start()
reply = huginn.sagax.chat("What can you do?")
print(reply)
```

### With an active session

```python
from huginn import build_huginn
from memory_module import MemoryAgent

muninn = MemoryAgent("artux.db")

huginn = build_huginn(
    muninn       = muninn,
    llm_backend  = "anthropic",
    fast_model   = "claude-haiku-4-5-20251001",
    sagax_model  = "claude-sonnet-4-6",
    logos_model  = "claude-sonnet-4-6",
    on_tts_token = lambda token: print(token, end="", flush=True),
)

huginn.orchestrator.new_session(
    entity_id        = "john",
    permission_scope = ["lights", "kettle", "calendar.read"],
    denied           = ["camera"],
)

huginn.start()
```

### Registering a tool

Tools are stored in Muninn LTM. Sagax discovers them via `recall()` — no registry, no hardcoded list.

```python
import json

huginn.tool_manager.register("tool.set_ceiling_lights.v1", my_lights_fn)

muninn.consolidate_ltm(
    narrative = json.dumps({
        "artifact_type":      "tool",
        "tool_id":            "tool.set_ceiling_lights.v1",
        "title":              "Set ceiling lights",
        "capability_summary": "Control ceiling ambient lighting. Use for mood, wakeup, movie scenes.",
        "polarity":           "write",
        "permission_scope":   ["lights.ceiling"],
        "inputs":  {"colour": {"type": "string"}, "brightness": {"type": "int", "default": 80}},
        "outputs": {"status": {"type": "string"}},
    }),
    class_type = "tool",
)

# Sagax discovers it automatically:
#   sagax.chat("Set the mood for a Christmas movie")
#   → recall("mood lighting") → tool.set_ceiling_lights.v1 ✓
```

### Registering an entity with grants

```python
muninn.create_entity(
    name    = "John",
    content = json.dumps({
        "grants": {
            "permission_scope":       ["microphone", "lights", "kettle", "calendar.read"],
            "denied":                 ["camera", "email.send"],
            "confirmation_required":  ["calendar.write"],
        },
        "signatures": [
            {"kind": "voiceprint", "embedding": john_voiceprint_embedding},
        ],
    }),
)

# When John speaks, the Perception Manager matches his voiceprint,
# confirms his identity, and the Orchestrator opens a session with
# his grants automatically. No login required.
```

### Registering a perception pipeline

```python
muninn.consolidate_ltm(
    narrative = json.dumps({
        "artifact_type": "skill",
        "class_type":    "pipeline",
        "title":         "Voice → STM",
        "source_type":   "user",
        "event_type":    "speech",
        "meta": {
            "active_by_default": True,
        },
        "steps": [
            {"order": 1, "artifact_id": "tool.audio_capture.v1",
             "args": {"sample_rate": 16000, "channels": 1}},
            {"order": 2, "artifact_id": "tool.asr.v1",
             "args": {"model": "whisper-base", "language": "en"}},
            {"order": 3, "artifact_id": "tool.embed_text.v1",
             "args": {"model": "nomic-embed-text"}},
        ],
    }),
    class_type = "skill",
)

huginn.tools.register("tool.audio_capture.v1", my_capture_fn)
huginn.tools.register("tool.asr.v1",           my_asr_fn)
huginn.tools.register("tool.embed_text.v1",    my_embed_fn)

# The Orchestrator starts this pipeline automatically on boot
# (meta.active_by_default = True) as a system HTM task.
# Sagax can pause, resume, or modify it at runtime.
```

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `fast_model` | `"qwen2.5:0.5b"` | Exilis triage + consN summarise (≤1B recommended) |
| `sagax_model` | `"llama3.2"` | Sagax planning loop (3–13B recommended) |
| `logos_model` | `"llama3.2"` | Logos consolidation (larger = better LTM quality) |
| `logos_interval_s` | `300` | Logos consolidation pass interval (seconds) |
| `HUGINN_LOGOS_BATCH_SIZE` | `50` | Max raw events per Logos pass |
| `HUGINN_CONS_N_MIN_NEW` | `8` | New events before consN update becomes eligible |
| `HUGINN_CONS_N_MAX_NEW` | `20` | New events before consN update is forced |
| `HUGINN_SKILL_MIN_RUNS` | `3` | Minimum successful runs before skill promotion |
| `HUGINN_SKILL_MIN_DAYS` | `2` | Minimum days spread for skill promotion |
| `HUGINN_SKILL_CONFIRM_RUNS` | `2` | Confirmation-required runs for newly synthesised skills |
| `HUGINN_SIG_THRESHOLD` | `0.88` | Minimum cosine similarity for voiceprint/faceprint match |

---

## File Layout

```
huginn/
├── __init__.py                   build_huginn() factory + HuginnInstance
├── agents/
│   ├── exilis.py                 Attention gate — poll loop, one LLM triage call
│   ├── sagax.py                  Planning agent — Narrator stream, HTM, consN
│   └── logos.py                  Consolidation daemon — LTM synthesis, skill promotion
├── runtime/
│   ├── stm.py                    STMStore — STMEvent, consN, logos_watermark
│   ├── htm.py                    HTM — ActiveSessionCache + Tasks
│   ├── perception.py             Perception Manager — pipeline runner, sig resolution
│   └── orchestrator.py           Routing bridge — token stream, permission gate, nudge
└── llm/
    ├── client.py                 Unified LLM client (Ollama + Anthropic), streaming
    └── prompts.py                All system prompts as versioned constants

design/
├── CognitiveModule.md            Architecture spec v1.0
├── CognitiveModule_Addendum.md   Corrections — supersedes §1 and §4.1
└── Orchestrator.md               Orchestrator routing bridge spec v1.0
```

---

## Relationship to Muninn

| | Muninn | Huginn |
|---|---|---|
| **Does** | Store, retrieve, decay, archive | Perceive, reason, plan, consolidate, learn |
| **Owns** | STM events, LTM entries, entities, sources | Agent logic, HTM, Narrator, skill synthesis |
| **Writes LTM** | Never (passive store) | Yes — via Logos only |
| **Uses `recall()`** | Implements it | Calls it constantly |
| **Can run alone** | Yes (as a library) | No (requires Muninn) |

---

## Status

Architecture stable. Initial implementation complete.

- [x] Architecture spec (`design/CognitiveModule.md` v1.0)
- [x] Orchestrator spec (`design/Orchestrator.md` v1.0)
- [x] Architecture addendum — P-1, P-3, P-4 resolved (`design/CognitiveModule_Addendum.md`)
- [x] `llm/client.py` — unified LLM client (Ollama + Anthropic, streaming, JSON mode, tool calling)
- [x] `llm/prompts.py` — all system prompts as versioned constants
- [x] `runtime/stm.py` — STMEvent envelope, consN, logos_watermark, flush
- [x] `runtime/htm.py` — Hot Task Manager (ActiveSessionCache + Tasks, SQLite-backed)
- [x] `runtime/perception.py` — pipeline runner, signature resolution, STM write
- [x] `runtime/orchestrator.py` — token stream router, permission gate, nudge, HTM scheduler
- [x] `agents/exilis.py` — 5 ms poll loop, batched LLM triage call
- [x] `agents/sagax.py` — planning loop, Narrator stream, consN updates, HTM integration
- [x] `agents/logos.py` — consolidation daemon, per-segment LTM, skill synthesis, ASC flush
- [x] `huginn/__init__.py` — `build_huginn()` factory
- [ ] Test suite
- [ ] Async support (`aiosqlite` — push notification replaces Exilis poll loop)
- [ ] Multi-agent STM write conflict handling (v0.5)

---

## Naming

In Norse cosmology, Odin sends both ravens out at dawn. Huginn flies to observe and reason; Muninn flies to retain. Odin fears losing Muninn more — *"Huginn I fear may not return, but I worry more for Muninn."* This captures the dependency correctly: without memory, thought has no ground to stand on.

The module split reflects this. Huginn is the active, volatile, reasoning layer. Muninn is the persistent, careful record. Each has a clear job. Neither does the other's.

---

## License

MIT
