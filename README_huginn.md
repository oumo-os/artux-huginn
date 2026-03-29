# 🦅 Huginn — Cognitive Module

**Huginn thinks. [Muninn](https://github.com/oumo-os/artux-muninn) remembers.**

In Norse mythology, Odin's two ravens fly out each day and return at dusk. **Muninn** brings back memory — everything seen and heard, preserved faithfully. **Huginn** brings back thought — meaning, judgement, and the capacity to act. Neither is useful without the other.

This repository is Huginn: the reasoning, planning, and learning layer of the Artux cognitive stack. It reads from and writes to Muninn, but it never *is* Muninn. The division is intentional and load-bearing.

---

## What Huginn Does

Huginn is a four-component cognitive stack that turns a stream of perceptual events into purposeful, improving action:

```
Perception → Exilis → STM (Muninn) → Sagax → Orchestrator → World
                 ↑            ↑           ↓            ↓
            Signature     consN +    Narrator     Tool Manager
           Resolution   HTM Tasks   token stream    / TTS / UI
                              │
                            Logos
                              ↓
                         LTM (Muninn)
                    Skills · Procedures · Tools
```

| Component | Speed | Role |
|---|---|---|
| **Exilis** | < 100 ms | Ingests percepts (speech, vision, sensors). Normalises, classifies, writes to STM. Runs signature-based entity resolution. Never actuates. |
| **Sagax** | 1–30 s | Recall-driven reasoning. Reads STM context, queries the Hot Task Manager, searches Muninn for capabilities, produces structured output via the Narrator token stream. |
| **Orchestrator** | < 50 ms | Routing bridge. No cognitive logic. Routes the Narrator stream to TTS, tools, UI, and HTM. Enforces session permissions. |
| **Logos** | Minutes → days | Background consolidation. Reads raw STM events, writes durable LTM. Synthesises skills from repeated successful execution traces. |

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

Artux doesn't ask you to log in. When John walks into the room and speaks, Exilis captures his voiceprint, runs it against the entity registry, and confirms his identity through signature matching. This confirmation flows into the Orchestrator as the active session's `entity_id`.

From that identity, Artux knows what John is allowed to do. Session grants are attached to entities:

- John's profile gives Artux access to his calendar, the smart lighting, and the kettle.
- The camera is denied — John has explicitly opted out of visual perception.
- Home assistant commands require verbal confirmation before execution.

When a second person, Sam, enters the room, Exilis captures an unrecognised voiceprint. Sam is an implied entity — Artux heard someone it doesn't know yet. It holds the voiceprint as a partial identity in the Hot Task Manager's active session cache, accumulates evidence (voice pattern, position, what Sam says about herself) and waits. If Sam introduces herself — *"Hi, I'm Sam"* — Sagax links the name claim to the voiceprint, updates the implied entity, and the next time Sam returns, Exilis will resolve her immediately.

Until Sam is confirmed, Artux treats her as a guest with restricted access. It won't read out private calendar entries or actuate personal devices — it doesn't know who Sam is. Identity is authority. Authority is scope.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                             HUGINN                                    │
│                                                                       │
│  ┌──────────┐  events   ┌─────────────────────────────────────────┐  │
│  │  Exilis  │──────────►│              MUNINN (STM)                │  │
│  │ < 100 ms │           │   events[]   consN   logos_watermark     │  │
│  │ sig.res. │           └───────────────────┬─────────────────────┘  │
│  └──────────┘                               │                         │
│       ▲ triage signals                      │ window / raw events     │
│       │                                     ▼                         │
│  ┌────┴──────────────────────────────┐  ┌─────────┐                  │
│  │          ORCHESTRATOR             │  │  Logos  │                  │
│  │  token router · permission gate   │  │  async  │                  │
│  │  nudge · aug_call · HTM scheduler │  └────┬────┘                  │
│  └──┬──────────────┬─────────────────┘       │ consolidate_ltm        │
│     │ Narrator     │ tool calls              │ skill synthesis        │
│     ▼ stream       ▼                         │ flush STM              │
│  ┌─────────┐  ┌──────────┐                  ▼                        │
│  │  Sagax  │  │   Tool   │  ┌──────────────────────────────────────┐ │
│  │ 1–30 s  │  │ Manager  │  │          MUNINN (LTM)                │ │
│  └────┬────┘  └──────────┘  │  tools · skills · procedures         │ │
│       │                      │  entities · facts · sources           │ │
│       │ asc.get()            └──────────────────────────────────────┘ │
│       ▼                               ▲ recall()                       │
│  ┌─────────────────────────────────┐  │                               │
│  │      HTM (Hot Task Manager)     │──┘                               │
│  │  ActiveSessionCache | Tasks     │                                  │
│  └─────────────────────────────────┘                                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### HTM: multi-tasking without context collision

The **Hot Task Manager** is a dual-surface store that lets Sagax hold multiple threads of work simultaneously without them colliding in STM.

**Surface 1 — ActiveSessionCache (ASC):** Per-session, ephemeral. Holds the session workbook (complete stream mirror), active entities, recently-used tools, and active recall results. Sagax queries it on demand — it's not pushed wholesale at wake-up. The workbook lets Sagax resume a paused task without re-executing confirmed steps.

**Surface 2 — Tasks:** Durable. Survives STM flush, consN compression, and session restart. Each task has a notebook (running commentary), a `resume_at` pointer, and a `persistence` flag. When Sagax is interrupted mid-skill, it writes a `<task_update>` block, parks the task, and picks up the calendar question. When it returns, it reads the HTM, finds the parked task, checks the workbook for the last confirmed step, and continues from there — not from the beginning.

This is also how Sagax avoids the "2-minute reconstruction problem": instead of reasoning through `<thinking>` to figure out what it has already done, it just reads the task notebook.

### consN is lossy by design

Sagax maintains a single rolling narrative (`consN`) that compresses older events as new ones arrive. It is intentionally lossy — approximate timestamps, merged repetition, smoothed detail. It is Sagax's private contextual shorthand, not a record of truth.

`consN` has three load-bearing properties:
1. **Single object.** There is never more than one. Each update replaces the previous version entirely.
2. **Triggers ASC garbage collection.** A consN update is a session boundary: stale topics, recalls, and tools are pruned from the cache. Unresolved implied entities are **never pruned** — they linger until Logos resolves them.
3. **Invisible to Logos.** Logos reads raw events. Building LTM from consN would compound lossiness across every consolidation pass.

### Identity is authority

Artux does not maintain a login system. Identity is established through **signature matching** — voiceprint, faceprint, or device ID — handled entirely by Exilis.

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

Progress is tracked in an HTM Task. When the skill completes, Logos finds the task notebook plus raw events and has a clean, structured execution trace. After three successful runs across two or more days, Logos promotes the sequence to a permanent skill artifact. The next time John asks for movie mode, Sagax finds the skill immediately and spends no time reconstructing the sequence.

### Logos earns skills through evidence

A sequence of tool calls does not become a skill because it worked once. Logos promotes a trace cluster to a `skill` artifact only after:
- ≥ 3 successful executions
- ≥ 0.85 structural similarity across runs
- 0 failures in the last 5 executions
- Executions spread across ≥ 2 different days

New skills run with human confirmation required for the first 2 executions (configurable) before the Orchestrator will proceed autonomously.

### Tools are memory artifacts

Tool descriptors, skills, and procedures are stored in Muninn LTM as first-class retrievable artifacts. Sagax calls `recall("adjust lighting")` and gets back `tool.set_ceiling_lights` — not because it scanned a registry, but because the tool descriptor lives in the same semantic space as everything else Sagax knows.

This means the capability ontology is self-organising: Logos adds skills as they are learned, marks failing ones for deprecation, and the next `recall()` reflects the updated state automatically.

---

## Repository Structure

```
huginn/
├── agents/
│   ├── exilis.py          # Attention gate: triage loop, signature resolution
│   ├── sagax.py           # Interactive agent: planning loop, Narrator token stream
│   └── logos.py           # Background daemon: consolidation, skill synthesis, ASC flush
│
├── runtime/
│   ├── stm.py             # consN helpers, logos_watermark, flush_up_to, get_raw_events
│   ├── htm.py             # Hot Task Manager: ASC (ephemeral) + Tasks (durable)
│   └── orchestrator.py    # Routing bridge: token stream, permission gate, nudge, aug_call
│
├── design/
│   ├── CognitiveModule.md # Full architecture spec (authoritative reference)
│   └── Orchestrator.md    # Execution runtime spec
│
├── specs/
│   ├── tool_descriptor_schema.json
│   ├── skill_schema.json
│   └── procedure_schema.json
│
└── tests/
    ├── test_stm.py                      # consN invariants, watermark, flush
    ├── test_htm.py                      # ASC GC, task persistence, entity retention
    ├── test_sagax_skill.py              # Guided skill execution end-to-end
    ├── test_logos_synthesis.py          # Skill promotion thresholds
    └── test_asc_entity_persistence.py   # Unresolved entities survive GC
```

**Huginn does not contain the memory implementation.** That lives in [Muninn](https://github.com/oumo-os/artux-muninn). You need both.

---

## Requirements

```bash
# Memory backend — required
git clone https://github.com/oumo-os/artux-muninn
pip install -e ./artux-muninn

# Huginn
pip install -e ./artux-huginn

# Semantic recall quality (strongly recommended — falls back to TF-IDF without it)
pip install sentence-transformers

# For Exilis audio/vision perception (optional)
pip install sounddevice opencv-python moonshine-onnx
# or: pip install faster-whisper
```

A local LLM backend is required for Sagax and Logos:

```bash
# Option A — Ollama (recommended for local setups)
ollama serve
ollama pull llama3.2        # Sagax
ollama pull llama3.2:70b    # Logos (larger = better LTM quality)

# Option B — Anthropic API
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Quickstart

### Minimal setup

```python
from huginn.agents.sagax import Sagax
from huginn.agents.logos import Logos
from huginn.runtime.htm import HTM
from memory_module import MemoryAgent

muninn = MemoryAgent("artux.db")
htm    = HTM()

sagax  = Sagax(muninn, htm, llm_backend="ollama", model="llama3.2")
logos  = Logos(muninn, htm, llm_backend="ollama", model="llama3.2")
logos.start()

reply = sagax.chat("What can you do?")
print(reply)
```

### With perception and multi-tasking

```python
from huginn.agents.exilis import Exilis
from huginn.agents.sagax import Sagax
from huginn.agents.logos import Logos
from huginn.runtime.htm import HTM
from huginn.runtime.orchestrator import Orchestrator
from memory_module import MemoryAgent

muninn       = MemoryAgent("artux.db")
htm          = HTM()
orchestrator = Orchestrator(htm)

exilis = Exilis(muninn, orchestrator)
sagax  = Sagax(muninn, htm, orchestrator, llm_backend="ollama", model="llama3.2")
logos  = Logos(muninn, htm, llm_backend="ollama", model="llama3.2")

exilis.start()  # begins recording audio/video → STM
logos.start()   # consolidates STM → LTM on a timer

# Sagax wakes on Exilis signals and handles multiple concurrent threads
# automatically via the HTM task manager.
sagax.run()
```

### Registering a tool

Tools must be registered in Muninn LTM before Sagax can discover them:

```python
from huginn.runtime.tools import register_tool

register_tool(muninn, {
    "tool_id":   "tool.set_ceiling_lights.v1",
    "title":     "Set ceiling lights",
    "capability_summary": "Control ceiling ambient lighting. Use for mood, wakeup, movie scenes.",
    "polarity":  "write",
    "modality":  "request",
    "inputs":    { "colour": {"type": "string"}, "brightness": {"type": "int", "default": 80} },
    "outputs":   { "status": {"type": "string"} },
    "permission_scope": ["lights.ceiling"],
    "handler":   my_lights_function,
})

# Sagax discovers it automatically:
#   sagax.chat("Set the mood for a Christmas movie")
#   → recall("mood lighting") → tool.set_ceiling_lights.v1 ✓
```

### Registering an entity with grants

```python
from huginn.runtime.identity import register_entity

register_entity(muninn, {
    "name":    "John",
    "grants": {
        "permission_scope":       ["microphone", "lights", "kettle", "calendar.read"],
        "denied":                 ["camera", "email.send"],
        "confirmation_required":  ["calendar.write"]
    },
    "voiceprint": john_voiceprint_embedding,  # float[] from your ASR pipeline
})

# When John speaks, Exilis matches his voiceprint, confirms his identity,
# and the Orchestrator opens a session with his grants automatically.
# No login required.
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HUGINN_LLM_BACKEND` | `"ollama"` | `"ollama"` or `"anthropic"` |
| `HUGINN_SAGAX_MODEL` | `"llama3.2"` | Model for Sagax (medium-large recommended) |
| `HUGINN_LOGOS_MODEL` | `"llama3.2"` | Model for Logos (larger = better LTM quality) |
| `HUGINN_LOGOS_INTERVAL_S` | `300` | Logos consolidation pass interval (seconds) |
| `HUGINN_LOGOS_BATCH_SIZE` | `50` | Max events per Logos consolidation pass |
| `HUGINN_CONS_N_MIN_NEW` | `8` | New events before consN update triggers |
| `HUGINN_SKILL_MIN_RUNS` | `3` | Minimum successful runs before skill promotion |
| `HUGINN_SKILL_MIN_DAYS` | `2` | Minimum days spread for skill promotion |
| `HUGINN_SKILL_CONFIRM_RUNS` | `2` | Confirmation-required runs for new skills |
| `HUGINN_SIG_THRESHOLD` | `0.88` | Minimum cosine similarity for voiceprint match |
| `HUGINN_DB_PATH` | `"artux.db"` | Muninn SQLite database path |

---

## Relationship to Muninn

| | Muninn | Huginn |
|---|---|---|
| **Does** | Store, retrieve, decay, archive | Ingest, reason, plan, consolidate, learn |
| **Owns** | STM events, LTM entries, entities, sources | Agent logic, HTM, planning loop, skill synthesis |
| **Writes LTM** | Never (passive store) | Yes — via Logos only |
| **Uses `recall()`** | Implements it | Calls it constantly |
| **Can run alone** | Yes (as a library) | No (requires Muninn) |

---

## Naming

In Norse cosmology, Odin sends both ravens out at dawn. Huginn flies to observe and reason; Muninn flies to retain. Odin fears losing Muninn more — *"Huginn I fear may not return, but I worry more for Muninn."* This captures the dependency correctly: without memory, thought has no ground to stand on.

The module split reflects this. Huginn is the active, volatile, reasoning layer. Muninn is the persistent, careful record. Each has a clear job. Neither does the other's.

---

## Status

Active development. Architecture spec stable; implementation in progress.

- [x] Architecture spec (`design/CognitiveModule.md` v1.0)
- [x] Orchestrator spec (`design/Orchestrator.md` v1.0)
- [x] Artifact schemas (tool descriptor, skill, procedure)
- [ ] `runtime/stm.py` — consN + watermark helpers
- [ ] `runtime/htm.py` — Hot Task Manager (ASC + Tasks)
- [ ] `runtime/orchestrator.py` — routing bridge
- [ ] `agents/exilis.py` — triage loop, signature resolution
- [ ] `agents/sagax.py` — planning loop, Narrator token stream
- [ ] `agents/logos.py` — consolidation daemon, skill synthesis
- [ ] Test suite

---

## License

MIT
