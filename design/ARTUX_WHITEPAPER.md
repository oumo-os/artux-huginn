# Artux: A Design for Ambient Cognitive Infrastructure

**Version 2.0 · March 2026**
**Repository: `oumo-os/artux-huginn` + `oumo-os/artux-muninn`**

---

## Preface and Origin

This document describes the design principles, architecture, and key decisions
behind Artux — a self-hosted ambient AI system built for personal environments.
It is written after implementation rather than before, which means it describes
what was actually built and why, not what was planned and hoped for. The gap
between those two things is usually where the real design lives.

Artux began as a conceptual document called ANIMA — *Autonomous Neurocognitive
Intelligence Modelling Architecture* — written out of frustration with a specific
and mundane problem: wake words. Every voice assistant in common use requires
you to summon it before it listens. This reveals something about the system's
self-conception: it assumes it needs permission to perceive. The wake word
is not a privacy feature. It is a philosophical concession — the machine
admitting that it does not know when it is relevant.

The central proposition of ANIMA was a restatement of Descartes: *"Percipio,
ita cogito agere"* — I perceive, therefore I think to act. Not waiting to be
addressed. Continuously aware, and deciding for itself when action is warranted.
The wake word problem is solved not by a better trigger mechanism, but by a
different conception of what the system is doing between commands. It is not
idle. It is attending.

ANIMA proposed two implementation spirits: **Sagax**, the stable backbone, and
**Mirai**, the adaptive imagination. Without Sagax, ANIMA would drift; without
Mirai, it would ossify. This framing was architecturally honest about a tension
that most systems pretend does not exist.

Artux is what happened when that conceptual framework met the constraints of
actual implementation. Several things transferred almost intact: the Sagax name
and role, the Orchestrator as a routing kernel with no cognitive logic, the
notion of internal thought as a first-class output rather than a side effect,
and the continuous perception loop as the primary driver of action.

Mirai did not survive as a named agent. The adaptive spirit it represented was
distributed across the architecture instead: Logos handles evolution and
consolidation, the tool ecosystem handles extensibility, the provider model
handles backend flexibility, the instruction system handles knowledge evolution.
Whether this is better than two explicitly named spirits in tension is a question
the implementation does not fully answer.

What Artux added that ANIMA did not anticipate: a memory architecture with clear
tier separation (STM, HTM, LTM), biometric identity as authority rather than
login, a live operational state surface (HTM States) that makes runtime
configuration as dynamic as runtime knowledge, and the insight that LLM inference
itself is just another tool — swappable, installable, configurable through the
same mechanisms as everything else.

The transparency argument ANIMA made on ethical grounds — that internal states
must be inspectable and revisable — became structural in Artux. The
`contemplation` block is mandated to write to STM. The raw event log is never
compressed away. The architecture enforces inspectability rather than promising
it as a design intention.

Artux is not a finished system. It is a minimal implementation of an idea that
is larger than any single codebase. This document describes what was built and
why those choices were made. The frustration with wake words is still the right
place to start.

This document describes the design principles, architecture, and key decisions
behind Artux — a self-hosted ambient AI system built for personal environments.
It is written after implementation rather than before, which means it describes
what was actually built and why, not what was planned and hoped for. The gap
between those two things is usually where the real design lives.

Artux is not a chatbot with plugins. It is not a local copy of a cloud AI
product. It is a cognitive architecture — a small set of components with clear
contracts, each doing one thing well, composable enough to handle the full
complexity of an ambient personal environment without any single component
becoming a monolith.

---

## 1. The Problem with Ambient AI

Most AI assistants are session-local. They know what you told them in the
current conversation. They may have tool access. They may have a document
context. But when you come back tomorrow, they start again. The system has no
memory of what you said, what it did, what worked, what didn't, or who you are
beyond your account identifier.

This is fine for a productivity tool. It is insufficient for an ambient system
that lives in your home, understands your routines, knows your preferences, and
should get better at serving you over time — not by fine-tuning a model, but by
accumulating structured knowledge about you specifically.

A second problem: cloud dependency. An ambient system in your home sees
everything. It hears conversations. It sees what your camera sees. It knows
when you are home and when you are not. That data should not leave your
possession. The system should be self-hosted by design, not as an afterthought
opt-in.

A third problem: rigidity. Fixed tool sets with hard-coded integrations mean
every new capability requires a software update. A system that can reason about
what tools exist, discover new ones at runtime, and learn from execution
experience is qualitatively different from one where capabilities are enumerated
in a config file.

Artux is built as a response to these three problems simultaneously.

---

## 2. The Architecture in One Paragraph

Artux is two repositories. **Muninn** is a passive SQLite-backed memory store —
it stores events, consolidates knowledge, tracks entities, and answers recall
queries. It has no opinions about what should be stored or why. **Huginn** is
the cognitive layer — five components that together turn a stream of perceptual
events into purposeful, improving action. Muninn can run alone as a library.
Huginn always requires Muninn. Neither knows what the other's internals look
like; they communicate through Muninn's public API.

---

## 3. The Memory Model

### 3.1 Three Tiers

Artux memory has three tiers with distinct roles that do not overlap.

**Short-term memory (STM)** is an append-only event log. Every perception —
speech, video frame, sensor reading, tool result, agent output — becomes a
typed event with a source, a timestamp, a confidence, and a structured payload.
Events are never modified once written. They are never deleted by compression.
They persist until a verified consolidation pass, at which point Logos advances
the flush watermark and removes them from the live event store. The raw event
is always ground truth.

**Hot state (HTM)** is the operational present. It has three surfaces. *Tasks*
are durable work records — goals that span multiple steps, may be interrupted,
must be resumable. *ActiveSessionCache* is ephemeral session context — the
working set of entities, recent recalls, tool invocations, and a complete
workbook of the current session's token stream and tool payloads. *States* is
a flat key-value store for live operational parameters: which LLM provider is
active, what speed the TTS daemon should run at, whether the microphone is
enabled. States persist to LTM at session end. They are operational ground
truth — the difference between a configuration file that requires a restart and
a parameter you can change mid-sentence.

**Long-term memory (LTM)** is durable knowledge. It is written only by Logos,
the background consolidation daemon. Every piece of LTM — episodic observations,
semantic assertions, entity ledgers, tool descriptors, skill artifacts,
instruction manuals, system configuration — is a first-class recall-able
artifact. The same query interface retrieves a kettle control tool, a
remembered fact about a person, and the operational manual for how to park an
interrupted task. There is no distinction at the retrieval layer.

### 3.2 Why Append-Only STM

The decision to never delete raw events during compression reflects a design
invariant: compression is a lossy summary for reasoning convenience, not a
faithful record. When Sagax updates `consN` — its rolling narrative of recent
events — it is making an editorial choice about what to keep in its active
context. That choice should not destroy the record those decisions were made
from.

Logos reads raw events for consolidation, not consN. Building LTM from a
summary of a summary would compound lossiness with each pass. Logos reads the
ground truth; Sagax reasons from the editorial summary; both are correct for
their purpose.

### 3.3 The Entity Model

Entities — people, objects, concepts — are historical ledgers, not records.
When something new is learned about an entity, it is appended. When something
is corrected, the correction is appended with its authority tier. Contradictions
coexist in the ledger, each weighted by source authority and recency. The
entity's identity is derived from this history, not imposed as a single current
state. This models how identity actually works — we do not overwrite what we
knew about someone when we learn something new. We accumulate.

---

## 4. The Cognitive Model

### 4.1 Five Components

**Perception Manager** has no language model. It runs active perception pipeline
tasks from HTM, chains tool calls in sequence (audio capture → ASR → signature
resolution), and writes canonical typed events to STM. It does not decide what
to do with what it perceives. It does not triage. It records.

**Exilis** is woken by each new STM event. It reads `consN` plus the new-event
window — the same context Sagax would read — and makes one LLM call with a
structured output: `ignore`, `act`, or `urgent`. That is its entire job. It
never actuates. It never writes. It never calls `recall()`. It decides whether
Sagax needs to wake up.

The key design decision in Exilis is that it shares Sagax's model and Sagax's
consN. Triage coherence requires shared priors — if Exilis classifies a
backchannel as `urgent` using a different world model than Sagax uses for
reasoning, the system will interrupt itself on its own speech. By sharing the
same model and the same rolling narrative, Exilis effectively asks: "Given what
Sagax knows right now, does Sagax need to wake up?" That is precisely the right
question.

**Sagax** is the reasoning and planning agent. It is event-driven — it sleeps
between Exilis signals and between tool result arrivals. When it wakes, it reads
consN plus the new-event window, checks HTM for active or due tasks, and
produces a structured Narrator token stream. It discovers capabilities by
calling `recall()`, not by consulting a registry. It tracks multi-step work in
HTM Tasks. It updates consN when its working context has grown stale. It reads
and writes HTM States for operational parameters.

Sagax's output is a single token stream with XML block delimiters. The
Orchestrator routes each block in real time — speech tokens go to TTS as they
arrive, tool calls are buffered and dispatched on close, contemplation is
written to STM, task updates go directly to HTM. The block grammar is the
interface contract between Sagax and the rest of the system.

**Logos** is the background consolidation daemon. It reads raw STM events,
synthesises LTM narratives, promotes execution traces to skill artifacts,
manages memory hygiene, and handles tool installation. Logos is the sole author
of durable LTM — Sagax can request LTM writes in narrow circumstances, but the
authoritative consolidation is always Logos. Logos also performs post-session
work: resolving implied entities, persisting dirty HTM States back to LTM
config, flushing the session cache.

**Orchestrator** is the routing bridge. It has no cognitive logic. It manages
the token stream state machine, the permission gate, the two-stage nudge
protocol, the speech chunker, the ActuationBus publisher, the HTM scheduler,
and the session lifecycle. Everything flows through it; it decides nothing.

### 4.2 consN — Lossy by Design

The rolling narrative summary (`consN`) is one of the more counterintuitive
decisions in the architecture. Most systems try to preserve information. Artux
deliberately compresses it into something lossy.

The reason: Sagax has a bounded context window. The full raw event log for a
long session would exceed it. Something has to summarise. The question is
whether that summary should be treated as a faithful record or an editorial
shorthand.

Artux treats it as editorial shorthand. `consN` is Sagax's private working
context — it is updated by Sagax when needed, it serves Sagax's reasoning, and
it is invisible to Logos. Logos always reads the raw events. This means the
full-fidelity record is always available for consolidation quality, even as
Sagax's working context shrinks to fit its window.

This also means `consN` updates are cheap in the right sense: they cannot
destroy information that matters for the long-term record. They can only affect
what Sagax has in its immediate reasoning context.

### 4.3 Skills as Guidance

When Sagax executes a skill, it does not run a script. It reads a guidance
sequence — an ordered list of steps with natural-language descriptions, tool
hints, and interaction flags — and reasons through each one. It fills in
arguments. It handles interactive beats by asking the user or recalling
preferences. It tracks progress in an HTM Task notebook.

This distinction matters for two reasons. First, it makes the system robust to
capability changes — if a tool is replaced, the guidance step still makes sense
even if the tool_hint is stale. Second, it produces rich execution traces.
Logos synthesises skills from execution traces, and a trace produced by
reasoned interpretation is far richer than one produced by mechanical execution.
The notebook entries Sagax writes as it works through a skill become the
evidence base for the next synthesis pass.

Skill synthesis thresholds are conservative by design: three successful
executions, structural similarity above 0.85, no failures in the last five
runs, spread across at least two days. New skills require human confirmation
for the first two autonomous runs. The system earns trust; it does not assume it.

---

## 5. The Provider Model

### 5.1 LLM Inference as a Tool

The most significant architectural decision in v2.0 is that LLM inference
is no longer special. A provider tool is a `.py` file with a `HUGINN_MANIFEST`
block declaring `mode: provider` and four callables: `complete`, `stream`,
`complete_json`, `complete_tools`. It is installed through the same staging
workflow as a kettle controller or a TTS daemon. It registers itself in
`LLMClient._PROVIDERS` at install time. Switching providers is two writes to
`HTM.states`.

This has practical consequences that extend beyond flexibility. An operator
wanting to run on a new local model or a new API does not need to modify any
Huginn source code. They write a provider tool with the appropriate API calls,
drop it in the staging directory, confirm it through Sagax, and it is active.
The cognitive stack is agnostic about what runs inference.

### 5.2 The Router

`LLMClient` is a thin router. At every call, it reads three values from
`HTM.states`:

- `{role}.provider` — which provider tool is active for this agent role
- `{role}.model` — the model name to pass to the provider
- `{role}.temperature` — the generation temperature

If a registered provider exists under that tool ID, it dispatches through
`_ProviderToolAdapter`, which injects `_htm` if the provider's callables
declare it. The `_htm` injection lets a provider read its own configuration
from HTM.states — host, api_key, timeout — without needing those values passed
explicitly by the caller.

If no provider is registered (first boot, or before any provider tool is
installed), `_BuiltinProvider` handles the call. It supports Ollama, Anthropic,
OpenAI, LM Studio, and llama.cpp as bootstrap backends, using the OpenAI-compat
`/v1/` interface for everything except Anthropic. The builtin provider keeps the
system working before any tool is installed; it is not the intended steady state.

### 5.3 Per-Role Configuration

Exilis, Sagax, and Logos can each run different providers and models. At boot,
the Orchestrator recalls config entries from Muninn LTM and calls
`load_from_config` on `HTM.states` for each role — populating `sagax.provider`,
`sagax.model`, `logos.provider`, etc. from the previously persisted
configuration. Changes Sagax makes at runtime (e.g., switching its own provider
mid-session) are dirty-tracked in states and written back to LTM by Logos at
session end, so they survive across reboots.

This means the model configuration for a running Artux instance is not in a
config file anywhere. It is in Muninn LTM, and it evolves as the operator or
Sagax modifies it. A user saying "switch to the bigger model" and Sagax
executing two `state_set` writes is a legitimate configuration management path.

---

## 6. The Tool Ecosystem

### 6.1 Capabilities as Memory

Tool descriptors live in Muninn LTM with `class_type: "tool"`. Sagax finds them
by calling `recall()` — the same way it finds facts about people, remembered
preferences, and skill guidance sequences. The search is semantic; a query like
"heat water for a hot drink" surfaces `tool.turn_on_kettle.v1` because its
`capability_summary` field ends up in the semantic embedding space.

There is no tool registry that Huginn maintains separately. The tool ontology
is self-organising: Logos writes descriptors at install time, updates them when
tools are deprecated, and the next recall reflects the current state of
capabilities automatically.

### 6.2 The Staging Lifecycle

A new tool's lifecycle:

1. Operator drops a `.py` file with a `HUGINN_MANIFEST` block into `tools/staging/`
2. Logos' next pass scans staging, parses the manifest, and creates a `waiting`
   HTM task with the tool details
3. Sagax sees the staging task in its next context, finds a natural pause in
   the conversation, and presents the tool to the user — what it does, what
   permissions it requests, whether it should be enabled as a live daemon
4. User confirms (or declines)
5. Sagax writes `user_affirmed: true` to the HTM task notebook
6. Logos' next pass reads the affirmation, installs pip dependencies, loads
   the module, writes the LTM descriptor, registers the handler, and optionally
   creates an HTM pipeline task for perception tools
7. If the user said "install it now", Sagax calls `request_early_logos_cycle`
   and Logos runs immediately

The key property of this flow: Sagax is the user-facing conversation layer and
Logos is the installation engine. They communicate through HTM. Neither needs
to know how the other works.

### 6.3 Tool Modes

Tools have three modes reflecting three different operational patterns:

**Callable** is the default. A single `handle(**args)` function called
synchronously, returning a result. Appropriate for tools that do one thing
and return.

**Service** tools are daemon threads. They expose `start(config)`, `stop()`,
and `handle(event)`. They subscribe to the `ActuationBus` and process output
events as they arrive — speech chunks for TTS, display updates for a screen
agent, tick events for a monitoring tool. They read their configuration from
`HTM.states` on every event cycle, so parameter changes (TTS speed, voice
selection) take effect immediately without a restart.

**Provider** tools are LLM inference backends. They expose four callables
matching the `LLMClient` router interface. They are discovered and installed
like any other tool; they just happen to provide the inference substrate that
other parts of the system depend on.

---

## 7. The Communication Model

### 7.1 The ActuationBus

Output events flow through an in-process publish/subscribe bus. The Orchestrator
publishes; live tool daemon threads subscribe with filter dicts. The bus is
non-blocking — full subscriber queues drop events silently rather than blocking
the Orchestrator's token stream processing. A TTS tool that falls behind does
not stall Sagax's generation.

Three event completeness levels reflect the needs of different downstream
consumers:

- `partial` — individual token, emitted on every token while speech is streaming.
  Consumed by avatar lip sync, which needs high-frequency position updates.
- `chunk` — phrase-boundary flush, triggered by punctuation after a minimum
  token count. The natural synthesis unit for TTS — sending full phrases rather
  than individual tokens dramatically improves prosody.
- `full` — complete speech block, written to STM on `</speech>` close. The unit
  Logos consolidates. Sagax only generates full events in STM; the sub-block
  granularity stays on the bus.

### 7.2 The Narrator Grammar

Sagax's output is a structured token stream. Each XML block type has a defined
routing contract:

- `<contemplation>` — buffered until close; written to STM as an output event;
  Logos reads it as part of the episodic record alongside raw tool calls
- `<speech>` — streamed live to TTS as tokens arrive; close writes full event
  to STM and publishes `full` event to ActuationBus
- `<speech_step>` — streamed like speech but sets a pending flag; the
  Orchestrator routes the next `chat()` input as the bound response rather than
  waking Sagax for a new planning cycle
- `<tool_call>` — buffered until close; dispatched through the permission gate;
  creates or updates an HTM task record; results surface as STM events
- `<aug_call>` — buffered until close; dispatched in parallel with per-tool
  timeout budgets; `<aug_result>` injected inline before Sagax generation
  resumes; only `polarity: read` tools eligible
- `<task_update>` — written directly to HTM; no STM event; supports task
  lifecycle actions and HTM States operations (`state_set`, `state_get`,
  `state_delete`)
- `<thinking>` / `<think>` — silently discarded; no STM write, no workbook
  entry, no TTS. Models emit these naturally; the explicit instruction to do
  so was removed from the system prompt when it was found to produce overuse.

### 7.3 Speech Step Suspension

`<speech_step>` deserves particular attention because it changes the
conversational model. Most agent architectures treat user input as something
that arrives at the start of a turn and triggers a new planning cycle. That
model makes it difficult to write skills with interactive beats — the skill
needs to emit a question, wait for an answer, and continue with the answer bound
to a specific variable.

`speech_step` suspends the planning cycle mid-execution. Sagax emits the
question, the Orchestrator sets a pending flag, and the next `chat()` call
routes to `receive_speech_step_response()` instead of waking Sagax for a new
planning cycle. The response is bound to the declared variable name in
`HTM.states`, and Sagax generation resumes with an injected `<speech_step_result>`
block. The skill continues exactly where it stopped.

This makes conversational skills — skills with branching based on user input —
first-class citizens rather than awkward workarounds.

---

## 8. The Instruction System

### 8.1 The Lean Prompt Problem

A system prompt that tries to be both an operating contract and a reference
manual succeeds at neither. The operating contract gets buried in prose; the
reference material is too compressed to be useful when precision matters.
The common solution — shorter prompt, more examples — just shifts the compression
problem.

The right solution separates the two responsibilities entirely. The operating
contract is in the system prompt: grammar, micro-examples for the constructs
most likely to cause first-use stumbles, a topic directory, hard rules. The
reference material is in Muninn LTM, retrievable inline during generation.

### 8.2 The Instruction Artifacts

Eight instruction artifacts are written to Muninn LTM by Logos on first boot,
each covering one operational domain:

- `htm_tasks` — task create/park/resume/complete cycle, persistence semantics,
  workbook recovery path
- `skill_execution` — guidance step interpretation, interactive beats,
  argument reasoning, notebook trace quality
- `memory` — recall parameter decision tree, entity operations, source
  references, the narrow exception for Sagax LTM writes
- `states` — namespace conventions, write/read/delete, provider switching
  end-to-end, states vs tasks
- `live_tools` — daemon lifecycle, configuration without restart, perception
  pipeline management
- `staging` — confirmation conversation flow, permissions walkthrough, urgent
  install path
- `entities` — implied entity accumulation, cross-modal evidence, authority
  tiers, guest grants
- `speech_step` — suspension mechanics, variable binding, interruption recovery,
  timeout and naming edge cases

Each artifact is action-oriented and self-contained — written so it can be
read in isolation, without reference to the system prompt, and acted upon
immediately. Each ends with a cross-reference to related topics so Sagax can
fetch adjacent context without guessing topic names.

Operators can update any artifact by writing a new LTM entry with the same
topic tags. Logos checks for existence before writing and never overwrites
custom content.

### 8.3 `get_instructions` as a Native Tool

`get_instructions(topic)` is registered in the tool manager's polarity map as
`polarity: read`, making it fully eligible for `aug_call` — Sagax can fetch an
instruction artifact inline during generation without interrupting its reasoning
arc. The call does a targeted `RecallQuery` with `topics=["instruction.<topic>"]`
and returns `entry.content` verbatim. The artifact arrives in the `<aug_result>`
block and Sagax continues with the full manual context.

The topic directory in `SAGAX_PLAN_v2` tells Sagax exactly when to fetch each
topic — specific enough that the decision cost is low, not so specific that
Sagax fetches unnecessarily. The rule is: fetch when a task type is unfamiliar,
when precision matters, or before a hard-to-undo operation. Skip for routine
recalls, simple speech, and tool calls already made this session.

---

## 9. Identity and Privacy

### 9.1 Biometric Identity

Artux does not use login screens. Identity is established through signature
matching — voiceprints, faceprints, or device identifiers emitted by perception
tools as structured payload fields. The Perception Manager resolves these
against the entity registry before writing events to STM. Matched events carry
`entity_id`. Unmatched events carry an implied entity identifier and
`signature_resolved: false`.

An implied entity accumulates evidence across events — voice patterns, name
claims, contextual associations with known entities, visual confirmation if
camera is available. When evidence from two or more independent sources
converges, Sagax creates a permanent entity and observes it with the accumulated
evidence. Logos performs post-session resolution, comparing embedded voiceprints
against the signature registry for entities that might have been identified after
the session ended.

Until an entity is confirmed, it has guest access only: no personal calendar
entries, no actuation of personal devices, no private information retrieval.
The system does not guess at identity and proceed; it acknowledges uncertainty
and constrains access accordingly.

### 9.2 Self-Hosted by Design

No component of Artux requires external network access during normal operation.
Muninn uses SQLite. The default inference backend is Ollama running locally.
Perception tools (ASR, vision) run local models. The only outbound connections
are optional: provider tools that call external APIs (Anthropic, OpenAI) if the
operator chooses to install them.

The privacy boundary is the machine. What Artux sees, hears, and knows stays
on the machine unless the operator explicitly chooses to send it elsewhere
through a tool they install and confirm.

---

## 10. From ANIMA to Artux — A Design Lineage

The table below maps ANIMA's original conceptual propositions to their
Artux implementation equivalents. It is a record of what transferred,
what transformed, and what was added.

| ANIMA concept | ANIMA framing | Artux implementation |
|---|---|---|
| Core motto | *Percipio, ita cogito agere* | Exilis — continuous perception, decides when to wake Sagax |
| Reactive vs proactive | System attends rather than waits | 5ms Exilis poll loop; no wake word |
| Sagax | Stable backbone, semi-fixed | Sagax — reasoning and planning agent, event-driven |
| Mirai | Adaptive imagination, self-extending | Distributed: Logos (evolution), tool ecosystem (extensibility), provider model (backend), instruction artifacts (knowledge) |
| Orchestrator | Cognitive kernel, sequences P→T→A | Orchestrator — routing bridge, no cognitive logic |
| ToolFactory | Dynamic capability generator | Staging workflow — manifest → confirm → install → LTM |
| Pipeline Registry | Catalog of cognitive flows | LTM tool/skill/pipeline artifacts, discovered via recall() |
| Diary of thoughts | Internal narrative, first-class output | `contemplation` blocks → STM → Logos reads for LTM |
| Continuity of state | Narrative thread across commands | STM (event log) + consN (rolling summary) + LTM |
| Dual spirit balance | Sagax and Mirai in tension | Single architecture with adaptive memory layer |
| Transparency imperative | Internal states must be inspectable | Structural: contemplation → STM, append-only events, full workbook |
| Self-state | Agent's current internal condition | HTM: Tasks (lifecycle) + ASC (session) + States (live params) |
| Identity | Implicit (not addressed) | Biometric signatures → entity grants → permission gate |
| Inference backend | Assumed fixed | Provider tools — swappable via two state_set writes |

The most significant departure from ANIMA is the memory architecture.
ANIMA describes memory as "short-term, long-term, and event RAM" without
specifying how these tiers interact, who writes to them, or how knowledge
moves between them. Artux's answer — Muninn as a passive store with
Logos as the sole LTM author, STM as an append-only log flushed only
after verified consolidation, and HTM States as live operational ground
truth — took the bulk of the implementation effort and most of the
architectural decisions documented in this paper.

The second significant departure is that Artux treats inference as a
service, not a substrate. ANIMA assumes a fixed LLM backbone. Artux
makes the LLM itself a tool — installed, configured, and swappable
through the same mechanisms as a kettle controller. This was not in the
original ANIMA conception; it emerged from the implementation insight
that hard-coding a backend creates the same rigidity that ANIMA was
designed to escape.

## 11. What Artux Is Not

**Not a chatbot with plugins.** A chatbot responds to turns. Artux runs
continuously, perceives its environment, maintains structured knowledge, and
acts proactively. The conversational interface is one modality among several.

**Not a local copy of a cloud product.** Cloud AI products are optimised for
breadth, safety at scale, and session-local interaction. Artux is optimised
for depth of knowledge about one environment and one set of people, continuity
across sessions, and the specific trust model of a personal home.

**Not a framework.** A framework gives you abstractions and lets you fill in
the logic. Artux has opinions — about memory architecture, cognitive component
separation, tool lifecycle, identity management, and what should and should not
be in a system prompt. Those opinions are the product.

**Not finished.** Async STM notification (replacing Exilis polling),
multi-agent write conflict resolution, the output event refactor (partial/chunk/full
in STM not just the bus), avatar integration, and the full skill synthesis
evaluation pipeline are pending. The architecture is stable; the implementation
continues.

---

## 12. Design Principles in Summary

**Memory is metabolic.** It breathes — growing through consolidation, shrinking
through forgetting, updating through reinforcement. The forgetting is not a
failure mode; it is how the system stays usable as it accumulates years of
experience.

**Raw events are ground truth.** Compression, summarisation, and narrative
construction are editorial processes that aid reasoning. They are not the
record. The record is the append-only event log. Nothing compresses it away.
Logos flushes it only after verified writes to durable LTM.

**All cognitive decisions are LLM calls.** No hardcoded classifiers, no rule
engines, no keyword matching. The small model does triage. The medium model
reasons and plans. The large model consolidates and synthesises. The models
are interchangeable components; the architecture is not.

**Capabilities are discovered, not enumerated.** Sagax finds out what it can
do the same way it finds out what it knows — by asking Muninn. New tools become
available through a structured lifecycle that includes user confirmation. The
system's capability set is as dynamic as its knowledge.

**Providers are tools.** LLM inference is not special infrastructure. It is a
tool with a known interface, installed through the same lifecycle as every other
capability, configurable through the same state mechanism. Switching inference
backends does not require changing any system code.

**The system prompt is a contract, not a manual.** Sagax's operating contract
is 84 lines. Everything else — the full detail of how to manage tasks, interpret
skills, handle entities, switch providers — lives in Muninn LTM and arrives on
demand inline during generation. The contract tells you what blocks to emit and
where to look for more. The manuals tell you how.

**Identity is authority.** Permissions are not session grants from a login.
They are properties of an identified entity, established through biometric
confirmation and maintained as part of that entity's ledger. An unknown presence
gets guest access. A confirmed presence gets their configured permissions.
The transition between the two is evidence-based, not credential-based.

**Artux earns trust by evidence.** Nothing becomes a capability until it has
proven itself. Nothing becomes a known entity until cross-modal evidence
converges. Nothing is stored in LTM until Logos has read it, evaluated it, and
judged it worth keeping. The system's confidence in everything it knows and
everything it can do is explicit, tracked, and subject to decay.

---

*Huginn thinks. Muninn remembers. Together they give Odin sight.*

---

**Repositories**
- `github.com/oumo-os/artux-muninn` — memory module (standalone library)
- `github.com/oumo-os/artux-huginn` — cognitive module (requires Muninn)

**License:** MIT
