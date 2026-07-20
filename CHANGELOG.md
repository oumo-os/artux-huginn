# Changelog

All notable changes to `artux-huginn` are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/), versioned by
[Semantic Versioning](https://semver.org/) — major for breaking
architectural changes, minor for additive capability, patch for docs/fixes.

---

## v3.1.0 — Performance: Denoise, Dedup, Hot-Reload

*191 tests.*

### Added
- `runtime/denoise.py` — `DenoiseEngine` for collapsing consecutive identical
  sensor events into range format before any LLM sees the data. Zero LLM
  cost (pure Python MD5 fingerprinting). 10-50x token savings on
  IoT-heavy setups
- `Exilis.denoise_engine` — denoises new events before triage using
  ConsN boundary as watermark. Reduces Exilis LLM input tokens
- `Logos._deduplicate_batch()` — pre-consolidation noise removal. Groups
  consecutive events by fingerprint (source + type + normalised payload),
  collapses runs into span events with metadata (count, from_ts, to_ts).
  Reduces Logos consolidation LLM cost 10-50x
- `Logos._source_cleanup_pass()` — deletes orphaned sources
  (retention="delete_when_orphaned") with no live LTM references.
  Prevents unbounded disk growth from captured media
- `ToolManager.uninstall_tool()` — removes tool from runtime, stops
  actuation daemon, unloads module from sys.modules
- `ToolManager.reload_tool()` — hot-reload: uninstall then re-install
  from same source path. Service tool daemons restarted, callable tool
  handler references updated immediately
- `ToolManager.set_actuation_manager()` — register ActuationManager
  reference for service tool lifecycle management
- `ToolManager.install_tool(skip_dependencies=)` — new parameter to skip
  pip install during reload (default True for fast offline reload)

### Changed
- `Logos._pass()` — dedup runs on raw batch before segmentation; source
  cleanup runs after skill synthesis; health event includes dedup and
  cleanup counts
- `llm/prompts.py` — ported note_type values, notebook_entry steps
  section, and improved skill completion docs from anima (SAGAX STATE
  retained)

### Design Decision: No pluggable memory backend (for now)
Huginn has 57 coupling points to Muninn across 7 files (34 actual API
calls: recall, consolidate_ltm, store_ltm, observe_entity, resolve_entity,
delete_orphaned, ToolExecutor, get_tools). All imports are soft/lazy —
Huginn runs without Muninn installed. A formal `MemoryBackend` protocol
would require abstracting 7 API methods with 34 call sites, which is a
large surface for unclear benefit. Muninn is the canonical backend
(see: muninn-ltm merge, commit `2a8bb1c`). If Huginn graduates from
"demo" to "framework", a protocol-based abstraction would make sense
at that point.

### Removed
- `tools/staging/.huginn_known.json` — duplicate calendar entry fixed

---

## v3.0.1 — Model Recommendation Refresh

*Docs only — no code or test changes.*

### Changed
- `CognitiveModule.md §4.3`, `README.md` — computation profile model
  recommendations updated (LFM2.5, Qwen3.5/3.6, Gemma4 replacing the
  earlier Qwen2.5/Llama-3.x generation)

### Added
- LFM2.5 Pythonic tool-call format compatibility note — LFM2.5 defaults to
  `<|tool_call_start|>`/`<|tool_call_end|>` Pythonic syntax rather than
  JSON; needs an explicit system-prompt override for `complete_tools`'
  JSON-extraction fallback path
- License verification caveat for commercial deployments — some model
  releases use dual-use or revenue-gated terms rather than a permissive
  license; verify against the model card at deployment time, not against
  static documentation

---

## v3.0.0 — Capture/Process Perception Architecture

*192 tests.*

### Breaking
- Perception model — step-based pipelines replaced with a capture/process
  split. `PerceptionManager` no longer runs per-sensor pipelines directly;
  it fans `CaptureFrame` objects out from capture tools to any number of
  process tools that declare `consumes`
- `tool_asr_moonshine.py` — `direction` changed from `input` to `process`;
  no longer depends on `sounddevice` (moved to a separate capture tool)

### Added
- `CaptureFrame` dataclass (`kind`, `data`, `ts`, `meta`)
- `ToolManifest.emits` / `ToolManifest.consumes` fields;
  `direction: capture` / `direction: process` manifest values;
  `is_capture_tool` / `is_process_tool` properties
- `PerceptionManager.start_capture_fanout()` — one dispatch thread per
  capture tool, routing frames to every processor that consumes that kind;
  one processor's failure never blocks the others
- `ToolManager.world_descriptors()` — enumerate installed world-tier tools
  by direction
- `tool_capture_microphone.py` (builtin) — pure `sounddevice` capture,
  emits `audio_chunk`, zero processing logic
- `tool_capture_camera.py` (builtin) — pure `opencv-python` capture,
  emits `video_frame`
- `tool_identity_voice.py` (staging) — ECAPA-TDNN speaker embeddings via
  SpeechBrain, consumes `audio_chunk`; corrected from a non-functional
  draft that called nonexistent Muninn methods
- `tool_identity_face.py` (staging) — InsightFace ArcFace 512-dim
  embeddings, consumes `video_frame`; replaces a placeholder that
  generated random vectors seeded by pixel sum rather than real embeddings

### Why
Moonshine ASR and a voice-identity tool can now consume the exact same
microphone capture stream with zero duplication — one capture loop, N
processors. Adding a new sensor consumer requires zero changes to
`PerceptionManager` or the capture tool; swapping capture hardware
requires zero changes to any processor.

---

## v2.4.0 — Notebooks and Cognitive States

*175 tests.*

### Added
- Typed notebook entries — `note_type` parameter on `Task.add_note` /
  `HTM.note` (`note`, `decision`, `result`, `observation`, `evidence`,
  `checkpoint`, `logos_examined`)
- `HTM.notebook_entries(task_id, note_type, limit)` — targeted query
  helper; Logos synthesis scan filters by type instead of scanning text
  for marker strings
- `action: "note"` in `<task_update>` — explicit typed note writes from
  Sagax
- `sagax.state` in `HTM.states` — written by Orchestrator on every
  Narrator block transition (`thinking` / `contemplating` / `speaking` /
  `aug_wait` / `tool_wait` / `step_wait` / `consolidating` / `sleep`)
- `logos.state` in `HTM.states` — written through the Logos pass
  lifecycle (`consolidating` / `synthesising` / `installing` / `sleep`)

### Changed
- `HTMStates.summary()` — cognitive states (`sagax.state`, `logos.state`)
  always surface first in the STATE SNAPSHOT injected into Sagax context
- `htm_tasks`, `skill_execution` instruction artifacts — documented note
  types with copy-adaptable examples; documented the `notebook_entry: true`
  step contract and checkpoint note format (`[step:N]` prefix)

### Fixed
- Both early-return paths in `Logos._pass()` (empty batch, error) now set
  `logos.state = "sleep"` before returning — previously left the state
  stuck on `"consolidating"` after a no-events pass

---

## v2.3.0 — Per-Role Model Assignment, Spec Rewrite

*156 tests.*

### Changed
- GGUF assignment — single-file "first file wins" detection replaced with
  `_assign_gguf_models()`: `models/models.yaml` → prefix-named files
  (`exilis_*.gguf`, `sagax_*.gguf`, `logos_*.gguf`) → size-ordered fallback
  → single shared file
- `CognitiveModule.md`, `Orchestrator.md`, `CognitiveModule_Addendum.md` —
  full v2 rewrite; addendum collapsed to a short historical note (its five
  corrections are fully incorporated into the main spec)

### Added
- `_role` injection through `_ProviderToolAdapter` and provider tool calls
  — providers can read `HTM.states["tool.llm.*.model_path.{role}"]` to
  serve a different model per agent role from one provider tool

---

## v2.2.0 — Builtin Default Stack

*156 tests.*

### Added
- `tools/builtin/` directory — auto-installed on first Logos pass, no
  staging confirmation required (distinct from `tools/staging/`)
- `tool_llm_llamacpp.py` — in-process GGUF inference provider
- `tool_tts_kokoro.py` — Kokoro ONNX TTS daemon
- `tool_asr_moonshine.py` — Moonshine ONNX ASR (first version;
  `direction: input`, later split in v3.0.0)
- `tool_ui_text.py` — terminal text I/O, the zero-dependency universal
  test harness
- `Logos._ensure_default_tools()` — first-pass builtin installation;
  skips tools with unavailable dependencies gracefully
- GGUF auto-detection in `build_huginn()` (single-file only; per-role
  assignment arrived in v2.3.0)

---

## v2.1.0 — Exilis Loop, Cognitive Correctness Fixes

*156 tests.*

### Changed
- Exilis — 5ms fixed poll interval replaced with a continuous loop gated
  on `events_pending()`; `poll_interval_s` renamed to `idle_yield_s`
  (default `0.0`, cooperative `os.sched_yield()`)
- Nudge interrupt handling — `status: "suspended"` replaced with
  `status: "full", interrupted: true` for both `<speech>` and
  `<speech_step>`; consistent with "everything in STM is a full event"
- Skill synthesis — hardcoded threshold counters (run count, day spread)
  replaced with observation-driven HTM evaluation tasks:
  `_identify_synthesis_candidates`, `_advance_evaluation_tasks`,
  `_propose_skill`. Evidence accumulates in task notebooks across
  sessions; no statistical gate, only LLM judgment against accumulated
  evidence
- Signature resolution — `resolve_entity_by_embedding` (never-existed
  Muninn API) replaced with `recall(RecallQuery(topics=["signature", kind]))`
  plus fallback to text-clue entity resolution

### Added
- `STMStore.events_pending(since_id)` — single `SELECT EXISTS` query,
  Exilis's own attention gate, no row loading
- `Orchestrator.on_consn_updated()` — wired to `Sagax._update_cons_n`;
  triggers ASC GC on every consN update (previously designed but never
  actually fired — session context accumulated indefinitely)

### Fixed
- ASC (ActiveSessionCache) garbage collection now actually runs

---

## v2.0.1 — Documentation Overhaul

*Docs only — no code or test changes.*

### Changed
- `README.md` — full rewrite reflecting the v2.0.0 architecture
  (states, providers, ActuationBus, lean prompt); previously described a
  pre-v2 constructor signature and had completed items still marked `[ ]`

### Added
- `ARTUX_WHITEPAPER.md` — "Preface and Origin" section tracing the
  ANIMA → Artux lineage, including the wake-word frustration that
  originated the project and the Cartesian motto
  (*Percipio, ita cogito agere*)
- §10 "From ANIMA to Artux — A Design Lineage" table mapping every ANIMA
  concept to its Artux implementation equivalent

---

## v2.0.0 — States, Actuation, Providers, Lean Prompt

*132 tests.*

### Breaking
- `LLMClient` constructor signature changed — `role` and `htm` are now
  primary; the old direct kwargs (`backend`, `model`, `host`, `api_key`,
  `temperature`, `timeout`) are retained only as fallback values for
  `_BuiltinProvider`

### Added
- `HTM.states` — flat key-value store for live operational parameters,
  a third surface on HTM alongside Tasks and ASC. Namespace convention:
  `{tool_id}.{param}` / `{role}.{param}`. Dirty-tracked; persisted to LTM
  by Logos at session end
- `ToolManifest` fields: `mode` (`callable`/`service`/`provider`),
  `direction` (`input`/`output`/`io`), `subscriptions`, `states`
- `ActuationBus` — in-process pub/sub for output events
  (`partial`/`chunk`/`full`), non-blocking, filter-based subscription
- `ActuationManager` — live tool daemon lifecycle; `start_from_htm()`
  auto-starts registered live tools at boot
- Speech chunker in Orchestrator — publishes at phrase boundaries for
  natural TTS synthesis units
- Provider tool registration — `LLMClient.register_provider()`, called
  automatically by `ToolManager.install_tool()` for `mode: provider` tools
- `tool_llm_ollama.py`, `tool_llm_anthropic.py` — first two concrete
  provider tools
- 8 LTM instruction artifacts (`htm_tasks`, `skill_execution`, `memory`,
  `states`, `live_tools`, `staging`, `entities`, `speech_step`)
- `get_instructions` / `htm_state_get` native tools — `polarity: read`,
  eligible for `aug_call`, bypass Muninn's `ToolExecutor`

### Changed
- `llm/client.py` — full rewrite; `LLMClient` is now a thin router reading
  `{role}.provider` / `{role}.model` / `{role}.temperature` from
  `HTM.states` at every call. `_BuiltinProvider` retained as pre-install
  fallback (Ollama, Anthropic, OpenAI, LM Studio, llama.cpp)
- `SAGAX_PLAN_v2` — system prompt cut from 256 lines / 12 KB to 84 lines /
  5 KB (58% reduction). Grammar table, six micro-examples, topic
  directory with fetch-when triggers, seven hard rules — all detailed
  walkthroughs moved to LTM instruction artifacts
- `<think>` / `<thinking>` — both tags now silently discarded by the
  Orchestrator's tag-state machine; no STM write, no workbook entry

---

## v1.0.0 — Foundation

*69 tests. Baseline prior to the changes tracked above.*

- Vigil / ActiveSessionCache layer — 7-surface design (`workbook`,
  `hot_entities`, `hot_capabilities`, `hot_topics`, `hot_recalls`,
  `hot_parameters`, `hot_state`)
- `speech_step` conversational suspension mechanism
- Startup procedure execution (`Sagax.execute_startup_procedure`)
- Muninn config-in-LTM pattern (`Logos._ensure_system_config`,
  `Orchestrator._recall_system_config` / `_apply_system_config`)
- `RecallQuery` upgrade — all `muninn.recall()` call sites moved from
  plain-string queries to structured `RecallQuery` objects
- Multi-arc Logos consolidation — LLM segments STM batches into coherent
  arcs rather than one monolithic narrative per pass
- Actuation Manager and `HTM.states` designed in this cycle but not yet
  implemented (landed in v2.0.0 and v2.2.0 respectively)
