# ADR: Exilis Triage Label Rename

**Date:** 2026-07-21
**Status:** Accepted
**Deciders:** oumo

## Context

Exilis is the attention gate of the Huginn cognitive layer. It triages incoming STM events into one of three signals that determine whether and how Sagax wakes. The original labels were `ignore`, `act`, and `urgent`.

Two problems emerged:

1. **`ignore` is semantically misleading.** When Exilis classifies an event as "ignore", the event is NOT dropped from STM. It persists and will be seen by Sagax on its next wake (for any reason) and by Logos during consolidation. "ignore" implies discarding, but the actual behavior is deferral. This mismatch can cause the LLM to over-dismiss events it should hold.

2. **`urgent` doesn't match its behavior.** The label "urgent" is a severity descriptor, but the actual behavior is interruption — Sagax is stopped mid-stream, pending work is parked, and a focused response mode is entered. The codebase already uses "interrupt" as the behavior name (`interrupted: True`, `urgent_interrupt` reason, `nudge_interrupt`).

3. **`task_expired` was misclassified.** Expired tasks were in the "ignore" list, but Sagax presents them opportunistically during normal reasoning cycles — they're not noise, they're deferred context.

## Decision

Rename the triage labels:

| Before | After | Rationale |
|---|---|---|
| `ignore` | `hold` | "hold" communicates deferral: real event, Sagax doesn't need to wake now, will be seen on next cycle |
| `act` | `act` | unchanged |
| `urgent` | `interrupt` | label matches behavior: Sagax is interrupted mid-stream |

Move `subtype=task_expired` from hold to act — Sagax presents expired tasks opportunistically, not as alarms.

## Semantics

**Exilis is an attention gate, not an event filter.** All triage signals leave events in STM. The signal controls *when* Sagax wakes, not *whether* the event is stored.

- `hold` — "I see this, it's real, but Sagax doesn't need to wake for it right now." Sagax and Logos will see it on their next cycle.
- `act` — "This warrants attention on Sagax's next natural cycle."
- `interrupt` — "Stop Sagax now. This needs immediate attention."

## Scope

### Changed
- `TriageLabel` enum: `HOLD`, `ACT`, `INTERRUPT`
- `WakeSignal.priority`: `"normal"` | `"interrupt"` (was `"urgent"`)
- `EXILIS_TRIAGE_v1` prompt: full rewording
- Sagax mode selector: checks `priority == "interrupt"` (was `"urgent"`)
- Orchestrator: emits `priority: "interrupt"` in nudge signal
- All tests, benchmarks, and sim tool hints updated

### Not changed
- User utterance "urgent" in tool staging prompts (natural language, not a triage label)
- `on_urgent` callback parameter name (internal API, behavior unchanged)
- Event payload `reason: "urgent_interrupt"` (internal metadata string)
- Logos `request_early_cycle()` trigger ("when user says 'urgent'" — user utterance)

## Files

```
anima/cognitive/agents/exilis.py        — TriageLabel enum, dispatch, validation
anima/cognitive/agents/sagax.py         — WakeSignal.priority, mode selector
anima/cognitive/runtime/orchestrator.py — nudge signal, comments
anima/cognitive/llm/prompts.py          — EXILIS_TRIAGE_v1 prompt text
anima/__init__.py                       — docstring
tests/test_anima.py                     — TriageLabel references
tests/test_behavioral.py                — triage mock returns
tests/bench_cognitive.py                — benchmark definitions
tools/sim/tool_sim_voice.py             — hint strings
tools/sim/tool_sim_social.py            — hint strings
tools/sim/tool_sim_desktop.py           — hint strings
tools/sim/tool_sim_asr.py               — hint strings
```

## Consequences

- LLM triage prompt uses words that match actual behavior — reduces semantic drift
- "hold" is a weaker cognitive frame than "ignore" — LLM is less likely to prematurely dismiss events
- "interrupt" makes the severity of the signal explicit — it's not just "urgent", it's "stop what you're doing"
- `task_expired` → `act` means expired tasks get Sagax attention on next cycle, enabling opportunistic presentation
- Internal event metadata (`reason: "urgent_interrupt"`) retains "urgent" as a historical marker — no semantic confusion since it's a payload value, not a triage label
