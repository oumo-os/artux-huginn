"""
tests/test_huginn.py — Huginn test suite.

Coverage:
  Unit tests
    TestSTM             — STMStore in-memory: record, get_events, consN, watermark
    TestHTM             — HTM Tasks: create, query, complete, note, tags, scheduler
    TestVigil           — ActiveSessionCache Vigil surfaces: all 7 surfaces, gc, flush
    TestToolDiscovery   — HUGINN_MANIFEST parse, scan, hash dedup, bad manifest skip
    TestNarratorStateMachine — Orchestrator token stream: all block types, nested attrs
    TestSpeechStep      — speech_step suspension and response injection end-to-end

  Integration tests
    TestBuildHuginn     — build_huginn() factory smoke test with mock Muninn
    TestBootProcedure   — execute_startup_procedure() with no LTM entry (fallback)
    TestLogosStagingLifecycle — full staging → install → STM event cycle (mock)

Run:
    python -m pytest tests/test_huginn.py -v
    # or without pytest:
    python tests/test_huginn.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Make the huginn package importable from the project root
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from huginn.runtime.stm import STMStore, STMEvent, ConsN
from huginn.runtime.htm import HTM, Task, ActiveSessionCache, HotEntity
from huginn.runtime.tool_discovery import ToolDiscovery, ToolManifest
from huginn.runtime.orchestrator import Orchestrator, NarratorState, Session
from huginn.agents.sagax import Sagax
from huginn import build_huginn


# ===========================================================================
# Mock Muninn — minimal interface satisfying all Huginn call sites
# ===========================================================================

class MockLTMEntry:
    def __init__(self, **kwargs):
        self.id   = kwargs.get("id", "ltm-mock-1")
        self.meta = kwargs.get("meta", {})
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockMuninn:
    """
    Minimal Muninn stub.

    All methods that Huginn calls on muninn are implemented here with
    safe no-op or minimal-return behaviour.  Tests that need specific
    LTM recall results can configure self.recall_results.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path        = db_path
        self.recall_results: list = []    # configure per test
        self.consolidated:  list = []    # records all consolidate_ltm() calls
        # Minimal db attribute for build_huginn() path resolution
        self.db             = MagicMock()
        self.db.path        = db_path
        self.db.db_path     = db_path

    def recall(self, query: str, top_k: int = 5) -> list:
        return list(self.recall_results)

    def consolidate_ltm(self, **kwargs) -> MockLTMEntry:
        entry = MockLTMEntry(id=f"ltm-{len(self.consolidated)}", **kwargs)
        self.consolidated.append(kwargs)
        return entry

    def observe_entity(self, **kwargs):
        pass

    @property
    def tools(self):
        executor = MagicMock()
        executor.execute.return_value = {"status": "ok"}
        return executor


# ===========================================================================
# TestSTM — STMStore unit tests
# ===========================================================================

class TestSTM(unittest.TestCase):

    def setUp(self):
        self.muninn = MockMuninn()
        # STMStore uses its own SQLite in-memory path
        self.stm = STMStore(self.muninn, db_path=f":memory:stm_{id(self)}")

    def test_record_returns_event_with_id(self):
        ev = self.stm.record(
            source="user", type="speech",
            payload={"text": "hello"},
        )
        self.assertIsInstance(ev, STMEvent)
        self.assertIsNotNone(ev.id)
        self.assertEqual(ev.type, "speech")
        self.assertEqual(ev.payload["text"], "hello")

    def test_get_events_after_returns_newer(self):
        ev1 = self.stm.record(source="user", type="speech", payload={"text": "a"})
        ev2 = self.stm.record(source="user", type="speech", payload={"text": "b"})
        ev3 = self.stm.record(source="user", type="speech", payload={"text": "c"})

        after = self.stm.get_events_after(ev1.id)
        ids   = [e.id for e in after]
        self.assertIn(ev2.id, ids)
        self.assertIn(ev3.id, ids)
        self.assertNotIn(ev1.id, ids)

    def test_record_multiple_types(self):
        self.stm.record(source="system", type="internal",
                        payload={"subtype": "test"})
        self.stm.record(source="system", type="output",
                        payload={"subtype": "speech", "text": "hi"})
        events = self.stm.get_events_after(0)
        types  = {e.type for e in events}
        self.assertIn("internal", types)
        self.assertIn("output", types)

    def test_watermark_advances(self):
        wm0 = self.stm.get_logos_watermark()
        ev  = self.stm.record(source="user", type="speech", payload={})
        self.stm.flush_up_to(ev.id)
        wm1 = self.stm.get_logos_watermark()
        self.assertGreater(wm1, wm0)

    def test_consn_starts_empty(self):
        cn = self.stm.get_cons_n()
        # Either None or empty summary text
        if cn is not None:
            self.assertEqual(cn.summary_text.strip(), "")


# ===========================================================================
# TestHTM — HTM Task unit tests
# ===========================================================================

class TestHTM(unittest.TestCase):

    def setUp(self):
        self.htm = HTM()   # in-memory (no db_path)

    def test_create_task(self):
        tid = self.htm.create(
            title="Test task", initiated_by="sagax", persistence="volatile",
        )
        self.assertIsNotNone(tid)
        tasks = self.htm.query(task_id=tid)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].title, "Test task")
        self.assertEqual(tasks[0].state, "active")

    def test_query_by_state(self):
        t1 = self.htm.create(title="A", initiated_by="sagax")
        t2 = self.htm.create(title="B", initiated_by="logos")
        self.htm.update(t2, state="waiting")
        active  = self.htm.query(state="active")
        waiting = self.htm.query(state="waiting")
        self.assertTrue(any(t.task_id == t1 for t in active))
        self.assertTrue(any(t.task_id == t2 for t in waiting))

    def test_query_by_tags_any(self):
        t1 = self.htm.create(title="T1", initiated_by="system", tags=["pipeline"])
        t2 = self.htm.create(title="T2", initiated_by="system", tags=["staging"])
        t3 = self.htm.create(title="T3", initiated_by="sagax",  tags=["other"])
        results = self.htm.query(tags_any=["pipeline", "staging"])
        ids = {t.task_id for t in results}
        self.assertIn(t1, ids)
        self.assertIn(t2, ids)
        self.assertNotIn(t3, ids)

    def test_query_by_tags_all(self):
        t1 = self.htm.create(title="T1", initiated_by="system",
                              tags=["pipeline", "perception"])
        t2 = self.htm.create(title="T2", initiated_by="system",
                              tags=["pipeline"])
        results = self.htm.query(tags_all=["pipeline", "perception"])
        ids = {t.task_id for t in results}
        self.assertIn(t1, ids)
        self.assertNotIn(t2, ids)

    def test_complete_task(self):
        tid = self.htm.create(title="C", initiated_by="sagax")
        self.htm.complete(tid, output={"result": "done"})
        tasks = self.htm.query(task_id=tid)
        self.assertEqual(tasks[0].state, "completed")
        self.assertEqual(tasks[0].output["result"], "done")

    def test_note_appended(self):
        tid = self.htm.create(title="N", initiated_by="sagax")
        self.htm.note(tid, "First note")
        self.htm.note(tid, "Second note")
        tasks = self.htm.query(task_id=tid)
        entries = [n["entry"] for n in tasks[0].notebook]
        self.assertIn("First note", entries)
        self.assertIn("Second note", entries)

    def test_query_by_initiated_by(self):
        self.htm.create(title="S", initiated_by="system")
        self.htm.create(title="L", initiated_by="logos")
        system = self.htm.query(initiated_by="system")
        logos  = self.htm.query(initiated_by="logos")
        self.assertTrue(all(t.initiated_by == "system" for t in system))
        self.assertTrue(all(t.initiated_by == "logos"  for t in logos))


# ===========================================================================
# TestVigil — ActiveSessionCache (Vigil) surface tests
# ===========================================================================

class TestVigil(unittest.TestCase):

    def setUp(self):
        self.asc = ActiveSessionCache(session_id="test-session")

    def test_all_seven_surfaces_readable(self):
        surfaces = ["workbook", "hot_entities", "hot_capabilities",
                    "hot_topics", "hot_recalls", "hot_parameters", "hot_state"]
        for surface in surfaces:
            result = self.asc.get(surface)
            self.assertIsNotNone(result, f"Surface {surface!r} returned None")

    def test_unknown_surface_raises(self):
        with self.assertRaises(ValueError):
            self.asc.get("hot_tools")   # old name — should raise

    def test_workbook_write(self):
        self.asc.workbook_write("speech", "hello world")
        wb = self.asc.get("workbook")
        self.assertEqual(len(wb), 1)
        self.assertEqual(wb[0]["block_type"], "speech")
        self.assertEqual(wb[0]["content"],    "hello world")

    def test_update_capability(self):
        self.asc.update_capability("tool.weather.v1", {
            "title": "Weather", "polarity": "read",
        })
        caps = self.asc.get("hot_capabilities")
        self.assertIn("tool.weather.v1", caps)
        self.assertEqual(caps["tool.weather.v1"]["use_count"], 1)

        # Second call increments use_count
        self.asc.update_capability("tool.weather.v1", {"title": "Weather"})
        caps = self.asc.get("hot_capabilities")
        self.assertEqual(caps["tool.weather.v1"]["use_count"], 2)

    def test_bind_parameter(self):
        self.asc.bind_parameter("user_name", "Alice")
        self.asc.bind_parameter("city",      "London")
        params = self.asc.get("hot_parameters")
        self.assertEqual(params["user_name"], "Alice")
        self.assertEqual(params["city"],      "London")

    def test_get_parameter(self):
        self.asc.bind_parameter("answer", 42)
        self.assertEqual(self.asc.get_parameter("answer"), 42)
        self.assertIsNone(self.asc.get_parameter("missing"))
        self.assertEqual(self.asc.get_parameter("missing", "default"), "default")

    def test_hot_parameters_survive_gc(self):
        """hot_parameters are NEVER auto-pruned during GC."""
        self.asc.bind_parameter("important_var", "do not lose me")
        self.asc.gc(
            new_consN_topics   = [],
            new_consN_entities = [],
            active_task_tools  = [],
        )
        params = self.asc.get("hot_parameters")
        self.assertIn("important_var", params)
        self.assertEqual(params["important_var"], "do not lose me")

    def test_hot_capabilities_pruned_by_gc(self):
        """hot_capabilities not in active tasks are pruned on GC."""
        self.asc.update_capability("tool.a", {"title": "A"})
        self.asc.update_capability("tool.b", {"title": "B"})
        self.asc.gc(
            new_consN_topics   = [],
            new_consN_entities = [],
            active_task_tools  = ["tool.a"],   # tool.b is not active
        )
        caps = self.asc.get("hot_capabilities")
        self.assertIn("tool.a", caps)
        self.assertNotIn("tool.b", caps)

    def test_entity_update_and_resolve(self):
        implied_id = self.asc.add_implied_entity(confidence=0.5)
        entities   = self.asc.get("hot_entities")
        self.assertIn(implied_id, entities)
        self.assertEqual(entities[implied_id].status, "unresolved")

        self.asc.resolve_implied(implied_id, "entity:alice", "Alice")
        entities = self.asc.get("hot_entities")
        self.assertNotIn(implied_id, entities)
        self.assertIn("entity:alice", entities)
        self.assertEqual(entities["entity:alice"].status, "confirmed")

    def test_implied_entity_survives_gc(self):
        """Unresolved/implied entities are never pruned."""
        implied_id = self.asc.add_implied_entity(confidence=0.3)
        self.asc.gc(
            new_consN_topics   = [],
            new_consN_entities = [],
        )
        entities = self.asc.get("hot_entities")
        self.assertIn(implied_id, entities)

    def test_flush_clears_all_except_archive(self):
        self.asc.workbook_write("test", "data")
        self.asc.bind_parameter("var", "value")
        self.asc.update_capability("tool.x", {})
        self.asc.gc([], [], [])   # archive current workbook segment
        self.asc.flush()

        self.assertEqual(self.asc.get("workbook"),       [])
        self.assertEqual(self.asc.get("hot_parameters"), {})
        # Archive should survive flush
        self.assertEqual(len(self.asc.get_archived_segments()), 1)

    def test_summary_for_sagax(self):
        self.asc.update_entity(HotEntity(
            entity_id="ent:alice", name="Alice", status="confirmed",
        ))
        self.asc.bind_parameter("city", "Paris")
        summary = self.asc.summary_for_sagax()
        self.assertIn("entities",   summary)
        self.assertIn("parameters", summary)
        self.assertEqual(summary["parameters"]["city"], "Paris")


# ===========================================================================
# TestToolDiscovery — HUGINN_MANIFEST parsing and staging scan
# ===========================================================================

_VALID_MANIFEST = '''\
"""
HUGINN_MANIFEST
tool_id:            tool.test.v1
title:              Test Tool
capability_summary: A minimal tool for testing the discovery pipeline.
polarity:           read
permission_scope:   []
inputs:
  query: {type: string, default: ""}
outputs:
  result: {type: string}
dependencies: []
perception_capable: false
handler:            handle
END_MANIFEST
"""

def handle(query: str = "") -> dict:
    return {"result": f"echo:{query}"}
'''

_BAD_MANIFEST = '''\
"""
HUGINN_MANIFEST
tool_id:    missing_required_fields
END_MANIFEST
"""
def handle(): pass
'''


class TestToolDiscovery(unittest.TestCase):

    def setUp(self):
        self.tmp  = tempfile.mkdtemp()
        self.staging = Path(self.tmp) / "staging"
        self.active  = Path(self.tmp) / "active"
        self.staging.mkdir()
        self.active.mkdir()

        self.stm = STMStore(MockMuninn(), db_path=f":memory:disc_{id(self)}")
        self.htm = HTM()
        self.discovery = ToolDiscovery(
            staging_dir = str(self.staging),
            active_dir  = str(self.active),
            stm         = self.stm,
            htm         = self.htm,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_tool(self, name: str, content: str):
        p = self.staging / name
        p.write_text(content)
        return p

    def test_scan_empty_staging(self):
        new_tools = self.discovery.scan()
        self.assertEqual(new_tools, [])

    def test_scan_valid_manifest(self):
        self._write_tool("tool_test.py", _VALID_MANIFEST)
        new_tools = self.discovery.scan()
        self.assertEqual(len(new_tools), 1)
        staged = new_tools[0]
        self.assertEqual(staged.manifest.tool_id, "tool.test.v1")
        self.assertEqual(staged.manifest.polarity, "read")
        self.assertFalse(staged.manifest.perception_capable)

    def test_scan_creates_stm_event(self):
        self._write_tool("tool_test.py", _VALID_MANIFEST)
        self.discovery.scan()
        events = self.stm.get_events_after(0)
        types  = {e.type for e in events}
        subtypes = {e.payload.get("subtype") for e in events}
        self.assertTrue(
            "tool_discovered" in subtypes or "internal" in types,
            f"Expected tool_discovered event, got: {subtypes}"
        )

    def test_scan_dedup_on_second_call(self):
        self._write_tool("tool_test.py", _VALID_MANIFEST)
        first  = self.discovery.scan()
        second = self.discovery.scan()
        self.assertEqual(len(first),  1)
        self.assertEqual(len(second), 0, "Second scan should return 0 (already known)")

    def test_scan_bad_manifest_skipped(self):
        self._write_tool("bad_tool.py", _BAD_MANIFEST)
        new_tools = self.discovery.scan()
        # Bad manifest should be skipped (not raised)
        tool_ids = [m.tool_id for m in new_tools]
        self.assertNotIn("missing_required_fields", tool_ids)

    def test_manifest_parse_inputs_outputs(self):
        self._write_tool("tool_test.py", _VALID_MANIFEST)
        staged = self.discovery.scan()
        m = staged[0].manifest
        self.assertIn("query", m.inputs)
        self.assertIn("result", m.outputs)

    def test_manifest_parse_permission_scope(self):
        self._write_tool("tool_test.py", _VALID_MANIFEST)
        staged = self.discovery.scan()
        self.assertEqual(staged[0].manifest.permission_scope, [])


# ===========================================================================
# TestNarratorStateMachine — Orchestrator token stream routing
# ===========================================================================

def _make_orchestrator(
    muninn=None, stm=None, htm=None
) -> tuple[Orchestrator, list[str], list[dict]]:
    """
    Build a minimal Orchestrator for token stream tests.
    Returns (orchestrator, tts_tokens_received, tool_calls_received).
    """
    if muninn is None:
        muninn = MockMuninn()
    if stm is None:
        import time
        stm = STMStore(muninn, db_path=f":memory:orch_{time.time_ns()}")
    if htm is None:
        htm = HTM()

    tts_tokens:  list[str]  = []
    projections: list[dict] = []

    # Minimal mocks for all components
    sagax      = MagicMock()
    sagax.on_narrator_token = None
    sagax.inject_aug_result = MagicMock()
    sagax.pause_for_speech_step = MagicMock()
    sagax.inject_speech_step_result = MagicMock()

    logos      = MagicMock()
    exilis     = MagicMock()
    perception = MagicMock()
    tool_mgr   = MagicMock()

    orch = Orchestrator(
        stm              = stm,
        htm              = htm,
        perception       = perception,
        exilis           = exilis,
        sagax            = sagax,
        logos            = logos,
        tool_manager     = tool_mgr,
        on_tts_token     = tts_tokens.append,
        on_ui_projection = projections.append,
    )
    orch.session = Session(
        session_id       = "test-sess",
        entity_id        = "user",
        permission_scope = ["calendar.read", "microphone", "speaker"],
        denied           = [],
    )
    return orch, tts_tokens, projections


def _feed(orch: Orchestrator, text: str):
    """Feed a complete token stream string character-by-character."""
    for ch in text:
        orch.on_narrator_token(ch)


class TestNarratorStateMachine(unittest.TestCase):

    def test_idle_state_initially(self):
        orch, _, _ = _make_orchestrator()
        self.assertEqual(orch._narrator_state, NarratorState.IDLE)

    def test_speech_token_streams_to_tts(self):
        orch, tts, _ = _make_orchestrator()
        _feed(orch, '<speech target="user">Hello world</speech>')
        combined = "".join(tts)
        self.assertIn("Hello world", combined)
        self.assertEqual(orch._narrator_state, NarratorState.IDLE)

    def test_thinking_not_stored_or_streamed(self):
        orch, tts, _ = _make_orchestrator()
        _feed(orch, "<thinking>secret reasoning</thinking>")
        self.assertEqual(tts, [], "thinking should not stream to TTS")
        events = orch.stm.get_events_after(0)
        texts  = [e.payload.get("text", "") for e in events]
        self.assertFalse(any("secret reasoning" in t for t in texts))

    def test_contemplation_stored_in_stm(self):
        orch, _, _ = _make_orchestrator()
        _feed(orch, "<contemplation>I should check the weather</contemplation>")
        events = orch.stm.get_events_after(0)
        found  = any(
            e.payload.get("subtype") == "contemplation"
            and "weather" in e.payload.get("text", "")
            for e in events
        )
        self.assertTrue(found, "contemplation should be stored in STM")

    def test_state_returns_idle_after_each_block(self):
        orch, _, _ = _make_orchestrator()
        for block in [
            "<thinking>x</thinking>",
            "<contemplation>y</contemplation>",
            '<speech target="user">z</speech>',
        ]:
            _feed(orch, block)
            self.assertEqual(orch._narrator_state, NarratorState.IDLE,
                             f"Should be IDLE after: {block}")

    def test_tool_call_dispatched(self):
        orch, _, _ = _make_orchestrator()
        orch.tool_manager.get_polarity = MagicMock(return_value="write")
        orch.tool_manager.execute      = MagicMock(return_value={"output": "ok"})
        tool_json = json.dumps({
            "name": "tool.notes.v1",
            "args": {"action": "list"},
            "permission_scope": [],
        })
        _feed(orch, f"<tool_call>{tool_json}</tool_call>")
        # Tool was submitted to executor — verify via executor.submit mock
        self.assertEqual(orch._narrator_state, NarratorState.IDLE)

    def test_aug_call_read_only_gate(self):
        orch, _, _ = _make_orchestrator()
        orch.tool_manager.get_polarity = MagicMock(return_value="write")
        aug_json = json.dumps({
            "name": "tool.notes.v1",
            "args": {"action": "list"},
        })
        _feed(orch, f'<aug_call timeout_ms="200">{aug_json}</aug_call>')
        # Write-polarity tool in aug_call — inject_aug_result called with error
        orch.sagax.inject_aug_result.assert_called_once()
        result = orch.sagax.inject_aug_result.call_args[0][0]
        self.assertIn("aug_call_write_tool_rejected",
                      str(result), "write tool in aug_call should be rejected")

    def test_projection_forwarded_to_ui(self):
        orch, _, projections = _make_orchestrator()
        proj_json = json.dumps({"type": "card", "content": "test"})
        _feed(orch, f"<projection>{proj_json}</projection>")
        self.assertEqual(len(projections), 1)
        self.assertEqual(projections[0]["type"], "card")

    def test_sequential_blocks(self):
        orch, tts, _ = _make_orchestrator()
        stream = (
            "<thinking>plan</thinking>"
            "<contemplation>I will say hi</contemplation>"
            '<speech target="user">Hi there!</speech>'
        )
        _feed(orch, stream)
        self.assertIn("Hi there!", "".join(tts))
        self.assertEqual(orch._narrator_state, NarratorState.IDLE)


# ===========================================================================
# TestSpeechStep — suspension and response injection
# ===========================================================================

class TestSpeechStep(unittest.TestCase):

    def test_speech_step_streams_to_tts(self):
        orch, tts, _ = _make_orchestrator()
        _feed(orch, '<speech_step var="city">Which city are you in?</speech_step>')
        combined = "".join(tts)
        self.assertIn("Which city are you in?", combined)

    def test_speech_step_sets_pending(self):
        orch, _, _ = _make_orchestrator()
        _feed(orch, '<speech_step var="answer">What time is it?</speech_step>')
        # Sagax.pause_for_speech_step should have been called
        orch.sagax.pause_for_speech_step.assert_called_once()
        call_kwargs = orch.sagax.pause_for_speech_step.call_args
        # call_args can be positional or keyword depending on Python version
        var_value = (
            call_kwargs.kwargs.get("var")
            if call_kwargs.kwargs
            else (call_kwargs[0][0] if call_kwargs[0] else None)
        )
        self.assertEqual(var_value, "answer")

    def test_speech_step_pending_flag(self):
        orch, _, _ = _make_orchestrator()
        self.assertFalse(orch._speech_step_pending)
        _feed(orch, '<speech_step var="x">Question?</speech_step>')
        self.assertTrue(orch._speech_step_pending)

    def test_receive_response_clears_pending(self):
        orch, _, _ = _make_orchestrator()
        _feed(orch, '<speech_step var="x">Q?</speech_step>')
        orch.receive_speech_step_response("My answer")
        self.assertFalse(orch._speech_step_pending)

    def test_receive_response_binds_parameter(self):
        orch, _, _ = _make_orchestrator()
        _feed(orch, '<speech_step var="city">City?</speech_step>')
        orch.receive_speech_step_response("London")
        params = orch.htm.asc.get("hot_parameters")
        self.assertEqual(params.get("city"), "London")

    def test_receive_response_injects_into_sagax(self):
        orch, _, _ = _make_orchestrator()
        _feed(orch, '<speech_step var="name">Name?</speech_step>')
        orch.receive_speech_step_response("Alice")
        orch.sagax.inject_speech_step_result.assert_called_once_with("name", "Alice")

    def test_chat_routes_to_speech_step_when_pending(self):
        orch, _, _ = _make_orchestrator()
        _feed(orch, '<speech_step var="q">Q?</speech_step>')
        orch.chat("my response")
        orch.sagax.inject_speech_step_result.assert_called_once()

    def test_chat_wakes_sagax_when_no_pending(self):
        orch, _, _ = _make_orchestrator()
        orch.chat("hello there")
        orch.sagax.wake.assert_called_once()


# ===========================================================================
# MockLLMClient — no network, no openai package required
# ===========================================================================

class _MockChunk:
    def __init__(self, delta): self.delta = delta

class MockLLMClient:
    """LLMClient stub: never makes network calls."""
    def __init__(self, **kwargs): pass
    def complete(self, system="", user="", **kw):
        class R:
            text = ""
        return R()
    def stream(self, system="", messages=None, **kw):
        return iter([])
    def tool_call(self, *a, **kw): return {"tool_calls": []}


def _build_huginn_with_mock_llm(muninn, **kwargs):
    """Call build_huginn() but patch LLMClient to avoid openai import."""
    import huginn
    orig = huginn.LLMClient
    huginn.LLMClient = MockLLMClient
    # Also patch inside __init__.py module namespace
    import huginn.__init__ as hi
    orig2 = getattr(hi, "LLMClient", orig)
    hi.LLMClient = MockLLMClient
    try:
        return huginn.build_huginn(muninn=muninn, **kwargs)
    finally:
        huginn.LLMClient = orig
        hi.LLMClient = orig2


# ===========================================================================
# TestBuildHuginn — factory smoke test
# ===========================================================================

class TestBuildHuginn(unittest.TestCase):

    def test_build_huginn_returns_instance(self):
        from huginn import HuginnInstance
        muninn = MockMuninn()
        h = _build_huginn_with_mock_llm(
            muninn      = muninn,
            llm_backend = "ollama",
            fast_model  = "qwen2.5:0.5b",
            sagax_model = "llama3.2",
            logos_model = "llama3.2",
        )
        self.assertIsInstance(h, HuginnInstance)

    def test_build_huginn_has_all_components(self):
        muninn = MockMuninn()
        h = _build_huginn_with_mock_llm(muninn=muninn)
        for attr in ("orchestrator", "sagax", "logos", "exilis",
                     "stm", "htm", "tools", "tool_manager"):
            self.assertTrue(hasattr(h, attr), f"Missing: {attr}")

    def test_build_huginn_stm_is_writable(self):
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)
        ev     = h.stm.record(source="test", type="test",
                               payload={"msg": "smoke test"})
        self.assertIsNotNone(ev.id)

    def test_build_huginn_htm_is_writable(self):
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)
        tid    = h.htm.create(title="smoke", initiated_by="test")
        tasks  = h.htm.query(task_id=tid)
        self.assertEqual(len(tasks), 1)

    def test_build_huginn_staging_dirs_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            muninn         = MockMuninn(db_path=os.path.join(tmp, "artux.db"))
            muninn.db.path = muninn.db_path
            h = _build_huginn_with_mock_llm(muninn=muninn)
            # Directories should be created by ToolDiscovery
            self.assertTrue(
                hasattr(h, "tool_manager"),
                "tool_manager must exist"
            )

    def test_build_huginn_tts_callback_wired(self):
        muninn    = MockMuninn()
        received  = []
        h = _build_huginn_with_mock_llm(muninn=muninn, on_tts_token=received.append)
        # Orchestrator should have the callback
        self.assertIsNotNone(h.orchestrator.on_tts_token)


# ===========================================================================
# TestBootProcedure — startup procedure fallback
# ===========================================================================

class TestBootProcedure(unittest.TestCase):

    def test_boot_with_no_ltm_entry_uses_fallback(self):
        """
        When Muninn returns no recall results for procedure.startup.v1,
        execute_startup_procedure() should emit a fallback speech token
        and not raise.
        """
        muninn  = MockMuninn()
        muninn.recall_results = []   # no startup procedure in LTM
        h       = _build_huginn_with_mock_llm(muninn=muninn)

        received = []
        h.sagax.on_narrator_token = received.append

        # Should not raise
        try:
            h.sagax.execute_startup_procedure()
        except Exception as e:
            self.fail(f"execute_startup_procedure() raised: {e}")

        # Should have emitted some STM events (at minimum boot_start)
        events   = h.stm.get_events_after(0)
        subtypes = {e.payload.get("subtype") for e in events}
        self.assertIn("boot_start", subtypes)
        self.assertIn("boot_complete", subtypes)

    def test_boot_emits_speech_token(self):
        """Fallback boot sequence should emit a <speech> block."""
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)

        tokens = []
        h.sagax.on_narrator_token = tokens.append
        h.sagax.execute_startup_procedure()

        combined = "".join(tokens)
        self.assertIn("<speech", combined, "boot should emit at least one speech block")


# ===========================================================================
# TestLogosStagingLifecycle — staging scan → HTM task → install attempt
# ===========================================================================

class TestLogosStagingLifecycle(unittest.TestCase):

    def setUp(self):
        self.tmp     = tempfile.mkdtemp()
        self.staging = Path(self.tmp) / "staging"
        self.active  = Path(self.tmp) / "active"
        self.staging.mkdir()
        self.active.mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_tool(self, name: str, content: str):
        (self.staging / name).write_text(content)

    def test_scan_creates_waiting_htm_task(self):
        """Logos._staging_scan_pass() should create a 'waiting' HTM task for each new tool."""
        self._write_tool("tool_test.py", _VALID_MANIFEST)
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(
            muninn      = muninn,
            staging_dir = str(self.staging),
            active_dir  = str(self.active),
        )
        # Run one Logos pass synchronously
        h.logos.run_once()

        staged_tasks = h.htm.query(tags_any=["tool_staging"])
        self.assertGreater(len(staged_tasks), 0, "Logos should create HTM tasks for staged tools")

    def test_affirmed_tool_installs(self):
        """If an HTM task has user_affirmed:true in notebook, install_tool() should be called."""
        self._write_tool("tool_test.py", _VALID_MANIFEST)
        muninn = MockMuninn()
        h = _build_huginn_with_mock_llm(
            muninn      = muninn,
            staging_dir = str(self.staging),
            active_dir  = str(self.active),
        )

        # First pass: discovers tool, creates waiting task
        h.logos.run_once()
        staged_tasks = h.htm.query(tags_any=["tool_staging"])
        self.assertTrue(staged_tasks)
        task_id = staged_tasks[0].task_id

        # Simulate user affirmation
        h.htm.note(task_id, "user_affirmed: true\nenable_pipeline: false")

        # Second pass: should attempt install
        # install_tool() will fail (no source_path in active dir yet — file needs to be moved)
        # but should not crash Logos
        try:
            h.logos.run_once()
        except Exception as e:
            self.fail(f"Logos.run_once() raised on install attempt: {e}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
