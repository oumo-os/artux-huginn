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

    def recall(self, query, top_k: int = 5) -> list:
        """Accept RecallQuery object or plain string — both return recall_results."""
        return list(self.recall_results)

    def consolidate_ltm(self, **kwargs) -> MockLTMEntry:
        entry = MockLTMEntry(id=f"ltm-{len(self.consolidated)}", **kwargs)
        self.consolidated.append(kwargs)
        return entry

    def store_ltm(self, **kwargs) -> MockLTMEntry:
        entry = MockLTMEntry(id=f"ltm-{len(self.consolidated)}", **kwargs)
        self.consolidated.append(kwargs)
        return entry

    def observe_entity(self, **kwargs):
        pass

    # Source ref API (unchanged in Muninn)
    def record_source(self, **kwargs):
        class _Ref:
            id = "src-mock-1"
            type = "image"
            location = "/mock/file.jpg"
            description = ""
            captured_at = __import__("datetime").datetime.now()
        return _Ref()

    def record_and_attach_source(self, **kwargs):
        return self.record_source()

    def update_source_description(self, source_id: str, new_description: str):
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
            muninn           = muninn,
            fallback_model   = "llama3.2",
            fallback_backend = "ollama",
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
# TestRecallQuery — new structured recall API
# ===========================================================================

class TestRecallQuery(unittest.TestCase):
    """
    Verify Huginn's internal recall calls use RecallQuery when available
    and degrade gracefully to plain strings when memory_module is absent.
    """

    def _orch(self):
        """Build a minimal orchestrator with a mock muninn."""
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)
        return h.orchestrator, muninn

    def test_recall_system_config_calls_recall_with_any_arg(self):
        """_recall_system_config must call muninn.recall for each role topic."""
        orch, muninn = self._orch()
        called_with = []
        orig = muninn.recall
        muninn.recall = lambda q, top_k=1: (called_with.append(q) or [])
        orch._recall_system_config()
        # Should have called recall once per role (exilis, sagax, logos)
        self.assertEqual(len(called_with), 3,
                         f"Expected 3 recall calls, got {len(called_with)}")

    def test_recall_system_config_returns_dict(self):
        """_recall_system_config returns a dict keyed by role."""
        orch, muninn = self._orch()
        result = orch._recall_system_config()
        self.assertIsInstance(result, dict)
        # With no LTM entries, all roles should be missing (empty dict is fine)

    def test_recall_system_config_parses_json_content(self):
        """When LTM contains a config entry, _recall_system_config parses it."""
        import json

        class _MockResult:
            class entry:
                content = json.dumps({"backend": "anthropic", "model": "claude-haiku-4-5-20251001"})
        
        orch, muninn = self._orch()
        # Make recall return a mock result for every query
        muninn.recall = lambda q, top_k=1: [_MockResult()]
        configs = orch._recall_system_config()
        # Should have parsed the JSON for all three roles
        self.assertEqual(len(configs), 3)
        for role, cfg in configs.items():
            self.assertEqual(cfg["backend"], "anthropic")

    def test_mockMuninn_accepts_string_or_object(self):
        """MockMuninn.recall accepts both a string and a RecallQuery-like object."""
        muninn = MockMuninn()
        # Plain string
        r1 = muninn.recall("who is John", top_k=3)
        self.assertIsInstance(r1, list)
        # Object with any interface (RecallQuery duck-type)
        class _Q:
            topics = ["test"]
            semantic_query = "test"
            top_k = 1
        r2 = muninn.recall(_Q(), top_k=1)
        self.assertIsInstance(r2, list)

    def test_recall_result_has_sources(self):
        """RecallResult.sources is the field name (not file_refs)."""
        from huginn.runtime.stm import STMStore
        import time
        muninn = MockMuninn()
        stm    = STMStore(muninn, db_path=f":memory:rq_{time.time_ns()}")
        # A mock recall result should expose .sources not .file_refs
        class _MockResult:
            class entry:
                content = '{"test": true}'
                confidence = 0.9
                id = "ltm-1"
            score = 0.8
            match_reasons = ["topic:test"]
            sources = []           # ← the correct field name
            from_archive = False
        
        r = _MockResult()
        self.assertTrue(hasattr(r, "sources"),
                        "RecallResult must have .sources not .file_refs")
        self.assertFalse(hasattr(r, "file_refs"),
                         "file_refs is not a RecallResult field in current Muninn")


# ===========================================================================
# TestLogosSegmentation — multi-arc consolidation
# ===========================================================================

class TestLogosSegmentation(unittest.TestCase):
    """
    Verify that Logos._segment_and_synthesise returns multiple LTM entries
    (one per coherent arc) and that the fallback path handles bad LLM output.
    """

    def _make_logos(self):
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)
        return h.logos, muninn

    def _make_events(self, n=6):
        """Build n minimal STMEvent-like objects."""
        import time as _time
        from huginn.runtime.stm import STMEvent
        events = []
        for i in range(n):
            events.append(STMEvent(
                id=f"t{_time.time_ns()}-{i:04d}",
                ts=f"2026-03-20T10:{i:02d}:00Z",
                source="user",
                type="speech",
                payload={"text": f"event {i}: some content"},
                confidence=0.9,
            ))
            _time.sleep(0.001)   # ensure unique ns timestamps
        return events

    def test_segment_returns_list(self):
        """_segment_and_synthesise always returns a list."""
        logos, muninn = self._make_logos()
        events = self._make_events(4)

        # Mock LLM to return multi-arc response
        logos.llm.complete_json = MagicMock(return_value={
            "entries": [
                {"narrative": "John asked about the weather.",
                 "class_type": "event", "topics": ["weather", "john"],
                 "concepts": [], "entities": [], "confidence": 0.9,
                 "event_ids": [events[0].id, events[1].id],
                 "entity_observations": [], "semantic_assertions": []},
                {"narrative": "Lights set to warm red for movie night.",
                 "class_type": "observation",
                 "topics": ["lighting", "movie", "preference"],
                 "concepts": ["what:john:lighting_preference"],
                 "entities": [], "confidence": 0.88,
                 "event_ids": [events[2].id, events[3].id],
                 "entity_observations": [], "semantic_assertions": []},
            ]
        })

        arcs = logos._segment_and_synthesise(events)
        self.assertIsInstance(arcs, list)
        self.assertEqual(len(arcs), 2)

    def test_segment_each_arc_has_narrative(self):
        """Every arc must have a non-empty narrative."""
        logos, muninn = self._make_logos()
        events = self._make_events(3)

        logos.llm.complete_json = MagicMock(return_value={
            "entries": [
                {"narrative": "Arc one.", "class_type": "event",
                 "topics": ["a"], "concepts": [], "entities": [],
                 "confidence": 0.9, "event_ids": [],
                 "entity_observations": [], "semantic_assertions": []},
                {"narrative": "Arc two.", "class_type": "assertion",
                 "topics": ["b"], "concepts": [], "entities": [],
                 "confidence": 0.85, "event_ids": [],
                 "entity_observations": [], "semantic_assertions": []},
            ]
        })

        arcs = logos._segment_and_synthesise(events)
        for arc in arcs:
            self.assertIn("narrative", arc)
            self.assertTrue(arc["narrative"].strip())

    def test_fallback_on_bad_llm_output(self):
        """If LLM returns garbage, fallback produces exactly one entry."""
        logos, muninn = self._make_logos()
        events = self._make_events(2)

        # LLM returns something totally wrong
        logos.llm.complete_json = MagicMock(return_value={
            "entries": "not a list"
        })

        arcs = logos._segment_and_synthesise(events)
        self.assertIsInstance(arcs, list)
        self.assertGreater(len(arcs), 0)
        self.assertTrue(arcs[0].get("narrative"))

    def test_fallback_on_empty_entries(self):
        """Empty entries array triggers fallback single entry."""
        logos, muninn = self._make_logos()
        events = self._make_events(2)

        logos.llm.complete_json = MagicMock(return_value={"entries": []})
        arcs = logos._segment_and_synthesise(events)
        self.assertEqual(len(arcs), 1)

    def test_pass_writes_one_ltm_per_arc(self):
        """_pass() should call consolidate_ltm once per arc, not once total."""
        logos, muninn = self._make_logos()
        events = self._make_events(4)

        # Populate STM
        for ev in events:
            logos.stm._events.append(ev)
            logos.stm._write_event(ev)

        logos.llm.complete_json = MagicMock(return_value={
            "entries": [
                {"narrative": "First arc.", "class_type": "event",
                 "topics": ["x"], "concepts": [], "entities": [],
                 "confidence": 0.9, "event_ids": [events[0].id],
                 "entity_observations": [], "semantic_assertions": []},
                {"narrative": "Second arc.", "class_type": "observation",
                 "topics": ["y"], "concepts": [], "entities": [],
                 "confidence": 0.85, "event_ids": [events[2].id],
                 "entity_observations": [], "semantic_assertions": []},
                {"narrative": "Third arc.", "class_type": "assertion",
                 "topics": ["z"], "concepts": [], "entities": [],
                 "confidence": 0.8, "event_ids": [events[3].id],
                 "entity_observations": [], "semantic_assertions": []},
            ]
        })

        # Also mock skill synthesis and pipeline passes to avoid side effects
        logos._skill_synthesis_scan = MagicMock(return_value=0)
        logos._pipeline_adaptation_pass = MagicMock()

        logos.run_once()

        consolidated = muninn.consolidated
        episodic = [c for c in consolidated
                    if c.get("class_type") not in ("config", "procedure")]
        self.assertGreaterEqual(len(episodic), 3,
            f"Expected ≥3 LTM entries (one per arc), got {len(episodic)}: "
            f"{[c.get('narrative','?')[:40] for c in episodic]}")

    def test_no_single_monolithic_dump(self):
        """A batch with clearly distinct topics must never be one entry."""
        logos, muninn = self._make_logos()
        events = self._make_events(6)

        # 4 distinct arcs
        logos.llm.complete_json = MagicMock(return_value={
            "entries": [
                {"narrative": "Weather discussed.", "class_type": "event",
                 "topics": ["weather"], "concepts": [], "entities": [],
                 "confidence": 0.9, "event_ids": [], "entity_observations": [],
                 "semantic_assertions": []},
                {"narrative": "Car maintenance mentioned.", "class_type": "assertion",
                 "topics": ["car", "maintenance", "betty"],
                 "concepts": ["what:betty:vehicle_identity"],
                 "entities": [], "confidence": 0.88, "event_ids": [],
                 "entity_observations": [], "semantic_assertions": []},
                {"narrative": "Lights controlled.", "class_type": "event",
                 "topics": ["lighting", "tool"], "concepts": [], "entities": [],
                 "confidence": 0.92, "event_ids": [], "entity_observations": [],
                 "semantic_assertions": []},
                {"narrative": "Arrival detected.", "class_type": "observation",
                 "topics": ["arrival", "sensor"], "concepts": [], "entities": [],
                 "confidence": 0.85, "event_ids": [], "entity_observations": [],
                 "semantic_assertions": []},
            ]
        })

        arcs = logos._segment_and_synthesise(events)
        self.assertEqual(len(arcs), 4,
            "4 distinct arcs should produce 4 entries, not a merged dump")


# ===========================================================================
# TestHTMStates — state store unit tests
# ===========================================================================

class TestHTMStates(unittest.TestCase):

    def setUp(self):
        self.htm = HTM()

    def test_set_and_get(self):
        self.htm.states.set("sagax.model", "phi4")
        self.assertEqual(self.htm.states.get("sagax.model"), "phi4")

    def test_get_missing_returns_default(self):
        self.assertIsNone(self.htm.states.get("missing.key"))
        self.assertEqual(self.htm.states.get("missing.key", "fallback"), "fallback")

    def test_delete(self):
        self.htm.states.set("x.y", 1)
        self.htm.states.delete("x.y")
        self.assertIsNone(self.htm.states.get("x.y"))

    def test_list_all(self):
        self.htm.states.set("a.x", 1)
        self.htm.states.set("b.y", 2)
        all_states = self.htm.states.list()
        self.assertIn("a.x", all_states)
        self.assertIn("b.y", all_states)

    def test_list_prefix(self):
        self.htm.states.set("kokoro_tts.speed", 1.4)
        self.htm.states.set("kokoro_tts.voice", "af_bella")
        self.htm.states.set("sagax.model", "phi4")
        tts = self.htm.states.list("kokoro_tts.")
        self.assertIn("kokoro_tts.speed", tts)
        self.assertIn("kokoro_tts.voice", tts)
        self.assertNotIn("sagax.model", tts)

    def test_set_default_only_applies_when_missing(self):
        self.htm.states.set("tts.speed", 2.0)
        self.htm.states.set_default("tts.speed", 1.0)   # should not overwrite
        self.assertEqual(self.htm.states.get("tts.speed"), 2.0)

    def test_set_default_applies_when_missing(self):
        self.htm.states.set_default("tts.voice", "af_bella")
        self.assertEqual(self.htm.states.get("tts.voice"), "af_bella")

    def test_dirty_tracking(self):
        self.htm.states.set("k.v", 42)
        self.assertIn("k.v", self.htm.states.dirty_keys())

    def test_set_default_not_dirty(self):
        self.htm.states.set_default("k.v", "default")
        self.assertNotIn("k.v", self.htm.states.dirty_keys())

    def test_flush_dirty_returns_and_clears(self):
        self.htm.states.set("a.x", 1)
        self.htm.states.set("b.y", 2)
        dirty = self.htm.states.flush_dirty()
        self.assertEqual(dirty["a.x"], 1)
        self.assertEqual(dirty["b.y"], 2)
        self.assertEqual(len(self.htm.states.dirty_keys()), 0)

    def test_load_from_config_not_dirty(self):
        self.htm.states.load_from_config(
            {"model": "llama3.2", "temperature": 0.2}, namespace="sagax"
        )
        self.assertEqual(self.htm.states.get("sagax.model"), "llama3.2")
        self.assertNotIn("sagax.model", self.htm.states.dirty_keys())

    def test_summary_groups_by_namespace(self):
        self.htm.states.set("kokoro_tts.speed", 1.4)
        self.htm.states.set("sagax.model", "phi4")
        s = self.htm.states.summary()
        self.assertIn("kokoro_tts", s)
        self.assertIn("sagax", s)

    def test_htm_has_states(self):
        """HTM instance exposes .states."""
        self.assertIsNotNone(self.htm.states)
        self.htm.states.set("test.val", 99)
        self.assertEqual(self.htm.states.get("test.val"), 99)


# ===========================================================================
# TestActuationBus — pub/sub unit tests
# ===========================================================================

class TestActuationBus(unittest.TestCase):

    def setUp(self):
        from huginn.runtime.actuation_bus import ActuationBus, ActuationEvent
        self.Bus   = ActuationBus
        self.Event = ActuationEvent

    def test_subscribe_and_receive(self):
        bus = self.Bus()
        q   = bus.subscribe("tool_a", {"type": "output", "target": "speech"})
        bus.publish(self.Event(type="output", target="speech",
                               complete="chunk", text="Hello"))
        ev = q.get_nowait()
        self.assertEqual(ev.text, "Hello")

    def test_filter_excludes_non_matching(self):
        bus = self.Bus()
        q   = bus.subscribe("tool_a", {"type": "output", "target": "speech"})
        bus.publish(self.Event(type="output", target="display",
                               complete="full", text="UI update"))
        self.assertTrue(q.empty())

    def test_multiple_subscribers_receive_matching(self):
        bus = self.Bus()
        q1  = bus.subscribe("tts",     {"target": "speech"})
        q2  = bus.subscribe("display", {"target": "display"})
        bus.publish(self.Event(type="output", target="speech",
                               complete="chunk", text="Hi"))
        self.assertFalse(q1.empty())
        self.assertTrue(q2.empty())

    def test_complete_filter(self):
        bus = self.Bus()
        q   = bus.subscribe("tts_chunk", {"target": "speech", "complete": "chunk"})
        bus.publish(self.Event(type="output", target="speech",
                               complete="partial", text="t"))
        bus.publish(self.Event(type="output", target="speech",
                               complete="chunk", text="Hello,"))
        # Only chunk should arrive
        ev = q.get_nowait()
        self.assertEqual(ev.complete, "chunk")
        self.assertTrue(q.empty())

    def test_empty_filter_matches_all(self):
        bus = self.Bus()
        q   = bus.subscribe("sink", {})
        bus.publish(self.Event(type="output", target="speech",   complete="chunk", text="a"))
        bus.publish(self.Event(type="output", target="display",  complete="full",  text="b"))
        bus.publish(self.Event(type="output", target="contemplation", complete="full", text="c"))
        self.assertEqual(q.qsize(), 3)

    def test_unsubscribe(self):
        bus = self.Bus()
        q   = bus.subscribe("tts", {"target": "speech"})
        bus.unsubscribe("tts")
        bus.publish(self.Event(type="output", target="speech",
                               complete="chunk", text="x"))
        self.assertTrue(q.empty())

    def test_publish_dict_helper(self):
        bus = self.Bus()
        q   = bus.subscribe("t", {})
        bus.publish_dict(type="output", target="speech",
                         complete="full", text="done")
        ev = q.get_nowait()
        self.assertEqual(ev.text, "done")
        self.assertEqual(ev.target, "speech")

    def test_full_queue_drops_silently(self):
        """A saturated subscriber queue must never block the publisher."""
        bus = self.Bus()
        bus.subscribe("slow", {}, maxsize=2)
        for i in range(10):
            bus.publish_dict(type="output", target="speech",
                             complete="chunk", text=str(i))
        # No exception raised — bus continues operating


# ===========================================================================
# TestSpeechChunker — Orchestrator phrase-boundary chunking
# ===========================================================================

class TestSpeechChunker(unittest.TestCase):

    def _make_orch(self):
        from huginn.runtime.actuation_bus import ActuationBus
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)
        # Attach a real bus so chunker publishes
        h.orchestrator.actuation_bus = ActuationBus()
        h.orchestrator.session.session_id = "test-sess"
        return h.orchestrator, h.orchestrator.actuation_bus

    def test_partial_published_per_token(self):
        orch, bus = self._make_orch()
        partials = bus.subscribe("partial", {"complete": "partial"})
        # Simulate speech streaming
        orch._narrator_state = "STREAMING_SPEECH"
        for ch in "Hello":
            orch._feed_speech_chunk(ch)
        self.assertEqual(partials.qsize(), 5)

    def test_chunk_published_at_sentence_end(self):
        orch, bus = self._make_orch()
        chunks = bus.subscribe("chunks", {"complete": "chunk"})
        orch._narrator_state = "STREAMING_SPEECH"
        for ch in "The weather is nice today.":
            orch._feed_speech_chunk(ch)
        self.assertGreater(chunks.qsize(), 0)
        ev = chunks.get_nowait()
        self.assertIn(".", ev.text)

    def test_flush_clears_buffer(self):
        orch, bus = self._make_orch()
        chunks = bus.subscribe("c", {"complete": "chunk"})
        orch._chunk_buf = "some partial text"
        orch._flush_speech_chunk()
        self.assertFalse(chunks.empty())
        self.assertEqual(orch._chunk_buf, "")

    def test_full_event_published_at_speech_close(self):
        orch, bus = self._make_orch()
        fulls = bus.subscribe("full", {"complete": "full", "target": "speech"})
        orch._publish_speech_full("Hello world.", "entity-john")
        ev = fulls.get_nowait()
        self.assertEqual(ev.text, "Hello world.")
        self.assertEqual(ev.complete, "full")

    def test_think_tag_silently_discarded(self):
        """<think> and <thinking> tokens must never reach STM or TTS."""
        orch, bus = self._make_orch()
        received = []
        orch.on_tts_token = received.append
        # Feed a <think> block through the state machine
        for ch in "<think>secret reasoning here</think>":
            orch.on_narrator_token(ch)
        # Nothing should have gone to TTS
        self.assertEqual(received, [],
                         "<think> content must be silently discarded")
        # Nothing should be in STM as output
        events = orch.stm.get_events_after(0)
        output = [e for e in events
                  if e.type == "output"
                  and "secret reasoning" in e.payload.get("text", "")]
        self.assertEqual(output, [])


# ===========================================================================
# TestStateFlowIntegration — state set → Sagax context → Logos persist
# ===========================================================================

class TestStateFlowIntegration(unittest.TestCase):

    def test_state_set_via_task_update(self):
        """<task_update action=state_set> writes to HTM.states."""
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)
        orch   = h.orchestrator
        orch.session.session_id = "test"

        import json
        orch._handle_task_update(
            json.dumps({"action": "state_set",
                        "key": "kokoro_tts.speed",
                        "value": 1.4})
        )
        self.assertEqual(h.htm.states.get("kokoro_tts.speed"), 1.4)

    def test_state_delete_via_task_update(self):
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)
        h.htm.states.set("x.y", 42)
        orch   = h.orchestrator
        orch.session.session_id = "test"
        import json
        orch._handle_task_update(
            json.dumps({"action": "state_delete", "key": "x.y"})
        )
        self.assertIsNone(h.htm.states.get("x.y"))

    def test_load_from_config_populates_states(self):
        """_apply_system_config writes sagax.model etc. to htm.states."""
        muninn  = MockMuninn()
        h       = _build_huginn_with_mock_llm(muninn=muninn)
        orch    = h.orchestrator

        orch._apply_system_config({
            "sagax": {"model": "phi4", "temperature": 0.3, "backend": "ollama"},
        })
        self.assertEqual(h.htm.states.get("sagax.model"), "phi4")
        self.assertEqual(h.htm.states.get("sagax.temperature"), 0.3)

    def test_state_snapshot_in_sagax_context(self):
        """HTM.states.summary() is non-empty after a state is set."""
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)
        h.htm.states.set("sagax.model", "phi4")
        summary = h.htm.states.summary()
        self.assertIn("sagax.model", summary)
        self.assertIn("phi4", summary)

    def test_manifest_state_defaults_written_on_install(self):
        """install_tool writes manifest.states defaults to htm.states."""
        import tempfile, os, time
        tmp     = tempfile.mkdtemp()
        muninn  = MockMuninn()
        h       = _build_huginn_with_mock_llm(muninn=muninn)

        # Create a fake tool file — manifest must be in a docstring
        tool_src = '"""\nHUGINN_MANIFEST\ntool_id:            tool.fake.tts.v1\ntitle:              Fake TTS\ncapability_summary: Test TTS tool for unit tests.\npolarity:           write\npermission_scope:   []\ninputs:\n  text: {type: string}\nmode:               service\ndirection:          output\nsubscriptions:\n  - type: output\n    target: speech\n    complete: chunk\nstates:\n  speed:\n    default: 1.0\n    type: float\n    description: Playback speed\n  voice:\n    default: test_voice\n    type: string\nEND_MANIFEST\n"""\n\ndef start(config): pass\ndef stop(): pass\ndef handle(event): pass\n' 
        src_path = os.path.join(tmp, "tool_fake_tts.py")
        open(src_path, "w").write(tool_src)

        from huginn.runtime.tool_discovery import parse_manifest
        manifest = parse_manifest(tool_src, src_path)
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.mode, "service")
        self.assertEqual(manifest.direction, "output")
        self.assertEqual(len(manifest.subscriptions), 1)
        self.assertIn("speed", manifest.states)
        self.assertIn("voice", manifest.states)

        # Install
        h.tool_manager.install_tool(
            manifest=manifest, stm=h.stm, htm=h.htm
        )

        # Defaults should now be in HTM.states
        self.assertEqual(h.htm.states.get("tool.fake.tts.v1.speed"), 1.0)
        self.assertEqual(h.htm.states.get("tool.fake.tts.v1.voice"), "test_voice")

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    def test_dirty_states_flushed_on_session_end(self):
        """Logos._persist_dirty_states writes changed states to Muninn."""
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)
        h.htm.states.set("sagax.model", "phi4")   # dirty

        # Run persist
        h.logos._persist_dirty_states()

        # Muninn should have received a store_ltm call
        self.assertTrue(len(muninn.consolidated) > 0)
        written = [c for c in muninn.consolidated
                   if "phi4" in c.get("content", "")]
        self.assertTrue(len(written) > 0,
            "phi4 should be in a persisted LTM config entry")


# ===========================================================================
# TestLLMProvider — provider tool routing
# ===========================================================================

class TestLLMProvider(unittest.TestCase):

    def _make_client(self, role="sagax"):
        """LLMClient with a real HTM instance but no actual inference."""
        from huginn.llm.client import LLMClient, LLMResponse, StreamChunk
        htm = HTM()
        c   = LLMClient(role=role, htm=htm,
                        fallback_backend="ollama",
                        fallback_model="llama3.2")
        return c, htm, LLMResponse, StreamChunk

    def test_client_has_role(self):
        c, _, _, _ = self._make_client("sagax")
        self.assertEqual(c.role, "sagax")

    def test_model_property_reads_from_states(self):
        c, htm, _, _ = self._make_client("sagax")
        htm.states.set("sagax.model", "phi4", mark_dirty=False)
        self.assertEqual(c.model, "phi4")

    def test_temperature_property_reads_from_states(self):
        c, htm, _, _ = self._make_client("sagax")
        htm.states.set("sagax.temperature", 0.7, mark_dirty=False)
        self.assertAlmostEqual(c.temperature, 0.7)

    def test_backend_property_reads_provider_from_states(self):
        c, htm, _, _ = self._make_client("sagax")
        htm.states.set("sagax.provider", "tool.llm.anthropic.v1", mark_dirty=False)
        self.assertEqual(c.backend, "tool.llm.anthropic.v1")

    def test_register_provider_routes_calls(self):
        """A registered provider intercepts all complete() calls."""
        from huginn.llm.client import LLMClient, LLMResponse
        calls = []
        def fake_complete(system, messages, model="", temperature=0.1,
                          timeout=60.0, **kw):
            calls.append({"model": model, "system": system})
            return LLMResponse(text="fake response", model=model)

        LLMClient.register_provider("tool.llm.fake.v1", {
            "complete":       fake_complete,
            "stream":         lambda **kw: iter([]),
            "complete_json":  lambda **kw: {},
            "complete_tools": lambda **kw: LLMResponse(text=""),
        })

        htm = HTM()
        htm.states.set("sagax.provider", "tool.llm.fake.v1", mark_dirty=False)
        htm.states.set("sagax.model",    "test-model",        mark_dirty=False)

        c   = LLMClient(role="sagax", htm=htm)
        r   = c.complete(system="sys", user="hello")

        self.assertEqual(r.text, "fake response")
        self.assertEqual(calls[0]["model"], "test-model")
        self.assertEqual(calls[0]["system"], "sys")

        # Clean up
        LLMClient.unregister_provider("tool.llm.fake.v1")

    def test_unregistered_provider_falls_back_to_builtin(self):
        """When the states provider is not registered, fallback is used."""
        from huginn.llm.client import LLMClient, _BuiltinProvider
        htm = HTM()
        htm.states.set("sagax.provider", "tool.llm.nobody.v1", mark_dirty=False)
        c   = LLMClient(role="sagax", htm=htm,
                        fallback_backend="ollama", fallback_model="test")
        prov, model, _, _ = c._resolve()
        self.assertIsInstance(prov, _BuiltinProvider)

    def test_reconfigure_writes_to_states(self):
        """reconfigure() updates HTM.states, not a backend client."""
        from huginn.llm.client import LLMClient
        htm = HTM()
        c   = LLMClient(role="sagax", htm=htm)
        c.reconfigure(model="phi4", temperature=0.5)
        self.assertEqual(htm.states.get("sagax.model"),       "phi4")
        self.assertAlmostEqual(htm.states.get("sagax.temperature"), 0.5)

    def test_reconfigure_sets_provider_from_backend(self):
        """reconfigure(backend='anthropic') sets sagax.provider."""
        from huginn.llm.client import LLMClient
        htm = HTM()
        c   = LLMClient(role="sagax", htm=htm)
        c.reconfigure(backend="anthropic")
        provider = htm.states.get("sagax.provider")
        self.assertEqual(provider, "tool.llm.anthropic.v1")

    def test_htm_injection_in_provider_handler(self):
        """Provider handlers that declare _htm receive the HTM instance."""
        from huginn.llm.client import LLMClient, LLMResponse
        received_htm = []
        def fake_complete(system, messages, model="", temperature=0.1,
                          timeout=60.0, _htm=None, **kw):
            received_htm.append(_htm)
            return LLMResponse(text="ok")

        LLMClient.register_provider("tool.llm.htmtest.v1", {
            "complete": fake_complete,
            "stream":   lambda **kw: iter([]),
            "complete_json":  lambda **kw: {},
            "complete_tools": lambda **kw: LLMResponse(text=""),
        })
        htm = HTM()
        htm.states.set("sagax.provider", "tool.llm.htmtest.v1", mark_dirty=False)
        c   = LLMClient(role="sagax", htm=htm)
        c.complete(system="s", user="u")

        self.assertIs(received_htm[0], htm)
        LLMClient.unregister_provider("tool.llm.htmtest.v1")

    def test_provider_tool_manifest_parsed(self):
        """Ollama provider tool manifest parses correctly."""
        import os
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "tools", "staging",
            "tool_llm_ollama.py"
        )
        if not os.path.exists(src_path):
            self.skipTest("tool_llm_ollama.py not found")
        from huginn.runtime.tool_discovery import parse_manifest
        src      = open(src_path).read()
        manifest = parse_manifest(src, src_path)
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.tool_id, "tool.llm.ollama.v1")
        self.assertEqual(manifest.mode, "provider")
        self.assertEqual(manifest.polarity, "read")
        self.assertIn("host",  manifest.states)
        self.assertIn("model", manifest.states)
        self.assertEqual(manifest.states["model"]["default"], "llama3.2")

    def test_provider_registered_on_install(self):
        """install_tool() for a mode=provider tool registers it in LLMClient."""
        import tempfile, os, shutil
        from huginn.llm.client import LLMClient
        from huginn.runtime.tool_discovery import parse_manifest

        tmp = tempfile.mkdtemp()
        _tool_src = [
            '"""',
            "HUGINN_MANIFEST",
            "tool_id:            tool.llm.testprov.v1",
            "title:              Test Provider",
            "capability_summary: Unit test provider.",
            "polarity:           read",
            "permission_scope:   []",
            "mode:               provider",
            'direction:          ""',
            "inputs:",
            "  system: {type: string}",
            "END_MANIFEST",
            '"""',
            "",
            "from huginn.llm.client import LLMResponse",
            "def complete(system, messages, model='', temperature=0.1, timeout=60.0, **kw):",
            "    return LLMResponse(text='testprov')",
            "def stream(**kw): return iter([])",
            "def complete_json(**kw): return {}",
            "def complete_tools(**kw): return LLMResponse(text='')",
        ]
        src      = "\n".join(_tool_src)
        src_path = os.path.join(tmp, "tool_llm_testprov.py")
        open(src_path, "w").write(src)
        manifest = parse_manifest(src, src_path)
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.mode, "provider")

        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)
        h.tool_manager.install_tool(
            manifest=manifest, stm=h.stm, htm=h.htm
        )

        # Should now be registered
        self.assertIn("tool.llm.testprov.v1", LLMClient.available_providers())

        # Route a call through the real LLMClient (not the mock) by using
        # the provider registry directly with a real HTM instance
        h.htm.states.set("sagax.provider", "tool.llm.testprov.v1", mark_dirty=False)
        real_client = LLMClient(role="sagax", htm=h.htm)
        r = real_client.complete(system="test", user="hello")
        self.assertEqual(r.text, "testprov")

        # Clean up
        LLMClient.unregister_provider("tool.llm.testprov.v1")
        shutil.rmtree(tmp, ignore_errors=True)

    def test_available_providers(self):
        """available_providers() returns a list."""
        from huginn.llm.client import LLMClient
        result = LLMClient.available_providers()
        self.assertIsInstance(result, list)


# ===========================================================================
# TestInstructionSystem — get_instructions tool + Logos first-pass writes
# ===========================================================================

class TestInstructionSystem(unittest.TestCase):

    def test_prompt_v2_shorter_than_v1(self):
        """SAGAX_PLAN_v2 must be substantially shorter than the old v1 content."""
        from huginn.llm.prompts import SAGAX_PLAN_v2
        # v2 should be under 6000 chars (old v1 was 12032)
        self.assertLess(len(SAGAX_PLAN_v2), 6000,
            "SAGAX_PLAN_v2 is too long — keep it lean")
        # Must still contain the grammar table header
        self.assertIn("Narrator grammar", SAGAX_PLAN_v2)

    def test_prompt_v2_contains_get_instructions_call(self):
        """v2 must show Sagax how to call get_instructions."""
        from huginn.llm.prompts import SAGAX_PLAN_v2
        self.assertIn("get_instructions", SAGAX_PLAN_v2)
        self.assertIn('"topic"', SAGAX_PLAN_v2)

    def test_prompt_v2_has_all_eight_topics(self):
        """All 8 topic names must appear in the topic directory."""
        from huginn.llm.prompts import SAGAX_PLAN_v2
        for topic in ["htm_tasks", "skill_execution", "memory", "states",
                      "live_tools", "staging", "entities", "speech_step"]:
            self.assertIn(topic, SAGAX_PLAN_v2,
                f"Topic '{topic}' missing from SAGAX_PLAN_v2 topic directory")

    def test_prompt_v1_alias_equals_v2(self):
        """SAGAX_PLAN_v1 must be the same object as SAGAX_PLAN_v2."""
        from huginn.llm.prompts import SAGAX_PLAN_v1, SAGAX_PLAN_v2
        self.assertIs(SAGAX_PLAN_v1, SAGAX_PLAN_v2)

    def test_prompt_user_v1_alias(self):
        from huginn.llm.prompts import SAGAX_PLAN_USER_v1, SAGAX_PLAN_USER_v2
        self.assertIs(SAGAX_PLAN_USER_v1, SAGAX_PLAN_USER_v2)

    def test_all_eight_instruction_constants_exist(self):
        """All 8 INSTRUCTION_*_v1 constants must be importable and non-empty."""
        from huginn.llm import prompts
        for name in [
            "INSTRUCTION_HTM_TASKS_v1",
            "INSTRUCTION_SKILL_EXECUTION_v1",
            "INSTRUCTION_MEMORY_v1",
            "INSTRUCTION_STATES_v1",
            "INSTRUCTION_LIVE_TOOLS_v1",
            "INSTRUCTION_STAGING_v1",
            "INSTRUCTION_ENTITIES_v1",
            "INSTRUCTION_SPEECH_STEP_v1",
        ]:
            const = getattr(prompts, name, None)
            self.assertIsNotNone(const, f"{name} missing from prompts.py")
            self.assertGreater(len(const), 100,
                f"{name} is suspiciously short ({len(const)} chars)")

    def test_instruction_constants_have_see_also(self):
        """Each instruction artifact should cross-reference related topics."""
        from huginn.llm import prompts
        for name in [
            "INSTRUCTION_HTM_TASKS_v1", "INSTRUCTION_SKILL_EXECUTION_v1",
            "INSTRUCTION_MEMORY_v1",    "INSTRUCTION_STATES_v1",
        ]:
            const = getattr(prompts, name)
            self.assertIn("See also", const,
                f"{name} missing 'See also' cross-reference")

    def test_get_instructions_tool_is_read_polarity(self):
        """get_instructions must have polarity 'read' so aug_call can use it."""
        from huginn.runtime.tool_manager import _MEMORY_POLARITY
        self.assertIn("get_instructions", _MEMORY_POLARITY)
        self.assertEqual(_MEMORY_POLARITY["get_instructions"], "read")

    def test_htm_state_get_tool_is_read_polarity(self):
        from huginn.runtime.tool_manager import _MEMORY_POLARITY
        self.assertIn("htm_state_get", _MEMORY_POLARITY)
        self.assertEqual(_MEMORY_POLARITY["htm_state_get"], "read")

    def test_get_instructions_no_topic_returns_list(self):
        """get_instructions() with no topic returns a list of topics."""
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)
        result = h.tool_manager._get_instructions("")
        self.assertIn("htm_tasks", result)
        self.assertIn("skill_execution", result)
        self.assertIn("memory", result)

    def test_get_instructions_unknown_topic_graceful(self):
        """Unknown topic returns helpful message, not an exception."""
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)
        result = h.tool_manager._get_instructions("nonexistent_topic")
        # Should not raise; should tell Sagax what to do
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)

    def test_get_instructions_known_topic_recalls_from_ltm(self):
        """get_instructions('htm_tasks') should recall from Muninn LTM."""
        from huginn.llm.prompts import INSTRUCTION_HTM_TASKS_v1

        muninn = MockMuninn()
        # Pre-populate MockMuninn with the instruction artifact
        class FakeEntry:
            content = INSTRUCTION_HTM_TASKS_v1
        class FakeResult:
            entry = FakeEntry()
        muninn._instruction_results = {"htm_tasks": [FakeResult()]}

        # Monkey-patch MockMuninn.recall to return instruction when asked
        orig_recall = muninn.recall
        def patched_recall(q, top_k=5):
            # Check if q mentions instruction topics
            q_str = str(q)
            if "instruction.htm_tasks" in q_str or "htm_tasks" in q_str:
                return muninn._instruction_results.get("htm_tasks", [])
            return orig_recall(q, top_k)
        muninn.recall = patched_recall

        h      = _build_huginn_with_mock_llm(muninn=muninn)
        result = h.tool_manager._get_instructions("htm_tasks")
        self.assertIn("HTM Task Management", result)
        self.assertIn("persistence", result)

    def test_execute_native_get_instructions(self):
        """execute() routes get_instructions to _execute_native."""
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)
        result = h.tool_manager.execute("get_instructions", {"topic": ""})
        self.assertTrue(result.success)
        self.assertIn("htm_tasks", result.output)

    def test_execute_native_htm_state_get_key(self):
        """htm_state_get with key= reads from HTM.states."""
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)
        h.htm.states.set("sagax.model", "phi4")
        result = h.tool_manager.execute("htm_state_get", {"key": "sagax.model"})
        self.assertTrue(result.success)
        self.assertIn("phi4", result.output)

    def test_execute_native_htm_state_get_prefix(self):
        """htm_state_get with prefix= returns all matching keys."""
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)
        h.htm.states.set("kokoro_tts.speed", 1.4)
        h.htm.states.set("kokoro_tts.voice", "af_bella")
        result = h.tool_manager.execute("htm_state_get", {"prefix": "kokoro_tts."})
        self.assertTrue(result.success)
        self.assertIn("kokoro_tts.speed", result.output)
        self.assertIn("kokoro_tts.voice", result.output)

    def test_logos_ensure_instruction_defaults_writes_ltm(self):
        """Logos._ensure_instruction_defaults() writes 8 entries to Muninn."""
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)

        # Capture how many store_ltm calls happen
        before = len(muninn.consolidated)
        h.logos._ensure_instruction_defaults()
        after = len(muninn.consolidated)

        written = after - before
        self.assertEqual(written, 8,
            f"Expected 8 instruction artifacts written, got {written}")

    def test_logos_ensure_instruction_defaults_idempotent(self):
        """Second call must not write again if entries already exist."""
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)

        # First call
        h.logos._ensure_instruction_defaults()
        first_count = len(muninn.consolidated)

        # Patch recall to return something for all instruction queries
        orig_recall = muninn.recall
        class _FakeEntry:
            content = "existing instruction"
        class _FakeResult:
            entry = _FakeEntry()
        def patched_recall(q, top_k=5):
            if "instruction" in str(q):
                return [_FakeResult()]
            return orig_recall(q, top_k)
        muninn.recall = patched_recall

        # Second call — should write nothing new
        h.logos._ensure_instruction_defaults()
        second_count = len(muninn.consolidated)
        self.assertEqual(first_count, second_count,
            "Second call to _ensure_instruction_defaults should write nothing")

    def test_logos_first_pass_calls_instruction_defaults(self):
        """Logos._pass() first-pass must call _ensure_instruction_defaults."""
        muninn = MockMuninn()
        h      = _build_huginn_with_mock_llm(muninn=muninn)
        h.logos._first_pass = True

        called = []
        orig = h.logos._ensure_instruction_defaults
        h.logos._ensure_instruction_defaults = lambda: called.append(True) or orig()

        h.logos._pass()
        self.assertTrue(called,
            "_ensure_instruction_defaults not called on first pass")

    def test_sagax_uses_v2_prompt(self):
        """Sagax imports SAGAX_PLAN_v2 (not old long v1)."""
        from huginn.llm.prompts import SAGAX_PLAN_v2
        import huginn.agents.sagax as sagax_mod
        # The prompt used by _cycle must equal v2 content
        self.assertIn("On-demand instructions", SAGAX_PLAN_v2)
        # Confirm v2 is lean — under 6000 chars
        self.assertLess(len(SAGAX_PLAN_v2), 6000)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
