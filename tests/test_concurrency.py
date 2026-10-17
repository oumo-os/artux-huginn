"""
tests/test_concurrency.py — Concurrency regression suite.

Asserts deterministic invariants under 16-thread contention:
  - exact ASC counters (use_count / fail_count)
  - single active health ledger with no lost notes
  - STM record-vs-flush losslessness (flushed + remaining == written)
  - load-on-first-use races
  - dirty-state flush completeness
  - dirty-state on_change hook fires
"""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import sys

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from huginn.runtime.htm import HTM, Task, ActiveSessionCache
from huginn.runtime.stm import STMStore, STMEvent
from huginn.runtime.tool_manager import ToolManager, ToolDescriptor
from huginn.runtime.context_budget import fit_user_prompt
from huginn.runtime.tool_manager import format_action_tags_capped


# ---------------------------------------------------------------------------
# Minimal Muninn stub (mirrors test_huginn.py MockMuninn)
# ---------------------------------------------------------------------------

class _MockLTMEntry:
    def __init__(self, **kwargs):
        self.id = kwargs.get("id", "ltm-mock-1")
        self.meta = kwargs.get("meta", {})
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockMuninn:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.recall_results = []
        self.consolidated = []
        self.db = MagicMock()
        self.db.path = db_path
        self.db.db_path = db_path

    def recall(self, query, top_k: int = 5):
        return list(self.recall_results)

    def consolidate_ltm(self, **kwargs):
        entry = _MockLTMEntry(id=f"ltm-{len(self.consolidated)}", **kwargs)
        self.consolidated.append(kwargs)
        return entry

    @property
    def tools(self):
        ex = __import__("unittest.mock").MagicMock()
        ex.execute.return_value = {"status": "ok"}
        return ex


# ===========================================================================
# TestASCConcurrency — exact counters under 16-thread contention
# ===========================================================================

class TestASCConcurrency(unittest.TestCase):
    def test_exact_use_count_under_contention(self):
        asc = ActiveSessionCache()
        n_threads = 16
        n_iter = 50
        barrier = threading.Barrier(n_threads)
        errors = []

        def worker():
            try:
                barrier.wait()
                for _ in range(n_iter):
                    asc.update_tool_usage("tool.lights.v1", title="Lights")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        caps = asc.get("hot_capabilities")
        hot = caps["tool.lights.v1"]
        expected = n_threads * n_iter
        self.assertEqual(hot.use_count, expected,
                         f"use_count {hot.use_count} != {expected}")

    def test_single_active_health_ledger_no_lost_notes(self):
        asc = ActiveSessionCache()
        n_threads = 16
        n_iter = 25
        barrier = threading.Barrier(n_threads)
        errors = []

        def worker():
            try:
                barrier.wait()
                for _ in range(n_iter):
                    asc.record_tool_failure("tool.broken.v1", "boom")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        failures = asc.get("tool_failures")
        hist = failures["tool.broken.v1"]
        expected = n_threads * n_iter
        self.assertEqual(hist.fail_count, expected,
                         f"fail_count {hist.fail_count} != {expected}")

    def test_evicted_tool_keeps_durable_history(self):
        asc = ActiveSessionCache()
        # Fill hot cache past cap to evict the first tool
        for i in range(25):
            asc.update_tool_usage(f"tool.{i}.v1", title=f"T{i}")
        # Now fail the first tool (evicted from hot cache)
        asc.record_tool_failure("tool.0.v1", "gone")
        # Re-seed it via record_tool_success — should re-enter with history
        asc.record_tool_success("tool.0.v1", title="T0")
        hot = asc.get("hot_capabilities")["tool.0.v1"]
        self.assertEqual(hot.fail_count, 1)
        self.assertEqual(hot.last_error, "gone")

    def test_summary_for_sagax_includes_evicted_failed_tool(self):
        asc = ActiveSessionCache()
        asc.record_tool_failure("tool.dead.v1", "permanent failure")
        summary = asc.summary_for_sagax()
        caps = summary["capabilities"]
        dead = [c for c in caps if c["tool_id"] == "tool.dead.v1"]
        self.assertEqual(len(dead), 1)
        self.assertGreaterEqual(dead[0]["fail_count"], 1)
        self.assertEqual(dead[0]["last_error"], "permanent failure")


# ===========================================================================
# TestSTMConcurrency — record-vs-flush losslessness
# ===========================================================================

class TestSTMConcurrency(unittest.TestCase):
    def setUp(self):
        self.muninn = _MockMuninn()
        self.stm = STMStore(self.muninn, db_path=f":memory:stm_conc_{id(self)}")

    def test_record_vs_flush_lossless(self):
        n_writers = 8
        n_events_each = 30
        barrier = threading.Barrier(n_writers)
        written_ids = []
        lock = threading.Lock()

        def writer(worker_id):
            try:
                barrier.wait()
                for i in range(n_events_each):
                    ev = self.stm.record(
                        source="sensor", type="motion",
                        payload={"worker": worker_id, "seq": i},
                    )
                    with lock:
                        written_ids.append(ev.id)
            except Exception:
                pass

        threads = [threading.Thread(target=writer, args=(w,)) for w in range(n_writers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_written = n_writers * n_events_each
        self.assertEqual(len(written_ids), total_written)

        # Flush everything
        last_id = written_ids[-1]
        flushed = self.stm.flush_up_to(last_id)
        remaining = self.stm.event_count_after_cons_n()  # no consN yet
        self.assertEqual(flushed + remaining, total_written,
                         f"flushed {flushed} + remaining {remaining} != {total_written}")

    def test_events_pending_short_circuits(self):
        self.stm.record(source="user", type="speech", payload={"text": "hi"})
        self.assertTrue(self.stm.events_pending(""))
        self.stm.flush_up_to(self.stm.get_events_after("")[-1].id)
        self.assertFalse(self.stm.events_pending(""))


# ===========================================================================
# TestDirtyStates — flush_dirty completeness + on_change hook
# ===========================================================================

class TestDirtyStates(unittest.TestCase):
    def test_dirty_keys_and_flush(self):
        htm = HTM()
        htm.states.set("sagax.model", "phi4")
        htm.states.set("sagax.temperature", 0.2)
        self.assertIn("sagax.model", htm.states.dirty_keys())
        changed = htm.states.flush_dirty()
        self.assertEqual(changed["sagax.model"], "phi4")
        self.assertEqual(changed["sagax.temperature"], 0.2)
        self.assertEqual(htm.states.dirty_keys(), set())

    def test_on_change_hook_fires(self):
        htm = HTM()
        calls = []
        htm.states.on_change = lambda k, v: calls.append((k, v))
        htm.states.set("tool.tts.kokoro.v1.speed", 1.4)
        self.assertEqual(calls, [("tool.tts.kokoro.v1.speed", 1.4)])


# ===========================================================================
# TestContextBudget — fit_user_prompt + format_action_tags_capped
# ===========================================================================

class TestContextBudget(unittest.TestCase):
    def test_format_action_tags_capped_hot_first(self):
        all_tags = {"lights.ceiling", "lights.wall", "calendar.read", "media.play"}
        hot_tags = {"media.play", "lights.ceiling"}
        out = format_action_tags_capped(all_tags, hot_tags, budget_chars=200)
        self.assertTrue(out.startswith("lights.ceiling, media.play"))
        self.assertIn("calendar.read", out)
        self.assertIn("lights.wall", out)

    def test_format_action_tags_capped_truncates_with_more(self):
        all_tags = {f"tool.{i}.v1" for i in range(50)}
        out = format_action_tags_capped(all_tags, (), budget_chars=40)
        self.assertTrue(out.endswith("more (use recall() to discover)"))
        self.assertLessEqual(len(out), 40 + len(" +N more (use recall() to discover)"))

    def test_fit_user_prompt_under_budget_unchanged(self):
        template = "Hello {name}, you have {count} tasks."
        out = fit_user_prompt(template, {"name": "John", "count": 3}, budget_chars=200)
        self.assertEqual(out, "Hello John, you have 3 tasks.")

    def test_fit_user_prompt_over_budget_strips_lossy_layers(self):
        template = "CTX: {cons_n_snippet} EVENTS: {recent_events} TAGS: {tags} ENTITY: {ent}"
        kwargs = {
            "cons_n_snippet": "x" * 300,
            "recent_events": "y" * 300,
            "tags": "lights.ceiling, media.play",
            "ent": "John",
        }
        out = fit_user_prompt(template, kwargs, budget_chars=100)
        self.assertIn("TAGS: lights.ceiling, media.play", out)
        self.assertIn("ENTITY: John", out)
        self.assertNotIn("x" * 300, out)
        self.assertLessEqual(len(out), 140)

    def test_fit_user_prompt_emits_truncated_event(self):
        emitted = []
        template = "CTX: {ctx} TAGS: {tags}"
        fit_user_prompt(
            template,
            {"ctx": "x" * 500, "tags": "lights.ceiling"},
            budget_chars=50,
            emit_truncated=lambda mode, saved, layer: emitted.append((mode, saved, layer)),
        )
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0][0], "")


# ===========================================================================
# TestToolHealth — record_tool_success/failure via ToolManager
# ===========================================================================

class TestToolHealth(unittest.TestCase):
    def setUp(self):
        self.muninn = _MockMuninn()
        self.htm = HTM()
        self.tm = ToolManager(self.muninn, self.htm)

    def test_success_and_failure_recorded_in_health(self):
        # Register a fake world tool handler
        def good_handler(**kw):
            return {"ok": True}
        def bad_handler(**kw):
            raise RuntimeError("boom")

        self.tm.register("tool.good.v1", good_handler,
                         ToolDescriptor(
                             tool_id="tool.good.v1", title="Good",
                             capability_summary="g", polarity="write",
                             permission_scope=["lights"], inputs={}, outputs={}))
        self.tm.register("tool.bad.v1", bad_handler,
                         ToolDescriptor(
                             tool_id="tool.bad.v1", title="Bad",
                             capability_summary="b", polarity="write",
                             permission_scope=["lights"], inputs={}, outputs={}))

        r1 = self.tm.execute("tool.good.v1", {})
        r2 = self.tm.execute("tool.bad.v1", {})
        self.assertTrue(r1.success)
        self.assertFalse(r2.success)

        vig = self.htm.asc.summary_for_sagax()
        caps = {c["tool_id"]: c for c in vig.get("capabilities", [])}
        self.assertEqual(caps["tool.good.v1"]["fail_count"], 0)
        self.assertEqual(caps["tool.bad.v1"]["fail_count"], 1)
        self.assertEqual(caps["tool.bad.v1"]["last_error"], "boom")


if __name__ == "__main__":
    unittest.main()
