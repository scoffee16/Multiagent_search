"""耗时埋点与指标存储的单测（对应需求 P0 §3.1 性能埋点）。

覆盖：
- percentile 分位数计算（P50/P95 是评测报告的口径，必须可复现）
- TaskTimer 分阶段累加、首字延迟只记一次、总耗时/llm 反推
- TaskMetricsStore 的 SQLite 写入、幂等覆盖、聚合（P50/P95）、重置
"""

from __future__ import annotations

import pytest

from agent.timing import (
    TaskMetricsStore,
    TaskTimer,
    get_timer,
    percentile,
    reset_timer,
    start_timer,
)


class TestPercentile:
    def test_empty_list_returns_zero(self):
        assert percentile([], 50) == 0.0

    def test_single_value(self):
        assert percentile([42.0], 95) == 42.0

    def test_median_of_two(self):
        assert percentile([10.0, 20.0], 50) == 15.0

    def test_p95_linear_interpolation(self):
        values = [float(i) for i in range(1, 21)]  # 1..20
        # k = (20-1)*0.95 = 18.05 -> 19 + 0.05*(20-19) = 19.05
        assert percentile(values, 95) == pytest.approx(19.05)

    def test_p0_and_p100(self):
        values = [3.0, 1.0, 2.0]
        assert percentile(values, 0) == 1.0
        assert percentile(values, 100) == 3.0


class TestTaskTimer:
    def test_stages_accumulate(self):
        timer = TaskTimer("t1")
        timer.add("tool_ms", 10)
        timer.add("tool_ms", 5)
        timer.add("retrieve_ms", 7)
        assert timer.stages["tool_ms"] == 15
        assert timer.stages["retrieve_ms"] == 7

    def test_unknown_stage_raises(self):
        with pytest.raises(ValueError):
            TaskTimer("t1").add("bogus_ms", 1)

    def test_negative_duration_clamped(self):
        timer = TaskTimer("t1")
        timer.add("tool_ms", -100)
        assert timer.stages["tool_ms"] == 0.0

    def test_first_token_only_recorded_once(self):
        timer = TaskTimer("t1")
        assert timer.mark_first_token() is True
        first = timer.first_token_ms
        assert first is not None
        assert first >= 0
        assert timer.mark_first_token() is False
        assert timer.first_token_ms == first

    def test_snapshot_contains_all_fields(self):
        timer = TaskTimer("t1", query="q")
        timer.add("retrieve_ms", 100)
        timer.add("tool_ms", 200)
        timer.mark_first_token()
        snap = timer.snapshot()
        assert set(snap) == {
            "thread_id",
            "total_ms",
            "first_token_ms",
            "retrieve_ms",
            "llm_ms",
            "tool_ms",
            "subagent_ms",
            "tool_calls",
        }
        assert snap["thread_id"] == "t1"
        assert snap["retrieve_ms"] == 100
        # 真实 wall time 远小于手工累加的 300ms，反推的 llm_ms 会被夹到 0
        assert snap["llm_ms"] >= 0
        assert snap["total_ms"] >= 0

    def test_tool_calls(self):
        timer = TaskTimer("t1")
        timer.incr_tool_call()
        timer.incr_tool_call()
        assert timer.snapshot()["tool_calls"] == 2

    def test_contextvar_lifecycle(self):
        reset_timer()
        assert get_timer() is None
        timer = start_timer("t2", query="问题")
        assert get_timer() is timer
        reset_timer()
        assert get_timer() is None


class TestTaskMetricsStore:
    @staticmethod
    def _store(tmp_path):
        return TaskMetricsStore(db_path=tmp_path / "metrics.db")

    @staticmethod
    def _run(thread_id: str, **overrides):
        run = {
            "thread_id": thread_id,
            "query": "2026 年 Q1 GMV 是多少",
            "intent": "database",
            "intent_confidence": 0.91,
            "route_layer": "L2",
            "latency_ms": 1000.0,
            "first_token_ms": 300.0,
            "retrieve_ms": 50.0,
            "llm_ms": 900.0,
            "tool_ms": 50.0,
            "subagent_ms": 0.0,
            "tool_calls": 2,
            "memory_hits": 1,
            "status": "ok",
            "created_at": "2026-11-01T10:00:00",
        }
        run.update(overrides)
        return run

    def test_record_and_get(self, tmp_path):
        store = self._store(tmp_path)
        store.record(self._run("t1"))
        row = store.get("t1")
        assert row is not None
        assert row["intent"] == "database"
        assert row["latency_ms"] == 1000.0
        assert row["memory_hits"] == 1
        assert store.get("missing") is None

    def test_record_is_upsert(self, tmp_path):
        store = self._store(tmp_path)
        store.record(self._run("t1"))
        store.record(self._run("t1", latency_ms=1234.0, status="error"))
        assert len(store.list_runs()) == 1
        row = store.get("t1")
        assert row["latency_ms"] == 1234.0
        assert row["status"] == "error"

    def test_record_requires_thread_id(self, tmp_path):
        store = self._store(tmp_path)
        with pytest.raises(ValueError):
            store.record({"latency_ms": 1.0})

    def test_list_runs_limit(self, tmp_path):
        store = self._store(tmp_path)
        for i in range(5):
            store.record(self._run(f"t{i}", created_at=f"2026-11-01T10:00:0{i}"))
        rows = store.list_runs(limit=3)
        assert len(rows) == 3
        assert rows[0]["thread_id"] == "t4"  # 按时间倒序

    def test_aggregate(self, tmp_path):
        store = self._store(tmp_path)
        store.record(self._run("t1", latency_ms=100.0, first_token_ms=10.0, route_layer="L1", intent="web"))
        store.record(self._run("t2", latency_ms=200.0, first_token_ms=20.0, route_layer="L2", intent="database"))
        store.record(self._run("t3", latency_ms=300.0, first_token_ms=30.0, route_layer="L2", intent="database"))
        agg = store.aggregate()
        assert agg["count"] == 3
        assert agg["latency_ms"]["avg"] == pytest.approx(200.0)
        assert agg["latency_ms"]["p50"] == pytest.approx(200.0)
        assert agg["latency_ms"]["p95"] == pytest.approx(290.0)
        assert agg["first_token_ms"]["p50"] == pytest.approx(20.0)
        assert agg["intent_distribution"] == {"web": 1, "database": 2}
        assert agg["route_layer_distribution"] == {"L1": 1, "L2": 2}
        assert agg["status_distribution"] == {"ok": 3}

    def test_reset(self, tmp_path):
        store = self._store(tmp_path)
        store.record(self._run("t1"))
        store.reset()
        assert store.list_runs() == []
        assert store.aggregate()["count"] == 0
