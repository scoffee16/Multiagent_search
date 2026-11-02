"""任务耗时埋点：分阶段计时（ContextVar 传递）+ SQLite 指标存储与 P50/P95 聚合。

设计目标（对应需求 P0 §3.1）：
1. 一次任务从开始到结束，记录四类数字：
   - 端到端耗时（latency_ms，P50/P95）
   - 首字延迟（first_token_ms，仅流式模式下可测）
   - 各阶段耗时占比（检索/准备 retrieve_ms、LLM 生成 llm_ms、工具调用 tool_ms、子智能体 subagent_ms）
   - 工具调用次数（tool_calls）
2. 指标写进 SQLite（data/metrics.db，随 data/ 一起被 .gitignore 忽略），
   通过 /api/metrics 与 /api/task/{thread_id}/timings 对外提供，供前端耗时面板与评测脚本使用。

为什么用 ContextVar：
    FastAPI 里每个会话是独立协程，工具的调用链由 asyncio 的 context 透传，
    用一个进程级 ContextVar 挂当前 TaskTimer，工具层（agent/tool_observer.py）无需显式传参就能写回耗时。
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections.abc import Iterable, Sequence
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "metrics.db"

# 阶段名 -> 语义；llm_ms 由主智能体在收尾时用总耗时反推，其余由埋点直接累加
STAGES: tuple[str, ...] = ("retrieve_ms", "llm_ms", "tool_ms", "subagent_ms")


def percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数；空输入返回 0.0。"""
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (max(0.0, min(100.0, p)) / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class TaskTimer:
    """单个任务的计时器。

    线程安全（工具可能在 worker 线程里调用 add()），且可以按阶段累加多次
    （例如一个任务里 LLM 会调用多轮工具，每次工具都累加到 tool_ms）。
    """

    def __init__(self, thread_id: str, query: str = ""):
        self.thread_id = thread_id
        self.query = query
        self.stages: dict[str, float] = {stage: 0.0 for stage in STAGES}
        self.first_token_ms: float | None = None
        self.tool_calls: int = 0
        self.memory_hits: int = 0
        self.intent: str = ""
        self.intent_confidence: float = 0.0
        self.route_layer: str = ""
        self.status: str = "running"
        self.started_at: float = time.perf_counter()
        self._lock = threading.Lock()

    def add(self, stage: str, ms: float) -> None:
        """累加某个阶段耗时（毫秒，负值按 0 处理）。"""
        if stage not in STAGES:
            raise ValueError(f"未知阶段: {stage}，可选: {STAGES}")
        with self._lock:
            self.stages[stage] += max(0.0, float(ms))

    def incr_tool_call(self) -> None:
        with self._lock:
            self.tool_calls += 1

    def mark_first_token(self) -> bool:
        """记录首字延迟；只认第一次。返回本次是否成功记录。"""
        with self._lock:
            if self.first_token_ms is not None:
                return False
            self.first_token_ms = (time.perf_counter() - self.started_at) * 1000.0
            return True

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000.0

    def snapshot(self) -> dict[str, Any]:
        """当前累计快照；llm_ms 用「总耗时 - 检索/工具/子智能体」反推（不小于 0）。"""
        with self._lock:
            stages = dict(self.stages)
            tool_calls = self.tool_calls
            first_token_ms = self.first_token_ms
        total_ms = self.elapsed_ms()
        measured = stages["retrieve_ms"] + stages["tool_ms"] + stages["subagent_ms"]
        stages["llm_ms"] = max(0.0, total_ms - measured)
        return {
            "thread_id": self.thread_id,
            "total_ms": round(total_ms, 1),
            "first_token_ms": round(first_token_ms, 1) if first_token_ms is not None else None,
            "retrieve_ms": round(stages["retrieve_ms"], 1),
            "llm_ms": round(stages["llm_ms"], 1),
            "tool_ms": round(stages["tool_ms"], 1),
            "subagent_ms": round(stages["subagent_ms"], 1),
            "tool_calls": tool_calls,
        }


# ---------------- ContextVar 传递 ----------------

_current_timer: ContextVar[TaskTimer | None] = ContextVar("task_timer", default=None)


def start_timer(thread_id: str, query: str = "") -> TaskTimer:
    timer = TaskTimer(thread_id, query)
    _current_timer.set(timer)
    return timer


def get_timer() -> TaskTimer | None:
    return _current_timer.get()


def reset_timer() -> None:
    _current_timer.set(None)


# ---------------- SQLite 指标存储 ----------------


class TaskMetricsStore:
    """任务指标 SQLite 存储：按 thread_id 主键幂等写入，并提供聚合分位数。"""

    def __init__(self, db_path: str | Path | None = None):
        env_path = os.getenv("METRICS_DB_PATH")
        self.db_path = Path(db_path or env_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_runs (
                    thread_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL DEFAULT '',
                    intent TEXT NOT NULL DEFAULT '',
                    intent_confidence REAL NOT NULL DEFAULT 0,
                    route_layer TEXT NOT NULL DEFAULT '',
                    latency_ms REAL NOT NULL DEFAULT 0,
                    first_token_ms REAL,
                    retrieve_ms REAL NOT NULL DEFAULT 0,
                    llm_ms REAL NOT NULL DEFAULT 0,
                    tool_ms REAL NOT NULL DEFAULT 0,
                    subagent_ms REAL NOT NULL DEFAULT 0,
                    tool_calls INTEGER NOT NULL DEFAULT 0,
                    memory_hits INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'ok',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_runs_created ON task_runs(created_at)")

    def record(self, run: dict[str, Any]) -> None:
        """写入/覆盖一条任务指标（thread_id 冲突则更新）。"""
        thread_id = str(run.get("thread_id") or "")
        if not thread_id:
            raise ValueError("record 需要 thread_id")
        row = {
            "thread_id": thread_id,
            "query": str(run.get("query") or "")[:500],
            "intent": str(run.get("intent") or ""),
            "intent_confidence": float(run.get("intent_confidence") or 0.0),
            "route_layer": str(run.get("route_layer") or ""),
            "latency_ms": float(run.get("latency_ms") or 0.0),
            "first_token_ms": run.get("first_token_ms"),
            "retrieve_ms": float(run.get("retrieve_ms") or 0.0),
            "llm_ms": float(run.get("llm_ms") or 0.0),
            "tool_ms": float(run.get("tool_ms") or 0.0),
            "subagent_ms": float(run.get("subagent_ms") or 0.0),
            "tool_calls": int(run.get("tool_calls") or 0),
            "memory_hits": int(run.get("memory_hits") or 0),
            "status": str(run.get("status") or "ok"),
            "created_at": str(run.get("created_at") or _now_iso()),
        }
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO task_runs (
                    thread_id, query, intent, intent_confidence, route_layer, latency_ms,
                    first_token_ms, retrieve_ms, llm_ms, tool_ms, subagent_ms,
                    tool_calls, memory_hits, status, created_at
                ) VALUES (
                    :thread_id, :query, :intent, :intent_confidence, :route_layer, :latency_ms,
                    :first_token_ms, :retrieve_ms, :llm_ms, :tool_ms, :subagent_ms,
                    :tool_calls, :memory_hits, :status, :created_at
                )
                ON CONFLICT(thread_id) DO UPDATE SET
                    query=excluded.query,
                    intent=excluded.intent,
                    intent_confidence=excluded.intent_confidence,
                    route_layer=excluded.route_layer,
                    latency_ms=excluded.latency_ms,
                    first_token_ms=excluded.first_token_ms,
                    retrieve_ms=excluded.retrieve_ms,
                    llm_ms=excluded.llm_ms,
                    tool_ms=excluded.tool_ms,
                    subagent_ms=excluded.subagent_ms,
                    tool_calls=excluded.tool_calls,
                    memory_hits=excluded.memory_hits,
                    status=excluded.status,
                    created_at=excluded.created_at
                """,
                row,
            )

    def get(self, thread_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM task_runs WHERE thread_id = ?", (thread_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM task_runs ORDER BY created_at DESC, rowid DESC LIMIT ?", (max(1, int(limit)),)
            ).fetchall()
        return [dict(row) for row in rows]

    def aggregate(self) -> dict[str, Any]:
        """聚合统计：整体 P50/P95、首字延迟、各阶段耗时、工具调用、意图/路由分布。"""
        runs = self.list_runs(limit=1000)
        latency = [r["latency_ms"] for r in runs if r.get("latency_ms")]
        first_token = [r["first_token_ms"] for r in runs if r.get("first_token_ms") is not None]
        stages = {
            stage: [r[stage] for r in runs if r.get(stage) is not None]
            for stage in STAGES
        }
        tool_calls = [r["tool_calls"] for r in runs if r.get("tool_calls")]

        def stats(values: list[float]) -> dict[str, float]:
            return {"avg": round(sum(values) / len(values), 1) if values else 0.0,
                    "p50": round(percentile(values, 50), 1),
                    "p95": round(percentile(values, 95), 1)}

        intent_dist: dict[str, int] = {}
        layer_dist: dict[str, int] = {}
        for r in runs:
            if r.get("intent"):
                intent_dist[r["intent"]] = intent_dist.get(r["intent"], 0) + 1
            if r.get("route_layer"):
                layer_dist[r["route_layer"]] = layer_dist.get(r["route_layer"], 0) + 1

        return {
            "count": len(runs),
            "latency_ms": stats(latency),
            "first_token_ms": stats(first_token),
            "stages": {stage: stats(values) for stage, values in stages.items()},
            "tool_calls": stats(tool_calls),
            "intent_distribution": intent_dist,
            "route_layer_distribution": layer_dist,
            "status_distribution": _count_by(runs, "status"),
        }

    def reset(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM task_runs")


def _count_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    dist: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if value:
            dist[value] = dist.get(value, 0) + 1
    return dist


# 全局单例：进程内共享一个 SQLite 指标库（服务端与评测脚本都可直接 import）
metrics_store = TaskMetricsStore()
