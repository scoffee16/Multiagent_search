"""monitor 事件总线的单测。

monitor 是全局单例，事件有三条出口：WebSocketManager（需要事件循环）、
`builtins.runtime.stream_writer`（脚本模式）、print。这里用最后一个出口做断言，
既不依赖事件循环，也不依赖前端。
"""

import builtins
import datetime
import types

import pytest

from api.monitor import ToolMonitor, monitor


class _Recorder:
    """冒名顶替 stream_writer，把 payload 收集起来。"""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def stream_writer(self, payload: dict) -> None:
        self.events.append(payload)


@pytest.fixture()
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(builtins, "runtime", types.SimpleNamespace(stream_writer=rec.stream_writer), raising=False)
    # 关掉 WebSocket 出口，否则单例会去找事件循环
    monkeypatch.setattr(monitor, "websocket_manager", None, raising=False)
    return rec


def test_monitor_is_singleton():
    assert ToolMonitor() is monitor


def test_report_tool_payload(recorder):
    monitor.report_tool("数据库表名查询工具：list_sql_tables", {"table_name": "orders"})

    assert len(recorder.events) == 1
    ev = recorder.events[0]
    assert ev["type"] == "monitor_event"
    assert ev["event"] == "tool_start"
    assert "list_sql_tables" in ev["message"]
    assert ev["data"]["tool_name"] == "数据库表名查询工具：list_sql_tables"
    assert ev["data"]["args"] == {"table_name": "orders"}


def test_report_assistant_payload(recorder):
    monitor.report_assistant("数据智能体", {"q": "统计订单"})

    ev = recorder.events[0]
    assert ev["event"] == "assistant_call"
    assert ev["data"]["assistant_name"] == "数据智能体"


def test_report_task_result_payload(recorder):
    monitor.report_task_result("# 报告\n内容")

    ev = recorder.events[0]
    assert ev["event"] == "task_result"
    assert ev["data"]["result"] == "# 报告\n内容"


def test_report_session_dir_payload(recorder):
    monitor.report_session_dir(r"D:\project\output\session_1")

    ev = recorder.events[0]
    assert ev["event"] == "session_created"
    assert ev["data"]["path"].endswith("session_1")


def test_timestamp_is_iso8601(recorder):
    monitor.report_tool("x")

    ts = recorder.events[0]["timestamp"]
    assert isinstance(ts, str)
    # 解析得动说明是合法的 ISO 时间，前端 new Date(ts) 才不会 NaN
    assert datetime.datetime.fromisoformat(ts).year >= 2024


def test_emit_without_runtime_still_prints(monkeypatch, capsys):
    """没有 runtime、没有 websocket 时也不能抛异常（服务端就是这种态）。"""
    monkeypatch.delattr(builtins, "runtime", raising=False)
    monkeypatch.setattr(monitor, "websocket_manager", None, raising=False)

    monitor.report_task_result("ok")

    out = capsys.readouterr().out
    assert "[Monitor:task_result]" in out


def test_report_tool_result_payload(recorder):
    monitor.report_tool_result("execute_sql_query", "col\n1", truncated=False)

    ev = recorder.events[0]
    assert ev["event"] == "tool_result"
    assert ev["data"]["tool"] == "execute_sql_query"
    assert ev["data"]["output"] == "col\n1"
    assert ev["data"]["chars"] == 5
    assert ev["data"]["truncated"] is False


def test_report_tool_result_truncates_long_output(recorder):
    monitor.report_tool_result("execute_sql_query", "x" * 5000, max_chars=100)

    ev = recorder.events[0]
    assert ev["data"]["truncated"] is True
    assert ev["data"]["chars"] == 5000
    assert len(ev["data"]["output"]) == 100


def test_report_intent_payload(recorder):
    intent = {
        "label": "database",
        "label_zh": "经营数据",
        "confidence": 0.87,
        "layer": "L2-classifier",
        "suggested_subagents": ["数据分析助手"],
        "scores": {"database": 0.87},
    }
    monitor.report_intent(intent)

    ev = recorder.events[0]
    assert ev["event"] == "intent"
    assert "经营数据" in ev["message"]
    assert "0.87" in ev["message"]
    assert ev["data"]["label"] == "database"
    assert ev["data"]["suggested_subagents"] == ["数据分析助手"]


def test_report_timing_payload(recorder):
    timings = {"total_ms": 1234.0, "first_token_ms": 321.0, "retrieve_ms": 40.0, "tool_calls": 2}
    monitor.report_timing(timings, {"note": "x"})

    ev = recorder.events[0]
    assert ev["event"] == "timing"
    assert ev["data"]["timings"] == timings
    assert ev["data"]["summary"] == {"note": "x"}


def test_report_memory_payload(recorder):
    monitor.report_memory({"retrieved": [{"id": 1, "content": "只看 Q1"}], "written": [2, 3]})

    ev = recorder.events[0]
    assert ev["event"] == "memory"
    assert ev["data"]["retrieved"][0]["id"] == 1
    assert ev["data"]["written"] == [2, 3]
    assert "召回 1 条" in ev["message"]


def test_report_token_is_websocket_only(recorder, capsys):
    """token 是高频事件：只走 WebSocket，不写 stream_writer / 控制台。"""
    monitor.report_token("你好")

    # 没有 websocket 时 token 事件不应出现在 stream_writer 与 stdout 里
    assert recorder.events == []
    assert "你好" not in capsys.readouterr().out


def test_report_token_ignores_empty_text(recorder):
    monitor.report_token("")
    assert recorder.events == []


def test_instrument_tool_clips_long_output(recorder):
    """超长工具输出先按 token 预算裁剪再交给模型，预览事件带 truncated=True。"""
    from langchain_core.tools import tool

    from agent.tool_observer import instrument_tool

    @tool
    def long_output_tool() -> str:
        """返回超长文本的测试工具。"""
        return "数" * 5000

    wrapped = instrument_tool(long_output_tool)
    result = wrapped.invoke({})

    assert len(result) < 5000
    assert "已裁剪" in result
    ev = recorder.events[-1]
    assert ev["event"] == "tool_result"
    assert ev["data"]["truncated"] is True
