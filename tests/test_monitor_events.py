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
