"""事件监控与 WebSocket 连接管理：工具/子智能体/流式 token/耗时/记忆事件上报与实时推送。

事件类型一览（前端 api/static/index.html 与评测脚本 evals/run_eval.py 消费）：
    session_created  工作目录创建
    tool_start       工具开始执行（tool_name/args）
    tool_result      工具执行结束（tool/output/chars/truncated）
    assistant_call   调用子智能体（assistant_name/args）
    intent           意图路由结果（label/label_zh/confidence/layer/suggested_subagents/scores）
    token            LLM 流式增量文本（高频，仅 WebSocket 推送）
    timing           任务耗时统计（timings/summary）
    memory           长期记忆召回/写入（retrieved/written）
    task_result      任务最终结果（result）
    error            异常
"""

import asyncio
import builtins
import datetime
import sys
from typing import Any, Dict, Optional

from fastapi import WebSocket

from api.context import get_thread_context

# 修复 Windows GBK 编码问题
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class ToolMonitor:
    """
    工具监控类，用于在任务执行过程中上报进度和状态。
    设计为单例模式，可在任何工具中直接导入使用。
    兼容 FastAPI WebSocket 和 脚本运行时的 stream_writer。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolMonitor, cls).__new__(cls)
            cls._instance.websocket_manager = None  # 预留给 FastAPI WebSocketManager
        return cls._instance

    def set_websocket_manager(self, manager):
        """设置 FastAPI 的 WebSocket 管理器"""
        self.websocket_manager = manager

    def _emit(
        self,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        broadcast_only: bool = False,
    ):
        """内部发送方法。

        broadcast_only=True 时只走 WebSocket（用于 token 这类高频增量事件，
        避免逐字 print / stream_writer 拖慢主流程、刷爆控制台）。
        """
        payload = {
            "type": "monitor_event",
            "event": event_type,
            "message": message,
            "data": data or {},
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # 1. 优先尝试通过 FastAPI WebSocket 发送 (定向推送)
        if self.websocket_manager:
            try:
                thread_id = get_thread_context()
                manager_loop = self.websocket_manager.loop
                if manager_loop and thread_id:
                    try:
                        current_loop = asyncio.get_running_loop()
                    except RuntimeError:
                        current_loop = None

                    if current_loop is not None and current_loop == manager_loop:
                        # 同一个事件循环：直接 create_task 效率最高
                        current_loop.create_task(
                            self.websocket_manager.send_to_thread(payload, thread_id)
                        )
                    else:
                        # 不同线程/循环：用线程安全方式投递，避免「协程在错误的循环中运行」
                        asyncio.run_coroutine_threadsafe(
                            self.websocket_manager.send_to_thread(payload, thread_id),
                            manager_loop,
                        )
                # thread_id 为空说明没有会话上下文（系统级消息），WebSocket 无目标，跳过
            except Exception as e:
                print(f"[Monitor] WebSocket send failed: {e}")

        if broadcast_only:
            return

        # 2. 尝试通过全局 runtime 输出 (DeepAgents 脚本模式)
        if hasattr(builtins, "runtime") and hasattr(builtins.runtime, "stream_writer"):
            try:
                builtins.runtime.stream_writer(payload)
            except Exception:
                pass

        # 3. 控制台保底输出 (方便调试)
        print(f"\n[Monitor:{event_type}] {message}")

    # ---------- 基础事件 ----------

    def report_tool(self, tool_name: str, args: Dict[str, Any] = None):
        """报告工具开始执行"""
        self._emit("tool_start", f"开始执行工具: {tool_name}", {"tool_name": tool_name, "args": args})

    def report_tool_result(
        self,
        tool_name: str,
        output: str,
        truncated: bool = False,
        max_chars: int = 1500,
    ):
        """报告工具执行结束及其输出（裁剪后），用于引用覆盖率统计与前端展示。"""
        full_text = output or ""
        preview = full_text
        if len(preview) > max_chars:
            preview = preview[:max_chars]
            truncated = True
        self._emit(
            "tool_result",
            f"工具执行完成: {tool_name}（{len(full_text)} 字符）",
            {
                "tool": tool_name,
                "output": preview,
                "chars": len(full_text),
                "truncated": truncated,
            },
        )

    def report_assistant(self, assistant_name: str, args: Dict[str, Any] = None):
        """报告正在调用的子智能体进度"""
        self._emit("assistant_call", f"正在调用助手: {assistant_name}",
                   {"assistant_name": assistant_name, "args": args})

    def report_intent(self, intent: Dict[str, Any]):
        """报告意图路由结果（分层路由：L1 规则 / L2 分类器 / L3 LLM / L4 澄清）"""
        label_zh = str(intent.get("label_zh") or intent.get("label") or "")
        try:
            confidence = float(intent.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        self._emit(
            "intent",
            f"意图路由: {label_zh}（置信度 {confidence:.2f}，路由层 {intent.get('layer', '')}）",
            dict(intent),
        )

    def report_token(self, text: str):
        """报告 LLM 流式增量文本（高频，仅 WebSocket 推送）。"""
        if not text:
            return
        self._emit("token", "", {"text": text}, broadcast_only=True)

    def report_timing(self, timings: Dict[str, Any], summary: Optional[Dict[str, Any]] = None):
        """报告任务耗时统计（端到端/首字/各阶段/工具调用次数）。"""
        self._emit("timing", "任务耗时统计完成", {"timings": timings, "summary": summary or {}})

    def report_memory(self, payload: Dict[str, Any]):
        """报告长期记忆的召回与写入情况。"""
        payload = payload or {}
        retrieved = payload.get("retrieved") or []
        written = payload.get("written") or []
        self._emit(
            "memory",
            f"长期记忆：召回 {len(retrieved)} 条 / 写入 {len(written)} 条",
            {"retrieved": retrieved, "written": written},
        )

    def report_task_result(self, result: str):
        """报告任务最终结果"""
        self._emit("task_result", "任务执行完成", {"result": result})

    def report_session_dir(self, path: str):
        """报告任务工作目录"""
        self._emit("session_created", f"工作目录已创建: {path}", {"path": path})


# 全局单例实例
monitor = ToolMonitor()


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        # 延迟绑定 loop，防止初始化时 loop 不一致
        self.loop = None

    def set_loop(self, loop):
        """显式设置事件循环"""
        self.loop = loop
        monitor.set_websocket_manager(self)
        print(f"[Monitor] ConnectionManager manually bound to loop: {id(self.loop)}")

    async def connect(self, websocket: WebSocket, thread_id: str):
        await websocket.accept()
        print(f"存储当前会话id:{thread_id}对应的:{websocket}")
        self.active_connections[thread_id] = websocket
        print(f"Client connected: {thread_id}")

    def disconnect(self, websocket: WebSocket, thread_id: str):
        if thread_id in self.active_connections:
            del self.active_connections[thread_id]
        print(f"Client disconnected: {thread_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def send_to_thread(self, message: dict, thread_id: str):
        if thread_id in self.active_connections:
            websocket = self.active_connections[thread_id]
            await websocket.send_json(message)


manager = ConnectionManager()
