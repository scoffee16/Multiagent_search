"""主智能体编排：意图路由 → 长期记忆检索 → 上下文组装 → 流式执行 → 记忆抽取。

相比早期版本，这个文件承担了 P0/P1 的几项集成工作：
1. P0 §3.4 分层意图路由：任务开始前先跑 classify_intent（L1 规则 → L2 分类器 →
   L3 LLM few-shot → L4 反问澄清），把路由结果作为参考提示注入，并通过意图事件上报；
2. P0 §3.5 流式输出：astream 同时订阅 updates 与 messages，把顶层 model 节点的
   增量 token 以 token 事件推给前端，从而可以统计首字延迟；
3. P0 §3.1 性能埋点：TaskTimer 记录检索/工具/子智能体/首字/端到端耗时，写入
   data/metrics.db，并通过 timing 事件与 /api/metrics 暴露；
4. P0 §3.7 长期记忆：任务前检索 top-5 记忆（计算类任务额外强制注入最近约束）注入
   `### 已知背景（长期记忆）`，任务后抽取 memory_candidates 写回（去重/置信度/时间衰减
   由 MemoryStore 负责）；
5. P1 §3.11 上下文管理：结构化槽位 + 记忆块 + 用户问题的分节组装与 token 预算。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Any

import aiosqlite
from deepagents import create_deep_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import memory as memory_module
from agent.context_manager import build_agent_input, extract_slots
from agent.intent_router import classify_intent
from agent.llm import model
from agent.memory import extract_memory_candidates
from agent.prompts import main_agent_content
from agent.subagents.database_query_agent import database_query_agent
from agent.subagents.knowledge_base_agent import knowledge_base_agent
from agent.subagents.network_search_agent import network_search_agent
from agent.timing import metrics_store, reset_timer, start_timer
from agent.tool_observer import instrument_tool
from api.context import (
    reset_session_context,
    set_session_context,
    set_thread_context,
)
from api.monitor import monitor
from tools.markdown_tools import generate_markdown
from tools.pdf_tools import convert_md_to_pdf
from tools.upload_file_read_tool import read_file_content

PROJECT_ROOT = Path(__file__).parents[1].resolve()
checkpoint_db_path = PROJECT_ROOT / "data" / "checkpoints.db"
checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)

# 全局懒加载（首次请求才建 SQLite 连接与 deep agent，避免 import 即连库）
_conn: aiosqlite.Connection | None = None
checkpointer: AsyncSqliteSaver | None = None
main_agent = None
_agent_lock = asyncio.Lock()

# token 事件聚合参数：攒够字数或间隔就推一批，避免逐字发 WebSocket 消息
_TOKEN_FLUSH_CHARS = 12
_TOKEN_FLUSH_SECONDS = 0.06

# 任务结束后是否自动抽取/写回长期记忆。
# 评测时建议设为 0：避免后台抽取的 LLM 调用与下一题并发竞争，干扰时延统计；
# 记忆链路由 evals/memory_eval.py 在开关打开的服务上单独验证。
_AUTO_MEMORY_WRITE = os.getenv("MEMORY_AUTO_WRITE", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
}


def get_memory_store() -> memory_module.MemoryStore:
    """进程级长期记忆仓库单例（与 agent.memory 默认仓库共享同一个 SQLite 连接）。"""
    return memory_module._get_default_store()


async def get_main_agent():
    """懒加载主智能体（SQLite checkpoint + 三个子智能体 + 工具观测包装）。"""
    global _conn, checkpointer, main_agent
    async with _agent_lock:
        if main_agent is not None:
            return main_agent
        print(f"[MainAgent] 初始化 checkpointer: {checkpoint_db_path}")
        _conn = await aiosqlite.connect(str(checkpoint_db_path))
        checkpointer = AsyncSqliteSaver(_conn)
        main_agent = create_deep_agent(
            model=model,
            system_prompt=main_agent_content["system_prompt"],
            tools=[
                instrument_tool(generate_markdown),
                instrument_tool(convert_md_to_pdf),
                instrument_tool(read_file_content),
            ],
            checkpointer=checkpointer,
            subagents=[database_query_agent, network_search_agent, knowledge_base_agent],
        )
        return main_agent


class _TokenEmitter:
    """增量 token 聚合器：攒够字数或超过时间阈值再调用 monitor.report_token。"""

    def __init__(self) -> None:
        self._buffer: list[str] = []
        self._buffered_chars = 0
        self._last_flush = time.perf_counter()

    def add(self, text: str) -> None:
        if not text:
            return
        self._buffer.append(text)
        self._buffered_chars += len(text)
        now = time.perf_counter()
        if self._buffered_chars >= _TOKEN_FLUSH_CHARS or now - self._last_flush >= _TOKEN_FLUSH_SECONDS:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        text = "".join(self._buffer)
        self._buffer.clear()
        self._buffered_chars = 0
        self._last_flush = time.perf_counter()
        monitor.report_token(text)


def _message_text(message: Any) -> str:
    """从 LangChain 消息里取出文本正文（兼容 str / 多模态 list）。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return ""


def _is_top_level_model_chunk(metadata: Any) -> bool:
    """只保留主图 model 节点的流式片段，过滤子智能体内部 LLM 的 token。

    LangGraph messages 模式会给每个片段带 metadata：
    - langgraph_node：产生该片段的节点名（主图/子图里都叫 model）；
    - langgraph_checkpoint_ns：命名空间，子图里会带 "|" 分隔的多级前缀。
    """
    if not isinstance(metadata, dict):
        return False
    if metadata.get("langgraph_node") != "model":
        return False
    namespace = str(metadata.get("langgraph_checkpoint_ns") or "")
    return "|" not in namespace


# 计算类问题的标记词：这类问题最容易踩口径（是否含退款/运费、保留几位小数、时间范围），
# 但口径类记忆与问题的字面重合非常低，向量检索召回不稳定，需要强制注入最近约束。
_CALCULATION_MARKERS: tuple[str, ...] = (
    "算",
    "统计",
    "多少",
    "合计",
    "汇总",
    "求和",
    "占比",
    "总额",
    "总量",
    "客单价",
    "gmv",
    "销售额",
    "同比",
    "环比",
    "利润率",
)


def _is_calculation_query(query: str) -> bool:
    """判断问题是否为数值计算/统计类（用于强制注入口径约束类记忆）。"""
    text = (query or "").lower()
    if not text:
        return False
    return any(marker in text for marker in _CALCULATION_MARKERS)


def _route_hint(intent_payload: dict) -> str:
    """把意图路由结果渲染成给主模型看的「参考提示」（不是硬约束）。"""
    label_zh = str(intent_payload.get("label_zh") or intent_payload.get("label") or "")
    try:
        confidence = float(intent_payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    layer = str(intent_payload.get("layer") or "")
    suggested = intent_payload.get("suggested_subagents") or []
    lines = [f"【意图路由提示】{label_zh}（置信度 {confidence:.2f}，路由层 {layer}）"]
    if suggested:
        lines.append("建议优先调用子智能体：" + "、".join(str(item) for item in suggested))
    lines.append("该提示来自分层意图路由，仅供参考，以问题实际意图为准。")
    return "\n".join(lines)


def _build_path_instruction(relative_session_dir_str: str, updated_info_prompt: str) -> str:
    return f"""
    【工作环境指令】
    工作目录: {relative_session_dir_str}
    {updated_info_prompt}

    规则：
    1. 新生成文件必须保存到工作目录：'{relative_session_dir_str}/filename'
    2. 读取已上传的文件时，请直接将文件名（例如：'开篇.txt'）作为 filename 参数传入（read_file_content）读取工具，不要带上任何目录前缀。
    3. 使用相对路径，禁止使用绝对路径
    4. 若存在上传文件，请先分析内容
    """


async def _read_final_answer_from_state(agent, config: dict) -> str:
    """兜底：流式过程中没拿到最终答案时，从 checkpoint 的最终状态里取最后一条 AI 文本。"""
    try:
        state = await agent.aget_state(config)
        messages = (state.values or {}).get("messages") or []
        for message in reversed(messages):
            if getattr(message, "type", None) != "ai":
                continue
            if getattr(message, "tool_calls", None):
                continue
            text = _message_text(message)
            if text:
                return text
    except Exception as exc:  # 兜底路径本身失败不能影响主流程
        print(f"[MainAgent] 读取最终状态失败: {exc}")
    return ""


def _prepare_session(task_query: str, session_id: str) -> tuple[str, str, str]:
    """创建工作目录、复制用户上传文件，返回 (session_dir_str, relative_dir, path_instruction)。"""
    session_dir = PROJECT_ROOT / "output" / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_dir_str = str(session_dir).replace("\\", "/")
    relative_session_dir_str = str(session_dir.relative_to(PROJECT_ROOT)).replace("\\", "/")

    updated_info_prompt = ""
    updated_dir_path = PROJECT_ROOT / "updated" / f"session_{session_id}"
    if updated_dir_path.exists() and updated_dir_path.is_dir():
        updated_files = [f for f in updated_dir_path.iterdir() if f.is_file()]
        if updated_files:
            for file_path in updated_files:
                try:
                    shutil.copy2(file_path, session_dir / file_path.name)
                except Exception as exc:
                    print(f"[MainAgent] 复制上传文件失败 {file_path.name}: {exc}")
            file_list = "\n".join(f"    - {f.name}" for f in updated_files)
            updated_info_prompt = (
                f"\n    [已上传文件] 已加载到工作目录:\n{file_list}\n"
                "    请优先使用工具（read_file_content）读取并参考这些文件。"
            )
    return (
        session_dir_str,
        relative_session_dir_str,
        _build_path_instruction(relative_session_dir_str, updated_info_prompt),
    )


async def run_deep_agent(task_query: str, session_id: str):
    """执行一次完整任务：路由 → 记忆检索 → 上下文组装 → 流式执行 → 指标 → 记忆写回。"""
    timer = start_timer(session_id, query=task_query)
    session_dir_token = None
    session_id_token = None
    final_answer = ""
    task_result_sent = False
    retrieved: list[dict] = []
    store = get_memory_store()
    status = "running"
    metrics_recorded = False

    def finish_metrics(final_status: str) -> None:
        """写入本次任务指标并推送 timing 事件（幂等）。"""
        nonlocal metrics_recorded
        if metrics_recorded:
            return
        metrics_recorded = True
        snapshot = timer.snapshot()
        try:
            metrics_store.record(
                {
                    "thread_id": session_id,
                    "query": task_query,
                    "intent": timer.intent,
                    "intent_confidence": timer.intent_confidence,
                    "route_layer": timer.route_layer,
                    "latency_ms": snapshot["total_ms"],
                    "first_token_ms": snapshot["first_token_ms"],
                    "retrieve_ms": snapshot["retrieve_ms"],
                    "llm_ms": snapshot["llm_ms"],
                    "tool_ms": snapshot["tool_ms"],
                    "subagent_ms": snapshot["subagent_ms"],
                    "tool_calls": snapshot["tool_calls"],
                    "memory_hits": timer.memory_hits,
                    "status": final_status,
                }
            )
        except Exception as exc:
            print(f"[MainAgent] 写入任务指标失败: {exc}")
        monitor.report_timing(
            snapshot,
            {
                "intent": timer.intent,
                "route_layer": timer.route_layer,
                "memory_hits": timer.memory_hits,
                "status": final_status,
            },
        )

    try:
        # 1. 工作目录与上传文件准备
        session_dir_str, _, path_instruction = _prepare_session(task_query, session_id)
        session_dir_token = set_session_context(session_dir_str)
        session_id_token = set_thread_context(session_id)
        monitor.report_session_dir(session_dir_str)

        # 2. 分层意图路由（P0 §3.4）
        prep_started = time.perf_counter()
        intent = await classify_intent(task_query)
        intent_payload = intent.to_dict() if hasattr(intent, "to_dict") else dict(intent)
        monitor.report_intent(intent_payload)
        timer.intent = str(intent_payload.get("label") or "")
        timer.intent_confidence = float(intent_payload.get("confidence") or 0.0)
        timer.route_layer = str(intent_payload.get("layer") or "")

        needs_clarification = bool(intent_payload.get("needs_clarification"))
        if needs_clarification:
            final_answer = str(
                intent_payload.get("clarification_question")
                or "为了更准确地回答，请补充说明具体的时间范围、指标口径或业务对象。"
            )
            monitor.report_task_result(final_answer)
            status = "clarified"
            finish_metrics(status)
            return

        # 3. 长期记忆检索 + 上下文组装（P0 §3.7 / P1 §3.11）
        # 取 top-5 而非常规的 top-3：实测约束类记忆（"统计时排除退款订单"）与数值类问题的
        # 字面重叠很低，top-3 很容易被事实类记忆挤出，导致模型看不到关键口径。
        items = store.search(task_query, top_k=5)
        # 省略指代兜底：用户说"那段时间/上次/之前提到的"时，查询与记忆措辞几乎不重叠，
        # 向量检索常常召不回，此时注入最近写入的记忆比什么都不注入更有用。
        if not items and memory_module.needs_recent_memory(task_query):
            items = store.recent(limit=3)
        # 计算类任务强制注入最近约束类记忆：口径/禁止项（"不要算退款订单""保留两位小数"）
        # 与问题本身几乎没有字面重合，再好的字面向量检索也容易漏，索性让模型每算必读。
        if _is_calculation_query(task_query):
            known_ids = {item.id for item in items}
            for extra in store.recent(limit=2, types=("constraint",)):
                if extra.id not in known_ids:
                    items.append(extra)
                    known_ids.add(extra.id)
        items = items[:7]
        memory_block = store.render_for_prompt(items) if items else ""
        retrieved = [item.to_dict() for item in items]
        timer.memory_hits = len(items)
        slots = extract_slots(task_query)
        user_content = (
            _route_hint(intent_payload)
            + "\n\n"
            + build_agent_input(task_query, memory_block=memory_block, slots=slots)
            + path_instruction
        )
        timer.add("retrieve_ms", (time.perf_counter() - prep_started) * 1000.0)

        # 4. 流式执行（P0 §3.5）
        config = {"configurable": {"thread_id": session_id}}
        agent = await get_main_agent()
        emitter = _TokenEmitter()
        streamed_text = ""
        task_started_at: float | None = None
        async for mode, payload in agent.astream(
            {"messages": [{"role": "user", "content": user_content}]},
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if mode == "messages":
                if isinstance(payload, tuple) and len(payload) == 2:
                    chunk, metadata = payload
                else:
                    chunk, metadata = payload, {}
                if not _is_top_level_model_chunk(metadata):
                    continue
                if getattr(chunk, "tool_call_chunks", None):
                    continue
                text = _message_text(chunk)
                if not text:
                    continue
                timer.mark_first_token()
                emitter.add(text)
                streamed_text += text
                continue

            # mode == "updates"：沿用早期版本对顶层节点状态的处理方式
            for node_name, state in (payload or {}).items():
                if not state or "messages" not in state:
                    continue
                messages = state.get("messages") or []
                if not isinstance(messages, list) or not messages:
                    continue
                last_msg = messages[-1]
                if node_name == "model":
                    tool_calls = getattr(last_msg, "tool_calls", None) or []
                    if tool_calls:
                        for tool_call in tool_calls:
                            if tool_call.get("name") != "task":
                                continue
                            args = tool_call.get("args") or {}
                            assistant_name = args.get("subagent_type") or ""
                            if assistant_name:
                                monitor.report_assistant(
                                    assistant_name,
                                    {"description": args.get("description")},
                                )
                            task_started_at = time.perf_counter()
                    else:
                        content = _message_text(last_msg)
                        if content:
                            final_answer = content
                            monitor.report_task_result(content)
                            task_result_sent = True
                elif node_name == "tools" and task_started_at is not None:
                    if any(getattr(msg, "name", None) == "task" for msg in messages):
                        timer.add("subagent_ms", (time.perf_counter() - task_started_at) * 1000.0)
                        task_started_at = None

        emitter.flush()
        final_answer = final_answer or streamed_text or await _read_final_answer_from_state(agent, config)
        if final_answer and not task_result_sent:
            monitor.report_task_result(final_answer)
            task_result_sent = True
        status = "ok"
        finish_metrics(status)

        # 5. 任务后记忆抽取与写回（失败不影响主流程；评测可关闭以隔离时延）
        written: list[int] = []
        if _AUTO_MEMORY_WRITE:
            try:
                candidates = await extract_memory_candidates(task_query, final_answer)
                if candidates:
                    written = store.add_candidates(session_id, candidates, source=f"thread:{session_id}")
            except Exception as exc:
                print(f"[MainAgent] 记忆抽取/写回失败（不影响主流程）: {exc}")
        if retrieved or written:
            monitor.report_memory({"retrieved": retrieved, "written": written})
    except Exception as e:
        status = "error"
        monitor._emit("error", f"执行主智能发生异常信息：{str(e)}")
        finish_metrics("error")
    finally:
        if not metrics_recorded:
            finish_metrics(status)
        if session_dir_token is not None and session_id_token is not None:
            reset_session_context(session_dir_token, session_id_token)
        reset_timer()
