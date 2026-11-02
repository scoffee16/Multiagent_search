"""工具观测包装：统一采集「工具耗时 + 工具输出」，供耗时埋点与引用覆盖率评测使用。

背景：
    各工具内部已经调用 monitor.report_tool(...) 上报「开始执行」，
    但没有统一的地方拿到「执行耗时」和「输出内容」。
    主智能体工具与子智能体工具都在注册处经过 instrument_tool() 包装后：
      1. 累加当前任务的 tool_ms（agent.timing）；
      2. 把输出（裁剪后）通过 monitor.report_tool_result(...) 推给前端/评测；
    工具本身的实现（tools/*.py）不需要改动。
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from agent.context_manager import clip_tool_output
from agent.timing import get_timer
from api.monitor import monitor

# 同一个工具对象只包装一次（模块可能被多个子智能体/主智能体重复导入）
_wrapped_cache: dict[int, BaseTool] = {}

# 工具输出事件里最多带多少字符（避免超长 CSV/搜索结果撑爆 WebSocket 与内存）
DEFAULT_OUTPUT_PREVIEW_CHARS = 1500


def _stringify(result: Any) -> str:
    """把工具返回值转成可传输的短文本。"""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    try:
        import json

        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(result)


def _preview(text: str, max_chars: int = DEFAULT_OUTPUT_PREVIEW_CHARS) -> tuple[str, bool]:
    """裁剪工具输出为预览：超过 max_chars 时保留头尾并标注裁剪字数。"""
    if len(text) <= max_chars:
        return text, False
    head_len = int(max_chars * 0.7)
    tail_len = max(0, max_chars - head_len)
    head = text[:head_len]
    tail = text[-tail_len:] if tail_len else ""
    omitted = len(text) - head_len - tail_len
    return f"{head}\n...[已裁剪 {omitted} 字符]...\n{tail}", True


def instrument_tool(tool_obj: BaseTool) -> BaseTool:
    """包装一个 LangChain 工具：记录耗时、推送输出预览。

    - 保持工具名 / 描述 / 参数 schema 不变，模型侧无感知；
    - 同步工具在 worker 线程执行，ContextVar 由 asyncio.to_thread 透传，因此能拿到 TaskTimer。
    """
    cached = _wrapped_cache.get(id(tool_obj))
    if cached is not None:
        return cached

    original = tool_obj
    tool_name = original.name
    description = original.description or ""
    args_schema = original.args_schema

    def _observed_run(**kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            result = original.invoke(kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            timer = get_timer()
            if timer is not None:
                timer.add("tool_ms", elapsed_ms)
                timer.incr_tool_call()
        # 上下文预算：超长工具输出先裁剪再交给模型（头 70% + 尾 + 裁剪标记），
        # 避免单次 SQL/检索结果把上下文撑爆；裁剪后的文本同时用于前端预览与引用覆盖率评测。
        if isinstance(result, str):
            result = clip_tool_output(result)
        # 成功路径：把（裁剪后的）工具输出推给前端/评测，用于引用覆盖率统计
        try:
            preview, truncated = _preview(_stringify(result))
            monitor.report_tool_result(tool_name, preview, truncated)
        except Exception:  # 上报失败不能影响工具本身
            pass
        return result

    wrapped = StructuredTool.from_function(
        func=_observed_run,
        name=tool_name,
        description=description,
        args_schema=args_schema,
    )
    _wrapped_cache[id(original)] = wrapped
    return wrapped


def instrument_tools(tools: list[BaseTool]) -> list[BaseTool]:
    """批量包装工具列表，返回新列表（不修改入参）。"""
    return [instrument_tool(tool) for tool in tools]
