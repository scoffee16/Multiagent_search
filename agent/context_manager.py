"""P1 §3.11 上下文管理：预算估算、裁剪、摘要、结构化槽位与提示词分节注入。

设计定位：deepagents 的 ``create_deep_agent`` 内部已经挂了 ``SummarizationMiddleware``
（见 ``deepagents/graph.py`` 的中间件栈），会在历史逼近模型窗口时自动摘要并把旧消息
落盘；本模块负责框架之外、业务侧可控的显式上下文策略：

* 预算：``ContextBudget`` 给记忆 / 历史 / 工具输出 / 槽位分别定上限，并提供总预算；
* 裁剪：``clip_text`` / ``clip_tool_output`` 保留头尾、砍掉中间，防止长工具输出撑爆窗口；
* 历史：``trim_history`` 保留 system 与最近若干轮，再按 query 关键词重叠补回旧的相关片段；
* 摘要：``summarize_history`` 提供不依赖外部服务的确定性兜底，以及基于 LLM 的要点压缩；
* 槽位：``StructuredSlots`` / ``extract_slots`` / ``render_slots`` 把时间、品类、指标、
  对比方式、样本量风险结构化后注入提示词；
* 注入：``build_agent_input`` 按「任务槽位 → 已知背景 → 用户问题」拼装模型输入，
  并按 ``ContextBudget.total_tokens`` 做总预算控制；
* 中间件：``summarize_middleware`` 生成 langchain 的 ``SummarizationMiddleware`` 配置。

token 估算口径与 ``evals/run_eval.py`` 的 ``CHARS_PER_TOKEN`` 一致：中文约 1.5 字符 ≈ 1 token，
这里统一向上取整（宁可高估，避免实际请求超窗口）。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

# 中文约 1.5 字符 1 token；与 evals/run_eval.py 的 CHARS_PER_TOKEN 保持一致
CHARS_PER_TOKEN = 1.5

# 提示词分节标题：集中定义，文档与测试直接引用这些常量
SECTION_SLOTS = "### 任务槽位"
SECTION_MEMORY = "### 已知背景（长期记忆）"
SECTION_QUERY = "### 用户问题"


# ---------------------------------------------------------------------------
# 一、token 估算与裁剪
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """按「中文约 1.5 字符 = 1 token」估算 token 数（向上取整）。

    Args:
        text: 待估算文本；空串 / ``None`` 返回 0。

    Returns:
        估算 token 数，非负整数。
    """
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _default_clip_marker(removed: int) -> str:
    """clip_text 未显式给 marker 时使用的默认裁剪标记。"""
    return f"\n...[已裁剪 {removed} 字符]...\n"


def clip_text(text: str, max_tokens: int, *, head_ratio: float = 0.7, marker: str | None = None) -> str:
    """把长文本裁剪到 ``max_tokens`` 以内，保留头尾、中间插入裁剪标记。

    Args:
        text: 原始文本。
        max_tokens: token 上限；``<= 0`` 时返回空串。
        head_ratio: 头部保留比例（0~1，越界自动夹取），默认 0.7。
        marker: 自定义裁剪标记；``None`` 时使用 ``"...[已裁剪 N 字符]..."`` 形式
            （带实际被裁字符数）。

    Returns:
        裁剪后的文本；原文本未超预算时原样返回。
    """
    if not text or max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    head_ratio = min(max(head_ratio, 0.0), 1.0)
    budget_chars = max(1, int(max_tokens * CHARS_PER_TOKEN))

    if marker is None:
        # 先按「被裁字符数最多 len(text) 位」预留标记长度，保证最终输出不超过字符预算
        digits = max(1, len(str(len(text))))
        reserve = len(_default_clip_marker(10**digits - 1))
        if budget_chars <= reserve + 2:
            # 预算太小放不下标记，退化为只保留头部
            return text[:budget_chars]
        available = budget_chars - reserve
        head_chars = int(available * head_ratio)
        tail_chars = available - head_chars
        removed = len(text) - head_chars - tail_chars
        head = text[:head_chars]
        tail = text[-tail_chars:] if tail_chars > 0 else ""
        return head + _default_clip_marker(removed) + tail

    reserve = len(marker)
    if budget_chars <= reserve + 2:
        return text[:budget_chars]
    available = budget_chars - reserve
    head_chars = int(available * head_ratio)
    tail_chars = available - head_chars
    head = text[:head_chars]
    tail = text[-tail_chars:] if tail_chars > 0 else ""
    return head + marker + tail


def clip_tool_output(text: str, max_tokens: int = 1500) -> str:
    """裁剪工具输出：保留头尾，中间插入 "...[已裁剪 N 字符]..." 标记。

    Args:
        text: 工具返回的原始文本。
        max_tokens: token 上限，默认 1500。

    Returns:
        裁剪后的文本；原文本未超预算时原样返回。
    """
    if not text or max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    budget_chars = max(1, int(max_tokens * CHARS_PER_TOKEN))
    digits = max(1, len(str(len(text))))
    reserve = len(f"...[已裁剪 {10**digits - 1} 字符]...")
    if budget_chars <= reserve + 2:
        return text[:budget_chars]

    available = budget_chars - reserve
    head_chars = int(available * 0.7)
    tail_chars = available - head_chars
    removed = len(text) - head_chars - tail_chars
    marker = f"...[已裁剪 {removed} 字符]..."
    head = text[:head_chars]
    tail = text[-tail_chars:] if tail_chars > 0 else ""
    return head + marker + tail


# ---------------------------------------------------------------------------
# 二、历史裁剪：保留 system + 最近轮，再按 query 相关性补回旧片段
# ---------------------------------------------------------------------------


def _content_text(content: Any) -> str:
    """把消息 content 统一转成纯文本（兼容 str 与多段 content 列表）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _message_text(message: Any) -> str:
    """取单条消息的正文文本（消息为 dict，字段 role/content）。"""
    if not isinstance(message, dict):
        return ""
    return _content_text(message.get("content"))


def _message_role(message: Any) -> str:
    """取单条消息的 role（不是 dict 或没有 role 时返回空串）。"""
    if isinstance(message, dict):
        return str(message.get("role") or "")
    return ""


_ASCII_TERM_RE = re.compile(r"[a-z0-9]{2,}")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def _query_terms(query: str) -> set[str]:
    """把 query 拆成用于重叠打分的词项：英文/数字词 + 中文相邻二字组。"""
    low = (query or "").casefold()
    terms = set(_ASCII_TERM_RE.findall(low))
    for run in _CJK_RUN_RE.findall(low):
        if len(run) < 2:
            terms.add(run)
        else:
            terms.update(run[i : i + 2] for i in range(len(run) - 1))
    return terms


def _relevance(message: Any, terms: set[str]) -> int:
    """旧消息与 query 的相关度：命中的词项个数。"""
    if not terms:
        return 0
    low = _message_text(message).casefold()
    return sum(1 for term in terms if term in low)


def trim_history(
    messages: list[dict[str, Any]],
    *,
    max_turns: int = 6,
    query: str = "",
    max_tokens: int = 1200,
) -> list[dict[str, Any]]:
    """裁剪历史消息：保留 system 消息与最近 ``max_turns`` 轮，再补回旧的相关片段。

    规则：
    1. 所有 ``role == "system"`` 的消息无条件保留，位置不变；
    2. 最近 ``max_turns`` 轮（以 user 消息为轮起点）无条件保留；
    3. 更早的消息按与 ``query`` 的关键词重叠度从高到低挑选，直到
       ``max_tokens`` 的剩余预算用完（最近轮已占用的 token 计入预算）；
    4. 返回新列表，消息保持原始先后顺序，消息对象本身不复制、不修改。

    Args:
        messages: 原始消息列表（dict，含 role/content）。
        max_turns: 保留最近多少轮；``<= 0`` 时只保留 system 消息。
        query: 当前用户问题，用于挑旧的相关消息；为空则不补旧消息。
        max_tokens: 非 system 消息的 token 预算。

    Returns:
        裁剪后的新消息列表。
    """
    if not messages:
        return []

    non_system: list[tuple[int, dict[str, Any]]] = [
        (index, message) for index, message in enumerate(messages) if _message_role(message) != "system"
    ]
    keep_indices: set[int] = {index for index, message in enumerate(messages) if _message_role(message) == "system"}

    if max_turns > 0:
        user_positions = [pos for pos, (_, message) in enumerate(non_system) if _message_role(message) == "user"]
        if len(user_positions) <= max_turns:
            recent_start = 0
        else:
            recent_start = user_positions[-max_turns]
    else:
        recent_start = len(non_system)

    recent = non_system[recent_start:]
    older = non_system[:recent_start]
    keep_indices.update(index for index, _ in recent)

    if query and older and max_tokens > 0:
        terms = _query_terms(query)
        used_tokens = sum(estimate_tokens(_message_text(message)) for _, message in recent)
        remaining = max_tokens - used_tokens
        # 按相关度从高到低挑；同分时优先离最近轮更近的旧消息
        scored = [(_relevance(message, terms), index) for index, message in older]
        for score, index in sorted(scored, key=lambda item: (item[0], item[1]), reverse=True):
            if score <= 0:
                break
            message = messages[index]
            cost = estimate_tokens(_message_text(message))
            if cost > remaining:
                continue
            remaining -= cost
            keep_indices.add(index)

    return [message for index, message in enumerate(messages) if index in keep_indices]


# ---------------------------------------------------------------------------
# 三、历史摘要：确定性兜底 + LLM 要点压缩
# ---------------------------------------------------------------------------


def _fallback_summary(messages: list[dict[str, Any]], max_tokens: int) -> str:
    """无 LLM 或 LLM 失败时的确定性摘要：role + 内容前 80 字符逐条拼接后裁剪。"""
    lines: list[str] = []
    for message in messages:
        role = _message_role(message) or "unknown"
        text = _message_text(message).replace("\n", " ").strip()
        lines.append(f"{role}: {text[:80]}")
    return clip_text("\n".join(lines), max_tokens, marker="\n...[摘要已裁剪]...\n")


def _summary_prompt(messages: list[dict[str, Any]]) -> str:
    """构造让 LLM 压缩历史为要点的提示词。"""
    lines: list[str] = []
    for message in messages:
        role = _message_role(message) or "unknown"
        text = _message_text(message).replace("\n", " ").strip()
        lines.append(f"{role}: {text[:300]}")
    dialog = "\n".join(lines)
    return (
        "请把下面这段多轮对话压缩成不超过 5 条要点的摘要，保留用户目标、关键约束、"
        "已经得到的结论和待办事项；只输出摘要正文，不要解释。\n\n"
        f"<对话>\n{dialog}\n</对话>"
    )


def _response_text(response: Any) -> str:
    """从 LLM 返回对象中取出文本正文。"""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    return _content_text(content)


async def summarize_history(llm: Any, messages: list[dict[str, Any]], *, max_tokens: int = 300) -> str:
    """把历史消息压缩成一段摘要文本。

    Args:
        llm: 支持异步 ``ainvoke`` 的聊天模型；为 ``None`` 时直接走确定性兜底。
        messages: 待摘要的历史消息。
        max_tokens: 摘要 token 上限。

    Returns:
        摘要文本；``max_tokens <= 0`` 时返回空串。
    """
    if not messages:
        return ""
    if max_tokens <= 0:
        return ""
    if llm is not None:
        try:
            response = await llm.ainvoke(_summary_prompt(messages))
            text = _response_text(response).strip()
        except Exception:  # noqa: BLE001 - 摘要失败不能拖垮主流程，统一走确定性兜底
            return _fallback_summary(messages, max_tokens)
        if text:
            return clip_text(text, max_tokens)
    return _fallback_summary(messages, max_tokens)


# ---------------------------------------------------------------------------
# 四、结构化槽位：时间 / 品类 / 指标 / 对比 / 样本量提醒
# ---------------------------------------------------------------------------


@dataclass
class StructuredSlots:
    """从用户问题里抽出的结构化任务槽位。

    Attributes:
        time_range: 时间范围，如 ``2026年Q1``、``2026-01``、``本月``、``近30天``。
        category: 品类，限定 ``数码电子`` / ``生活家电`` / ``美妆护肤`` / ``运动鞋服``。
        metric: 指标，如 ``GMV``、``客单价``、``订单数``、``退款率``、``品类占比``。
        comparison: 对比方式，``同比`` / ``环比`` / ``趋势`` / ``无``。
        sample_warning: 问题里出现「N 条 / 样本 / 够不够」等样本量风险提示时为 True。
    """

    time_range: str | None = None
    category: str | None = None
    metric: str | None = None
    comparison: str | None = None
    sample_warning: bool = False

    def is_empty(self) -> bool:
        """是否没有任何有效槽位。

        注意 ``comparison="无"`` 只表示「用户没提对比」，不算有效信号；
        只有时间 / 品类 / 指标 / 明确对比方式（同比/环比/趋势）/ 样本量提醒
        至少命中一个时才非空。
        """
        has_signal = bool(self.time_range or self.category or self.metric or self.sample_warning)
        has_comparison = self.comparison in {"同比", "环比", "趋势"}
        return not (has_signal or has_comparison)


_CN_DIGITS = {"一": "一", "二": "二", "三": "三", "四": "四", "1": "一", "2": "二", "3": "三", "4": "四"}

_YEAR_QUARTER_RE = re.compile(r"(20\d{2})\s*年?\s*[Qq]\s*([1-4])")
_YEAR_QUARTER_CN_RE = re.compile(r"(20\d{2})\s*年?\s*第\s*([一二三四1-4])\s*季度")
_YEAR_MONTH_RE = re.compile(r"(20\d{2})\s*[-/.年]\s*(\d{1,2})\s*月?")
_QUARTER_CN_RE = re.compile(r"第\s*([一二三四1-4])\s*季度")
_RECENT_DAYS_RE = re.compile(r"(?:近|最近)\s*(\d+)\s*天")
_RECENT_MONTHS_RE = re.compile(r"(?:近|最近)\s*(\d+)\s*个?\s*月")
_RECENT_WEEKS_RE = re.compile(r"(?:近|最近)\s*(\d+)\s*(?:周|星期)")
_RELATIVE_MONTH_RE = re.compile(r"(本|上|下)\s*月")
_RELATIVE_WEEK_RE = re.compile(r"(本|上|下)\s*(?:周|星期)")
_RELATIVE_QUARTER_RE = re.compile(r"(本|上|下)\s*季度")
_YEAR_WORD_RE = re.compile(r"(今年|去年|前年)")
_YEAR_RE = re.compile(r"(20\d{2})\s*年")

# 品类词：先整体判定，命中即返回规范写法
_CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("数码电子", ("数码电子", "数码产品", "数码", "电子产品", "手机", "笔记本电脑")),
    ("生活家电", ("生活家电", "小家电", "家电", "冰箱", "洗衣机", "空调")),
    ("美妆护肤", ("美妆护肤", "美妆", "护肤", "化妆品")),
    ("运动鞋服", ("运动鞋服", "运动服饰", "鞋服", "跑鞋", "服装")),
]

# 指标词：值为规范指标名，键为可能的同义表达
_METRIC_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("GMV", ("gmv", "成交额", "销售额", "交易额", "营收")),
    ("客单价", ("客单价", "客单", "平均订单金额")),
    ("订单数", ("订单数", "订单量", "单量", "订单笔数")),
    ("退款率", ("退款率", "退货率")),
    ("品类占比", ("品类占比", "占比", "份额")),
    ("转化率", ("转化率",)),
    ("毛利率", ("毛利率", "利润率", "利润")),
]

_SAMPLE_WARN_RE = re.compile(r"\d+\s*条|几条|样本|够不够|数据量|数据够|太少")


def _extract_time_range(query: str) -> str | None:
    """按优先级抽取时间范围，命中即返回规范写法。"""
    match = _YEAR_QUARTER_RE.search(query)
    if match:
        return f"{match.group(1)}年Q{match.group(2)}"
    match = _YEAR_QUARTER_CN_RE.search(query)
    if match:
        quarter = _CN_DIGITS.get(match.group(2), match.group(2))
        if quarter in "一二三四":
            return f"{match.group(1)}年Q{'一二三四'.index(quarter) + 1}"
    match = _YEAR_MONTH_RE.search(query)
    if match:
        month = int(match.group(2))
        if 1 <= month <= 12:
            return f"{match.group(1)}-{month:02d}"
    match = _QUARTER_CN_RE.search(query)
    if match:
        return f"第{_CN_DIGITS.get(match.group(1), match.group(1))}季度"
    match = _RECENT_DAYS_RE.search(query)
    if match:
        return f"近{match.group(1)}天"
    match = _RECENT_MONTHS_RE.search(query)
    if match:
        return f"近{match.group(1)}个月"
    match = _RECENT_WEEKS_RE.search(query)
    if match:
        return f"近{match.group(1)}周"
    match = _RELATIVE_MONTH_RE.search(query)
    if match:
        return f"{match.group(1)}月"
    match = _RELATIVE_WEEK_RE.search(query)
    if match:
        return f"{match.group(1)}周"
    match = _RELATIVE_QUARTER_RE.search(query)
    if match:
        return f"{match.group(1)}季度"
    match = _YEAR_WORD_RE.search(query)
    if match:
        return match.group(1)
    match = _YEAR_RE.search(query)
    if match:
        return f"{match.group(1)}年"
    return None


def _extract_category(query: str) -> str | None:
    """抽取品类，返回规范名。"""
    for canonical, keywords in _CATEGORY_RULES:
        if any(keyword in query for keyword in keywords):
            return canonical
    return None


def _extract_metric(query: str) -> str | None:
    """抽取指标，返回规范名。"""
    low = query.casefold()
    for canonical, keywords in _METRIC_RULES:
        if any(keyword in low for keyword in keywords):
            return canonical
    return None


def _extract_comparison(query: str) -> str:
    """抽取对比方式；未提及对比时返回 ``"无"``（表示「不主动做对比」）。"""
    for word in ("同比", "环比"):
        if word in query:
            return word
    if "趋势" in query or "走势" in query:
        return "趋势"
    return "无"


def extract_slots(query: str) -> StructuredSlots:
    """纯规则抽取用户问题的结构化槽位（不调用 LLM，零成本、可预测）。

    Args:
        query: 用户原始问题。

    Returns:
        ``StructuredSlots`` 实例；没有命中任何信号时 ``is_empty()`` 为 True。
    """
    query = query or ""
    return StructuredSlots(
        time_range=_extract_time_range(query),
        category=_extract_category(query),
        metric=_extract_metric(query),
        comparison=_extract_comparison(query),
        sample_warning=bool(_SAMPLE_WARN_RE.search(query)),
    )


def _slot_lines(slots: StructuredSlots) -> list[str]:
    """渲染槽位正文行（不含分节标题）。"""
    lines: list[str] = []
    if slots.time_range:
        lines.append(f"- 时间范围：{slots.time_range}")
    if slots.category:
        lines.append(f"- 品类：{slots.category}")
    if slots.metric:
        lines.append(f"- 指标：{slots.metric}")
    if slots.comparison:
        lines.append(f"- 对比：{slots.comparison}")
    if slots.sample_warning:
        lines.append("- 样本量提醒：数据样本可能偏少，结论需注明样本量不足的风险")
    return lines


def render_slots(slots: StructuredSlots) -> str:
    """把结构化槽位渲染成提示词分节；空槽返回空串。

    Args:
        slots: 待渲染槽位。

    Returns:
        以 ``### 任务槽位`` 开头的文本；``slots.is_empty()`` 为 True 时返回 ``""``。
    """
    if slots is None or slots.is_empty():
        return ""
    return SECTION_SLOTS + "\n" + "\n".join(_slot_lines(slots))


# ---------------------------------------------------------------------------
# 五、提示词注入：槽位 / 记忆 / 用户问题分节 + 总预算控制
# ---------------------------------------------------------------------------


@dataclass
class ContextBudget:
    """上下文预算：各分节单独的 token 上限 + 模型输入总上限。

    Attributes:
        memory_tokens: 长期记忆分节 token 上限。
        history_tokens: 历史消息 token 上限（供 ``trim_history`` 使用）。
        tool_output_tokens: 单次工具输出 token 上限（供 ``clip_tool_output`` 使用）。
        slots_tokens: 任务槽位分节 token 上限。
        total_tokens: 拼装后的模型输入总 token 上限。
    """

    memory_tokens: int = 600
    history_tokens: int = 1200
    tool_output_tokens: int = 1500
    slots_tokens: int = 200
    total_tokens: int = 6000


def _assemble(query_section: str, slot_section: str, memory_section: str) -> str:
    """按「槽位 → 记忆 → 用户问题」拼装分节，空分节省略。"""
    return "\n\n".join(section for section in (slot_section, memory_section, query_section) if section)


def build_agent_input(
    query: str,
    *,
    memory_block: str = "",
    slots: StructuredSlots | None = None,
    budget: ContextBudget | None = None,
) -> str:
    """拼装模型输入：任务槽位 → 已知背景（长期记忆）→ 用户问题，并控制总 token 预算。

    Args:
        query: 用户问题；该分节始终最后输出且不做裁剪。
        memory_block: 长期记忆文本（由 ``agent/memory.py`` 检索得到，``main_agent`` 传入）。
        slots: 结构化槽位；``None`` 或空槽时省略该分节。
        budget: 预算配置；``None`` 时用默认 ``ContextBudget()``。

    Returns:
        分节后的模型输入文本。总长度按 ``budget.total_tokens`` 控制：超出时按
        「先裁记忆、再裁槽位」的顺序压缩，用户问题永不被裁剪；只有 query 本身
        就超过总预算时返回值才会超预算。
    """
    budget = budget or ContextBudget()
    query = query or ""
    query_section = f"{SECTION_QUERY}\n{query}"
    if budget.total_tokens <= 0:
        return query_section
    total_chars = max(1, int(budget.total_tokens * CHARS_PER_TOKEN))

    # 1) 各分节先按自身上限裁剪
    slot_body = ""
    if slots is not None and not slots.is_empty() and budget.slots_tokens > 0:
        slot_body = clip_text("\n".join(_slot_lines(slots)), budget.slots_tokens)
    slot_section = f"{SECTION_SLOTS}\n{slot_body}" if slot_body else ""

    memory_body = ""
    if memory_block and memory_block.strip() and budget.memory_tokens > 0:
        memory_body = clip_text(memory_block.strip(), budget.memory_tokens)
    memory_section = f"{SECTION_MEMORY}\n{memory_body}" if memory_body else ""

    # 2) 超出总预算：先裁记忆，再裁槽位（绝不动 query）
    if len(_assemble(query_section, slot_section, memory_section)) > total_chars and memory_section:
        base_chars = len(_assemble(query_section, slot_section, ""))
        allowance = total_chars - base_chars - 2 - len(SECTION_MEMORY) - 1
        memory_body = clip_text(memory_body, int(allowance / CHARS_PER_TOKEN)) if allowance > 0 else ""
        memory_section = f"{SECTION_MEMORY}\n{memory_body}" if memory_body else ""

    if len(_assemble(query_section, slot_section, memory_section)) > total_chars and slot_section:
        base_chars = len(_assemble(query_section, "", memory_section))
        allowance = total_chars - base_chars - 2 - len(SECTION_SLOTS) - 1
        slot_body = clip_text(slot_body, int(allowance / CHARS_PER_TOKEN)) if allowance > 0 else ""
        slot_section = f"{SECTION_SLOTS}\n{slot_body}" if slot_body else ""

    result = _assemble(query_section, slot_section, memory_section)
    if len(result) > total_chars:
        # 记忆、槽位都裁空后仍超预算：保底只返回用户问题
        return query_section
    return result


# ---------------------------------------------------------------------------
# 六、可选中间件：langchain SummarizationMiddleware 配置入口
# ---------------------------------------------------------------------------


def summarize_middleware(model: Any, *, trigger_tokens: int = 80000, keep_messages: int = 20) -> Any:
    """构造 langchain 的 ``SummarizationMiddleware``，供 ``create_deep_agent`` 使用。

    注意：deepagents 0.4.3 的 ``create_deep_agent`` 默认中间件栈里已经内置
    ``deepagents`` 版 ``SummarizationMiddleware``；本函数构造的是 langchain 版，
    传入 ``create_deep_agent(middleware=[...])`` 时是「追加」而非替换内置项。

    Args:
        model: 聊天模型实例或模型名字符串。
        trigger_tokens: 历史 token 达到该值时触发摘要。
        keep_messages: 摘要后保留最近多少条消息。

    Returns:
        ``SummarizationMiddleware`` 实例；langchain 不可用或构造失败时返回 ``None``。
    """
    try:
        from langchain.agents.middleware import SummarizationMiddleware
    except ImportError:  # pragma: no cover - 已安装 langchain 的环境不会走到这里
        return None
    try:
        return SummarizationMiddleware(
            model=model,
            trigger=("tokens", trigger_tokens),
            keep=("messages", keep_messages),
        )
    except Exception:  # noqa: BLE001 - 配置失败降级为 None，由调用方决定是否重试
        return None
