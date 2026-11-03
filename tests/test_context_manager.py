"""P1 §3.11 上下文管理模块的离线单测。

覆盖 ``agent/context_manager.py`` 的公开 API：token 估算、头尾裁剪、工具输出裁剪、
历史裁剪、异步摘要（含无 LLM / LLM 抛异常的确定性兜底）、结构化槽位抽取与渲染、
分节注入的总预算控制，以及 langchain ``SummarizationMiddleware`` 配置入口。

全部用例纯离线：不发网络请求、不连数据库、不启动服务；摘要用的 LLM 是本地假对象。
"""

import asyncio

import pytest

from agent.context_manager import (
    SECTION_MEMORY,
    SECTION_QUERY,
    SECTION_SLOTS,
    ContextBudget,
    StructuredSlots,
    build_agent_input,
    clip_text,
    clip_tool_output,
    estimate_tokens,
    extract_slots,
    render_slots,
    summarize_history,
    summarize_middleware,
    trim_history,
)


def _msg(role: str, content: object) -> dict:
    """构造一条历史消息（role/content 结构）。"""
    return {"role": role, "content": content}


# ---------------------------------------------------------------------------
# token 估算与裁剪
# ---------------------------------------------------------------------------


def test_estimate_tokens_boundaries():
    assert estimate_tokens("") == 0
    assert estimate_tokens("中" * 150) == 100
    assert estimate_tokens("中" * 3) == 2  # 向上取整
    assert estimate_tokens("abc") == 2


def test_clip_text_keeps_head_and_tail_with_marker():
    text = "头" * 2000 + "尾" * 500
    clipped = clip_text(text, 100)
    assert clipped.startswith("头头")
    assert clipped.endswith("尾尾")
    assert "已裁剪" in clipped
    assert len(clipped) < len(text)
    assert estimate_tokens(clipped) <= 100


def test_clip_text_short_text_unchanged_and_custom_marker():
    assert clip_text("很短的文本", 100) == "很短的文本"
    clipped = clip_text("A" * 1000, 20, marker="[...]")
    assert clipped.startswith("A")
    assert "[...]" in clipped
    assert estimate_tokens(clipped) <= 20


def test_clip_text_non_positive_budget_returns_empty():
    assert clip_text("任意文本", 0) == ""
    assert clip_text("任意文本", -5) == ""
    assert clip_text("", 100) == ""


def test_clip_tool_output_truncates_large_output():
    text = "A" * 5000 + "B" * 5000
    clipped = clip_tool_output(text, 100)
    assert clipped.startswith("A")
    assert clipped.endswith("B")
    assert "已裁剪" in clipped
    assert "字符" in clipped
    assert estimate_tokens(clipped) <= 100
    assert clip_tool_output("小输出", 1500) == "小输出"


# ---------------------------------------------------------------------------
# 历史裁剪
# ---------------------------------------------------------------------------


def _sample_history() -> list[dict]:
    return [
        _msg("system", "你是电商运营分析师"),
        _msg("user", "2026年Q1 数码电子 GMV 同比是多少"),
        _msg("assistant", "需要查询订单库"),
        _msg("user", "今天天气怎么样"),
        _msg("assistant", "晴"),
        _msg("user", "帮我写一首诗"),
        _msg("assistant", "好诗"),
        _msg("user", "最后一个问题"),
        _msg("assistant", "最后一个回答"),
    ]


def test_trim_history_keeps_system_and_recent_turns():
    messages = _sample_history()
    trimmed = trim_history(messages, max_turns=2)
    assert trimmed is not messages  # 返回新列表
    assert [m["content"] for m in trimmed] == [
        "你是电商运营分析师",
        "帮我写一首诗",
        "好诗",
        "最后一个问题",
        "最后一个回答",
    ]
    # 不修改原始列表
    assert len(messages) == 9


def test_trim_history_keeps_older_relevant_by_query():
    messages = _sample_history()
    trimmed = trim_history(messages, max_turns=2, query="2026年Q1 数码电子 GMV 同比", max_tokens=1200)
    contents = [m["content"] for m in trimmed]
    assert "2026年Q1 数码电子 GMV 同比是多少" in contents
    assert "今天天气怎么样" not in contents
    # 顺序与原始消息一致
    assert contents == [m["content"] for m in messages if m["content"] in set(contents)]


def test_trim_history_edge_cases():
    assert trim_history([]) == []
    messages = _sample_history()
    assert [m["content"] for m in trim_history(messages, max_turns=0)] == ["你是电商运营分析师"]


# ---------------------------------------------------------------------------
# 历史摘要
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    """记录 prompt 并返回固定要点的假 LLM。"""

    def __init__(self, reply: str = "要点：GMV 同比增长 20%") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> _FakeResponse:
        self.prompts.append(prompt)
        return _FakeResponse(self.reply)


class _RaisingLLM:
    async def ainvoke(self, prompt: str) -> None:
        raise RuntimeError("模拟模型调用失败")


def test_summarize_history_fallback_without_llm():
    messages = [_msg("user", "查 2026 年 Q1 数据"), _msg("assistant", "数据如下" + "长" * 500)]
    summary = asyncio.run(summarize_history(None, messages, max_tokens=40))
    assert summary
    assert "user" in summary
    assert estimate_tokens(summary) <= 40
    assert asyncio.run(summarize_history(None, [])) == ""


def test_summarize_history_fallback_on_llm_error():
    messages = [_msg("user", "查 2026 年 Q1 数据"), _msg("assistant", "数据如下")]
    summary = asyncio.run(summarize_history(_RaisingLLM(), messages, max_tokens=60))
    assert "user" in summary
    assert estimate_tokens(summary) <= 60


def test_summarize_history_uses_llm_points():
    messages = [_msg("user", "2026年Q1 数码电子 GMV 同比"), _msg("assistant", "GMV 同比增长 20%")]
    llm = _FakeLLM()
    summary = asyncio.run(summarize_history(llm, messages, max_tokens=100))
    assert summary == "要点：GMV 同比增长 20%"
    assert "数码电子" in llm.prompts[0]  # 历史内容确实进了提示词
    assert len(llm.prompts) == 1


# ---------------------------------------------------------------------------
# 结构化槽位
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("2026年Q1 数码电子 GMV", "2026年Q1"),
        ("2026Q2 销售趋势", "2026年Q2"),
        ("2026-01 的退款率", "2026-01"),
        ("2026年3月订单", "2026-03"),
        ("本月销售额", "本月"),
        ("上月的数据", "上月"),
        ("近30天 GMV", "近30天"),
        ("第一季度表现", "第一季度"),
        ("今年整体情况", "今年"),
    ],
)
def test_extract_slots_time_range(query: str, expected: str):
    assert extract_slots(query).time_range == expected


def test_extract_slots_category_and_metric():
    slots = extract_slots("上月生活家电的客单价环比")
    assert slots.time_range == "上月"
    assert slots.category == "生活家电"
    assert slots.metric == "客单价"
    assert slots.comparison == "环比"
    assert extract_slots("数码电子销售额").metric == "GMV"
    assert extract_slots("查询各品类占比").metric == "品类占比"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("2026年Q1 GMV 同比", "同比"),
        ("本月 GMV 环比", "环比"),
        ("近30天销售趋势", "趋势"),
        ("2026年Q1 GMV", "无"),
    ],
)
def test_extract_slots_comparison(query: str, expected: str):
    assert extract_slots(query).comparison == expected


def test_extract_slots_sample_warning():
    slots = extract_slots("只有8条数据，样本够不够？")
    assert slots.sample_warning is True
    assert extract_slots("2026年Q1 数码电子 GMV").sample_warning is False


def test_extract_slots_empty_query_and_render_empty():
    slots = extract_slots("你好，帮我看看")
    assert slots.is_empty()
    assert slots.comparison == "无"  # 未提对比时为「无」，但不构成有效槽位
    assert render_slots(slots) == ""
    assert render_slots(StructuredSlots()) == ""


def test_render_slots_non_empty():
    slots = StructuredSlots(
        time_range="2026年Q1",
        category="数码电子",
        metric="GMV",
        comparison="同比",
        sample_warning=True,
    )
    text = render_slots(slots)
    assert text.startswith(SECTION_SLOTS)
    assert "- 时间范围：2026年Q1" in text
    assert "- 品类：数码电子" in text
    assert "- 指标：GMV" in text
    assert "- 对比：同比" in text
    assert "样本量" in text


# ---------------------------------------------------------------------------
# 分节注入与预算控制
# ---------------------------------------------------------------------------


def test_build_agent_input_section_order_and_query_intact():
    query = "2026年Q1数码电子GMV同比"
    out = build_agent_input(query, memory_block="背景：上季度 GMV 增长 20%", slots=extract_slots(query))
    assert out.endswith(f"{SECTION_QUERY}\n{query}")
    assert out.index(SECTION_SLOTS) < out.index(SECTION_MEMORY) < out.index(SECTION_QUERY)
    assert "- 时间范围：2026年Q1" in out
    assert "背景：上季度 GMV 增长 20%" in out


def test_build_agent_input_omits_empty_sections():
    out = build_agent_input("今天天气如何")
    assert out == f"{SECTION_QUERY}\n今天天气如何"


def test_build_agent_input_respects_total_budget():
    query = "2026年Q1数码电子GMV同比"
    long_memory = "长期记忆：" + "数码电子 GMV 同比增长。" * 1000
    slots = extract_slots(query)

    out = build_agent_input(query, memory_block=long_memory, slots=slots, budget=ContextBudget())
    assert estimate_tokens(out) <= 6000
    assert out.endswith(f"{SECTION_QUERY}\n{query}")
    assert "已裁剪" in out  # 超长记忆被裁剪

    small = ContextBudget(memory_tokens=600, slots_tokens=200, total_tokens=120)
    out_small = build_agent_input(query, memory_block=long_memory, slots=slots, budget=small)
    assert estimate_tokens(out_small) <= 120
    assert out_small.endswith(f"{SECTION_QUERY}\n{query}")


def test_build_agent_input_never_trims_query():
    query = "这是一个必须完整保留的用户问题"
    out = build_agent_input(query, memory_block="记忆" * 1000, budget=ContextBudget(total_tokens=1))
    assert out.endswith(query)
    assert "记忆" not in out


# ---------------------------------------------------------------------------
# SummarizationMiddleware 配置入口
# ---------------------------------------------------------------------------


def _fake_model() -> object:
    """用 langchain-core 自带的假聊天模型，避免依赖 langchain 内部构造细节。

    裸对象（只有一个 ``_llm_type``）在 langchain 1.2.x 能构造成功，
    但 1.4.x 的 ``SummarizationMiddleware.__init__`` 会调用 ``model.with_retry(...)``
    等方法，裸对象直接抛 AttributeError，用例会误报。使用真正的
    ``BaseChatModel`` 子类，跨版本稳定。
    """
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    return FakeListChatModel(responses=["fake summary"])


def test_summarize_middleware_configures_and_degrades():
    middleware = summarize_middleware(_fake_model())
    assert middleware is not None
    assert getattr(middleware, "trigger", None) == ("tokens", 80000)
    assert getattr(middleware, "keep", None) == ("messages", 20)

    custom = summarize_middleware(_fake_model(), trigger_tokens=1000, keep_messages=4)
    assert custom is not None
    assert custom.trigger == ("tokens", 1000)
    assert custom.keep == ("messages", 4)

    # 模型无效（None）时构造失败，降级为 None
    assert summarize_middleware(None) is None
