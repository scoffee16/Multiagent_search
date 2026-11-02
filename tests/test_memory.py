"""长期记忆子系统单测（全部离线，数据库落 tmp_path，不联网、不起服务）。

覆盖点：
* embedding：确定性、维度、L2 归一、空文本；
* 写入：add / add_candidates / 跨 thread 去重（余弦 >0.9 合并更新）；
* 检索：命中、排序、top_k、min_score、时间衰减（200 天前降权）、hit_count / last_used_at 回写；
* 注入：render_for_prompt 固定格式、retrieve_memory_context 预算与空结果；
* 维护：list / delete / clear / stats / to_dict；
* 抽取：假 llm（字符串 / AIMessage / 同步 ainvoke）、代码围栏 JSON、非法 JSON、5 条上限、
  无 llm 且默认模型不可用时返回 []。
"""

import asyncio
import datetime
import json
import math
import sqlite3

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from agent.memory import (
    DEFAULT_DB_PATH,
    EMBED_DIM,
    MemoryCandidate,
    MemoryItem,
    MemoryStore,
    cosine_similarity,
    embed_text,
    extract_memory_candidates,
    retrieve_memory_context,
)

Q1_PREFERENCE = "用户只看2026年Q1的数据"
UNRELATED_QUERY = "量子力学与天体物理有什么最新进展"


@pytest.fixture()
def store(tmp_path):
    """每个用例一个独立数据库，用完即关。"""
    memory_store = MemoryStore(tmp_path / "memory.db")
    yield memory_store
    memory_store.close()


def _iso_days_ago(days: float) -> str:
    return (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat(timespec="seconds")


class _FakeLLM:
    """假 LLM：记录 prompt，ainvoke 返回预设结果（异步版）。"""

    def __init__(self, response):
        self.response = response
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str):
        self.prompts.append(prompt)
        return self.response


class _SyncFakeLLM:
    """假 LLM：ainvoke 是同步函数，返回 AIMessage。"""

    def __init__(self, response):
        self.response = response

    def ainvoke(self, prompt: str):
        return self.response


# --------------------------------------------------------------------------- embedding
def test_embed_deterministic_dim_and_normalized():
    first = embed_text(Q1_PREFERENCE)
    second = embed_text(Q1_PREFERENCE)

    assert first == second  # 同文本必须完全确定
    assert len(first) == EMBED_DIM
    assert math.isclose(sum(value * value for value in first), 1.0, rel_tol=1e-9)
    assert len(embed_text(Q1_PREFERENCE, dim=64)) == 64
    # 相似文本 > 无关文本
    assert cosine_similarity(first, embed_text("用户只看2026年Q1数据")) > cosine_similarity(
        first, embed_text("统计时不要算退款订单")
    )
    assert cosine_similarity(first, second) == pytest.approx(1.0, rel=1e-6)


def test_embed_empty_text_and_cosine_edges():
    zero = embed_text("")
    assert zero == [0.0] * EMBED_DIM
    assert cosine_similarity(zero, zero) == 0.0
    assert cosine_similarity(embed_text(Q1_PREFERENCE), []) == 0.0
    assert cosine_similarity(embed_text(Q1_PREFERENCE), [0.0] * EMBED_DIM) == 0.0
    assert cosine_similarity(embed_text(Q1_PREFERENCE), embed_text(Q1_PREFERENCE, dim=64)) == 0.0
    with pytest.raises(ValueError):
        embed_text("x", dim=0)


# --------------------------------------------------------------------------- 写入 / 去重
def test_add_then_search_hits_and_updates_usage(store):
    memory_id = store.add("thread-a", Q1_PREFERENCE, type="preference", confidence=0.9, source="extract")

    assert memory_id > 0
    before = {row["id"]: row for row in store.list()}[memory_id]
    assert before["hit_count"] == 0
    assert before["last_used_at"] == before["created_at"]

    items = store.search("2026年Q1 GMV数据", top_k=3)
    assert [item.id for item in items] == [memory_id]
    assert items[0].score > 0.0
    assert items[0].hit_count == 1
    assert datetime.datetime.fromisoformat(items[0].last_used_at) >= datetime.datetime.fromisoformat(
        before["created_at"]
    )

    store.search("2026年Q1 GMV数据", top_k=3)  # 再命中一次
    after = store.list()[0]
    assert after["hit_count"] == 2


def test_dedupe_similar_content_updates_existing_row(store):
    original = "用户只看2026年第一季度GMV数据"
    updated = "用户只看2026年第一季度GMV数据表"
    # 前提：两条内容余弦 >0.9，才会走去重更新分支
    assert cosine_similarity(embed_text(original), embed_text(updated)) > 0.9

    first_id = store.add("thread-a", original, type="preference", confidence=0.9, source="manual")
    second_id = store.add("thread-b", updated, confidence=0.7, source="extract")

    assert second_id == first_id
    assert store.stats()["total"] == 1
    row = store.list()[0]
    assert row["content"] == updated  # 内容取新的
    assert row["confidence"] == 0.7  # 置信度取新的
    assert row["source"] == "extract"  # source 取新的
    assert row["thread_id"] == "thread-a"  # 去重不迁移归属 thread

    # 阈值调高到 1.0 时不合并，用于需要保留两条近似记忆的场景
    store.add("thread-c", original, dedupe_threshold=1.0)
    assert store.stats()["total"] == 2


def test_add_empty_content_raises(store):
    with pytest.raises(ValueError):
        store.add("thread-a", "   ")


def test_add_candidates_sources_and_idempotent(store):
    candidates = [
        MemoryCandidate(type="preference", content=Q1_PREFERENCE, confidence=0.9),
        MemoryCandidate(type="constraint", content="统计时不要算退款订单", confidence=0.8, source="cand-src"),
    ]
    ids = store.add_candidates("thread-a", candidates, source="llm")

    assert len(ids) == 2
    rows = {row["id"]: row for row in store.list()}
    assert rows[ids[0]]["source"] == "llm"  # 候选没给 source 时回退到参数
    assert rows[ids[1]]["source"] == "cand-src"  # 候选自带 source 优先
    assert rows[ids[0]]["type"] == "preference"
    assert rows[ids[1]]["type"] == "constraint"

    # 同一候选重复写入同一线程：被去重合并，返回同一 id，不新增行
    again = store.add_candidates("thread-b", [candidates[0], candidates[0]])
    assert again == [ids[0]]
    assert store.stats()["total"] == 2

    with pytest.raises(TypeError):
        store.add_candidates("thread-a", ["不是 MemoryCandidate"])  # type: ignore[list-item]


def test_candidate_validation_normalizes(store):
    candidate = MemoryCandidate(type="Preference", content="  用户只看2026年Q1数据  ", confidence=1.5, source=" x ")
    assert candidate.type == "preference"
    assert candidate.content == "用户只看2026年Q1数据"
    assert candidate.confidence == 1.0
    assert candidate.source == "x"
    assert MemoryCandidate(type="unknown-type", content="随便").type == "fact"
    with pytest.raises(ValidationError):
        MemoryCandidate(content="   ")


# --------------------------------------------------------------------------- 检索排序
def test_search_top_k_sorted_and_only_top_hits_counted(store):
    store.add("t", "用户只看2026年Q1的GMV数据", dedupe_threshold=1.0)
    store.add("t", "用户只看2026年Q1的订单数据", dedupe_threshold=1.0)
    store.add("t", "用户只看2026年Q1的退款数据", dedupe_threshold=1.0)
    items = store.search("2026年Q1 GMV数据", top_k=2)

    assert len(items) == 2
    assert items[0].score >= items[1].score
    hit = {row["id"]: row["hit_count"] for row in store.list()}
    assert sum(hit.values()) == 2  # 只有返回的 top_k 被计为命中
    assert sorted(hit.values()) == [0, 1, 1]


def test_search_min_score_filters(store):
    store.add("t", Q1_PREFERENCE)

    assert store.search("2026年Q1 GMV数据", min_score=1.01) == []
    assert store.search(UNRELATED_QUERY, min_score=0.05) == []
    assert store.search("2026年Q1 GMV数据", min_score=0.05) != []


def test_search_time_decay_penalizes_old_memory(store):
    old_content = "用户偏好查看2026年Q1的GMV数据"
    new_content = "用户偏好查看2026年Q1的GMV数据明细"
    assert cosine_similarity(embed_text(old_content), embed_text(new_content)) > 0.9
    old_id = store.add("t", old_content, confidence=1.0, dedupe_threshold=1.0)
    new_id = store.add("t", new_content, confidence=1.0, dedupe_threshold=1.0)

    # 手工把旧记忆的 last_used_at 改到 200 天前（模拟长期未命中）
    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute("UPDATE memories SET last_used_at = ? WHERE id = ?", (_iso_days_ago(200), old_id))
        conn.commit()

    query = "2026年Q1 GMV数据"
    items = store.search(query, top_k=3)
    ids = [item.id for item in items]
    assert ids == [new_id, old_id]  # 新的排前面
    old_item = next(item for item in items if item.id == old_id)
    new_item = next(item for item in items if item.id == new_id)

    # 理论得分 = 余弦 × 置信度 × 0.5 ** (200 / 90)；容忍 BLOB float32 与秒级时间的误差
    expected_old = cosine_similarity(embed_text(query), embed_text(old_content)) * 1.0 * 0.5 ** (200.0 / 90.0)
    assert old_item.score == pytest.approx(expected_old, rel=0.05)
    assert old_item.score < new_item.score * 0.4  # 显著低于新记忆

    # 可用 half_life_days 调衰减强度：半衰期很大时旧记忆几乎不被惩罚
    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute("UPDATE memories SET last_used_at = ? WHERE id = ?", (_iso_days_ago(200), old_id))
        conn.commit()
    loose = store.search(query, top_k=3, half_life_days=100000.0)
    loose_old = next(item for item in loose if item.id == old_id)
    assert loose_old.score > old_item.score * 3


# --------------------------------------------------------------------------- 跨 thread
def test_search_cross_thread_by_default(store):
    memory_id = store.add("thread-a", "用户主要负责数码电子品类", type="fact")

    assert [item.id for item in store.search("数码电子品类")] == [memory_id]
    assert store.search("数码电子品类", thread_id="thread-b") == []
    assert [item.id for item in store.search("数码电子品类", thread_id="thread-a")] == [memory_id]


def test_recent_returns_latest_and_filters_type(store):
    older = store.add("t1", "统计时不要算退款订单", type="constraint")
    newer = store.add("t2", "GMV 保留两位小数", type="constraint")
    store.add("t3", "用户喜欢结论先行", type="preference")

    recent = store.recent(limit=2)
    assert recent[0].id > recent[1].id
    assert all(item.score == 0.0 for item in recent)
    # 只取 constraint 类型，preference 不应出现
    constraints = store.recent(limit=5, types=("constraint",))
    assert {item.type for item in constraints} == {"constraint"}
    assert [item.id for item in constraints] == [newer, older]
    assert store.recent(limit=0) == []


# --------------------------------------------------------------------------- 注入渲染
def test_render_for_prompt_format(store):
    store.add("t", Q1_PREFERENCE, type="preference", confidence=0.9)
    items = store.search("2026年Q1数据")

    text = store.render_for_prompt(items)
    lines = text.splitlines()
    assert lines[0] == "### 已知背景（长期记忆）"
    assert lines[1] == f"- [记忆 #{items[0].id}] {Q1_PREFERENCE}（preference，置信度 0.90）"
    assert store.render_for_prompt([]) == ""


def test_retrieve_memory_context(store):
    store.add("thread-a", Q1_PREFERENCE, type="preference", confidence=0.9)

    context = retrieve_memory_context("2026年Q1 GMV数据", store=store)
    assert context.startswith("### 已知背景（长期记忆）")
    assert Q1_PREFERENCE in context
    assert "[记忆 #" in context

    # 无命中 / 预算为 0 / 预算小到放不下任何条目 → 空串
    assert retrieve_memory_context(UNRELATED_QUERY, store=store) == ""
    assert retrieve_memory_context("2026年Q1 GMV数据", store=store, max_tokens=0) == ""
    assert retrieve_memory_context("2026年Q1 GMV数据", store=store, max_tokens=3) == ""

    # 预算充足时包含全部命中条目
    store.add("thread-a", "统计时不要算退款订单", type="constraint", confidence=0.8)
    wide = retrieve_memory_context("2026年Q1 GMV数据 不要算退款订单", store=store, top_k=3, max_tokens=600)
    assert "2026年Q1" in wide


def test_retrieve_memory_context_empty_store(tmp_path):
    empty_store = MemoryStore(tmp_path / "empty.db")
    try:
        assert retrieve_memory_context("2026年Q1 GMV数据", store=empty_store) == ""
    finally:
        empty_store.close()


# --------------------------------------------------------------------------- 维护接口
def test_list_delete_clear_stats(store):
    store.add("t1", "用户偏好蓝色主题看板", type="preference")
    store.add("t2", "统计时不要算退款订单", type="constraint")
    store.add("t1", "GMV 按 SOP 口径计算", type="glossary")

    rows = store.list()
    assert len(rows) == 3
    expected_keys = {
        "id",
        "thread_id",
        "type",
        "content",
        "confidence",
        "source",
        "created_at",
        "last_used_at",
        "hit_count",
        "score",
    }
    assert set(rows[0]) == expected_keys
    assert json.loads(json.dumps(rows, ensure_ascii=False))  # 可直接 JSON 序列化
    assert len(store.list(thread_id="t1")) == 2
    assert len(store.list(limit=1)) == 1
    assert store.list(limit=0) == []

    stats = store.stats()
    assert stats["total"] == 3
    assert stats["by_type"] == {"preference": 1, "constraint": 1, "glossary": 1}
    assert stats["by_thread"] == {"t1": 2, "t2": 1}
    assert datetime.datetime.fromisoformat(stats["latest_created_at"]).year >= 2024

    victim = rows[0]["id"]
    assert store.delete(victim) is True
    assert store.delete(victim) is False
    assert store.stats()["total"] == 2

    assert store.clear(thread_id="t1") == 1
    assert store.stats()["total"] == 1
    assert store.clear() == 1
    assert store.stats() == {
        "total": 0,
        "by_type": {},
        "by_thread": {},
        "latest_created_at": None,
    }


def test_memory_item_to_dict():
    item = MemoryItem(
        id=7,
        thread_id="t",
        type="preference",
        content="用户只看2026年Q1数据",
        confidence=0.9,
        source="extract",
        created_at=_iso_days_ago(1),
        last_used_at=_iso_days_ago(0),
        hit_count=3,
        score=0.123456,
    )
    data = item.to_dict()
    assert data["score"] == 0.1235
    assert data["hit_count"] == 3
    assert data["id"] == 7


def test_default_db_path_points_into_project_data():
    assert DEFAULT_DB_PATH.name == "memory.db"
    assert DEFAULT_DB_PATH.parent.name == "data"
    assert DEFAULT_DB_PATH.parent.exists()


# --------------------------------------------------------------------------- LLM 抽取
def test_extract_parses_json_string_with_code_fence():
    content = (
        "```json\n"
        '{"memories": ['
        '{"type": "preference", "content": "用户只看2026年Q1数据", "confidence": 0.9},'
        '{"type": "constraint", "content": "统计时不要算退款订单", "confidence": 0.8}'
        "]}\n```"
    )
    fake = _FakeLLM(content)
    candidates = asyncio.run(
        extract_memory_candidates("我们只看2026年Q1的数据", "好的，之后默认只看2026年Q1。", llm=fake)
    )

    assert [candidate.type for candidate in candidates] == ["preference", "constraint"]
    assert candidates[0].content == "用户只看2026年Q1数据"
    assert candidates[1].confidence == 0.8
    # 提示词里必须带上一轮问答，且明确要求 JSON 与最多 5 条
    assert "2026年Q1" in fake.prompts[0]
    assert "memories" in fake.prompts[0]
    assert "5 条" in fake.prompts[0]


def test_extract_parses_aimessage_and_sync_ainvoke():
    message = AIMessage(
        content='{"memories": [{"type": "glossary", "content": "GMV 按 SOP 口径计算", "confidence": 0.85}]}'
    )
    candidates = asyncio.run(extract_memory_candidates("GMV 怎么算", "按 SOP 口径计算。", llm=_SyncFakeLLM(message)))
    assert len(candidates) == 1
    assert candidates[0].type == "glossary"


def test_extract_filters_invalid_entries_and_limits_five():
    raw = {
        "memories": [
            {"type": "preference", "content": "用户只看2026年Q1数据", "confidence": 0.9},
            {"type": "preference", "confidence": 0.9},  # 缺 content，应被跳过
            {"type": "unknown", "content": "未知类型落回 fact", "confidence": 2.0},
            {"content": "   "},  # 空内容，应被跳过
            {"type": "constraint", "content": "统计时不要算退款订单"},
            {"type": "fact", "content": "用户负责数码电子品类"},
            {"type": "fact", "content": "用户负责家电品类"},
            {"type": "fact", "content": "用户负责服饰品类"},
            {"type": "fact", "content": "用户负责食品品类"},
        ]
    }
    fake = _FakeLLM(json.dumps(raw, ensure_ascii=False))
    candidates = asyncio.run(extract_memory_candidates("问题", "回答", llm=fake))

    assert len(candidates) == 5  # 最多 5 条
    assert candidates[1].type == "fact"  # 未知类型归一
    assert candidates[1].confidence == 1.0  # 置信度截断
    assert all(candidate.content for candidate in candidates)


def test_extract_returns_empty_on_bad_output():
    cases = [
        "抱歉，我无法抽取记忆。",  # 完全没有 JSON
        '{"foo": 1}',  # 没有 memories 字段
        '{"memories": []}',  # 空数组
        '{"memories": "not-a-list"}',  # 类型不对
        "",  # 空返回
    ]
    for response in cases:
        candidates = asyncio.run(extract_memory_candidates("问题", "回答", llm=_FakeLLM(response)))
        assert candidates == []


def test_extract_returns_empty_when_llm_raises():
    class _BoomLLM:
        async def ainvoke(self, prompt: str):
            raise RuntimeError("model unavailable")

    assert asyncio.run(extract_memory_candidates("问题", "回答", llm=_BoomLLM())) == []


def test_extract_skips_empty_qa_without_calling_llm():
    fake = _FakeLLM('{"memories": []}')
    assert asyncio.run(extract_memory_candidates("", "回答", llm=fake)) == []
    assert asyncio.run(extract_memory_candidates("问题", "  ", llm=fake)) == []
    assert fake.prompts == []


def test_extract_without_llm_returns_empty_when_default_unavailable(monkeypatch):
    import agent.memory as memory_module

    def _raise() -> None:
        raise RuntimeError("没有配置 LLM")

    monkeypatch.setattr(memory_module, "_load_default_llm", _raise)
    assert asyncio.run(extract_memory_candidates("问题", "回答", llm=None)) == []


def test_extract_without_llm_uses_default_model(monkeypatch):
    import agent.memory as memory_module

    fake = _FakeLLM('{"memories": [{"type": "preference", "content": "用户只看2026年Q1数据"}]}')
    monkeypatch.setattr(memory_module, "_load_default_llm", lambda: fake)

    candidates = asyncio.run(extract_memory_candidates("我们只看2026年Q1的数据", "好的。", llm=None))
    assert len(candidates) == 1
    assert fake.prompts
