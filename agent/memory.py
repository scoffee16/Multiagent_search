"""长期记忆子系统（P0 §3.7）：跨会话记忆的写入、检索、注入与结构化抽取。

背景
----
主智能体的对话状态已经由 ``AsyncSqliteSaver`` 持久化在 ``data/checkpoints.db``，
但 checkpoint 只是「某个 thread 的图状态」，不是长期记忆：

* 它按 thread_id 隔离，换一个会话（新 thread_id）就完全读不到旧会话的用户偏好；
* 它没有相关性召回，只能整段回放历史消息，无法回答「这次该注入哪几条」；
* 它没有语义相似度、去重、生命周期与可观测接口。

本模块用一张独立的 SQLite 表 + 本地字符 n-gram 哈希向量补上这一层：
写入（LLM 抽取候选 → Pydantic 校验 → 余弦去重）、检索（余弦 × 置信度 × 时间衰减）、
注入（``render_for_prompt`` 生成固定格式的「已知背景」块）。

设计要点
--------
* **零依赖可离线**：embedding 用字符 n-gram + blake2b 特征哈希，确定性、不联网；
* **单连接 + RLock**：sqlite3 连接以 ``check_same_thread=False`` 创建，所有读写都在
  同一把 ``threading.RLock`` 下完成，适配 FastAPI 单事件循环里的并发调用；
* **时间统一 ISO8601**（``datetime.now().isoformat(timespec="seconds")``），
  SQLite 里存字符串，方便人工排查与 ``max()`` 取最近写入时间。

对外 API 由 Lead 在 ``agent/main_agent.py``（任务前注入 + 任务后抽取）与
``api/server.py``（``GET /api/memory``、``GET /api/memory/stats``、
``DELETE /api/memory/{id}``、WebSocket memory 事件）中调用，签名见各函数文档。
"""

import datetime
import hashlib
import inspect
import json
import math
import sqlite3
import struct
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

# 项目根目录 / data / memory.db（data/ 已在 .gitignore 中，运行时生成，不入库）
DEFAULT_DB_PATH: Path = Path(__file__).resolve().parents[1] / "data" / "memory.db"

# 本地字符 n-gram 哈希向量维度；将来替换成真 embedding 时保持该维度即可零迁移
EMBED_DIM = 256

# 记忆类型白名单：事实 / 偏好 / 约束 / 术语表（未知类型统一落回 fact）
MEMORY_TYPES: tuple[str, ...] = ("fact", "preference", "constraint", "glossary")

# 中文场景下 1 token ≈ 1.5 字符（与 evals/run_eval.py 的成本估算口径一致）
CHARS_PER_TOKEN = 1.5

# 建表语句：embedding 用 BLOB（float32 打包）存，避免 JSON 文本膨胀
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL DEFAULT 'fact',
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    source TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    last_used_at TEXT NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memories_thread ON memories(thread_id);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
"""

# 抽取提示词：只输出 JSON，最多 5 条，只抽长期有效信息
_EXTRACT_PROMPT_HEAD = """你是一个长期记忆抽取器。请判断下面这一轮「用户提问 + 助手回答」里，
有哪些信息值得跨会话长期记住，供以后的新会话直接参考。

只抽取长期有效的信息，归入以下四类之一：
- fact：稳定事实（用户负责的业务范围、固定数据源、组织约定等）
- preference：用户偏好（只看某时间段、报告先结论后过程、喜欢什么格式等）
- constraint：明确约束（统计时不算退款订单、禁止使用某张表等）
- glossary：术语/口径定义（如 GMV 按 SOP 口径计算）

不要抽取：本次的临时结论、一次性数字、助手本次的计算过程、与长期无关的闲聊。
最多 5 条；如果没有值得记住的内容，返回空数组。
严格只输出 JSON，不要 Markdown 代码块，不要任何解释，格式：
{"memories":[{"type":"preference","content":"用户只看2026年Q1数据","confidence":0.9}]}

【用户提问】
"""

_EXTRACT_PROMPT_MIDDLE = "\n\n【助手回答】\n"


def _now_iso() -> str:
    """当前时间，ISO8601 字符串（秒级），全模块统一口径。"""
    return datetime.datetime.now().isoformat(timespec="seconds")


def _parse_iso(value: str) -> datetime.datetime:
    """解析 ISO8601 字符串；坏数据不抛异常，按「刚刚」处理，避免拖垮检索。"""
    try:
        return datetime.datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return datetime.datetime.now()


def _char_ngrams(text: str, max_n: int = 3) -> list[str]:
    """把文本切成字符 1~3-gram；去空白 + 转小写后切，中文无需分词即可工作。"""
    cleaned = "".join(ch for ch in text if not ch.isspace()).lower()
    grams: list[str] = []
    for n in range(1, max_n + 1):
        for start in range(len(cleaned) - n + 1):
            grams.append(cleaned[start : start + n])
    return grams


def embed_text(text: str, dim: int = EMBED_DIM) -> list[float]:
    """把文本编码为 L2 归一化的字符 n-gram 哈希向量（确定性、不联网）。

    做法：字符 1~3-gram 各用 blake2b 摘要映射到 ``[0, dim)`` 的桶并随机取 ±1，
    累加后整体 L2 归一化。相同文本必然得到相同向量；语义相近（字面重合多）的
    文本余弦相似度高。空文本返回零向量（余弦为 0）。
    """
    if dim <= 0:
        raise ValueError("dim 必须为正整数")
    vector = [0.0] * dim
    for gram in _char_ngrams(text or ""):
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度；任一向量为空/零向量/维度不一致时返回 0.0。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for left, right in zip(a, b, strict=True):
        dot += left * right
        norm_a += left * left
        norm_b += right * right
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _pack_embedding(vector: Sequence[float]) -> bytes:
    """list[float] -> BLOB（float32 little-endian），1024B/256 维，比 JSON 省一半。"""
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_embedding(blob: bytes) -> list[float]:
    """BLOB -> list[float]；维度由 BLOB 长度反推，兼容历史数据。"""
    size = len(blob) // 4
    return list(struct.unpack(f"<{size}f", blob[: size * 4]))


def _estimate_tokens(text: str) -> int:
    """粗略 token 估算（中文约 1.5 字符/token），只用于注入块的预算控制。"""
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


class MemoryCandidate(BaseModel):
    """LLM 抽取出来的单条候选记忆；写入前必须过这层校验。"""

    type: str = "fact"
    content: str
    confidence: float = 1.0
    source: str = ""

    @field_validator("type")
    @classmethod
    def _normalize_type(cls, value: str) -> str:
        value = (value or "").strip().lower()
        return value if value in MEMORY_TYPES else "fact"

    @field_validator("content")
    @classmethod
    def _require_content(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("content 不能为空")
        return value

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    @field_validator("source")
    @classmethod
    def _strip_source(cls, value: str) -> str:
        return (value or "").strip()


@dataclass
class MemoryItem:
    """一条命中的记忆（检索结果），``score`` 为排序得分，由检索时回填。"""

    id: int
    thread_id: str
    type: str
    content: str
    confidence: float
    source: str
    created_at: str
    last_used_at: str
    hit_count: int
    score: float = 1.0

    def to_dict(self) -> dict:
        """转成可直接 JSON 序列化的 dict（API / 评测脚本用）。"""
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "type": self.type,
            "content": self.content,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "hit_count": self.hit_count,
            "score": round(self.score, 4),
        }


class MemoryStore:
    """SQLite 长期记忆仓库：add / search / list / delete / clear / stats / render。

    一条连接 + 一把 RLock，可在 FastAPI 单进程内跨线程安全使用；测试里注入
    ``tmp_path`` 数据库即可，互不影响。
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ 写入
    def add(
        self,
        thread_id: str,
        content: str,
        type: str = "fact",
        confidence: float = 1.0,
        source: str = "",
        dedupe_threshold: float = 0.9,
    ) -> int:
        """写入一条记忆，返回其 id。

        写入前与**全部已有记忆（跨 thread）**比余弦相似度：若最相似的一条超过
        ``dedupe_threshold``，则视为同一件长期信息，更新其内容 / 置信度 /
        last_used_at / source，并返回已有记录的 id，不新增行。
        """
        content = (content or "").strip()
        if not content:
            raise ValueError("content 不能为空")
        memory_type = type if type in MEMORY_TYPES else "fact"
        confidence = min(1.0, max(0.0, float(confidence)))
        vector = embed_text(content)
        now = _now_iso()

        with self._lock:
            rows = self._conn.execute("SELECT id, embedding FROM memories").fetchall()
            best_id: int | None = None
            best_similarity = -1.0
            for row in rows:
                similarity = cosine_similarity(vector, _unpack_embedding(row["embedding"]))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_id = int(row["id"])
            if best_id is not None and best_similarity > dedupe_threshold:
                self._conn.execute(
                    "UPDATE memories SET content = ?, embedding = ?, confidence = ?, source = ?, "
                    "last_used_at = ? WHERE id = ?",
                    (content, _pack_embedding(vector), confidence, source or "", now, best_id),
                )
                self._conn.commit()
                return best_id
            cursor = self._conn.execute(
                "INSERT INTO memories "
                "(thread_id, type, content, embedding, confidence, source, created_at, last_used_at, hit_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    thread_id or "",
                    memory_type,
                    content,
                    _pack_embedding(vector),
                    confidence,
                    source or "",
                    now,
                    now,
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid or 0)

    def add_candidates(self, thread_id: str, candidates: Sequence[MemoryCandidate], source: str = "") -> list[int]:
        """批量写入抽取出来的候选记忆，返回去重后的 id 列表（保持首次出现顺序）。

        空内容候选直接跳过；``source`` 为空时回退到 ``candidate.source``。
        同一批里内容高度相似的候选会被 ``add`` 的跨库去重合并成同一个 id。
        """
        ids: list[int] = []
        seen: set[int] = set()
        for candidate in candidates:
            if not isinstance(candidate, MemoryCandidate):
                raise TypeError("candidates 必须是 MemoryCandidate 列表")
            memory_id = self.add(
                thread_id=thread_id,
                content=candidate.content,
                type=candidate.type,
                confidence=candidate.confidence,
                source=candidate.source or source,
            )
            if memory_id not in seen:
                seen.add(memory_id)
                ids.append(memory_id)
        return ids

    # ------------------------------------------------------------------ 检索
    def search(
        self,
        query: str,
        top_k: int = 3,
        thread_id: str | None = None,
        min_score: float = 0.05,
        half_life_days: float = 90.0,
    ) -> list[MemoryItem]:
        """按相关性检索记忆，返回 top_k 条并回写命中统计。

        得分公式：``cosine(query, memory) * confidence * 0.5 ** (days / half_life_days)``，
        其中 days 是距 ``last_used_at`` 的天数（90 天半衰）。默认跨 thread 检索
        （长期记忆本就该跨会话），``thread_id`` 只作为可选的会话内过滤。
        命中项 ``hit_count + 1``、``last_used_at`` 刷新为当前时间。
        """
        limit = max(0, int(top_k))
        if limit == 0:
            return []
        query_vector = embed_text(query or "")
        now = datetime.datetime.now()
        sql = "SELECT * FROM memories"
        params: tuple[Any, ...] = ()
        if thread_id is not None:
            sql += " WHERE thread_id = ?"
            params = (thread_id,)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        half_life_days = float(half_life_days)
        items: list[MemoryItem] = []
        for row in rows:
            similarity = cosine_similarity(query_vector, _unpack_embedding(row["embedding"]))
            last_used = _parse_iso(row["last_used_at"])
            days = max(0.0, (now - last_used).total_seconds() / 86400.0)
            decay = 0.5 ** (days / half_life_days) if half_life_days > 0 else 1.0
            score = similarity * float(row["confidence"]) * decay
            if score < min_score:
                continue
            items.append(
                MemoryItem(
                    id=int(row["id"]),
                    thread_id=row["thread_id"],
                    type=row["type"],
                    content=row["content"],
                    confidence=float(row["confidence"]),
                    source=row["source"],
                    created_at=row["created_at"],
                    last_used_at=row["last_used_at"],
                    hit_count=int(row["hit_count"]),
                    score=score,
                )
            )
        items.sort(key=lambda item: (item.score, item.id), reverse=True)
        top_items = items[:limit]

        if top_items:
            used_at = _now_iso()
            with self._lock:
                self._conn.executemany(
                    "UPDATE memories SET hit_count = hit_count + 1, last_used_at = ? WHERE id = ?",
                    [(used_at, item.id) for item in top_items],
                )
                self._conn.commit()
            for item in top_items:
                item.hit_count += 1
                item.last_used_at = used_at
        return top_items

    def recent(
        self,
        limit: int = 3,
        thread_id: str | None = None,
        types: Sequence[str] | None = None,
    ) -> list[MemoryItem]:
        """按写入时间倒序取最近的记忆（不更新命中统计）。

        用于两类「向量检索不可靠」的兜底：
        - 省略指代（"那段时间/上次/之前提到的"）：查询与记忆措辞几乎不重叠；
        - 计算类任务强制注入最近约束（``types=("constraint",)``）：
          口径/禁止项天然缺少与问题的字面重合，但算数前必须被模型看到。
        """
        limit = max(0, int(limit))
        if limit == 0:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if types:
            placeholders = ", ".join("?" for _ in types)
            clauses.append(f"type IN ({placeholders})")
            params.extend(types)
        sql = "SELECT * FROM memories"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [
            MemoryItem(
                id=int(row["id"]),
                thread_id=row["thread_id"],
                type=row["type"],
                content=row["content"],
                confidence=float(row["confidence"]),
                source=row["source"],
                created_at=row["created_at"],
                last_used_at=row["last_used_at"],
                hit_count=int(row["hit_count"]),
                score=0.0,
            )
            for row in rows
        ]

    # ------------------------------------------------------------------ 维护
    def list(self, thread_id: str | None = None, limit: int = 100) -> list[dict]:
        """列出记忆（最新写入的在前），返回 dict 列表；可按 thread_id 过滤。"""
        limit = max(0, int(limit))
        sql = "SELECT * FROM memories"
        params: tuple[Any, ...] = ()
        if thread_id is not None:
            sql += " WHERE thread_id = ?"
            params = (thread_id,)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params += (limit,)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            MemoryItem(
                id=int(row["id"]),
                thread_id=row["thread_id"],
                type=row["type"],
                content=row["content"],
                confidence=float(row["confidence"]),
                source=row["source"],
                created_at=row["created_at"],
                last_used_at=row["last_used_at"],
                hit_count=int(row["hit_count"]),
            ).to_dict()
            for row in rows
        ]

    def delete(self, memory_id: int) -> bool:
        """删除一条记忆；返回是否真的删掉了。"""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (int(memory_id),))
            self._conn.commit()
            return cursor.rowcount > 0

    def clear(self, thread_id: str | None = None) -> int:
        """清空记忆；``thread_id`` 为空表示全清，返回删除条数。"""
        with self._lock:
            if thread_id is None:
                cursor = self._conn.execute("DELETE FROM memories")
            else:
                cursor = self._conn.execute("DELETE FROM memories WHERE thread_id = ?", (thread_id,))
            self._conn.commit()
            return int(cursor.rowcount or 0)

    def stats(self) -> dict:
        """统计信息：总量、按 type / thread 分布、最近写入时间。"""
        with self._lock:
            total = int(self._conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"])
            by_type = {
                row["type"]: int(row["c"])
                for row in self._conn.execute("SELECT type, COUNT(*) AS c FROM memories GROUP BY type")
            }
            by_thread = {
                row["thread_id"]: int(row["c"])
                for row in self._conn.execute("SELECT thread_id, COUNT(*) AS c FROM memories GROUP BY thread_id")
            }
            latest_created_at = self._conn.execute("SELECT MAX(created_at) AS latest FROM memories").fetchone()[
                "latest"
            ]
        return {
            "total": total,
            "by_type": by_type,
            "by_thread": by_thread,
            "latest_created_at": latest_created_at,
        }

    # ------------------------------------------------------------------ 注入
    def render_for_prompt(self, items: Sequence[MemoryItem]) -> str:
        """把命中记忆渲染成注入块；无内容返回空串。

        固定格式（Lead 会直接拼进用户消息，不要随意改）：

        .. code-block:: text

            ### 已知背景（长期记忆）
            - [记忆 #12] 用户只看 2026 年 Q1 数据（preference，置信度 0.90）
        """
        if not items:
            return ""
        header = "### 已知背景（长期记忆）"
        lines = [header]
        for item in items:
            lines.append(self._render_line(item))
        return "\n".join(lines)

    @staticmethod
    def _render_line(item: MemoryItem) -> str:
        """单条记忆的渲染行，``render_for_prompt`` 与检索注入共用。"""
        return f"- [记忆 #{item.id}] {item.content}（{item.type}，置信度 {item.confidence:.2f}）"

    def close(self) -> None:
        """关闭底层连接（测试 / 进程退出时用）。"""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


# 默认仓库单例：retrieve_memory_context 未显式传 store 时惰性创建，
# 避免 import agent.memory 就去碰 data/ 目录。
_default_store_lock = threading.Lock()
_default_store: MemoryStore | None = None


def _get_default_store() -> MemoryStore:
    """惰性创建进程级默认 MemoryStore（data/memory.db）。"""
    global _default_store
    with _default_store_lock:
        if _default_store is None:
            _default_store = MemoryStore()
        return _default_store


# 「省略指代」标记词：用户不重复说明范围，而是指向上文（跨会话时只能靠长期记忆补全）。
_ELLIPTICAL_MARKERS: tuple[str, ...] = (
    "那段时间",
    "这段时间",
    "那会儿",
    "之前",
    "上次",
    "刚才",
    "前面提到",
    "上面说",
    "照旧",
    "依旧",
    "还是按",
    "按之前的",
    "按上次",
    "继续用",
    "同样口径",
    "那个范围",
)


def needs_recent_memory(query: str) -> bool:
    """判断问题是否包含「省略指代」，需要最近记忆兜底注入。"""
    text = (query or "").strip()
    if not text:
        return False
    return any(marker in text for marker in _ELLIPTICAL_MARKERS)


def retrieve_memory_context(
    query: str,
    top_k: int = 3,
    store: MemoryStore | None = None,
    max_tokens: int = 600,
) -> str:
    """任务前检索并渲染注入块；无命中/异常返回空串（绝不阻断主流程）。

    ``max_tokens`` 是注入预算，超过时按顺序保留能放下的条目；一条都放不下返回空串。
    """
    if max_tokens <= 0:
        return ""
    active_store = store or _get_default_store()
    try:
        items = active_store.search(query or "", top_k=top_k)
    except Exception:  # 记忆检索失败不能拖垮提问主流程
        return ""
    if not items:
        return ""
    header = "### 已知背景（长期记忆）"
    used = _estimate_tokens(header)
    kept: list[MemoryItem] = []
    for item in items:
        cost = _estimate_tokens(active_store._render_line(item)) + 1
        if used + cost > max_tokens:
            break
        kept.append(item)
        used += cost
    if not kept:
        return ""
    return active_store.render_for_prompt(kept)


def _load_default_llm() -> Any:
    """惰性加载全局对话模型（agent.llm.model），配置缺失时抛异常由调用方兜底。"""
    from agent.llm import model

    return model


def _build_extract_prompt(query: str, answer: str) -> str:
    """拼抽取提示词；回答截断到 4000 字，避免一次性把超长报告灌给模型。"""
    return _EXTRACT_PROMPT_HEAD + (query or "") + _EXTRACT_PROMPT_MIDDLE + (answer or "")[:4000]


def _message_text(response: Any) -> str:
    """从假 LLM / 真 LLM 的返回值里取出文本（str、AIMessage、多模态 content 都兼容）。"""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return "" if response is None else str(response)


def _json_candidates(text: str) -> Iterator[str]:
    """从模型输出里依次产出可能合法的 JSON 片段（去代码围栏 → 原文 → 花括号子串）。"""
    cleaned = (text or "").strip()
    if not cleaned:
        return
    if "```" in cleaned:
        for segment in cleaned.split("```")[1:]:
            body = segment.strip()
            if body[:4].lower() == "json":
                body = body[4:].strip()
            if body:
                yield body
    yield cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if 0 <= start < end:
        yield cleaned[start : end + 1]


def _parse_memories_payload(text: str) -> list[dict] | None:
    """解析 ``{"memories": [...]}`` 或裸数组；都失败返回 None。"""
    for candidate in _json_candidates(text):
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        raw = payload.get("memories") if isinstance(payload, dict) else payload
        if not isinstance(raw, list):
            continue
        return [entry for entry in raw if isinstance(entry, dict)]
    return None


async def extract_memory_candidates(query: str, answer: str, llm: Any = None) -> list[MemoryCandidate]:
    """从一轮问答中异步抽取长期记忆候选（最多 5 条）。

    ``llm`` 缺省时惰性使用 ``agent.llm.model``；任何异常（配置缺失、模型报错、
    JSON 解析失败、Pydantic 校验失败）都降级为空列表，保证任务后写入不阻断主流程。
    测试可注入假 llm（``ainvoke`` 返回字符串或 ``AIMessage`` 即可）。
    """
    if not (query or "").strip() or not (answer or "").strip():
        return []
    active_llm = llm
    if active_llm is None:
        try:
            active_llm = _load_default_llm()
        except Exception:  # 没有配置 LLM 时静默跳过
            return []
    prompt = _build_extract_prompt(query, answer)
    try:
        response = active_llm.ainvoke(prompt)
        if inspect.isawaitable(response):
            response = await response
    except Exception:  # 模型调用失败不阻断主流程
        return []
    raw_entries = _parse_memories_payload(_message_text(response))
    if not raw_entries:
        return []
    candidates: list[MemoryCandidate] = []
    for entry in raw_entries:
        if len(candidates) >= 5:  # 先校验再去重截断，保证「最多 5 条」是有效条数
            break
        try:
            candidates.append(MemoryCandidate.model_validate(entry))
        except (ValidationError, TypeError, ValueError):
            continue
    return candidates
