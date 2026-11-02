# 长期记忆子系统设计（P0 §3.7）

> 模块：`agent/memory.py`（实现 + 单元测试：`tests/test_memory.py`）
> 存储：`data/memory.db`（SQLite，运行时自动创建，已在 `.gitignore` 中）
> 跨会话评测：`evals/memory_eval.py`（报告输出 `evals/memory_report.md`）
> 集成方：`agent/main_agent.py` 的 `run_deep_agent()`、`api/server.py` 的记忆 API

## 0. 一页纸概览

长期记忆解决的是「**新会话如何记得旧会话里说过的事**」。整体链路只有两条：

- **写**：一轮任务结束后，把 query + answer 交给 LLM 抽取「长期有效的记忆候选」，经 Pydantic 校验后写入 SQLite，写入时跨会话去重。
- **读**：下一轮任务开始前，用当前 query 检索 top-5 记忆，渲染成固定格式的文本块，拼进用户消息。另有两个兜底：省略指代（"那段时间/上次"）时注入最近记忆；计算类任务强制注入最近约束类记忆。

对外 API（Lead 集成按此调用，签名冻结）：

| 函数 / 类型 | 作用 |
| --- | --- |
| `DEFAULT_DB_PATH` | 默认数据库路径（项目根 `data/memory.db`） |
| `EMBED_DIM = 256` | 字符 n-gram 哈希向量维度 |
| `embed_text(text, dim=EMBED_DIM) -> list[float]` | 文本 → L2 归一向量（确定性、可离线） |
| `cosine_similarity(a, b) -> float` | 余弦相似度 |
| `MemoryCandidate` | pydantic v2 候选模型，抽取结果的校验层 |
| `MemoryItem` | dataclass，检索结果（带 `score`） |
| `MemoryStore` | SQLite 存储：`add / add_candidates / search / list / delete / clear / stats / render_for_prompt` |
| `retrieve_memory_context(...) -> str` | 检索 + 渲染成注入块；无记忆返回 `""` |
| `extract_memory_candidates(...) -> list[MemoryCandidate]` | 任务结束后抽取候选（async） |

`import agent.memory` 不触碰数据库、不读 `.env`：默认库是惰性单例，默认 LLM 在抽取时才 `from agent.llm import model`，保证纯逻辑可测、离线可跑。

## 1. 为什么 checkpoint ≠ 长期记忆

`agent/main_agent.py` 目前用 `AsyncSqliteSaver` 把对话状态持久化到 `data/checkpoints.db`，`thread_id` 取自 `session_id`。这解决的是「**同一个会话中途断掉后能续跑**」，不是「**跨会话记得住**」。对照本仓库现状，五宗罪如下：

1. **放文件夹**：checkpoint 严格按 `thread_id` 隔离。新会话 = 新 `thread_id` = 新命名空间，旧会话的消息**根本不会被读进来**。它像档案柜，只能按编号找到整袋卷宗，不能按内容取出一页。
2. **无相关性召回**：checkpoint 的读取粒度是「整条会话的消息列表」，没有 Top-N、没有打分、没有阈值。要回答问题只能把历史全量回放，上下文一长就爆 token，且噪声淹没信号。
3. **无语义匹配**：消息是自然语言原文，没有向量化。用户以前说「只看 2026 年 Q1」，今天问「那段时间 GMV 多少」时，系统没有任何手段知道这两句相关。
4. **无生命周期**：checkpoint 没有置信度、来源、生效次数、衰减或去重。同一偏好说过三次就存三份，新旧冲突时无法取舍；过期信息（如「上个月只统计退款单」）会永远留在历史里。
5. **不可观测**：checkpoint 是内部二进制格式，没有 API 能列出「系统现在记住了什么」。运营/测试无法核对记忆质量，出问题只能整库删掉重来。

因此本子系统**不替换 checkpoint**，而是补一张 `memories` 表：checkpoint 负责「对话可续跑」，长期记忆负责「跨会话可检索」。两者互补，都不越界。

## 2. 存储：表结构与字段

`agent/memory.py:57` 的 `_SCHEMA` 在库文件首次打开时执行 `CREATE TABLE IF NOT EXISTS`：

```sql
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT    NOT NULL DEFAULT '',   -- 首次写入该记忆的会话
    type        TEXT    NOT NULL DEFAULT 'fact',
    content     TEXT    NOT NULL,              -- 记忆正文（用户原话的规范化陈述）
    embedding   BLOB    NOT NULL,              -- 256 维 float32 little-endian
    confidence  REAL    NOT NULL DEFAULT 1.0,  -- 0.0 ~ 1.0
    source      TEXT    NOT NULL DEFAULT '',   -- 来源标记（如抽取器名）
    created_at  TEXT    NOT NULL,              -- ISO8601
    last_used_at TEXT   NOT NULL,              -- ISO8601，命中时刷新（衰减基准）
    hit_count   INTEGER NOT NULL DEFAULT 0     -- 被检索命中次数
);
CREATE INDEX IF NOT EXISTS idx_memories_thread ON memories(thread_id);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
```

字段要点：

- **type** 取值固定为 `fact / preference / constraint / glossary` 四类（`MEMORY_TYPES`，`agent/memory.py:51`），非法类型在 Pydantic 层归一为 `fact`。
- **embedding** 由 `_pack_embedding()` / `_unpack_embedding()`（`agent/memory.py:155`、`160`）用 `struct` 打包 float32；256 维 = 1024 字节，读回时按 `len(blob) // 4` 反推维度。存 BLOB 而非 JSON 是为了省空间、省解析开销。
- **时间** 统一 `datetime.now().isoformat(timespec="seconds")`（`_now_iso()`，`agent/memory.py:95`），字符串便于直接 SQL 排序和 API 透出。
- **并发**：`sqlite3.connect(..., check_same_thread=False)` + `threading.RLock()`。FastAPI 的 WS/HTTP 会在不同线程触碰存储，所有读写都进同一把锁，简单可靠；当前规模（< 10⁴ 条）不需要 WAL 或连接池。
- `data/` 已在 `.gitignore` 中，`memory.db` 是运行时产物，不入库。

## 3. 写入路径

```
run_deep_agent() 任务结束
   └─ monitor.report_task_result(answer)
   └─ Lead 调用 await extract_memory_candidates(query, answer)
         ├─ 构造中文抽取提示词（_build_extract_prompt，最多 5 条、只输出 JSON）
         ├─ LLM.ainvoke(...)  → 文本
         ├─ _json_candidates / _parse_memories_payload：容忍 Markdown JSON 代码围栏、前后废话
         ├─ MemoryCandidate.model_validate：逐条校验/归一，非法条目跳过
         └─ 先校验、后截断到最多 5 条
   └─ store.add_candidates(thread_id, candidates)
         └─ 逐条 add()：与库内已有记忆余弦 > 0.9 → 更新；否则插入
```

细节与理由：

- **只抽长期有效信息**：提示词明确四类对象（用户偏好、统计口径、约束/禁止项、术语约定），并要求不抽「本次结论/临时数据」。临时结论进记忆会污染后续所有会话。
- **Pydantic 校验不可省**：模型输出不可信。`MemoryCandidate`（`agent/memory.py:171`）的 validator 会把 `type` 不在白名单的归为 `fact`、`confidence` 夹到 `[0,1]`、`content` strip 后为空则报错丢弃、`source` strip。
- **先校验后截断**：如果先把原始数组截到 5 条再校验，前几条非法会导致最终有效条数不足；实现里先逐条 `model_validate` 再截断前 5 条（这个顺序在单元测试里专门覆盖）。
- **全链路失败降级**：LLM 异常、非 JSON、结构不符、`content` 空，任一环节出错都返回 `[]` 或跳过该条，**永不向主流程抛异常**。记忆是增强能力，不能成为任务失败的新原因。
- **跨会话去重**：`add()` 与库内**所有 thread** 的记忆比较余弦，`best > dedupe_threshold`（默认 0.9）时执行 `UPDATE content/embedding/confidence/source/last_used_at`，返回已有 id，不新增行。同一句偏好不该因为换了会话就再存一份。更新时**不迁移 `thread_id`、不改 `type`**：thread 记录「最早由哪个会话说出来的」，type 记录「首次分类」，避免同一记忆被后到的弱证据改写归属。
- **`add_candidates` 的 source 回退**：候选自带 `source` 优先，否则用调用方传入的 `source`，再否则空串。

例：query「我们只看2026年Q1的数据」+ 回答 → 候选 `{"type": "constraint", "content": "用户只看2026年Q1数据", "confidence": 0.9}`。

## 4. 检索与排序

`MemoryStore.search()`（`agent/memory.py`，`MemoryStore.search`）的排序公式：

```
score = cosine(embed(query), embed(content))
        × confidence
        × 0.5 ** (days_since(last_used_at) / half_life_days)
```

- **默认参数**：`top_k=3`、`min_score=0.05`、`half_life_days=90.0`；`thread_id` 为 `None` 时**跨所有会话检索**（长期记忆的默认语义），传入时才按会话过滤。
- **主流程为何取 top-5**：`agent/main_agent.py` 调 `search(task_query, top_k=5)`。端到端评测发现约束类记忆（"统计时不要算退款订单"）与数值问题的字面重合极低，top-3 时常被事实/术语类记忆挤出，导致模型不知道口径；取 5 条后命中率明显改善。
- **兜底 1：省略指代**。查询命中 `needs_recent_memory`（"那段时间/上次/之前提到的/照旧…"）且检索为空时，用 `recent(limit=3)` 注入最近写入记忆——查询与记忆措辞几乎不重叠时，向量检索必然失效。
- **兜底 2：计算类任务强制注入约束**。查询命中计算标记词（算/统计/GMV/销售额/占比/客单价…）时，用 `recent(limit=2, types=("constraint",))` 把最近两条约束类记忆追加进注入块（与检索结果按 id 去重，总条数上限 7）。依据是：口径/禁止项天然缺少与问题的字面重合，但算数前必须被模型看到。实测该项把 mem-05 从 FAIL 变为 PASS。
- **余弦**：对 256 维、L2 归一的向量就是点积，零向量/空向量/维度不一致返回 `0.0`，不会抛异常。
- **置信度**：抽取模型给的分值，低置信度记忆即使相关也排在后面；`1.0` 为人工/默认写入。
- **时间衰减**：以 `last_used_at` 而非 `created_at` 为基准——**被用过就续命**，符合「常用的记忆更重要」。90 天半衰的直观数值：0 天 1.0、90 天 0.5、180 天 0.25、365 天 ≈0.06。`half_life_days <= 0` 时关闭衰减（测试里用大/非正半衰期验证）。
- **两阶段写**：先全量算分、过滤 `min_score`、排序，**只对最终返回的 top_k 更新** `hit_count += 1`、`last_used_at = now`，并把新值回填到返回对象。这样未命中的记忆不会因为被算过一次就「假装被使用」。
- 复杂度 O(N) 全表扫描 + Python 内点积；N 在万级以内完全够用。未来换 ANN 索引只需改 `search()` 内部，对外签名与 `MemoryItem` 不变。

## 5. 注入位置与格式

**位置**：在 `agent/main_agent.py` 的 `run_deep_agent()` 里，构造用户消息/调用 `agent.astream(...)` 之前，用 `task_query` 调 `retrieve_memory_context()`，把返回块拼在用户消息前面。选择用户消息而不是 system prompt 的原因：

- 不污染主/子智能体的系统提示词，子智能体仍按原职责工作；
- 注入内容随本轮消息进入 checkpoint，便于事后审计「这轮回答参考了哪条记忆」；
- 用户问题与背景挨在一起，模型更容易正确引用。

**格式**（严格固定，Lead 直接拼接，不要再加工）：

```text
### 已知背景（长期记忆）
- [记忆 #12] 用户偏好：只看 2026 年 Q1 数据（preference，置信度 0.90）
```

渲染由 `render_for_prompt()` 完成：无条目返回 `""`（Lead 不拼任何额外分隔符，保证「无记忆」与「有记忆但为空」两种 prompt 完全一致）；每条一行，`id` 用于 API 删除/纠错时定位。

`retrieve_memory_context()`（`agent/memory.py:525`）额外做 **600 token 预算**控制（`_estimate_tokens` 按 1.5 字符/token 估算）：标题行计入预算，逐条整条加入，放不下就停，**绝不切半条**；一条都放不下返回 `""`。`max_tokens <= 0` 直接返回 `""`（用于回归测试/紧急关闭）。

## 6. 维护策略

| 机制 | 规则 | 目的 |
| --- | --- | --- |
| 去重更新 | 余弦 > 0.9 时更新内容/置信度/source/last_used_at | 同一偏好只留一条，越说越准 |
| 时间半衰 | 90 天半衰期，基准 `last_used_at` | 旧信息自然降权，无需人工过期 |
| 命中计数 | `hit_count` 每次命中 +1 | 观测哪些记忆真被用上；后续可做冷记忆清理/晋升 |
| 使用续命 | 命中即刷新 `last_used_at` | 常用记忆保持高分，避免误衰减 |
| API 删除 | `DELETE /api/memory/{id}` 硬删 | 运营可纠正错误记忆 |
| 清空 | `clear(thread_id=None)` 全清 / 按会话清 | 调试与重置 |
| 失败降级 | 检索异常返回 `""`、抽取异常返回 `[]` | 记忆子系统故障不拖垮主任务 |

未来可补（当前不做，避免过度设计）：`hit_count = 0` 且分数长期低于阈值的冷记忆 TTL 清理、按 type 的手工审核工作流、多用户 `user_id` 隔离。

## 7. embedding 选型：本地字符 n-gram 哈希向量

`embed_text()`（`agent/memory.py:118`）：文本去空白、转小写后切 1~3 字符 n-gram；每个 gram 用 `blake2b(..., digest_size=8)` 哈希，低 4 字节定维度下标、第 5 字节定符号 ±1，累加到 `dim=256` 的桶里；最后 L2 归一。空文本返回零向量，`dim <= 0` 抛 `ValueError`。确定性：同文本在任何进程/机器都得到同一向量。

选它的原因：

1. **当前 LLM 接口没有 embedding endpoint**：`agent/llm.py` 用 `init_chat_model(model=os.getenv("LLM_QWEN_MAX"), model_provider="openai")` 对接 DeepSeek 兼容接口，只有 chat completion，没有 `/embeddings`；为少量记忆再引入一个 embedding 服务/API key 不划算。
2. **零依赖、可离线**：不新增三方包，不需要联网，`pytest` 里也能验证排序逻辑；评测不产生额外费用。
3. **对短中文偏好够用**：记忆条目通常是一两句话，字符 n-gram 对「2026 年 Q1」「保留两位小数」「数码电子」这类关键词/数字/术语的字面重合很敏感。实测去重对近似句（如「…GMV 数据」vs「…GMV 数据表」）余弦可达 0.95，相关 query 与记忆的余弦 0.7~0.75，无关文本接近 0。
4. **可替换**：未来接入真 embedding 时，只需替换 `embed_text()` 实现（输入 str、输出 L2 归一 `list[float]`），`cosine_similarity`、存储格式（BLOB）与 `search()` 排序全都不变；维度若变，改 `EMBED_DIM` 并做一次库迁移即可。

局限也要写清楚：字符 n-gram 只捕捉**字面**相似，同义改写（「不要退款单」vs「排除退款订单」）余弦可能偏低。缓解手段是抽取侧把用户原话尽量规范化后存储；真正解决要靠语义 embedding，届时换实现即可。

## 8. 跨会话评测设计

目标：证明「**会话 A 说一遍，全新会话 B 能用上**」——这正是 checkpoint 做不到、长期记忆必须做到的事。脚本 `evals/memory_eval.py`（只做客户端，不启动服务）：

每组用例流程：

1. 会话 A：`thread_id = uuidA`，先连 `ws://.../ws/{thread_id}` 再 `POST /api/task`，等 `task_result` 事件（`evals/run_eval.py` 同款收包方式）；
2. 轮询 `GET /api/memory?thread_id=uuidA&limit=20`，直到出现 ≥1 条记忆或 30s 超时（写入是任务结束后异步抽取的，必须轮询）；
3. 会话 B：**全新** `thread_id = uuidB`，问一个依赖该记忆的问题，等 `task_result`；
4. 确定性判分：关键词组（组内任一命中）+ 正则（如两位小数 `\d+\.\d{2}(?!\d)`）+ 顺序约束（「结论」须早于「过程」）；不用 LLM 裁判，保证可复现、零成本、争议小。

5 组用例（`CASES`）：

| ID | 会话 A（说） | 会话 B（问） | 判分期望 |
| --- | --- | --- | --- |
| mem-01 | 我们只看2026年Q1的数据 | 那段时间GMV大概多少？ | 回答沿用 2026 与 Q1/1-3 月 |
| mem-02 | 以后GMV都按SOP口径，金额保留两位小数 | 算一下客单价，记住要保留两位 | 出现客单价且数字为两位小数 |
| mem-03 | 以后报告先给结论再给过程 | 给我出个品类占比简报 | 开头 400 字内有「结论」，且早于「过程」 |
| mem-04 | 我这边主要负责数码电子品类 | 分析一下我们品类的销售情况 | 回答聚焦数码电子 |
| mem-05 | 统计时不要算退款订单 | 算一下总销售额 | 提及退款，并说明排除/过滤/status 条件 |

CLI：`--base`（默认 `http://127.0.0.1:8010`）、`--timeout`（单轮超时，默认 240s）、`--only`（只跑某组）、`--out`（报告路径，默认 `evals/memory_report.md`）。脚本在**服务未启动**（探活 `/api/files` 失败）或**记忆接口未集成**（`/api/memory/stats` 返回 404）时给出可操作提示并以非 0 退出；单组失败不影响其余用例，最终写 Markdown 报告（汇总表 + 逐用例明细 + B 的回答原文）。

## 9. 端到端评测结果

- 运行时间：2026-11-02 21:54；服务 `http://127.0.0.1:8010`（`MEMORY_AUTO_WRITE=1`，每轮任务结束后自动抽取写回）；命令 `python evals/memory_eval.py --timeout 240`；完整报告在 `evals/memory_report.md`。

### 9.1 结果汇总

| 用例 | 结果 | 会话A耗时(s) | 会话B耗时(s) | 记忆条数 | 未命中/说明 |
| --- | --- | --- | --- | --- | --- |
| mem-01 时间范围偏好 | ✅ | 4.8 | 19.5 | 1 | B 沿用 2026Q1 口径，GMV 23,096.00 |
| mem-02 口径约定：两位小数 | ✅ | 33.6 | 24.0 | 4 | 客单价 9,893.00 / 4,946.50 均为两位小数 |
| mem-03 报告结构偏好 | ✅ | 3.8 | 43.2 | 2 | 开头即「核心结论」，结构符合结论先行 |
| mem-04 关注业务范围 | ✅ | 3.4 | 92.6 | 1 | 分析聚焦数码电子品类 |
| mem-05 统计口径约束 | ✅ | 2.5 | 19.5 | 4 | 明确写出「已排除退款、取消和待支付的订单」 |

整体 5/5 通过（100%），记忆写入 5/5 组在会话 A 后可通过 `/api/memory` 查到。会话 A 平均 9.6s（最长 33.6s），会话 B 平均 39.7s（最长 92.6s）。

### 9.2 结论

- **通过率**：5/5（100%）。会话 A 不需要任何特殊指令，正常陈述偏好/约束即可；会话 B 全部是全新 `thread_id`，证明信息确实来自跨会话记忆而非 checkpoint。
- **修复前的对照**：未加兜底时该套用例为 4/5 —— 首轮 mem-01（"那段时间 GMV 多少"召不回"只看 2026Q1"）、加上省略指代兜底后 mem-05（"算一下总销售额"召不回"统计时不要算退款订单"）先后失败，两次失败都不是抽取没写入，而是**检索没召回**：字符 n-gram 只抓字面重合，而这两类记忆与问题几乎没有共同字面。
- **有效修复**：① 省略指代时用 `recent()` 注入最近记忆；② 计算类任务强制注入最近 `constraint` 记忆；③ 主流程检索 `top_k` 从 3 提到 5。三项都落在集成层，`MemoryStore` 的检索公式与对外签名未变。
- **失败/局限（如实记录）**：字符 n-gram 无法处理同义改写，真正语义相关要靠 embedding（接口可平滑替换）；抽取模型每轮最多输出 5 条、置信度有波动；跨用例多轮运行后记忆会累积（本次库内 30+ 条），无关记忆可能挤占注入名额，生产环境需要按用户隔离或做冷记忆清理；记忆抽取是任务收尾的同步 LLM 调用，开启写回会延长任务总时长（评测 20 题时用 `MEMORY_AUTO_WRITE=0` 关闭以免干扰时延统计）。

