# 设计取舍：MySQL vs RAGFlow、FC vs MCP 与 P0/P1 关键决策

> 本文记录 P0/P1 阶段反复被问到、也最容易做错的几组技术选型。每条统一按 **问题 → 两个候选 → 结论 → 理由 → 本项目落点** 展开，落点全部指向仓库内真实文件/函数；所有落点均已在仓库内落盘（2026-11-02 核对），引用前仍建议按路径确认存在。
>
> 相关需求编号以 `docs/requirements-p0-p1.md` 为准：§3.1 性能埋点、§3.2 数据源分工、§3.3 工具接入协议、§3.4 意图方案、§3.5 流式输出、§3.6 数据建模、§3.7 记忆、§3.9 评测升级、§3.11 上下文管理。

## 结论速览

| 编号 | 核心问题 | 结论 | 主要落点 |
| --- | --- | --- | --- |
| §3.2 | 事实数值从哪取、口径从哪来 | MySQL 取数 + RAGFlow 查口径，**串联编排**而非合并成一个检索端点 | `tools/db_tools.py`、`tools/ragflow_tools.py`、`agent/subagents/`、`docs/kb/` |
| §3.3 | 工具怎么暴露给模型 | 本项目全部用 **Function Calling**；MCP 是未来"工具层独立部署/跨宿主复用"时的演进方向，两者分层不互斥 | `tools/*.py`、`agent/main_agent.py`、`agent/subagents/*.py` |
| §3.4 | 问题分给哪个子智能体 | **分层路由**：L1 规则 → L2 字符 n-gram 朴素贝叶斯 → L3 LLM 兜底 → L4 反问澄清 | `agent/intent_router.py`、`evals/intent_eval.py` |
| §3.5 | 5–10 秒等待怎么变成"秒开" | 复用已有 WebSocket，新增 `token` 事件；服务端聚合推送、前端增量渲染 + 首字延迟指标 | `api/monitor.py`、`agent/main_agent.py`、`api/static/index.html` |
| §3.6 | 订单数据怎么建模 | 当前只有一张 `orders` 表，字段类型/索引/状态枚举按 `sql/init.sql` 为准，金额一律 `DECIMAL` | `sql/init.sql`、`docs/kb/orders数据字典.md`、`docs/kb/订单分析口径SOP.md` |
| §3.7 | 跨会话偏好/约束存哪 | 长期记忆是**独立存储 + 向量检索 + 生命周期**，与 checkpoint 的图状态互补，不是替代 | `agent/memory.py`、`docs/memory-design.md` |
| §3.11 | 长上下文怎么不爆 | 预算驱动：token 估算 + 分节组装（槽位/记忆/问题）+ 分层裁剪，而不是无脑截断 | `agent/context_manager.py`、`docs/context-management.md` |

## 通用判断原则

后面所有取舍都可以还原成四条原则，写在这里避免逐条重复：

1. **可观测优先**：任何链路都要能回答"上一次任务慢在哪、错在哪、走的哪条路径"。所以计时（`agent/timing.py`）、意图（`intent.layer`）、工具输出摘要（`agent/tool_observer.py`）都是先埋进去，再谈优化。
2. **确定性优先、LLM 兜底**：能用规则/离线模型给出稳定答案的，不交给 LLM 随机发挥；LLM 只做它擅长的开放式环节（分类兜底、记忆抽取、报告撰写）。
3. **失败要降级，不能拖垮主流程**：记忆检索失败返回空串、记忆面板接口失败静默隐藏、工具输出预览失败不影响工具本身、流式 token 没到就等最终答案。
4. **零新增重型依赖**：单机演示 + 评测可复现，优先标准库/已有依赖（SQLite、纯 Python 分类器、字符 n-gram 哈希向量），不引入向量数据库、不引入前端构建链。

---

## §3.2 MySQL vs RAGFlow：一个答"是多少"，一个答"怎么算"

### 问题

电商运营分析的问题天然分两类：

- "orders 表里总 GMV 是多少？""哪个品类订单数最多？"——要的是**事实与数值**，可精确聚合、有 schema 约束、答案唯一。
- "GMV 的口径是什么？""清库存一级预警标准是多少天？""哪些订单状态不计入 GMV？"——要的是**规则与口径**，靠语义匹配、必须引用原文，答案会随文档版本变化。

两类问题混在一起时，系统很容易"用错数据源"：该查文档的拿 SQL 结构硬答，该算数的拿一段文字糊弄。

### 两个候选

| 候选 | 做法 | 表面收益 | 真实代价 |
| --- | --- | --- | --- |
| A. 合并检索端点 | 做一个统一的 `search` 工具/端点，内部按 query 路由到 MySQL 或 RAGFlow，对模型只暴露一个入口 | 工具数量少、模型选择简单 | 模型看不到"结构化取数"与"语义查规"的语义差异，失去工具选择的分离信号；路由失败时整题失败，无法分工评测；两类数据源的生命周期/权限/失败模式被耦合进一个端点 |
| B. 保留两个数据源、两个工具，编排层串联 | MySQL 工具归数据分析助手，RAGFlow 工具归内部知识助手；主智能体/子智能体按"先定口径、再取数、最后合并"的顺序协作 | 职责清晰、可分别评测、可分别降级 | 需要编排层（提示词 + 意图路由）明确先后顺序，否则模型可能只取数不查口径 |

### 结论

**选 B：串联，不是并列。**

先由知识库/文档确定口径与规则，再由 SQL 取数，最后在报告里同时给出 **结论 + 口径 + 数据来源**。两个工具、两个子智能体、两套评测关键词，不做合并端点。

### 理由

1. **回答对象不同**：一个是"是多少"（事实/数值/可精确聚合/有 schema/答案唯一），一个是"怎么算、能不能这么算"（规则/口径/语义匹配/必须引用原文）。同一个检索入口无法同时表达这两种契约。
2. **失败模式不同**：SQL 失败是连接/语法/只读守卫问题，文档检索失败是知识库不可达/召回为空；合并后错误信息被抹平，排查要翻两层。
3. **合并会退化成工具误选，本项目早期评测实测到过**：首轮真实链路评测（`evals/report-round1.md`）查文档类 3/5 通过——其中 `rag-04`《根据数据字典，orders 表里哪些订单状态不计入 GMV》被答成 SQLite/库结构信息，没有命中期望的"取消/退款"口径；`rag-01`《SOP 里样本量少于多少禁止趋势推断》直接 240s 超时。这正是"口径类问题被当成数据问题"的典型症状，说明数据源边界必须靠分工和编排显式表达，而不是藏进一个检索端点。
4. **报告质量要求三要素**：运营分析报告如果只给数字不给口径，数字没有可解释性；只给口径不给数字，没有决策价值。串联天然支持"结论 + 口径 + 来源"，合并端点则需要模型自己拆分，稳定性差。
5. **评测可分解**：`evals/dataset.json` 的 `expect` 同时覆盖数值（如 `39572`）与口径关键词（如 `取消/cancelled`），串行链路可以分别定位"取数错"还是"口径错"。

### 分工与串联时序

| 数据源 | 回答类型 | 工具 | 归属子智能体 | 知识/数据载体 |
| --- | --- | --- | --- | --- |
| MySQL `deep_search.orders` | 事实、数值、聚合 | `list_sql_tables` / `get_table_data` / `execute_sql_query` | 数据分析助手 | `sql/init.sql`、`docs/kb/orders数据字典.md` |
| RAGFlow 知识库 | 规则、口径、SOP、数据字典释义 | `get_assistant_list` / `create_ask_delete` | 内部知识助手 | `docs/kb/订单分析口径SOP.md` 等 |
| Tavily | 外部市场情报 | `internet_search` | 市场情报助手 | 公网 |
| 主智能体 | 汇总、报告产出 | `generate_markdown` / `convert_md_to_pdf` / `read_file_content` | 主智能体 | `output/` |

串联时序（以"按 SOP 口径算各品类 GMV"为例）：

```
用户问题
  └─ 意图路由（mixed / document / database，见 §3.4）
       ├─ 内部知识助手：查 SOP/数据字典 → 拿到口径（GMV=SUM(total_amount)、status 过滤、品类占比用 GMV）
       ├─ 数据分析助手：按口径写只读 SQL → 取数（WHERE status NOT IN ('cancelled','refunded')）
       └─ 主智能体：汇总成报告 = 结论 + 口径 + 数据来源
```

关键约束：**口径必须来自文档而不是模型记忆**；SQL 里的过滤条件必须能追溯到数据字典（`status` 的 `cancelled/refunded` 不计 GMV）；报告里引用了哪条口径、查了哪张表、用的是哪套数据，要写清楚。跨源不一致时（SOP 与数据字典冲突），以数据字典的字段定义优先，并在报告里标注冲突。

### 本项目落点

| 落点 | 说明 |
| --- | --- |
| `tools/db_tools.py` | `list_sql_tables` / `get_table_data` / `execute_sql_query` 三个只读工具；`assert_read_only_sql` 用 `_ALLOWED_SQL_PREFIXES`（select/with/show/desc/describe/explain）+ `_FORBIDDEN_SQL_KEYWORDS` 黑名单 + 禁止多语句做守卫；`get_db_config()` 读 `MYSQL_*` 环境变量 |
| `tools/ragflow_tools.py` | `get_assistant_list` / `create_ask_delete`，RAGFlow SDK 封装，凭据经 `rawflow.rag_config._load_ragflow_env()` 读取 |
| `agent/subagents/database_query_agent.py` | 数据分析助手（`name/description/system_prompt/tools` 字典，提示词约束"先看数据字典/口径，再只读查询"） |
| `agent/subagents/knowledge_base_agent.py` | 内部知识助手，负责 SOP、数据字典、项目文档的口径问答 |
| `agent/main_agent.py` | `create_deep_agent(subagents=[database_query_agent, network_search_agent, knowledge_base_agent])`，编排层按意图给路由提示（`_route_hint`）；报告产出工具也在主智能体 |
| `prompt/prompts.yml`（经 `agent/prompts.py` 加载） | 主/子智能体职责、口径要求、报告结构（结论/口径/来源） |
| `docs/kb/订单分析口径SOP.md` | GMV、订单数、AOV、ASP、品类占比口径，样本量规则，清库存三级标准 |
| `docs/kb/orders数据字典.md` | 字段类型、`status` 枚举与 GMV 计入规则、单表边界（无商品/库存/流量表） |
| `evals/dataset.json` / `evals/run_eval.py` | 20 题 4 类（查库/查文档/联网/综合），关键词组判定；串行链路按类别分别看通过率 |

---

## §3.3 Function Calling vs MCP：不是替代关系，是不同层次

### 问题

工具如何暴露给模型？是全部走 Function Calling（进程内函数签名 + JSON Schema），还是起一个 MCP server 做标准化工具接入？

### 两个候选

| 候选 | 做法 | 适用前提 |
| --- | --- | --- |
| A. Function Calling（FC） | 在进程内用 `@tool` / `StructuredTool` 定义工具，`name/description/args_schema` 随 agent 构建注册，模型输出 `tool_calls` 由框架本地执行 | 工具与 agent 同进程、同一代码库、单机部署 |
| B. MCP（Model Context Protocol） | 工具封装成独立 MCP server（stdio/HTTP），宿主通过协议 `list_tools` 动态发现、远程调用 | 工具跨进程/跨主机、多宿主复用、需要独立权限与生命周期 |

### 结论

本项目 **全部 6 个业务工具 + 3 个产出工具都用 FC**（进程内），MCP 作为未来演进方向预留，不引入当下实现。

关键认知：**FC 与 MCP 不是替代关系，而是不同层次**。

- FC 是 **模型 ↔ 工具** 的调用协议：模型输出结构化 `tool_calls`，由宿主执行后回填结果。
- MCP 是 **工具 ↔ 宿主** 的接入协议：宿主通过标准协议发现并调用外部工具进程。

两者可以叠加：宿主先用 MCP 拿到远端工具，再以 FC 的形式暴露给模型。本项目在"宿主=本进程、工具=本仓库函数"的形态下，MCP 的那一层就是空的，硬加只会多一跳。

### 理由（对比维度）

| 维度 | Function Calling | MCP | 本项目取舍 |
| --- | --- | --- | --- |
| 部署边界 | 与 agent 同进程 | 跨进程/跨主机，需要管理 server 生命周期 | 单机单进程，FC 足够 |
| 发现能力 | 构建时静态注册 schema | 运行时 `list_tools` 动态发现 | 工具集固定（9 个），不需要动态发现 |
| 复用 | 代码级 import，仅限本仓库 | 协议级复用，可被 IDE/其他 agent 宿主消费 | 无多宿主需求 |
| 延迟 | 本地函数调用，微秒级 | 多一跳 IPC/HTTP/stdio 序列化 | P50 目标 < 10s，省掉不必要的跳数 |
| 调试 | 断点/日志直接打在进程内 | 需同时看 server 日志与协议帧，排障成本高 | 单人维护，优先可调试 |
| 权限与隔离 | 与 agent 同权限 | 可独立部署、独立授权 | 当前无隔离需求 |
| 适用规模 | 中小型单体 agent | 多团队工具市场、工具即服务 | 当前规模选 FC |

补充：三个业务子智能体的工具都是**只读**（只读 SQL、知识库查询、联网搜索），进程内调用不存在越权面扩大的问题；`assert_read_only_sql` 已经把 SQL 风险收敛在工具内部。

### 未来如何切到 MCP（不改编排层）

1. 把 `tools/*.py` 的 `@tool` 函数包一层 MCP server（优先 stdio，便于本地进程管理），`name/description/args_schema` 原样暴露。
2. 宿主侧改用 MCP 客户端适配器（如 `langchain-mcp-adapters`）把 MCP 工具转成 `BaseTool`，替换子智能体 `tools` 列表里的本地函数。
3. **`agent/main_agent.py` 的编排逻辑、意图路由、计时、记忆注入都不用改**——它们只依赖 `BaseTool` 接口，不依赖工具是本地还是远端。
4. 触发条件：需要把工具给其他宿主（IDE/另一个 agent 项目）复用，或工具需要独立部署/独立权限隔离，或团队出现"工具即服务"的协作需求。没有这些触发条件前不引入。

### 本项目落点

| 落点 | 说明 |
| --- | --- |
| `tools/db_tools.py` | 3 个 `@tool`：`list_sql_tables` / `get_table_data` / `execute_sql_query` |
| `tools/ragflow_tools.py` | 2 个 `@tool`：`get_assistant_list` / `create_ask_delete` |
| `tools/tavily_tool.py` | 1 个 `@tool`：`internet_search`（Tavily + 退避重试） |
| `tools/markdown_tools.py` / `pdf_tools.py` / `upload_file_read_tool.py` | 3 个产出类 `@tool`：`generate_markdown` / `convert_md_to_pdf` / `read_file_content` |
| `agent/tool_observer.py` | `instrument_tool()` 包装 `BaseTool`（`StructuredTool.from_function` 保留 name/description/args_schema，仅加计时与输出预览），证明工具层可插拔、不依赖 FC/MCP 具体形态 |
| `agent/main_agent.py` | `create_deep_agent(tools=[instrument_tool(generate_markdown), instrument_tool(convert_md_to_pdf), instrument_tool(read_file_content)], ...)`；子智能体工具在各自 `agent/subagents/*.py` 中声明 |
| `agent/subagents/*.py` | `{name, description, system_prompt, tools}` 字典，`tools` 即 FC 工具列表；未来换 MCP 只改这里的列表 |
| `tests/test_sql_guard.py` | 只读守卫单测，保证工具边界不因接入方式变化而失效 |

---

## §3.4 意图方案取舍：为什么不用"LLM 直接分类"

### 问题

用户问题需要被路由到正确的子智能体（数据分析/内部知识/市场情报/综合）。最直接的做法是让 LLM 分类，但这条路在"稳定性、可校准性、可评估性"上都有硬伤。

### 两个候选

| 候选 | 做法 | 问题 |
| --- | --- | --- |
| A. LLM 直接分类 | 一次 prompt 让模型输出 label（+置信度） | ① 不稳定：同义改写/温度/上下文顺序都会导致标签漂移；② 不可校准：LLM 自报置信度不是概率，无法设有效阈值；③ 不可评估：每题一次 LLM 调用，离线评测贵且不可复现，无法回归 |
| B. 分层路由 | L1 规则 → L2 离线分类器 → L3 LLM 兜底 → L4 反问澄清 | 需要维护规则与训练集，工程量大一些，但每层都可测、可降级、可解释 |

### 结论

**选 B：分层路由。** 四层职责如下：

| 层 | 手段 | 触发条件 | 输出 |
| --- | --- | --- | --- |
| L1-rule | 关键词/正则强规则（`_DATABASE_RULES` / `_DOCUMENT_RULES` / `_WEB_RULES` / `_MIXED_RULES`） | 强命中且规则间不冲突 | 直接返回，`confidence=min(0.95, 0.78+0.04*命中数)` |
| L2-classifier | 字符 n-gram（1–3）+ 朴素贝叶斯，纯 Python 离线训练 | L1 未命中或弱命中 | `label + confidence + scores`；要求 `confidence ≥ 0.55` 且 `margin(top1-top2) ≥ 0.15` 才采纳 |
| L3-llm | few-shot LLM 分类 | L2 置信度/边际不足 | 返回解析后的 label；失败降级 L4 |
| L4-clarify | 反问澄清（`CLARIFICATION_QUESTION`） | 空问题、离线模式、以上全部失败 | `needs_clarification=True`，主智能体直接返回追问而不是硬猜 |

标签体系固定 4 类：`database`（数据分析）、`document`（内部知识）、`web`（市场情报）、`mixed`（综合 → 三个子智能体）。映射表：`INTENT_TO_SUBAGENTS`。

### 理由

1. **稳定性**：L1/L2 对同一输入完全确定；L2 的朴素贝叶斯在字符 n-gram 上对中文近义改写比较鲁棒，且训练/推理无随机性。
2. **可校准**：L2 输出的是后验近似概率，配合 margin 双阈值（`L2_MIN_CONFIDENCE=0.55`、`L2_MIN_MARGIN=0.15`）能真正"低置信就兜底"，而不是拿 LLM 的口头置信度当概率。
3. **可评估**：`classify_intent(use_llm=False)` 提供完全离线的确定路径，可对 `evals/intent_dataset.json`（72 条，4 类各 18 条）跑混淆矩阵与每类 P/R/F1（`confusion_matrix` / `classification_metrics` 已在 `agent/intent_router.py` 内实现，供评测脚本与单测复用）。类别均衡（18/18/18/18）让 accuracy 有意义，但仍要求报告误判方向而不只看总分。
4. **成本可控**：LLM 只在低置信尾部兜底，绝大多数请求零额外 token；L1 命中是纯字符串匹配，微秒级。
5. **错误前置**：宁可在 L4 明确反问一句，也不要在低置信下猜错子智能体、浪费 5–10 秒和几次工具调用。

### 风险与边界

- 训练集只有 72 条（每类 18），L2 泛化有限：新领域词汇可能落入低置信区而频繁走 L3；因此 L1 规则要覆盖高频业务说法（orders/SOP/GMV/竞品等），用规则消化头部流量。
- L2 的置信度是近似值，**不能当严格概率用于对外承诺**；阈值是工程调参结果，改动需重跑 `evals/intent_eval.py` 回归。
- `mixed` 类与 `database/document` 边界天然模糊（"结合知识库和数据库回答"）：`_MIXED_RULES` 优先匹配显式"结合/先查…再对照…"句式，避免 mixed 被误判成单源。
- 意图路由只决定"给主智能体什么提示 + 哪个子智能体优先"，**不是硬性执行路径**：deep agent 仍可自主调整，因此路由错误不会直接导致任务失败，只会降低效率与准确性。

### 本项目落点

| 落点 | 说明 |
| --- | --- |
| `agent/intent_router.py` | 全部四层实现：`classify_intent` / `classify_by_rules` / `classify_by_classifier` / `NaiveBayesIntentClassifier` / `_llm_classify` / `_clarify_result`；`IntentResult.to_dict()`；`confusion_matrix` / `classification_metrics`；常量 `INTENTS`、`INTENT_LABELS_ZH`、`INTENT_TO_SUBAGENTS`、`L2_MIN_CONFIDENCE`、`L2_MIN_MARGIN` |
| `agent/main_agent.py` | 任务入口调用 `classify_intent(task_query)`，`monitor.report_intent(intent.to_dict())` 上报（前端徽标），并把 `intent/intent_confidence/route_layer` 写进计时与 metrics；`needs_clarification` 时直接返回澄清问题，不调工具 |
| `api/monitor.py` | `report_intent(intent)` → WebSocket `event="intent"` |
| `api/static/index.html` | 意图徽标：`🎯 意图：{label_zh} · 置信度 {0.86} · L2-classifier` + 建议子智能体列表 |
| `evals/intent_dataset.json` | 72 条标注评测集（database/document/web/mixed 各 18） |
| `evals/intent_eval.py` | 混淆矩阵 + 每类 P/R/F1 报告；离线 `use_llm=False` 保证可复现 |
| `tests/test_intent_router.py`（35 个测试） | 规则/分类器/降级路径单测 |

---

## §3.5 流式方案：复用 WebSocket + token 事件

### 问题

主智能体一次完整回答约 5–10 秒（2026-10-26 基线评测 P50 8.3s），用户在页面上长时间只看到"工具调用中"，首字等待体验差；"首字延迟"本身也是 P0 性能指标，需要可采集。

### 两个候选

| 候选 | 做法 | 代价 |
| --- | --- | --- |
| A. 新增 SSE / HTTP chunk 流式端点 | 为流式单开 `/api/task/stream`，用 `text/event-stream` 推送增量 | 新协议、新端点、新连接生命周期；与现有 WS 事件（tool_start/assistant_call/task_result）分属两条通道，前端要合并两路事件排序；断线重连语义要重写 |
| B. 复用现有 WebSocket + 新增 `token` 事件 | 在已有 `/ws/{thread_id}` 事件流中增加 `event="token"`，与其它 monitor_event 同通道、同 thread 隔离 | 高频小消息（需服务端聚合）；前端要做增量渲染与最终答案替换 |

### 结论

**选 B：WebSocket + `token` 事件。**

`token` 事件 payload：`{"type":"monitor_event","event":"token","message":"","data":{"text":"增量片段"},"timestamp":...}`。同一任务会产生多条；客户端把 `text` 依次追加到当前会话的"流式回答"气泡，`task_result` 到达后丢弃流式状态，用最终完整文本重新做 markdown 渲染。

### 理由

1. **零新增通道**：`api/monitor.py` 已有 `ToolMonitor._emit` + `ConnectionManager` 的事件契约与 `thread_id → 连接` 映射，加一个事件类型即可；前端 `ws.onmessage` 也只多一个分支。
2. **天然有序**：token 与工具事件走同一队列，前端按到达顺序渲染，不会出现 SSE + WS 双通道的跨通道乱序。
3. **与首字指标同源**：服务端 `TaskTimer.mark_first_token()` 记录"开始 → 第一个 token"的耗时，`snapshot().first_token_ms` 直接进 metrics，前端 timing 面板展示同一数字，避免两套口径。
4. **代价可控制**：token 是高频消息，服务端用 `_TokenEmitter` 聚合（攒够 `_TOKEN_FLUSH_CHARS=12` 字符或 `_TOKEN_FLUSH_SECONDS=0.06` 秒才发一批，结束前 `flush()`），前端用 `requestAnimationFrame` 合并渲染（`scheduleStreamRender`），不逐 token 触发 DOM 重绘。
5. **兼容降级**：后端不推 token 事件时（旧版本/异常），前端不会出现流式卡片，仍等 `task_result` 正常显示——同一个前端兼容两种后端。

### 实现要点

| 层 | 做法 | 落点 |
| --- | --- | --- |
| 服务端计时 | 任务开始 `start_timer()`；首个 token `timer.mark_first_token()`（只认第一次）；`snapshot()` 输出 `first_token_ms` 与各阶段毫秒 | `agent/timing.py`（`TaskTimer`）、`agent/main_agent.py` |
| 服务端推送 | `monitor.report_token(text)`（空文本忽略，`broadcast_only=True` 不写 stream_writer 不 print） | `api/monitor.py` |
| 主链路取 token | `agent.astream(stream_mode=["updates","messages"])`，只取**主图 model 节点**的 `AIMessageChunk.content`，避免把子智能体内部 token 混进最终回答；`_TokenEmitter` 攒批 | `agent/main_agent.py` |
| 最终答案兜底 | 流式结束后从图状态读最终回答（`_read_final_answer_from_state`），保证有完整文本可替换 | `agent/main_agent.py` |
| 前端增量渲染 | `session.streamText += data.text`（仅当前会话且 running）；`streamCardHTML` 创建/复用 `#stream_{threadId}` 气泡；`updateStream` 用 `mdToHtml` 渲染 + 打字光标，自动滚底 | `api/static/index.html` |
| 前端清理 | `task_result` 到达：`streamText=''`、`streaming=false`、移除流式气泡，再用完整 `result` 走原有 markdown 大卡片；`error`/`onclose` 同样清理并停止光标 | `api/static/index.html` |
| 指标展示 | `timing` 事件 → `timingPanelHTML`：首字延迟、总耗时、工具调用次数 chips + 检索/LLM/工具/子智能体横向条形 | `api/static/index.html`、`api/monitor.py` |
| 指标查询 | `GET /api/metrics` → `TaskMetricsStore.aggregate()` 的 P50/P95/分阶段统计 | `api/server.py`、`agent/timing.py` |

### 验收

- `tests/test_timing.py`：`percentile`、`TaskTimer.mark_first_token/add/snapshot`、`TaskMetricsStore` 落库/聚合（已存在）。
- `tests/test_monitor_events.py`：`report_token` / `report_intent` / `report_timing` / `report_tool_result` / `report_memory` 的 payload 结构（已存在，15 个测试）。
- 前端：`node --check` 内联脚本语法校验；真实链路手测（或 `scripts/e2e_frontend_check.py` 产出的 `task_result`）验证流式追加、替换、自动滚底。
- 评测：`evals/run_eval.py` 的 P50/P95 与 `first_token_ms` 进入报告，P1 §3.9 汇总。

---

## §3.6 重要字段清单：以 `sql/init.sql` 的 `orders` 表为准

### 问题

数据层的建模直接决定口径能不能算对：金额用什么类型、时间用哪个字段、状态枚举怎么影响 GMV、索引建给谁用。当前示例库只有 `orders` 一张表（无商品/库存/流量表），需要把"有什么、没有什么、能算什么、不能算什么"写清楚，避免报告编造指标。

### 两个候选

| 候选 | 做法 | 问题 |
| --- | --- | --- |
| A. 订单宽表 | 把商品、客户、支付、库存信息都塞进 `orders` | 演示数据少时看似省事，但字段语义混杂、冗余大，后续接真实数据源要拆表重写 |
| B. 规范化多表 | `orders` + `order_items` + `products` + `customers` + `inventory` | 标准，但当前只有 8 条演示数据、且没有商品/库存/流量数据源，多表是过度设计，还会让 SQL 题目与知识库口径超出可验证范围 |

### 结论

**当前阶段以单表 `orders` 为准**（`sql/init.sql`），字段精简、类型精确、索引只建在真实会用的过滤/分组列上；同时在数据字典里显式声明"只有一张表"的边界，指标不可计算就明说不可计算。等真实数据源接入时再按 B 演进。

### 字段清单（与 `sql/init.sql` 一致）

| 字段 | 类型 | 约束/默认 | 业务含义 | 建模要点 |
| --- | --- | --- | --- | --- |
| `id` | `INT` | `AUTO_INCREMENT PRIMARY KEY` | 行主键 | 仅内部关联用，不对外展示；订单号用 `order_no` |
| `order_no` | `VARCHAR(32)` | `NOT NULL` | 订单号 | 规则 ORD + 4 位年份 + 3 位流水（见数据字典）；唯一性规则保证 `COUNT(DISTINCT order_no)` 与 `COUNT(*)` 同义 |
| `product_name` | `VARCHAR(128)` | `NOT NULL` | 商品名 | 演示数据粒度；无商品维表，商品改名/重名无法追踪 |
| `category` | `VARCHAR(64)` | 可空 | 品类 | 枚举四值：数码电子/生活家电/美妆护肤/运动鞋服；**建索引**支持 `GROUP BY category`；品类占比口径用 GMV 而非订单数占比 |
| `price` | `DECIMAL(10,2)` | `NOT NULL` | 单价（元） | 精确两位小数，参与 ASP=SUM(price)/SUM(quantity) |
| `quantity` | `INT` | `NOT NULL DEFAULT 1` | 数量 | 数据质量规则要求 `total_amount = price * quantity` |
| `total_amount` | `DECIMAL(10,2)` | `NOT NULL` | 订单总金额（元） | **GMV 口径字段**：`SUM(total_amount)` 直接得 GMV；含税；金额一律 DECIMAL 不用 FLOAT |
| `customer_name` | `VARCHAR(64)` | 可空 | 客户名 | 无客户维表；报告展示需脱敏（SOP 要求） |
| `status` | `VARCHAR(16)` | `DEFAULT 'completed'` | 订单状态 | 枚举 completed/paid/cancelled/refunded：前两者计 GMV，后两者不计；SQL 模板与知识库口径必须一致 |
| `created_at` | `DATETIME` | `DEFAULT CURRENT_TIMESTAMP` | 下单时间 | **所有时间口径统一用此字段**（SOP）；**建索引**支持时间范围过滤/趋势/同比类查询 |

索引：

- `INDEX idx_category (category)`：按品类分组、品类筛选是运营分析最高频查询；没有索引时 `GROUP BY category` 全表扫。
- `INDEX idx_created_at (created_at)`：时间范围/趋势/同环比查询的过滤列；SOP 规定自然周/自然月/自然年口径都落在此列。
- 引擎/字符集：`ENGINE=InnoDB`、`utf8mb4` / `utf8mb4_unicode_ci`——中文品类与客户名必须 utf8mb4，InnoDB 保证事务与行级锁（只读场景也作为默认规范）。

### 理由

单表 + 精确类型 + 精准索引 + 显式口径，这四点合起来才保证"报告里的数字是对的"：金额类型决定精度、时间字段决定口径一致性、状态枚举决定 GMV 能否算对、单表边界决定哪些指标不能编。

#### 金额为什么必须 `DECIMAL(10,2)` 而不是 `FLOAT`/`DOUBLE`

- 二进制浮点无法精确表示 0.1/0.2 这类十进制小数，`SUM(total_amount)` 会出现 `39571.999999...` 一类误差，报告里就是"数不对"。
- 金额有明确精度语义（分），DECIMAL 的定点运算与四舍五入规则确定，跨 MySQL 与 Python（`decimal.Decimal`）一致。
- FLOAT/DOUBLE 只适合比率/度量类科学计算；金额、计数、ID 一律整数或定点。

#### `status` 枚举如何影响 GMV（口径落点）

数据字典约定：

| status | 含义 | 是否计入 GMV |
| --- | --- | --- |
| `completed` | 已完成 | 计入 |
| `paid` | 已支付未发货 | 计入 |
| `cancelled` | 已取消 | **不计入** |
| `refunded` | 已退款 | **不计入** |

因此"查 GMV"的标准 SQL 形态是 `WHERE status NOT IN ('cancelled','refunded')`（或显式白名单 `IN ('completed','paid')`），而不是无脑 `SUM(total_amount)`。当前 8 条演示数据全部是 `completed`，所以两种写法结果相同；但口径必须按数据字典写，否则接真实数据后会错。另：退款率当前**不可计算**（无退款金额/退款时间字段），SOP 已明确，报告不得编造。

#### 当前单表边界（能不能算）

| 想算的指标 | 数据是否支持 | 说明 |
| --- | --- | --- |
| GMV / 订单数 / AOV / ASP / 品类占比 / 时间趋势 | ✅ | 现有字段 + 口径 SOP 可算 |
| 退款率 / 退货率 | ❌ | 无退款字段 |
| 库存周转 / 清库存预警 | ❌（仅规则可讲） | 无库存表/库龄字段，SOP 的"90 天无动销"等只作为口径知识回答，不能用数据算 |
| 转化率 / 流量漏斗 | ❌ | 无流量/曝光/加购表 |
| 商品维度分析（单品排行/毛利） | 部分 | 有 `product_name/price/quantity`，无成本/毛利字段，毛利不可算 |

### 本项目落点

| 落点 | 说明 |
| --- | --- |
| `sql/init.sql` | 建库建表 DDL + 8 条演示数据（ORD2026001–008，2026-01-15–2026-07-01，`total_amount` 合计 39572） |
| `docs/kb/orders数据字典.md` | 字段类型/枚举/订单号规则/单表边界/GMV 过滤建议 |
| `docs/kb/订单分析口径SOP.md` | GMV、订单数、AOV、ASP、品类占比、时间口径、数据质量规则、样本量规则、清库存三级 |
| `tools/db_tools.py` | `list_sql_tables` / `get_table_data` / `execute_sql_query` + `assert_read_only_sql` 只读守卫（禁止写操作与多语句） |
| `agent/subagents/database_query_agent.py` | 数据分析助手提示词要求先看数据字典/SOP 再写 SQL |
| `evals/dataset.json` | 数值类断言（如 db-03 期望 39572、db-01 期望 8）与口径类断言（rag-04 期望 取消/退款） |
| `docs/requirements-p0-p1.md` | §3.6 需求→实现→验证映射 |

---

## §3.7 记忆 vs checkpoint：一个保会话连续，一个保跨会话知识

### 问题

用户在前一个会话里说过"报告都用中文、金额保留两位、只看 2026 年上半年"，下一个会话（新 thread）还能不能记住？deepagents 的 `AsyncSqliteSaver` checkpoint 能不能直接当记忆用？

### 两个候选

| 候选 | 做法 | 行为 |
| --- | --- | --- |
| A. 复用 checkpoint | 把偏好/约束写进图状态字段，靠 `checkpointer` 持久化，按 `thread_id` 恢复 | checkpoint 本意是**图运行状态**（消息历史、节点状态），按 thread 隔离，新会话读不到旧会话；恢复=整段回放，没有相关性检索 |
| B. 独立长期记忆层 | 独立 SQLite 表 `memories` + 向量检索 + 生命周期管理（去重/衰减/命中统计/删除）+ 提示词注入 | 跨 thread 全局可检索；只注入 top-k；可被 API/前端管理；与图状态解耦，失败可静默降级 |

### 结论

**选 B，但不替代 checkpoint：两者互补。**

- checkpoint（`AsyncSqliteSaver`，`data/checkpoints.db`）负责"同一会话内多轮对话与图状态"的连续性——deep agent 多轮追问、工具中间状态都靠它。
- 长期记忆（`agent/memory.py`，`data/memory.db`）负责"跨会话的用户事实/偏好/约束/术语"——换 thread 仍然记得。

### 理由（对比）

| 维度 | checkpoint | 长期记忆 |
| --- | --- | --- |
| 存储对象 | 图状态 + 消息历史 | 结构化事实/偏好/约束/术语（`fact/preference/constraint/glossary`） |
| 作用域 | `thread_id` 隔离，新会话读不到 | 默认跨 thread 全局，`thread_id` 仅作来源标记/可选过滤 |
| 召回方式 | 按 thread 整段恢复（回放） | 相关性 top-k：`cosine × confidence × 0.5^(days/half_life_days)`（half-life 90 天），带 `min_score` 阈值 |
| 生命周期 | 随会话存在，无内在淘汰 | `created_at/last_used_at/hit_count` 追踪；`add()` 余弦 >0.9 去重合并；可 `delete/clear` |
| 上下文成本 | 历史越长占用越大 | 只注入 top-k（`retrieve_memory_context` 限 600 tokens） |
| 失败影响 | checkpointer 故障 = 任务状态异常 | 记忆检索/写入失败 = 返回空串/跳过写库，主流程照常 |
| 可管理性 | 无面向用户的接口 | `GET /api/memory` / `DELETE /api/memory/{id}`，前端记忆面板可删除 |
| 实时性/一致性 | 与图状态强一致 | 最终一致（任务结束后异步抽取写入） |

### 设计要点（为什么这样实现）

- **向量检索不引入外部服务**：`embed_text` 用字符 n-gram + `blake2b` 特征哈希到固定 256 维并归一化，纯 Python、确定性、离线可复现；对中文短句（偏好/约束/术语）足够。将来换真 embedding 服务只需替换 `embed_text` 与 `EMBED_DIM`，`search`/`add` 接口不变。
- **写入门控**：不是每轮都写记忆。任务结束后 `extract_memory_candidates(query, answer)` 让 LLM 抽取最多 5 条候选项，经 Pydantic `MemoryCandidate` 校验（类型白名单、content 非空、confidence clamp 到 0–1）后才入库；失败返回 `[]`，不阻塞主流程。
- **注入格式固定**：`render_for_prompt(items)` 输出 `### 已知背景（长期记忆）` + `- [记忆 #12] 内容（preference，置信度 0.90）`，与 `agent/context_manager.build_agent_input` 的 `memory_block` 分节对齐，受 600 token 预算约束。
- **可观测**：写入后发 `event="memory"`（`retrieved/written` 列表），前端展示"本轮召回/写入"；任务结束刷新右侧记忆面板，与 `GET /api/memory` 保持同步。

### 本项目落点

| 落点 | 说明 |
| --- | --- |
| `agent/memory.py` | `MemoryStore`（`add` 去重合并、`add_candidates`、`search`、`list/delete/clear/stats`、`render_for_prompt`）、`MemoryCandidate`/`MemoryItem`、`embed_text`/`cosine_similarity`、`retrieve_memory_context`、`extract_memory_candidates`、`_get_default_store`；`data/memory.db` |
| `agent/main_agent.py` | 任务开始：记忆检索 → `memory_block` 注入 `build_agent_input`；任务结束：`extract_memory_candidates` → `add_candidates` → `monitor.report_memory`；`timer.memory_hits` 记录召回数 |
| `agent/context_manager.py` | `build_agent_input(...)` 的 `memory_block` 分节与 600 token 预算；`ContextBudget(memory_tokens=600)` |
| `api/monitor.py` | `report_memory({"retrieved": [...], "written": [...]})` → `event="memory"` |
| `api/server.py` | `GET /api/memory?limit=20`、`DELETE /api/memory/{id}`；前端记忆面板依赖 |
| `api/static/index.html` | 右侧"🧠 长期记忆"面板：列表（类型/置信/命中）、删除按钮、刷新；接口失败静默隐藏/降级，不弹错 |
| `tests/test_memory.py` | 30 个测试：embedding/相似度、去重合并、检索排序与衰减、类型校验、`clear/delete/stats` 等 |
| `evals/memory_eval.py` | 跨会话端到端评测：会话 A 陈述偏好 → 轮询 `GET /api/memory` 确认写入 → 会话 B 在**新 thread** 提问依赖记忆的问题 → 确定性判分 → `evals/memory_report.md`（5 组用例） |
| `docs/memory-design.md` | 记忆数据模型、检索评分、生命周期与评测细节的完整设计文档 |

---

## §3.11（附）上下文管理一页概述

### 问题

多轮对话 + 工具原始输出（SQL 结果、检索文本）会迅速挤爆上下文窗口，直接带来 token 成本、首字延迟与"中间信息丢失"；同时用户问题里的时间范围/品类/指标等槽位在多轮之间容易漂移。

### 两个候选

| 候选 | 做法 | 问题 |
| --- | --- | --- |
| A. 简单截断 | 保留最近 N 条消息、工具输出截到固定字符 | 可能截掉关键约束（如"只看 2026 上半年"）；无 token 预算概念，长度不可预期 |
| B. 预算驱动 + 分节组装 | token 估算 → 分层裁剪（工具输出/历史/记忆）→ 结构化槽位 + 记忆块 + 用户问题分节拼装 | 需要额外实现与测试，但长度可预期、信息优先级明确 |

### 结论

**选 B 的轻量实现**：`agent/context_manager.py` 提供 token 估算、裁剪、槽位提取与分节组装；deepagents 自带摘要中间件保持默认，`summarize_middleware` 作为可选追加能力。

### 理由与要点

- `estimate_tokens` 用 `字符数 / 1.5` 近似（与评测 `CHARS_PER_TOKEN=1.5` 一致），零依赖、足够做预算判断。
- `extract_slots(query)` 把时间范围/品类/指标/对比等结构化槽位抽出来，多轮/改写问法时槽位稳定，减少"重复啰嗦地复述约束"。
- `build_agent_input` 分节顺序：**槽位 → 记忆 → 用户问题**；`query` 分节永远最后输出且**不裁剪**；先按各分节上限裁剪（记忆 600 / 历史 1200 / 工具输出 1500 / 槽位 200 token），总预算（`total_tokens=6000`，按 `chars = tokens * 1.5`）超限时**先裁记忆、再裁槽位**，最终保底只留用户问题，保证"问题本身永不丢"。
- 工具输出优先用 `clip_tool_output`（默认 1500 tokens，保留头 70% + 尾 30%）——头是结论/表头，尾是汇总行，中间省略。
- `summarize_middleware(model, trigger_tokens=80000, keep_messages=20)` 构造 langchain 版 `SummarizationMiddleware`，作为 deepagents 内置摘要能力的**追加**选项；构造失败返回 `None`，不影响主链路。

### 本项目落点

| 落点 | 说明 |
| --- | --- |
| `agent/context_manager.py` | `estimate_tokens`、`clip_text`、`clip_tool_output`、`trim_history`、`summarize_history`、`StructuredSlots`/`extract_slots`/`render_slots`、`ContextBudget`、`build_agent_input`、`summarize_middleware` |
| `agent/main_agent.py` | `build_agent_input(task_query, memory_block=..., slots=extract_slots(...))` 组装本轮输入 |
| `tests/test_context_manager.py` | token 估算、裁剪优先级、槽位提取、预算保底路径的单测 |
| `docs/context-management.md` | 分层策略、预算表、与 deepagents 中间件的衔接细节 |

---

## 附录 A：术语表

| 术语 | 解释 |
| --- | --- |
| FC（Function Calling） | 模型输出结构化 `tool_calls`、宿主执行工具并回填结果的调用协议 |
| MCP（Model Context Protocol） | 工具 ↔ 宿主的标准化接入协议（跨进程、可发现），与 FC 分层可叠加 |
| GMV | `SUM(total_amount)`，只计 `completed/paid`，不含税口径见 SOP |
| AOV / ASP | 客单价 = GMV / 订单数；件单价 = SUM(price) / SUM(quantity) |
| L1–L4 路由层 | 规则 / n-gram 朴素贝叶斯 / LLM few-shot / 反问澄清 |
| checkpoint | deepagents/LangGraph 的图状态持久化（`AsyncSqliteSaver`，`data/checkpoints.db`） |
| 长期记忆 | 独立存储的跨会话事实/偏好/约束/术语（`data/memory.db`） |
| first_token_ms | 任务开始到第一个流式 token 的毫秒数（P0 性能指标） |

## 附录 B：相关文档与产物

- `docs/requirements-p0-p1.md`：P0/P1 需求→实现→验证映射表（配套本文）。
- `docs/kb/订单分析口径SOP.md`、`docs/kb/orders数据字典.md`：口径与字段的权威来源。
- `docs/memory-design.md`、`docs/context-management.md`：记忆与上下文管理的详细设计。
- `evals/report.md`、`evals/report-round1.md`：真实链路评测结果（当前 / 首轮）。
- `README.md`：项目结构、接口表与启动方式（由 Lead 统一维护）。
