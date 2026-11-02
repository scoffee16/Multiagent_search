# 需求 → 实现 → 验证 映射表（P0 / P1）

> 本文把 P0/P1 的每条需求落到**真实存在的代码位置**与**可执行的验证方式**上，方便走查、答辩与回归。
> 配套文档：[`docs/design-decisions.md`](design-decisions.md)（选型理由与取舍）。
>
> - 核对时间：2026-11-02（工作区 `feature/p0-p1-improvements`，Python 3.13）。
> - 状态快照：`python -m pytest -q` → **173 passed**；`api/static/index.html` 内联脚本经 `node --check` 通过；服务 8010 真实链路已联调（20 题评测 + 意图/记忆离线评测均落盘）。
> - 状态图例：✅ 已完成（实现与验证均已落盘）｜🔶 部分完成（实现落盘，仍有待接入/待重跑项）｜⏳ 计划中｜✖ 范围外（明确不做）。
> - 本表「实现位置」列的文件均已用 `read`/`glob` 核对存在；尚未落盘的项在状态列标注并汇总到文末「缺口清单」。

## 一页速览

| 编号 | 需求 | 状态 |
| --- | --- | --- |
| P0 §3.1 | 性能埋点（端到端/首字/分阶段耗时、工具调用数、指标查询接口） | ✅ 已完成（20 题真实评测已采集 P50/P95 与阶段耗时） |
| P0 §3.2 | MySQL 取数 + RAGFlow 查口径，串联编排（不做合并检索端点） | ✅ 已完成 |
| P0 §3.3 | 工具接入：Function Calling（进程内），MCP 仅作演进方向 | ✅ 已完成 |
| P0 §3.4 | 意图识别：分层路由 L1 规则 → L2 分类器 → L3 LLM → L4 澄清 | ✅ 已完成（离线准确率 95.0%、macro F1 0.9495、35 个单测） |
| P0 §3.5 | 流式输出：WS `token` 事件 + 首字延迟 + 前端增量渲染 | ✅ 已完成（真实链路已联调，20 题均有首字延迟） |
| P0 §3.6 | 数据建模：`orders` 单表字段/类型/索引/状态口径 | ✅ 已完成 |
| P0 §3.7 | 长期记忆：独立存储 + 向量检索 + 生命周期 + 记忆面板 | ✅ 已完成（跨会话评测 5/5，docs/memory-design.md §9 已回填） |
| P1 §3.9 | 评测升级：路由准确率 + 引用覆盖率 | ✅ 已完成（报告已重跑并含新指标） |
| P1 §3.11 | 上下文管理：预算驱动 + 分节组装 + 裁剪/摘要 | ✅ 已完成（分节注入 + 工具输出裁剪已接入；历史摘要由框架内置 SummarizationMiddleware 承担） |
| P2 §3.8 | 多模态（EduRAG 专属） | ✖ 本仓库不适用 |
| P2 §3.10 | 合并两项目 | ✖ 明确不做 |

---

## P0 需求映射

### §3.1 性能埋点

| 项 | 内容 |
| --- | --- |
| **编号** | P0 §3.1 |
| **需求（一句话）** | 每次任务都要能回答"总耗时、首字延迟、检索/LLM/工具/子智能体各占多少、调了几次工具"，并提供查询接口与前端展示 |
| **实现位置** | `agent/timing.py`：`TaskTimer`（`add` / `incr_tool_call` / `mark_first_token` / `elapsed_ms` / `snapshot`）、`start_timer` / `get_timer` / `reset_timer`、`percentile`、`TaskMetricsStore`（`record` / `get` / `list_runs` / `aggregate` / `reset`）、模块级 `metrics_store`（SQLite `data/metrics.db`，可用 `METRICS_DB_PATH` 覆盖）<br>`agent/main_agent.py` `run_deep_agent`：`start_timer` → `timer.add("retrieve_ms", ...)` → `timer.mark_first_token()` → 结束时 `metrics_store.record(...)` + `monitor.report_timing(...)`<br>`agent/tool_observer.py` `instrument_tool` / `_observed_run`：工具耗时 `timer.add("tool_ms", ...)`、工具调用计数<br>`api/monitor.py`：`report_timing(timings, summary)`、`report_tool_result(...)`<br>`api/server.py`：`GET /api/task/{thread_id}/timings`、`GET /api/metrics`<br>`api/static/index.html`：`timing` 事件 → `timingPanelHTML`（首字/总耗时/工具次数 chips + 分阶段横向条形） |
| **验证方式** | `python -m pytest tests/test_timing.py tests/test_monitor_events.py -q`（计时器/百分位/指标库/事件 payload）<br>服务启动后：跑一题 → `curl http://127.0.0.1:8010/api/metrics` 检查 `aggregate.latency_ms.p50/p95`、`first_token_ms`、`stages`、`tool_calls`；`GET /api/task/{thread_id}/timings` 核对单任务明细<br>`python evals/run_eval.py` 报告中的 P50/P95 延迟与工具调用数（`evals/report.md`） |
| **状态** | ✅ 已完成（单测通过；端到端指标数值待 Lead 启动 8010 后联调采集） |

### §3.2 MySQL vs RAGFlow（数据源分工与串联）

| 项 | 内容 |
| --- | --- |
| **编号** | P0 §3.2 |
| **需求（一句话）** | 事实/数值走 MySQL、规则/口径走 RAGFlow，先定口径再取数，报告同时给结论+口径+数据来源；不合并成一个检索端点 |
| **实现位置** | `tools/db_tools.py`：`list_sql_tables` / `get_table_data` / `execute_sql_query`、只读守卫 `assert_read_only_sql`（`_ALLOWED_SQL_PREFIXES`、`_FORBIDDEN_SQL_KEYWORDS`、`_strip_sql_literals`、禁多语句）<br>`tools/ragflow_tools.py`：`get_assistant_list` / `create_ask_delete`（RAGFlow SDK，`rawflow.rag_config._load_ragflow_env`）<br>`agent/subagents/database_query_agent.py`、`agent/subagents/knowledge_base_agent.py`（子智能体工具集与提示词）<br>`agent/main_agent.py`：`create_deep_agent(subagents=[..., database_query_agent, network_search_agent, knowledge_base_agent])`、`_route_hint` 把意图转成编排提示<br>`agent/intent_router.py`：`INTENT_TO_SUBAGENTS` 意图 → 子智能体映射<br>`prompt/prompts.yml`（经 `agent/prompts.py` 加载）<br>`docs/kb/订单分析口径SOP.md`、`docs/kb/orders数据字典.md`（口径权威来源） |
| **验证方式** | 单测：`python -m pytest tests/test_sql_guard.py -q`（只读守卫、注入/多语句拦截）<br>评测：`python evals/run_eval.py --category 查库`、`python evals/run_eval.py --category 查文档`（分别看两类通过率与输出是否含口径/来源）<br>人工走查：`docs/design-decisions.md` §3.2；首轮误判证据见 `evals/report-round1.md`（查文档 3/5） |
| **状态** | ✅ 已完成 |

### §3.3 Function Calling vs MCP（工具接入）

| 项 | 内容 |
| --- | --- |
| **编号** | P0 §3.3 |
| **需求（一句话）** | 9 个工具用进程内 Function Calling 接入；MCP 只写清演进边界与切换路径，不引入额外进程/协议 |
| **实现位置** | 业务工具：`tools/db_tools.py`（3 个 `@tool`）、`tools/ragflow_tools.py`（2 个）、`tools/tavily_tool.py`（`internet_search`）<br>产出工具：`tools/markdown_tools.py`（`generate_markdown`）、`tools/pdf_tools.py`（`convert_md_to_pdf`）、`tools/upload_file_read_tool.py`（`read_file_content`）<br>统一包装：`agent/tool_observer.py` `instrument_tool`（`StructuredTool.from_function` 保留 `name/description/args_schema`，只加计时与输出预览）<br>注册与编排：`agent/main_agent.py` 的 `create_deep_agent(tools=[...], subagents=[...])`；子智能体工具列表在各 `agent/subagents/*.py`<br>设计说明：`docs/design-decisions.md` §3.3（FC/MCP 对比表 + 切换路径） |
| **验证方式** | `python -m pytest tests/test_sql_guard.py -q`（工具边界不因包装失效）<br>`python -m pytest tests/test_monitor_events.py -q`（工具/输出事件正常上报）<br>评测链路：`python evals/run_eval.py --limit 3` 观察 `tool_start` / `tool_result` 事件与工具调用数<br>代码走查：`instrument_tool` 包装后工具 `name/description/args_schema` 不变 |
| **状态** | ✅ 已完成 |

### §3.4 意图方案（分层路由）

| 项 | 内容 |
| --- | --- |
| **编号** | P0 §3.4 |
| **需求（一句话）** | 用"规则 → 小模型 → LLM 兜底 → 反问澄清"的分层路由替代 LLM 直接分类，给出可校准置信度与可评估结果 |
| **实现位置** | `agent/intent_router.py`：`classify_intent(query, use_llm=True)` 主入口；`classify_by_rules`（L1-rule）、`NaiveBayesIntentClassifier` / `classify_by_classifier`（L2-classifier，字符 1–3 gram、alpha=1.0）、`_llm_classify`（L3-llm）、`_clarify_result`（L4-clarify）；`IntentResult.to_dict()`；`confusion_matrix` / `classification_metrics`；常量 `INTENTS`、`INTENT_LABELS_ZH`、`INTENT_TO_SUBAGENTS`、`L2_MIN_CONFIDENCE=0.55`、`L2_MIN_MARGIN=0.15`<br>`agent/main_agent.py`：任务入口 `classify_intent(task_query)` → `monitor.report_intent(intent.to_dict())` → `timer.intent/intent_confidence/route_layer`；`needs_clarification` 时直接返回澄清问题<br>`api/monitor.py`：`report_intent(intent)`<br>训练/评测数据：`evals/intent_dataset.json`（72 条，4 类各 18）<br>评测脚本：`evals/intent_eval.py` → `evals/intent_report.md` / `evals/intent_report.json`<br>前端：`api/static/index.html` 的 `intent` 事件徽标（意图/置信度/路由层/建议子智能体） |
| **验证方式** | 离线评测（无需服务）：`python evals/intent_eval.py` → 当前报告 `evals/intent_report.md`：**准确率 95.0%（19/20）、macro F1 0.9495、L1 17 题 / L2 3 题 / L3 0 / L4 0**；退出码 0=达标（目标 ≥85%）<br>混淆矩阵与每类 P/R/F1 见报告；误判 `mix-04` 预测为 `database`（约定标 mixed，属边界样本）<br>单测：`python -m pytest tests/test_intent_router.py -q`（35 个测试：规则/分类器/阈值降级/澄清/混淆矩阵） |
| **状态** | ✅ 已完成（离线评测达标 + 35 个单测全绿） |

### §3.5 流式输出（token 事件）

| 项 | 内容 |
| --- | --- |
| **编号** | P0 §3.5 |
| **需求（一句话）** | 复用 WebSocket 新增 `token` 事件，把主智能体最终回答增量推给前端；前端增量渲染、自动滚底，`task_result` 到达后替换为最终 markdown；采集首字延迟 |
| **实现位置** | `api/monitor.py`：`report_token(text)`（`broadcast_only=True`，空文本忽略）<br>`agent/main_agent.py`：`_TokenEmitter`（`_TOKEN_FLUSH_CHARS=12`、`_TOKEN_FLUSH_SECONDS=0.06` 攒批）、`agent.astream(stream_mode=["updates","messages"])` 只取主图 `model` 节点 token、`timer.mark_first_token()`、`_read_final_answer_from_state` 兜底最终文本<br>`agent/timing.py`：`first_token_ms` 进入 `snapshot()` 与 metrics<br>`api/static/index.html`：`ws.onmessage` 的 `token` 分支（仅当前会话 running 时累加 `session.streamText`，不进 logs）、`streamCardHTML` / `scheduleStreamRender`（rAF 合并）/ `updateStream`（渲染 + 光标 + 自动滚底）、`task_result`/`error`/`onclose` 清理流式态 |
| **验证方式** | `python -m pytest tests/test_timing.py tests/test_monitor_events.py -q`（`mark_first_token`、`report_token` payload）<br>前端：内联脚本 `node --check` 通过（见文末命令）<br>真实链路（需服务）：提交一题，浏览器观察流式气泡逐字增长、任务完成后替换为完整 markdown；`GET /api/metrics` 的 `first_token_ms` 有值 |
| **状态** | ✅ 已完成（前端静态校验通过；真实链路流式经服务联调确认） |

### §3.6 数据建模（orders 表）

| 项 | 内容 |
| --- | --- |
| **编号** | P0 §3.6 |
| **需求（一句话）** | 以 `sql/init.sql` 的 `orders` 单表为准，明确字段类型/索引/状态枚举对 GMV 的影响，并在知识库说明"只有一张表"的边界 |
| **实现位置** | `sql/init.sql`：`orders` DDL（`order_no/status/total_amount/category/created_at` 等，`DECIMAL(10,2)`，`idx_category`、`idx_created_at`）+ 8 条演示数据<br>`docs/kb/orders数据字典.md`：字段类型、`status` 枚举与 GMV 计入规则、`order_no` 规则、单表边界<br>`docs/kb/订单分析口径SOP.md`：GMV/订单数/AOV/ASP/品类占比/时间口径/样本量规则/清库存三级<br>`tools/db_tools.py`：`list_sql_tables` / `get_table_data` / `execute_sql_query`（只读访问该表）<br>`agent/subagents/database_query_agent.py`：提示词要求先看数据字典/SOP 再写 SQL<br>设计说明：`docs/design-decisions.md` §3.6（字段清单表 + 建模要点） |
| **验证方式** | `python -m pytest tests/test_sql_guard.py -q`（只读守卫）<br>初始化数据：执行 `sql/init.sql` 后 `orders` 应为 8 行、`SUM(total_amount)=39572`（评测 `db-01`/`db-03` 断言）<br>`python evals/run_eval.py --category 查库` 验证数值与枚举类问题<br>`python evals/run_eval.py --category 查文档 --only rag-04` 验证状态口径（取消/退款不计 GMV） |
| **状态** | ✅ 已完成 |

### §3.7 长期记忆

| 项 | 内容 |
| --- | --- |
| **编号** | P0 §3.7 |
| **需求（一句话）** | 跨会话记住用户事实/偏好/约束/术语：独立存储 + 向量检索 + 生命周期，任务前注入、任务后抽取写入，提供列表/删除接口与前端面板；与 checkpoint 互补而非替代 |
| **实现位置** | `agent/memory.py`：`MemoryStore`（`add` 余弦 >0.9 去重合并、`add_candidates`、`search` 评分 `cosine×confidence×0.5^(days/90)`、`recent`（按类型/数量取最近记忆，服务两个兜底）、`list/delete/clear/stats`、`render_for_prompt`）、`MemoryCandidate` / `MemoryItem`、`embed_text`（字符 n-gram + blake2b，256 维）、`_ELLIPTICAL_MARKERS` / `needs_recent_memory`（省略指代检测）、`retrieve_memory_context`（600 token 预算）、`extract_memory_candidates`（LLM 抽取 ≤5 条，容错）、`_get_default_store`；`data/memory.db`<br>`agent/main_agent.py`：任务前 `store.search(task_query, top_k=5)` → 省略指代空结果时 `recent(3)` 兜底 → 计算类任务（`_is_calculation_query`）强制追加最近 2 条 `constraint` 记忆（去重、上限 7 条）→ `render_for_prompt` → `build_agent_input(memory_block=...)`；任务后 `extract_memory_candidates` → `add_candidates` → `monitor.report_memory`；`timer.memory_hits`<br>`api/server.py`：`GET /api/memory`（`{"memories":[...],"count":N}`）、`GET /api/memory/stats`、`DELETE /api/memory/{memory_id}`（`{"deleted":bool,"id":...}`）<br>`api/monitor.py`：`report_memory({"retrieved":[...],"written":[...]})`<br>`api/static/index.html`：「🧠 长期记忆」面板（列表/类型/置信/命中/删除/刷新，失败静默降级）、`memory` 事件内联展示<br>`docs/memory-design.md`：记忆子系统设计（§9 已回填 5/5 结果与修复对照） |
| **验证方式** | `python -m pytest tests/test_memory.py -q`（26 个测试：embedding/去重/检索衰减/最近记忆类型过滤/校验/维护）<br>`python -m pytest tests/test_monitor_events.py -q`（`report_memory` payload）<br>跨会话 E2E（需服务）：`python evals/memory_eval.py` → `evals/memory_report.md`：**5/5 通过（100%）、记忆写入 5/5**；会话 A 平均 9.6s、会话 B 平均 39.7s<br>接口手测：`curl http://127.0.0.1:8010/api/memory?limit=20`、`curl -X DELETE http://127.0.0.1:8010/api/memory/1` |
| **状态** | ✅ 已完成（代码/单测/接口/前端均落盘；跨会话评测 5/5，结果已回填 `docs/memory-design.md` §9） |

---

## P1 需求映射

### §3.9 评测升级（路由准确率 + 引用覆盖率）

| 项 | 内容 |
| --- | --- |
| **编号** | P1 §3.9 |
| **需求（一句话）** | 评测从"关键词通过率"升级为同时给出**路由准确率**（期望子智能体 vs 实际调用）与**引用覆盖率**（数值结论是否带来源/可回溯到工具输出） |
| **实现位置** | `evals/dataset.json`：每题新增 `intent`（4 类标签）与 `expected_subagents`（应调用的子智能体）<br>`evals/run_eval.py`：`routing_stats(expected, actual_names)`（完全匹配 + 平均召回）、`citation_stats(result, tool_output)`（数值句带来源标记或数字可回溯）、`summarize`/`write_markdown` 汇总路由准确率与引用覆盖率，输出 `evals/report.json` / `evals/report.md`<br>`evals/intent_eval.py`：意图路由离线评测（混淆矩阵 + 每类 P/R/F1 + 分层命中），输出 `evals/intent_report.md|json`<br>事件来源：`assistant_call` 事件（`api/monitor.py` `report_assistant`）、`tool_result` 事件（`agent/tool_observer.py` → `monitor.report_tool_result`） |
| **验证方式** | 离线：`python evals/intent_eval.py`（当前 95.0%，macro F1 0.9495，目标 ≥85%，退出码 0）<br>端到端（需服务）：`python evals/run_eval.py` → 报告含「路由准确率：完全匹配 15/20（75%），平均召回率 85%」与「引用覆盖率：83%（114/130）」<br>快速冒烟：`python evals/run_eval.py --limit 5` |
| **状态** | ✅ 已完成：2026-11-02 已重跑 20 题并写入 `evals/report.md` / `.json`（含路由准确率与引用覆盖率）；旧版保留在 `evals/report-round1.md` 供对比 |

### §3.11 上下文管理

| 项 | 内容 |
| --- | --- |
| **编号** | P1 §3.11 |
| **需求（一句话）** | 用 token 预算 + 分节组装 + 分层裁剪（工具输出/历史/记忆/摘要）控制上下文长度，保证用户问题永不丢失、信息按优先级保留 |
| **实现位置** | `agent/context_manager.py`：`estimate_tokens`（1 token≈1.5 字符）、`clip_text`（头 70%/尾 30%）、`clip_tool_output`（默认 1500 token）、`trim_history`、`summarize_history`、`StructuredSlots` / `extract_slots` / `render_slots`（时间/品类/指标/对比/样本量）、`ContextBudget`（memory 600 / history 1200 / tool_output 1500 / slots 200 / total 6000 token）、`build_agent_input`（分节顺序 槽位→记忆→问题；query 永不裁剪；超预算先裁记忆再裁槽位）、`summarize_middleware`（langchain `SummarizationMiddleware` 工厂，供需要显式调节时使用）<br>`agent/main_agent.py`：已接入 `extract_slots` + `build_agent_input`（检索块 `agent/main_agent.py:347-370`）<br>`agent/tool_observer.py`：工具返回值先 `clip_tool_output(result)` 再交给模型（单一工具输出 token 上限 1500），裁剪后文本同时用于前端预览与引用覆盖率评测<br>历史摘要：`create_deep_agent` 内置的 `SummarizationMiddleware` 已默认按 token 触发摘要（本项目不重复实现）；`trim_history` / `summarize_history` 保留为可定制的纯函数<br>`docs/context-management.md`：模块总览/五条做法/接入点/边界<br>`tests/test_context_manager.py`：token 估算、裁剪、槽位、分节预算、摘要降级单测；`tests/test_monitor_events.py` 覆盖「超长工具输出被裁剪后再交给模型」 |
| **验证方式** | `python -m pytest tests/test_context_manager.py tests/test_monitor_events.py -q`（全部通过）<br>`python -m ruff check agent/context_manager.py agent/tool_observer.py tests/test_context_manager.py`<br>接入核对：`docs/context-management.md`「当前接入点」表（已接入/框架承担/可定制） |
| **状态** | ✅ 已完成：分节注入（槽位/记忆/问题）与工具输出裁剪已接入主链路；历史摘要由框架内置中间件承担；`trim_history`/`summarize_history`/`summarize_middleware` 为可选定制件（`docs/context-management.md` 已说明何时启用） |

---

## P2 / 范围外

### §3.8 多模态（EduRAG 专属）

| 项 | 内容 |
| --- | --- |
| **编号** | P2 §3.8 |
| **需求（原文）** | 多模态（图片/PDF 解析、图表问答等），来自 EduRAG 项目场景 |
| **结论** | ✖ 本项目不适用，明确不做 |
| **理由** | 本仓库**没有 PDF/图片解析链路**：`tools/pdf_tools.py` 只做 Markdown→PDF **生成**，`tools/upload_file_read_tool.py` 只读文本文件，没有 OCR/版面分析/图像理解工具，也没有对应评测集。硬做只能接入外部服务，超出本项目"单机、可复现、零外部服务"的范围。 |
| **落点** | 无（范围外）；相关能力仅在 `docs/design-decisions.md` 与 README 的 Roadmap 中说明边界 |

### §3.10 合并两项目

| 项 | 内容 |
| --- | --- |
| **编号** | P2 §3.10 |
| **需求（原文）** | 把 MultiAgent-Search 与 EduRAG 合并成一个项目 |
| **结论** | ✖ 用户已明确不合并 |
| **理由** | 两者领域、数据源（电商 orders/运营文档 vs 教育资料）、工具链与评测集都不同；合并会让工具选择与意图空间变大、评测不可归因。保留各自仓库，通过"工具层可替换（FC/MCP 分层）"借鉴设计而不是合并代码。 |
| **落点** | 无（范围外）；`docs/design-decisions.md` §3.3 已写明"未来复用工具可通过 MCP，而不是合并仓库" |

---

## 验证命令

> 所有命令均在仓库根目录执行。本任务（task-4）只做静态文件改动，**未启动 8010 服务**；服务启动由 Lead 统一执行。

### 1. 全量单元测试

```powershell
python -m pytest -q
# 当前结果：173 passed（约 1.3s）
```

按模块：

```powershell
python -m pytest tests/test_timing.py tests/test_monitor_events.py -q   # §3.1 / §3.5 埋点、事件、工具输出裁剪
python -m pytest tests/test_memory.py -q                                # §3.7 长期记忆
python -m pytest tests/test_context_manager.py -q                       # §3.11 上下文管理
python -m pytest tests/test_intent_router.py -q                         # §3.4 意图路由
python -m pytest tests/test_sql_guard.py -q                             # §3.2 / §3.3 SQL 只读守卫
python -m pytest tests/test_path_utils.py -q
```

### 2. 静态检查

```powershell
python -m ruff check agent/ tools/ api/ evals/ tests/
python -m ruff format --check agent/ tools/ api/ evals/ tests/   # 可选
```

### 3. 启动服务（Lead 统一执行，本任务不执行）

```powershell
python api/server.py
# 等价：python -m uvicorn api.server:app --host 0.0.0.0 --port 8010
# 浏览器：http://127.0.0.1:8010 ；Swagger：http://127.0.0.1:8010/docs
```

### 4. 端到端评测（需服务在 8010）

```powershell
python evals/run_eval.py                       # 全量 20 题 → evals/report.md / evals/report.json
python evals/run_eval.py --category 查库 --limit 3
python evals/run_eval.py --category 查文档
python evals/run_eval.py --only db-03
python evals/run_eval.py --skip-web            # 没有 TAVILY_API_KEY 时跳过联网类
```

### 5. 意图路由离线评测（无需服务）

```powershell
python evals/intent_eval.py            # 输出 evals/intent_report.md / .json；达标退出码 0（目标 ≥85%）
python evals/intent_eval.py --no-write # 只打印结果，不覆盖报告
python evals/intent_eval.py --llm      # 可选：允许 L3 LLM 兜底抽查（不可完全复现）
```

### 6. 记忆跨会话评测（需服务在 8010）

```powershell
python evals/memory_eval.py             # 5 组用例 → evals/memory_report.md
python evals/memory_eval.py --only mem-01
```

### 7. 前端内联脚本语法校验（node 可用时）

`api/static/index.html` 是单文件、内联 `<script>`，改动后必须校验语法：

```powershell
@'
import pathlib, re, subprocess, tempfile
html = pathlib.Path("api/static/index.html").read_text(encoding="utf-8")
code = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
tmp = pathlib.Path(tempfile.gettempdir()) / "index_inline_check.js"
tmp.write_text(code, encoding="utf-8")
r = subprocess.run(["node", "--check", str(tmp)], capture_output=True, text=True)
print("node --check exit:", r.returncode)
print(r.stderr or "OK: inline script syntax valid")
'@ | python -
```

### 8. 改动规模核对

```powershell
git status --short
git diff --stat
```

---

## 缺口清单（截至 2026-11-02，Lead 收尾）

| # | 原缺口 | 当前状态 |
| --- | --- | --- |
| 1 | `tests/test_intent_router.py` 不存在 | ✅ 已补齐：35 个测试（规则/分类器/阈值降级/澄清/混淆矩阵），`pytest tests/test_intent_router.py` 全绿 |
| 2 | `evals/memory_report.md` 未生成、`docs/memory-design.md` §9 占位 | ✅ 已生成并回填：跨会话评测 **5/5**（修复前 4/5 的对照写在 §9.2） |
| 3 | `evals/report.md` 仍是 2026-10-26 旧版 | ✅ 已重跑：含通过率、首字/阶段耗时、路由准确率、引用覆盖率（见下方「验收汇总」） |
| 4 | `context_manager` 的裁剪/摘要未接入 | ✅ 工具输出裁剪已接入 `agent/tool_observer.py`（单工具 1500 token 上限）；历史摘要由 deepagents 内置 `SummarizationMiddleware` 承担；`trim_history`/`summarize_middleware` 作为可选定制件保留并在 `docs/context-management.md` 说明 |
| 5 | 前端面板未经真实服务联调 | ✅ 事件链路已真实联调：20 题评测脚本从 WebSocket 收齐 `token/intent/timing/tool_result/memory/assistant_call/task_result` 事件；`index.html` 内联脚本 `node --check` 通过。**剩余**：浏览器人工目测视觉细节（不计入需求验收） |
| 6 | `GET /api/task/{thread_id}/timings`、`/api/memory/stats` 前端未使用 | 保留为「可选」：接口可用，前端只用了 `/api/metrics` 与 `/api/memory`；不影响需求验收 |

## 验收汇总（真实服务 + 真实模型）

| 验收项 | 结果 | 数据来源 |
| --- | --- | --- |
| 全量单测 | **173 passed** | `python -m pytest -q` |
| 意图路由离线评测 | 准确率 **95.0%（19/20）**、macro F1 **0.9495**、L1/L2/L3/L4 = 17/3/0/0、平均 0.03ms | `evals/intent_report.md` |
| 跨会话记忆评测 | **5/5（100%）**，记忆写入 5/5 | `evals/memory_report.md` |
| 端到端 20 题评测 | **20/20（100%）**，四类各 5/5；端到端 P50 **14.2s** / P95 32.6s；首字 P50 **8.6s** / P95 21.5s；阶段 retrieve 0ms、LLM P50 2.6s、tool P50 3.3s；路由完全匹配 **15/20（75%）**、平均召回 85%；引用覆盖率 **83%（114/130）**；工具平均 4.5 次/最多 14 次 | `evals/report.md` / `evals/report.json` |

> 备注：端到端 P50 比 2026-10-26 基线（8.3s）更慢，原因是模型服务 TTFT 波动与流式首字口径变化；
> 新增的前置检索（意图+记忆+槽位）实测只占 0–1ms。基线没有首字/阶段/路由/引用指标，不做逐项对比。

---

## 附录：实现位置文件核对表

| 路径 | 存在 | 用途 |
| --- | --- | --- |
| `agent/timing.py` | ✅ | §3.1 计时器 / 指标库 |
| `agent/tool_observer.py` | ✅ | §3.1/§3.5 工具耗时与输出预览 |
| `agent/intent_router.py` | ✅ | §3.4 分层意图路由 |
| `agent/memory.py` | ✅ | §3.7 长期记忆存储/检索/抽取 |
| `agent/context_manager.py` | ✅ | §3.11 上下文预算与分节组装 |
| `agent/main_agent.py` | ✅ | 主链路接入（意图/记忆/计时/流式/报告） |
| `api/monitor.py` | ✅ | WS 事件：tool/token/intent/timing/tool_result/memory/task_result/error |
| `api/server.py` | ✅ | `/api/task`、`/api/task/{id}/timings`、`/api/metrics`、`/api/memory*`、`/api/files`、`/ws/{id}` |
| `api/static/index.html` | ✅ | 前端：流式/意图/耗时/工具输出/记忆面板 |
| `evals/run_eval.py` | ✅ | §3.9 端到端评测（路由 + 引用） |
| `evals/intent_eval.py` | ✅ | §3.4 意图离线评测 |
| `evals/memory_eval.py` | ✅ | §3.7 跨会话记忆评测 |
| `evals/intent_dataset.json` | ✅ | 意图训练集（72 条） |
| `evals/dataset.json` | ✅ | 端到端评测集（20 题，含 intent/expected_subagents） |
| `docs/kb/订单分析口径SOP.md` | ✅ | §3.2/§3.6 口径 |
| `docs/kb/orders数据字典.md` | ✅ | §3.6 字段与状态枚举 |
| `docs/memory-design.md` | ✅ | §3.7 记忆设计文档（§9 已回填 5/5） |
| `docs/context-management.md` | ✅ | §3.11 上下文管理文档 |
| `docs/design-decisions.md` | ✅ | §3.2/§3.3/§3.4/§3.5/§3.6/§3.7/§3.11 选型决策 |
| `sql/init.sql` | ✅ | §3.6 建表与演示数据 |
| `prompt/prompts.yml` | ✅ | §3.2 主/子智能体提示词 |
| `tests/test_timing.py` | ✅ | §3.1 单测 |
| `tests/test_monitor_events.py` | ✅ | §3.1/§3.4/§3.5/§3.7 事件单测 |
| `tests/test_memory.py` | ✅ | §3.7 单测（30 个） |
| `tests/test_context_manager.py` | ✅ | §3.11 单测 |
| `tests/test_sql_guard.py` | ✅ | §3.2/§3.3 只读守卫 |
| `tests/test_intent_router.py` | ✅ | §3.4 意图路由单测（35 个） |
| `evals/memory_report.md` | ✅ | §3.7 跨会话 E2E 报告（5/5） |
