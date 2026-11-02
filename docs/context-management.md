# 上下文管理（P1 §3.11）

主智能体在长任务里要同时面对：不断增长的对话历史、动辄上千字符的工具输出（SQL 结果、网页正文）、
跨会话检索到的长期记忆，以及用户问题里隐含的时间/品类/指标口径。上下文越长，**成本越高、延迟越大，
且模型越容易丢掉重点**。本模块在 deepagents 内置摘要机制之上，补齐「显式预算 + 裁剪 + 槽位 + 注入」
这层可控策略，所有函数都可以离线单测。

实现文件：[agent/context_manager.py](../agent/context_manager.py)；接入文件：[agent/main_agent.py](../agent/main_agent.py)、
[agent/memory.py](../agent/memory.py)。

## 模块总览

`agent/context_manager.py` 是纯函数 + dataclass 模块，不联网、不碰全局状态，公开 API 如下：

| API | 作用 |
| --- | --- |
| `estimate_tokens(text)` | 按 `len / 1.5` 估算 token 数（中文约 1.5 字符 1 token），向上取整保守高估 |
| `clip_text(text, max_tokens, *, head_ratio=0.7, marker=None)` | 头尾保留式裁剪，中间插入裁剪标记 |
| `clip_tool_output(text, max_tokens=1500)` | 工具输出专用裁剪，标记形如 `...[已裁剪 N 字符]...` |
| `trim_history(messages, *, max_turns=6, query="", max_tokens=1200)` | 保留 system + 最近若干轮，并按 query 相关性回填旧消息 |
| `summarize_history(llm, messages, *, max_tokens=300)` | 异步历史摘要；LLM 不可用时走确定性兜底 |
| `StructuredSlots` / `extract_slots(query)` / `render_slots(slots)` | 结构化任务槽位抽取与渲染 |
| `ContextBudget` | 记忆/历史/工具输出/槽位/总预算的上限配置 |
| `build_agent_input(query, *, memory_block="", slots=None, budget=None)` | 按「槽位 → 记忆 → 用户问题」分节组装模型输入并控总预算 |
| `summarize_middleware(model, *, trigger_tokens=80000, keep_messages=20)` | 生成 langchain `SummarizationMiddleware` 配置实例，导入/构造失败返回 `None` |

## 五条做法

### 1. 裁剪：历史只留最近的，旧的按相关性补回

- 历史裁剪由 `trim_history`（[agent/context_manager.py:201](../agent/context_manager.py#L201)）负责：
  - `system` 消息无条件全部保留；
  - 以 user 消息为轮起点，最近 `max_turns` 轮整块保留；
  - 更早的消息按与 `query` 的关键词重叠打分（中文相邻二字组 + 英文/数字词），在剩余 `max_tokens`
    预算内按分数从高到低挑选；
  - 返回**新列表**且保持原始时间顺序，不修改调用方数据。
- 工具输出裁剪由 `clip_tool_output`（[agent/context_manager.py:108](../agent/context_manager.py#L108)）负责，
  它复用 `clip_text`（[:61](../agent/context_manager.py#L61)）：保留头部约 70%、尾部约 30%，
  中间插 `...[已裁剪 N 字符]...`，保证裁剪后估算 token 不超上限。通用场景可用
  `clip_text(text, max_tokens, head_ratio=..., marker=...)` 自定义比例和标记。

### 2. 摘要：显式兜底 + 复用框架内置摘要

- 项目内的显式摘要入口是 `summarize_history`（[agent/context_manager.py:304](../agent/context_manager.py#L304)）：
  - 传了 `llm`：把历史压缩成不超过 5 条要点（用户目标 / 约束 / 结论 / 待办）；
  - `llm=None`、`ainvoke` 抛异常或返回空内容：走确定性兜底——每条消息取 `role: 内容前 80 字符` 拼接，
    再按 `max_tokens` 裁剪。因此该函数在离线单测、LLM 超时场景下都不会失败。
- deepagents 0.4.3 的 `create_deep_agent` 标准中间件栈里**本来就包含**一个 deepagents 版
  `SummarizationMiddleware`（`deepagents/middleware/summarization.py`），默认阈值由
  `_compute_summarization_defaults(model)` 按模型 profile 计算：
  - 模型 profile 带 `max_input_tokens` 时：`trigger=("fraction", 0.85)`、`keep=("fraction", 0.10)`；
  - 否则：`trigger=("tokens", 170000)`、`keep=("messages", 6)`。
  它在 `wrap_model_call` 阶段生效，摘要前会把旧消息落盘到 backend 的
  `/conversation_history/{thread_id}.md`，并把摘要事件写入图状态 `_summarization_event`。
  这层机制由框架自动触发，**本模块不替换它**。
- 需要自定义触发阈值时用 `summarize_middleware`（[:615](../agent/context_manager.py#L615)）构造
  langchain 版 `SummarizationMiddleware`，在 `create_deep_agent(middleware=[...])` 处传入。
  注意 `middleware=[...]` 是**追加**到 deepagents 标准栈之后（见 `deepagents/graph.py` 里
  `deepagent_middleware.extend(middleware)`），并非替换内置摘要；若两者同时启用，需观察触发顺序与
  实际预算是否符合预期。导入失败或模型对象无效时该函数返回 `None`，调用方需判空。

### 3. 检索注入：长期记忆只注入 top-k，并单独成节

- 记忆检索在 [agent/memory.py:337](../agent/memory.py#L337) `MemoryStore.search`，格式化在
  [agent/memory.py:475](../agent/memory.py#L475) `MemoryStore.render_for_prompt`；
  [agent/main_agent.py:347-360](../agent/main_agent.py#L347-L360) 默认取 `top_k=5` 并生成 `memory_block`。
- 两个「向量检索不可靠」的兜底（都在 `run_deep_agent` 内、渲染前）：
  - **省略指代**（"那段时间/上次/之前提到的"）：检索为空时用 `MemoryStore.recent(limit=3)` 注入最近记忆；
  - **计算类任务**：命中计算标记词（算/统计/GMV/销售额/占比…）时用
    `MemoryStore.recent(limit=2, types=("constraint",))` 强制注入最近约束（口径/禁止项），
    避免「统计时不要算退款订单」这类记忆因为与问题字面不重合而被漏掉。最终条数上限 7。
- `build_agent_input` 把记忆块放进独立的 `### 已知背景（长期记忆）` 节，位于任务槽位之后、用户问题之前，
  与用户指令明确分区，避免模型把历史记忆当成当前任务要求。
- 接入状态：已接入（[agent/main_agent.py:360-370](../agent/main_agent.py#L360-L370)）。

### 4. 结构化槽位：把时间/品类/指标/对比显式化

- `StructuredSlots`（[agent/context_manager.py:336](../agent/context_manager.py#L336)）承载
  `time_range`、`category`、`metric`、`comparison`、`sample_warning` 五个字段；`is_empty()` 判断是否
  有有效信号（`comparison="无"` 不算）。
- `extract_slots`（[:474](../agent/context_manager.py#L474)）用纯规则抽取：时间表达式
  （`2026年Q1`、`2026-01`、`本月/上月`、`近30天`、`第一季度` 等）、品类词（数码电子/生活家电/
  美妆护肤/运动鞋服）、指标词（GMV/客单价/订单数/退款率/品类占比等）、对比词（同比/环比/趋势）、
  样本量风险提示（"8条/样本/够不够"等）。
- `render_slots`（[:509](../agent/context_manager.py#L509)）渲染成 `### 任务槽位` 分节，空槽不输出。
- 接入状态：已接入（[agent/main_agent.py:364](../agent/main_agent.py#L364)、
  [:368](../agent/main_agent.py#L368)）。

### 5. 预算控制：所有上限集中在一个 dataclass

- `ContextBudget`（[agent/context_manager.py:529](../agent/context_manager.py#L529)）默认：
  `memory_tokens=600`、`history_tokens=1200`、`tool_output_tokens=1500`、`slots_tokens=200`、
  `total_tokens=6000`。
- 估算口径与 [evals/run_eval.py:32](../evals/run_eval.py#L32) 的 `CHARS_PER_TOKEN = 1.5` 保持一致，
  评测与线上预算不会出现两套口径。
- `build_agent_input`（[:552](../agent/context_manager.py#L552)）的组装规则：
  1. 顺序固定为「任务槽位 → 已知背景（长期记忆）→ 用户问题」，空节直接省略；
  2. 用户问题**始终最后且永不裁剪**；
  3. 槽位、记忆分别先按 `slots_tokens` / `memory_tokens` 裁剪；
  4. 全文字符数超过 `total_tokens * 1.5` 时，先按剩余空间重裁记忆，再重裁槽位；仍放不下时只保留
     用户问题节（用户问题本身超预算属于唯一例外，已在函数 docstring 说明）。
- 接入状态：已使用默认预算（[agent/main_agent.py:368](../agent/main_agent.py#L368)）；
  需要按任务调参时在调用处传 `budget=ContextBudget(...)`。

## 当前接入点

> 行号为撰写文档时的快照，函数名保持稳定；`agent/main_agent.py` 由 Lead 负责集成。

| 能力 | 函数 | 调用位置 | 状态 |
| --- | --- | --- | --- |
| 长期记忆检索 + 格式化 | `MemoryStore.search` / `render_for_prompt` | `agent/main_agent.py` `run_deep_agent` :347/:361（含两个兜底） | 已接入 |
| 槽位抽取 | `extract_slots` | `agent/main_agent.py` `run_deep_agent` :364 | 已接入 |
| 分节注入 + 预算 | `build_agent_input` | `agent/main_agent.py` `run_deep_agent` :366-370 | 已接入 |
| 工具输出裁剪 | `clip_tool_output` | `agent/tool_observer.py` `instrument_tool`/`_observed_run`：工具输出先裁剪再回填模型 | 已接入 |
| 历史裁剪 | `trim_history` | 由框架内置 `SummarizationMiddleware` 的 keep 策略承担，无需额外接入 | 框架承担 |
| 历史摘要兜底 | `summarize_history` | 纯函数，供后续自定义压缩旧轮时使用 | 可选定制 |
| 框架摘要中间件 | `summarize_middleware` | deepagents 已默认启用内置摘要；需要改阈值时在 `create_deep_agent(middleware=[...])` 传入 | 可选定制 |
| 预算配置 | `ContextBudget` | 通过 `build_agent_input(budget=...)` 传入，当前用默认值 | 已接入（默认预算） |

## 测试与验证

离线单测在 [tests/test_context_manager.py](../tests/test_context_manager.py)，覆盖 token 估算边界、
头尾裁剪、工具输出裁剪、历史裁剪（保留 system/最近轮/旧相关消息/顺序/空列表）、摘要的三种路径
（无 LLM、LLM 异常、LLM 正常）、槽位各时间/品类/指标/对比/样本量、分节顺序与总预算、以及
`summarize_middleware` 的正常配置与 `None` 降级。

```powershell
python -m pytest tests/test_context_manager.py -q --basetemp=.pytest_tmp\ctx -p no:cacheprovider
python -m ruff check agent/context_manager.py tests/test_context_manager.py
```

## 边界说明

- `agent/context_manager.py` 只做纯计算：不读文件、不联网、不建连接、不依赖全局可变状态。
- 本模块不改变 deepagents 内置摘要的默认行为；`summarize_middleware` 只是提供显式配置入口，
  `create_deep_agent(middleware=[...])` 的追加语义见上文。
- 记忆的写入/去重/生命周期在 `agent/memory.py`；本模块只负责把检索结果按预算注入。
