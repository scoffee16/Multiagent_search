"""分层意图识别路由（P0 §3.4）：规则 → 字符 n-gram 朴素贝叶斯 → LLM → 澄清。

让「这句话该交给哪个子智能体」变得可复现、可离线评测，而不是每次都由 LLM 直接分类。

分层策略
--------
* L1 规则/关键词：秒级、零成本。只在「强命中且不冲突」时返回（例如明确出现
  SOP / 数据字典 / 最新政策 等专属说法），其余情况一律往下走。
* L2 字符 n-gram 朴素贝叶斯：纯 Python 实现（不依赖 sklearn），首次调用时用
  evals/intent_dataset.json 训练并缓存。输出 label + confidence + 各类 scores，
  并计算 top1 与 top2 的 margin。
* L3 LLM few-shot 兜底：L2 置信度不足时，用 agent/llm.py 里的 model 做一次
  few-shot 分类（惰性导入，避免离线 / 无密钥场景被拖累）。
* L4 澄清：L3 不可用、解析失败或仍不确定时，needs_clarification=True 并给出反问话术。

路由阈值：L2 接受条件为 confidence >= L2_MIN_CONFIDENCE 且 margin >= L2_MIN_MARGIN；
不满足且 use_llm=False 时直接返回 L4（离线评测走的就是这条确定性路径）。
"""

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

INTENTS: tuple[str, ...] = ("database", "document", "web", "mixed")

INTENT_LABELS_ZH = {
    "database": "数据分析",
    "document": "内部知识",
    "web": "市场情报",
    "mixed": "综合",
}

INTENT_TO_SUBAGENTS = {
    "database": ["数据分析助手"],
    "document": ["内部知识助手"],
    "web": ["市场情报助手"],
    "mixed": ["市场情报助手", "数据分析助手", "内部知识助手"],
}

# 路由阈值：L2 置信度与 top1-top2 间隔
L2_MIN_CONFIDENCE = 0.55
L2_MIN_MARGIN = 0.15

# 字符 n-gram 范围与朴素贝叶斯平滑系数
DEFAULT_NGRAM_RANGE: tuple[int, int] = (1, 3)
DEFAULT_ALPHA = 1.0

DEFAULT_TRAINING_PATH = Path(__file__).resolve().parents[1] / "evals" / "intent_dataset.json"

# L4 反问话术：把可选意图列清楚，引导用户补充信息
CLARIFICATION_QUESTION = (
    "我暂时还判断不太准你的问题类型，能再具体说明一下吗？比如："
    "1) 查询数据库里的订单数据；"
    "2) 查阅内部文档 / SOP / 数据字典；"
    "3) 联网查询外部公开信息（新闻、竞品、政策等）；"
    "4) 需要多来源综合（数据库 + 文档 / 联网，或整理成报告）。"
)


@dataclass
class IntentResult:
    """一次意图识别的结果。

    layer 取值："L1-rule" | "L2-classifier" | "L3-llm" | "L4-clarify"。
    """

    label: str
    confidence: float
    layer: str
    scores: dict[str, float] = field(default_factory=dict)
    matched: list[str] = field(default_factory=list)
    suggested_subagents: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转成可直接 JSON 序列化的字典（供 API / 日志 / 评测复用）。"""
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "layer": self.layer,
            "scores": {label: round(float(score), 4) for label, score in self.scores.items()},
            "matched": list(self.matched),
            "suggested_subagents": list(self.suggested_subagents),
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
        }


# ---------------------------------------------------------------------------
# 训练数据
# ---------------------------------------------------------------------------


def load_training_data(path: str | Path | None = None) -> list[dict[str, str]]:
    """读取意图训练集，逐条校验 query / label 合法性。

    :param path: 训练集路径，默认 evals/intent_dataset.json。
    :return: [{"query": str, "label": str}, ...]，label 一定属于 INTENTS。
    """
    target = Path(path) if path is not None else DEFAULT_TRAINING_PATH
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"意图训练数据应为列表：{target}")

    samples: list[dict[str, str]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"训练数据第 {index} 条不是对象：{item!r}")
        query = str(item.get("query") or "").strip()
        label = str(item.get("label") or "").strip()
        if not query:
            raise ValueError(f"训练数据第 {index} 条缺少 query")
        if label not in INTENTS:
            raise ValueError(f"训练数据第 {index} 条 label 非法：{label!r}，可选 {INTENTS}")
        samples.append({"query": query, "label": label})
    if not samples:
        raise ValueError(f"训练数据为空：{target}")
    return samples


# ---------------------------------------------------------------------------
# L2：字符 n-gram 朴素贝叶斯
# ---------------------------------------------------------------------------

_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)


def _normalize_text(text: str) -> str:
    """归一化：小写、去掉空白与标点，只保留中英文 / 数字 / 下划线。"""
    return _NON_WORD_RE.sub("", (text or "").casefold())


class NaiveBayesIntentClassifier:
    """字符 n-gram 多项式朴素贝叶斯意图分类器（纯标准库实现，结果确定可复现）。

    采用「每个类别独立词表 + 拉普拉斯平滑」的多项式模型：
    log P(c) + Σ_t count(t) * log((count_c(t) + α) / (total_c + α * V))。
    训练文本很短，字符 1~3 gram 足以覆盖「订单数 / 口径 / 最新」这类说法。
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
        alpha: float = DEFAULT_ALPHA,
    ) -> None:
        low, high = ngram_range
        if not 1 <= low <= high:
            raise ValueError(f"非法 ngram_range：{ngram_range}")
        if alpha <= 0:
            raise ValueError(f"alpha 必须为正数：{alpha}")
        self.ngram_range = (low, high)
        self.alpha = float(alpha)
        self._reset_state()

    def _reset_state(self) -> None:
        """清空训练状态（供 __init__ 与重复 fit 复用）。"""
        self._class_counts: Counter[str] = Counter()
        self._token_counts: dict[str, Counter[str]] = {}
        self._class_totals: dict[str, int] = {}
        self._vocab: set[str] = set()
        self._labels: list[str] = []
        self._fitted = False

    def _features(self, text: str) -> list[str]:
        """抽取字符 n-gram 特征。"""
        normalized = _normalize_text(text)
        low, high = self.ngram_range
        features: list[str] = []
        for size in range(low, high + 1):
            for start in range(len(normalized) - size + 1):
                features.append(normalized[start : start + size])
        return features

    def fit(self, samples: Sequence[dict[str, str]]) -> "NaiveBayesIntentClassifier":
        """用标注样本训练分类器（可重复调用，会重置旧状态）。"""
        self._reset_state()

        for index, sample in enumerate(samples):
            query = str(sample.get("query") or "").strip()
            label = str(sample.get("label") or "").strip()
            if not query:
                raise ValueError(f"第 {index} 条训练样本缺少 query")
            if label not in INTENTS:
                raise ValueError(f"第 {index} 条训练样本 label 非法：{label!r}")
            self._class_counts[label] += 1
            counter = self._token_counts.setdefault(label, Counter())
            tokens = self._features(query)
            counter.update(tokens)
            self._class_totals[label] = self._class_totals.get(label, 0) + len(tokens)
            self._vocab.update(tokens)

        self._labels = [label for label in INTENTS if self._class_counts[label] > 0]
        if not self._labels:
            raise ValueError("训练样本为空，无法训练意图分类器")
        self._fitted = True
        return self

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        """预测单条文本。

        :return: (label, confidence, scores)；scores 为 4 类归一化概率，和为 1。
        """
        if not self._fitted:
            raise RuntimeError("分类器尚未训练，请先调用 fit()")

        features = self._features(text)
        vocab_size = max(len(self._vocab), 1)
        total_samples = sum(self._class_counts.values())
        log_scores: dict[str, float] = {}
        for label in self._labels:
            log_prior = math.log(self._class_counts[label] / total_samples)
            denominator = self._class_totals.get(label, 0) + self.alpha * vocab_size
            counter = self._token_counts.get(label, Counter())
            score = log_prior
            for feature in features:
                score += math.log((counter.get(feature, 0) + self.alpha) / denominator)
            log_scores[label] = score

        # softmax（先减最大值，避免 exp 溢出），结果归一化为概率
        max_log = max(log_scores.values())
        exp_scores = {label: math.exp(value - max_log) for label, value in log_scores.items()}
        normalizer = sum(exp_scores.values()) or 1.0
        scores = {label: exp_scores[label] / normalizer for label in INTENTS if label in exp_scores}
        # 并列时按 INTENTS 顺序取（dict 迭代顺序即 INTENTS 顺序），保证确定性
        best_label = max(scores, key=lambda label: scores[label])
        return best_label, float(scores[best_label]), scores


# 模块级缓存：首次调用 classify_by_classifier 时训练
_CLASSIFIER: NaiveBayesIntentClassifier | None = None


def _get_classifier() -> NaiveBayesIntentClassifier:
    """懒加载并缓存默认分类器（用 evals/intent_dataset.json 训练）。"""
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = NaiveBayesIntentClassifier().fit(load_training_data())
    return _CLASSIFIER


def _score_margin(scores: dict[str, float]) -> float:
    """top1 与 top2 的概率差（margin），用于判断分类是否犹豫。"""
    if len(scores) < 2:
        return 1.0
    ordered = sorted(scores.values(), reverse=True)
    return float(ordered[0] - ordered[1])


# ---------------------------------------------------------------------------
# L1：规则 / 关键词
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Rule:
    """一条关键词规则：命中后把 query 归入 intent。"""

    intent: str
    name: str
    pattern: re.Pattern[str]


_DATABASE_RULES: tuple[_Rule, ...] = (
    _Rule(
        "database",
        "数量统计",
        re.compile(
            r"多少条|几条|多少个|几个不同|多少笔订单|一共多少|总共多少|"
            r"订单总数|记录总数|订单数|订单数量|订单量"
        ),
    ),
    _Rule(
        "database",
        "聚合计算",
        re.compile(r"总和|合计|求和|汇总|平均值|平均|最高|最低|最多|最少|最早|最晚|排名|排序|前三|前五"),
    ),
    _Rule("database", "SQL查库", re.compile(r"sql|查库|数据库查询|查询订单|订单查询|查询\s*orders|检索订单", re.I)),
    _Rule("database", "字段取值", re.compile(r"哪几种取值|哪些取值|几种取值|不同取值|有哪些取值|字段有哪")),
    _Rule("database", "订单明细", re.compile(r"订单号|客户姓名|订单详情|订单记录")),
)

_DOCUMENT_RULES: tuple[_Rule, ...] = (
    _Rule("document", "SOP", re.compile(r"\bsop\b", re.I)),
    _Rule(
        "document",
        "口径定义",
        re.compile(r"口径|计算规则|定义|含义|规范|制度|流程|判定标准|预警标准|标准是什么|计算方法|计算公式"),
    ),
    _Rule(
        "document",
        "数据字典",
        re.compile(r"数据字典|字段默认值|默认值|字段类型|编码规则|枚举|字段.{0,6}(说明|含义|允许为空)"),
    ),
    _Rule("document", "内部文档", re.compile(r"内部文档|内部资料|内部规范|内部制度|知识库|文档里|文档中|说明书|手册")),
    _Rule("document", "样本置信度", re.compile(r"样本量|样本少于|置信度|趋势推断")),
)

_WEB_RULES: tuple[_Rule, ...] = (
    _Rule("web", "联网搜索", re.compile(r"联网|上网|网上|搜索|搜一下|查一下最新|公开报道|公开信息|外部信息")),
    # 注意：不要把「最近 / 最新」单独当强信号——「最近 7 天的订单量」这类数据库问法也常出现，
    # 交给 L2 结合上下文判断；这里的新闻词本身已经足够明确。
    _Rule("web", "时效新闻", re.compile(r"新闻|动态|报道|消息|行情|政策")),
    _Rule("web", "竞品行业", re.compile(r"竞品|竞争对手|行业|市场|战报|促销活动")),
    _Rule("web", "外部事实", re.compile(r"哪一年|什么时候发布|发布|获奖|授予|颁发|诺贝尔")),
)

_MIXED_RULES: tuple[_Rule, ...] = (
    _Rule("mixed", "结合综合", re.compile(r"结合|综合|同时|整合|兼顾|一并")),
    _Rule("mixed", "先查再答", re.compile(r"先.{0,10}再|先按|然后再结合|最后汇总")),
    _Rule("mixed", "项目级", re.compile(r"本项目|本系统|我们系统|整个项目|系统架构|三个子智能体|子智能体")),
    # 报表类请求需要「数据查询 + 结构化报告输出」，在 multi-agent 语义下归为综合
    _Rule("mixed", "报表输出", re.compile(r"表格|报告|周报|月报|汇报|看板|仪表盘|\bppt\b", re.I)),
)

_ALL_RULES: tuple[_Rule, ...] = _DATABASE_RULES + _DOCUMENT_RULES + _WEB_RULES + _MIXED_RULES

# 规则命中后按「来源意图」聚合时使用的顺序（与 INTENTS 保持一致）
_SOURCE_INTENTS: tuple[str, ...] = ("database", "document", "web")


def _match_rules(query: str) -> dict[str, list[str]]:
    """返回每个意图命中的规则名列表（保持 INTENTS 顺序）。"""
    hits: dict[str, list[str]] = {intent: [] for intent in INTENTS}
    for rule in _ALL_RULES:
        if rule.pattern.search(query):
            hits[rule.intent].append(rule.name)
    return hits


def classify_by_rules(query: str) -> IntentResult | None:
    """L1 规则路由：强命中且不冲突时返回结果，否则返回 None 交给 L2。

    冲突处理：同时命中 database / document / web 中多个来源时，不硬猜，交给 L2。
    例外：项目级问题（本项目 / 三个子智能体…）与「报表输出」类请求按综合处理。
    """
    text = (query or "").strip()
    if not text:
        return None

    hits = _match_rules(text)
    source_hits = {intent: hits[intent] for intent in _SOURCE_INTENTS if hits[intent]}
    mixed_hits = hits["mixed"]
    project_level = "项目级" in mixed_hits
    report_output = "报表输出" in mixed_hits
    combo_marker = any(name in mixed_hits for name in ("结合综合", "先查再答"))

    label: str | None = None
    if project_level:
        # 项目级问题往往横跨 README / 文档 / 配置多个来源，按综合意图处理
        label = "mixed"
    elif combo_marker and len(source_hits) >= 2:
        # 明确要求「结合 / 先…再」，且真的涉及两个来源 → 综合
        label = "mixed"
    elif report_output and len(source_hits) == 1:
        # 单一数据源 + 报表/表格输出：数据查询 + 报告组织，按综合处理
        label = "mixed"
    elif len(source_hits) == 1 and not mixed_hits:
        label = next(iter(source_hits))
    if label is None:
        return None

    matched = [name for intent in INTENTS for name in hits[intent]]
    confidence = min(0.95, 0.78 + 0.04 * len(matched))
    return IntentResult(
        label=label,
        confidence=round(confidence, 4),
        layer="L1-rule",
        scores={label: round(confidence, 4)},
        matched=matched,
        suggested_subagents=list(INTENT_TO_SUBAGENTS[label]),
    )


# ---------------------------------------------------------------------------
# L2：分类器入口
# ---------------------------------------------------------------------------


def classify_by_classifier(query: str) -> IntentResult:
    """L2：用字符 n-gram 朴素贝叶斯分类（首次调用自动训练并缓存）。"""
    label, confidence, scores = _get_classifier().predict(query)
    return IntentResult(
        label=label,
        confidence=float(confidence),
        layer="L2-classifier",
        scores=dict(scores),
        matched=[],
        suggested_subagents=list(INTENT_TO_SUBAGENTS[label]),
    )


# ---------------------------------------------------------------------------
# L3：LLM few-shot 兜底 / L4：澄清
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """你是电商运营分析系统的意图分类器，负责判断用户问题该交给哪个子智能体。
可选意图只有 4 个：
- database：查询数据库里的订单数据（数量、金额、分组统计、明细等）；
- document：查阅内部文档 / 知识库（SOP 口径、数据字典、内部规则流程等）；
- web：需要联网获取外部信息（新闻、竞品动态、政策、公开事实等）；
- mixed：需要多个来源综合（数据库 + 文档、数据库 + 联网，或项目级综合问题）。

示例：
问题：orders 表里哪个品类的订单数最多？
回答：{"label": "database", "confidence": 0.95, "reason": "需要查询订单表数据"}

问题：内部 SOP 里 AOV 的计算口径是什么？
回答：{"label": "document", "confidence": 0.95, "reason": "内部文档口径问题"}

问题：竞品最近上了哪些新品？
回答：{"label": "web", "confidence": 0.93, "reason": "需要联网获取竞品动态"}

问题：结合知识库和数据库，说明本月 GMV 情况
回答：{"label": "mixed", "confidence": 0.92, "reason": "需要文档与数据库综合"}

只输出一行 JSON，不要输出解释、markdown 代码块或其他内容。"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)


def _build_llm_prompt(query: str) -> str:
    return f"{_LLM_SYSTEM_PROMPT}\n\n问题：{query}\n回答："


def _parse_llm_json(content: str) -> dict[str, Any] | None:
    """从模型输出里抠出第一个 JSON 对象；解析失败返回 None。"""
    match = _JSON_OBJECT_RE.search(content or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


async def _llm_classify(query: str) -> IntentResult | None:
    """L3：调用现有 LLM 做 few-shot 分类；不可用 / 解析失败时返回 None。"""
    try:
        # 惰性导入：离线 / 测试路径（use_llm=False）不会加载 LLM 配置
        from agent.llm import model
    except Exception:
        return None

    try:
        response = await model.ainvoke(_build_llm_prompt(query))
    except Exception:
        return None

    content = getattr(response, "content", response)
    if isinstance(content, list):  # 部分模型返回分段内容
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(part))
        content = "".join(parts)

    payload = _parse_llm_json(str(content))
    if not payload:
        return None

    label = str(payload.get("label") or "").strip().lower()
    if label not in INTENTS:
        return None
    try:
        confidence = float(payload.get("confidence") or 0.6)
    except (TypeError, ValueError):
        confidence = 0.6
    confidence = min(1.0, max(0.0, confidence))

    return IntentResult(
        label=label,
        confidence=confidence,
        layer="L3-llm",
        scores={label: confidence},
        matched=[],
        suggested_subagents=list(INTENT_TO_SUBAGENTS[label]),
    )


def _clarify_result(fallback: IntentResult | None = None) -> IntentResult:
    """构造 L4 澄清结果（label 取 L2 的最佳猜测，方便日志排查）。"""
    label = fallback.label if fallback is not None else "mixed"
    return IntentResult(
        label=label,
        confidence=float(fallback.confidence) if fallback is not None else 0.0,
        layer="L4-clarify",
        scores=dict(fallback.scores) if fallback is not None else {},
        matched=list(fallback.matched) if fallback is not None else [],
        suggested_subagents=list(INTENT_TO_SUBAGENTS[label]),
        needs_clarification=True,
        clarification_question=CLARIFICATION_QUESTION,
    )


async def classify_intent(query: str, *, use_llm: bool = True) -> IntentResult:
    """分层意图识别主入口：L1 → L2 →（可选 L3）→ L4。

    :param query: 用户原始问题。
    :param use_llm: 是否允许 L3 调用 LLM；离线评测 / 单元测试传 False。
    """
    text = (query or "").strip()
    if not text:
        return _clarify_result(None)

    rule_result = classify_by_rules(text)
    if rule_result is not None:
        return rule_result

    classifier_result = classify_by_classifier(text)
    margin = _score_margin(classifier_result.scores)
    if classifier_result.confidence >= L2_MIN_CONFIDENCE and margin >= L2_MIN_MARGIN:
        return classifier_result

    if not use_llm:
        return _clarify_result(classifier_result)

    llm_result = await _llm_classify(text)
    if llm_result is not None:
        return llm_result
    return _clarify_result(classifier_result)


# ---------------------------------------------------------------------------
# 评测指标（evals/intent_eval.py 与测试复用）
# ---------------------------------------------------------------------------


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = INTENTS,
) -> list[list[int]]:
    """混淆矩阵：行 = 真实标签，列 = 预测标签，顺序与 labels 一致。"""
    label_list = list(labels)
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true 与 y_pred 长度不一致：{len(y_true)} != {len(y_pred)}")
    index = {label: position for position, label in enumerate(label_list)}
    matrix = [[0 for _ in label_list] for _ in label_list]
    for truth, prediction in zip(y_true, y_pred, strict=True):
        if truth not in index:
            raise ValueError(f"y_true 中出现未知标签：{truth!r}")
        if prediction not in index:
            raise ValueError(f"y_pred 中出现未知标签：{prediction!r}")
        matrix[index[truth]][index[prediction]] += 1
    return matrix


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = INTENTS,
) -> dict[str, Any]:
    """多分类指标：accuracy、macro_f1 以及每类 precision / recall / f1 / support。"""
    label_list = list(labels)
    matrix = confusion_matrix(y_true, y_pred, label_list)
    total = len(y_true)
    correct = sum(matrix[position][position] for position in range(len(label_list)))
    accuracy = correct / total if total else 0.0

    per_class: dict[str, dict[str, float | int]] = {}
    for position, label in enumerate(label_list):
        true_positive = matrix[position][position]
        support = sum(matrix[position])
        predicted = sum(matrix[row][position] for row in range(len(label_list)))
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    macro_f1 = sum(float(per_class[label]["f1"]) for label in label_list) / len(label_list) if label_list else 0.0
    return {
        "labels": label_list,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }
