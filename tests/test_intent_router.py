"""分层意图识别（agent/intent_router.py）单元测试。

覆盖 P0 §3.4 的关键约定：
- L1 规则：强命中才返回，多来源冲突交给 L2，纯闲聊不硬猜；
- L2 分类器：纯 Python 实现、确定性、scores 归一、label 合法；
- 分层兜底：低置信度 + use_llm=False 时落 L4 澄清，且完全不触碰 LLM / 网络；
- 训练集与评测集标注：条数、标签合法、无重叠原句；
- 混淆矩阵 / 分类指标的正确性。

全部使用假数据与假模型，不依赖网络、MySQL、RAGFlow 和真实 LLM。
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

from agent import intent_router as router

ROOT = Path(__file__).resolve().parents[1]

# 假训练集：每类 3 条，词面足够区分，用于验证分类器的确定性与归一化
FAKE_TRAINING: list[dict[str, str]] = [
    {"query": "订单表里一共有多少条记录", "label": "database"},
    {"query": "统计所有订单的金额总和", "label": "database"},
    {"query": "按品类分组统计订单数量", "label": "database"},
    {"query": "内部 SOP 里这个指标口径是什么", "label": "document"},
    {"query": "数据字典里这个字段是什么意思", "label": "document"},
    {"query": "知识库文档里怎么规定的", "label": "document"},
    {"query": "联网搜索一下行业新闻", "label": "web"},
    {"query": "竞品最近有什么动态", "label": "web"},
    {"query": "最新政策报道说了什么", "label": "web"},
    {"query": "结合知识库和数据库一起分析", "label": "mixed"},
    {"query": "先查文档再查数据库最后汇总", "label": "mixed"},
    {"query": "综合多个来源写一份报告", "label": "mixed"},
]


class _StubClassifier:
    """L2 假分类器：不训练，直接返回预设结果。"""

    def __init__(self, label: str, confidence: float, scores: dict[str, float]) -> None:
        self._label = label
        self._confidence = confidence
        self._scores = scores

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        return self._label, self._confidence, dict(self._scores)


def _stub_high_confidence() -> _StubClassifier:
    scores = {"database": 0.9, "document": 0.05, "web": 0.03, "mixed": 0.02}
    return _StubClassifier("database", 0.9, scores)


def _stub_low_confidence() -> _StubClassifier:
    scores = {"database": 0.3, "document": 0.1, "web": 0.2, "mixed": 0.4}
    return _StubClassifier("mixed", 0.4, scores)


class _FakeResponse:
    """模拟 langchain 的 AIMessage（这里只需要 content 字段）。"""

    def __init__(self, content: str) -> None:
        self.content = content


class _FakeModel:
    """假 LLM：记录收到的 prompt，返回固定文本，不产生任何网络请求。"""

    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> _FakeResponse:
        self.prompts.append(prompt)
        return _FakeResponse(self.content)


def _install_fake_llm(monkeypatch: pytest.MonkeyPatch, content: str) -> _FakeModel:
    """把 agent.llm 替换成假模块，让 L3 的惰性导入拿到假 model。"""
    model = _FakeModel(content)
    module = types.ModuleType("agent.llm")
    module.model = model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent.llm", module)
    return model


class TestRuleRouting:
    """L1 规则：强命中 / 冲突 / 不命中。"""

    def test_sop_routes_to_document(self):
        result = router.classify_by_rules("内部 SOP 里 AOV 的计算口径是什么？")
        assert result is not None
        assert result.label == "document"
        assert result.layer == "L1-rule"
        assert result.needs_clarification is False
        assert result.matched
        assert result.suggested_subagents == router.INTENT_TO_SUBAGENTS["document"]

    def test_aggregate_routes_to_database(self):
        result = router.classify_by_rules("orders 表里金额最高的订单号是哪一个？")
        assert result is not None
        assert result.label == "database"
        assert result.suggested_subagents == ["数据分析助手"]

    def test_project_level_routes_to_mixed(self):
        result = router.classify_by_rules("本项目的三个子智能体分别负责什么？")
        assert result is not None
        assert result.label == "mixed"
        assert result.suggested_subagents == router.INTENT_TO_SUBAGENTS["mixed"]

    def test_combo_marker_with_two_sources_routes_mixed(self):
        result = router.classify_by_rules("结合知识库和数据库，orders 表有多少条记录？")
        assert result is not None
        assert result.label == "mixed"

    def test_multi_source_conflict_returns_none(self):
        # 同时命中 document / database / web 三个来源且无综合标记：不硬猜，交给 L2
        query = "内部 SOP 里退款率口径是什么？orders 表里退款订单总数是多少？竞品最新动态如何？"
        assert router.classify_by_rules(query) is None

    def test_plain_chitchat_returns_none(self):
        assert router.classify_by_rules("你好呀") is None
        assert router.classify_by_rules("") is None

    def test_result_to_dict_is_json_serializable(self):
        result = router.classify_by_rules("内部 SOP 里 AOV 的计算口径是什么？")
        assert result is not None
        payload = result.to_dict()
        expected_keys = {
            "label",
            "confidence",
            "layer",
            "scores",
            "matched",
            "suggested_subagents",
            "needs_clarification",
            "clarification_question",
        }
        assert set(payload) == expected_keys
        assert json.dumps(payload, ensure_ascii=False)


class TestNaiveBayesClassifier:
    """L2 字符 n-gram 朴素贝叶斯。"""

    def test_deterministic_for_same_input(self):
        classifier = router.NaiveBayesIntentClassifier().fit(FAKE_TRAINING)
        first = classifier.predict("订单表里一共有多少条记录")
        second = classifier.predict("订单表里一共有多少条记录")
        assert first == second
        # 两个独立实例训练结果也应一致（无随机性）
        other = router.NaiveBayesIntentClassifier().fit(FAKE_TRAINING)
        assert other.predict("订单表里一共有多少条记录") == first

    def test_predict_returns_legal_label_and_normalized_scores(self):
        classifier = router.NaiveBayesIntentClassifier().fit(FAKE_TRAINING)
        label, confidence, scores = classifier.predict("订单表里一共有多少条记录")
        assert label in router.INTENTS
        assert set(scores) == set(router.INTENTS)
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-9)
        assert all(0.0 <= value <= 1.0 for value in scores.values())
        assert confidence == pytest.approx(scores[label])
        assert label == max(scores, key=scores.get)

    def test_requires_fit_before_predict(self):
        classifier = router.NaiveBayesIntentClassifier()
        with pytest.raises(RuntimeError):
            classifier.predict("订单表里一共有多少条记录")

    def test_fit_rejects_invalid_label(self):
        with pytest.raises(ValueError):
            router.NaiveBayesIntentClassifier().fit([{"query": "订单查询", "label": "unknown"}])

    def test_fit_rejects_empty_query(self):
        with pytest.raises(ValueError):
            router.NaiveBayesIntentClassifier().fit([{"query": "", "label": "database"}])

    def test_invalid_hyper_parameters_rejected(self):
        with pytest.raises(ValueError):
            router.NaiveBayesIntentClassifier(ngram_range=(3, 1))
        with pytest.raises(ValueError):
            router.NaiveBayesIntentClassifier(alpha=0)


class TestClassifyIntentLayers:
    """分层入口 classify_intent：L1 → L2 →（L3）→ L4。"""

    def test_rules_short_circuit_before_classifier_and_llm(self, monkeypatch):
        def _should_not_be_called():
            raise AssertionError("L1 命中后不应加载 L2 分类器")

        async def _should_not_be_called_async(_query: str):
            raise AssertionError("L1 命中后不应调用 L3")

        monkeypatch.setattr(router, "_get_classifier", _should_not_be_called)
        monkeypatch.setattr(router, "_llm_classify", _should_not_be_called_async)
        result = asyncio.run(router.classify_intent("内部 SOP 里 AOV 的计算口径是什么？", use_llm=False))
        assert result.layer == "L1-rule"
        assert result.label == "document"

    def test_high_confidence_returns_l2(self, monkeypatch):
        monkeypatch.setattr(router, "_get_classifier", _stub_high_confidence)
        result = asyncio.run(router.classify_intent("随便说点什么", use_llm=False))
        assert result.layer == "L2-classifier"
        assert result.label == "database"
        assert result.confidence == pytest.approx(0.9)
        assert result.needs_clarification is False
        assert result.suggested_subagents == ["数据分析助手"]

    def test_low_confidence_disabled_llm_returns_clarification(self, monkeypatch):
        async def _no_llm(_query: str):
            raise AssertionError("use_llm=False 时不应调用 L3")

        monkeypatch.setattr(router, "_get_classifier", _stub_low_confidence)
        monkeypatch.setattr(router, "_llm_classify", _no_llm)
        result = asyncio.run(router.classify_intent("随便说点什么", use_llm=False))
        assert result.layer == "L4-clarify"
        assert result.needs_clarification is True
        assert result.clarification_question == router.CLARIFICATION_QUESTION
        assert result.label in router.INTENTS
        assert result.suggested_subagents == router.INTENT_TO_SUBAGENTS[result.label]

    def test_empty_query_goes_clarify_without_classifier(self, monkeypatch):
        def _should_not_be_called():
            raise AssertionError("空问题不应加载 L2 分类器")

        monkeypatch.setattr(router, "_get_classifier", _should_not_be_called)
        result = asyncio.run(router.classify_intent("   ", use_llm=False))
        assert result.layer == "L4-clarify"
        assert result.needs_clarification is True

    def test_l3_result_is_used_when_available(self, monkeypatch):
        model = _install_fake_llm(monkeypatch, '{"label": "web", "confidence": 0.9, "reason": "需要联网"}')
        monkeypatch.setattr(router, "_get_classifier", _stub_low_confidence)
        result = asyncio.run(router.classify_intent("随便说点什么", use_llm=True))
        assert result.layer == "L3-llm"
        assert result.label == "web"
        assert result.confidence == pytest.approx(0.9)
        assert result.suggested_subagents == ["市场情报助手"]
        assert model.prompts and "随便说点什么" in model.prompts[0]

    @pytest.mark.parametrize(
        "bad_output",
        [
            "我认为是 web 类，但我不太确定",  # 没有 JSON
            '{"label": "unknown", "confidence": 0.9}',  # label 非法
            "```json\n{broken json}\n```",  # JSON 解析失败
        ],
    )
    def test_l3_bad_output_falls_back_to_clarify(self, monkeypatch, bad_output):
        _install_fake_llm(monkeypatch, bad_output)
        monkeypatch.setattr(router, "_get_classifier", _stub_low_confidence)
        result = asyncio.run(router.classify_intent("随便说点什么", use_llm=True))
        assert result.needs_clarification is True
        assert result.layer == "L4-clarify"

    def test_l3_missing_confidence_uses_default(self, monkeypatch):
        _install_fake_llm(monkeypatch, '{"label": "document", "reason": "内部文档"}')
        monkeypatch.setattr(router, "_get_classifier", _stub_low_confidence)
        result = asyncio.run(router.classify_intent("随便说点什么", use_llm=True))
        assert result.layer == "L3-llm"
        assert result.label == "document"
        assert result.confidence == pytest.approx(0.6)


class TestTrainingData:
    """训练集加载与标注。"""

    def test_default_training_data_is_balanced_and_valid(self):
        samples = router.load_training_data()
        assert len(samples) >= 60
        counts = Counter(sample["label"] for sample in samples)
        assert set(counts) == set(router.INTENTS)
        assert min(counts.values()) >= 10
        assert len({sample["query"] for sample in samples}) == len(samples)

    def test_training_queries_do_not_overlap_eval_questions(self):
        dataset = json.loads((ROOT / "evals" / "dataset.json").read_text(encoding="utf-8"))
        eval_questions = {str(case["question"]).strip() for case in dataset["cases"]}
        training_queries = {sample["query"] for sample in router.load_training_data()}
        assert eval_questions.isdisjoint(training_queries)

    def test_load_training_data_from_custom_path(self, tmp_path):
        path = tmp_path / "train.json"
        path.write_text(json.dumps([{"query": "查订单", "label": "database"}]), encoding="utf-8")
        assert router.load_training_data(path) == [{"query": "查订单", "label": "database"}]

    def test_load_training_data_rejects_bad_label(self, tmp_path):
        path = tmp_path / "train.json"
        path.write_text(json.dumps([{"query": "查订单", "label": "unknown"}]), encoding="utf-8")
        with pytest.raises(ValueError):
            router.load_training_data(path)

    def test_load_training_data_rejects_empty_list(self, tmp_path):
        path = tmp_path / "train.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError):
            router.load_training_data(path)


class TestEvalDatasetAnnotation:
    """evals/dataset.json 的 intent / expected_subagents 标注约定。"""

    def test_intent_follows_id_convention(self):
        dataset = json.loads((ROOT / "evals" / "dataset.json").read_text(encoding="utf-8"))
        prefix_to_intent = {"db": "database", "rag": "document", "web": "web", "mix": "mixed"}
        for case in dataset["cases"]:
            prefix = str(case["id"]).split("-")[0]
            assert case["intent"] == prefix_to_intent[prefix], case["id"]

    def test_expected_subagents_are_valid_names(self):
        allowed = {name for names in router.INTENT_TO_SUBAGENTS.values() for name in names}
        dataset = json.loads((ROOT / "evals" / "dataset.json").read_text(encoding="utf-8"))
        for case in dataset["cases"]:
            subagents = case["expected_subagents"]
            assert subagents, case["id"]
            assert set(subagents) <= allowed, case["id"]


class TestMetrics:
    """混淆矩阵与分类指标。"""

    def test_confusion_matrix_counts(self):
        matrix = router.confusion_matrix(
            ["database", "database", "web", "web", "mixed"],
            ["database", "web", "web", "web", "database"],
        )
        assert matrix == [
            [1, 0, 1, 0],  # 真实 database：1 次判对，1 次误判为 web
            [0, 0, 0, 0],
            [0, 0, 2, 0],  # 真实 web：2 次判对
            [1, 0, 0, 0],  # 真实 mixed：1 次误判为 database
        ]

    def test_confusion_matrix_rejects_length_mismatch(self):
        with pytest.raises(ValueError):
            router.confusion_matrix(["database"], ["database", "web"])

    def test_confusion_matrix_rejects_unknown_label(self):
        with pytest.raises(ValueError):
            router.confusion_matrix(["database"], ["unknown"])

    def test_classification_metrics_values(self):
        y_true = ["database", "database", "web", "web"]
        y_pred = ["database", "web", "web", "web"]
        metrics = router.classification_metrics(y_true, y_pred)
        assert metrics["labels"] == list(router.INTENTS)
        assert metrics["accuracy"] == pytest.approx(0.75)
        assert metrics["per_class"]["database"]["precision"] == pytest.approx(1.0)
        assert metrics["per_class"]["database"]["recall"] == pytest.approx(0.5)
        assert metrics["per_class"]["database"]["f1"] == pytest.approx(2 / 3, abs=1e-4)
        assert metrics["per_class"]["database"]["support"] == 2
        assert metrics["per_class"]["document"]["support"] == 0
        assert metrics["per_class"]["document"]["f1"] == pytest.approx(0.0)
        assert metrics["per_class"]["web"]["precision"] == pytest.approx(2 / 3, abs=1e-4)
        assert metrics["per_class"]["web"]["recall"] == pytest.approx(1.0)
        assert metrics["per_class"]["web"]["f1"] == pytest.approx(0.8)
        assert metrics["macro_f1"] == pytest.approx(0.3667, abs=2e-4)

    def test_classification_metrics_perfect(self):
        labels = list(router.INTENTS)
        metrics = router.classification_metrics(labels, labels)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["macro_f1"] == pytest.approx(1.0)
        assert all(metrics["per_class"][label]["precision"] == pytest.approx(1.0) for label in labels)

    def test_suggested_subagents_mapping(self):
        assert router.INTENT_TO_SUBAGENTS == {
            "database": ["数据分析助手"],
            "document": ["内部知识助手"],
            "web": ["市场情报助手"],
            "mixed": ["市场情报助手", "数据分析助手", "内部知识助手"],
        }
        assert set(router.INTENT_TO_SUBAGENTS) == set(router.INTENTS)
