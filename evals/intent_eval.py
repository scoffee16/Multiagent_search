"""离线意图识别评测（P0 §3.4）：训练 L2 分类器，然后在 evals/dataset.json 上跑分层路由。

跑法：
    python evals/intent_eval.py           # 默认只评测 L1+L2（use_llm=False）：确定性、可离线、不花钱
    python evals/intent_eval.py --llm     # 额外启用 L3 LLM 兜底抽查（需要 .env 里的模型配置）

设计要点：
* 训练集 = evals/intent_dataset.json；测试集 = evals/dataset.json 的 20 题（question + intent 标注）。
  两套数据覆盖同一领域词汇但没有重叠原句，因此可以离线量化「分层路由」而不是只看端到端答案。
* 默认只走确定性路径：L1 规则 + L2 朴素贝叶斯，阈值不满足时直接落 L4 澄清；
  不依赖网络 / MySQL / RAGFlow / LLM，结果可复现，适合当回归基线。
* 指标：准确率、4x4 混淆矩阵、每类 P/R/F1/support、L1/L2/L3/L4 覆盖率、逐题分类耗时。
* 报告落盘 evals/intent_report.md 与 evals/intent_report.json；准确率低于目标时进程退出码为 1。
"""

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # 允许直接 python evals/intent_eval.py
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import intent_router as router  # noqa: E402  (需先补 sys.path)

# 离线准确率目标（低于它退出码为 1，方便 CI / 手动回归时立即发现退化）
TARGET_ACCURACY = 0.85

DEFAULT_DATASET = PROJECT_ROOT / "evals" / "dataset.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "evals" / "intent_report.md"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "evals" / "intent_report.json"

LAYER_ORDER: tuple[str, ...] = ("L1-rule", "L2-classifier", "L3-llm", "L4-clarify")
LAYER_LABELS = {
    "L1-rule": "L1 规则",
    "L2-classifier": "L2 分类器",
    "L3-llm": "L3 LLM",
    "L4-clarify": "L4 澄清",
}


def load_eval_cases(path: Path = DEFAULT_DATASET) -> list[dict[str, str]]:
    """读取评测集，只保留 id / question / intent 三个字段并校验标注。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"评测数据为空或格式错误：{path}")

    rows: list[dict[str, str]] = []
    for index, item in enumerate(cases):
        if not isinstance(item, dict):
            raise ValueError(f"评测数据第 {index} 条不是对象：{item!r}")
        question = str(item.get("question") or "").strip()
        intent = str(item.get("intent") or "").strip()
        case_id = str(item.get("id") or f"case-{index + 1}")
        if not question:
            raise ValueError(f"评测数据第 {index} 条（{case_id}）缺少 question")
        if intent not in router.INTENTS:
            raise ValueError(f"评测数据 {case_id} 的 intent 非法：{intent!r}，可选 {router.INTENTS}")
        rows.append({"id": case_id, "question": question, "intent": intent})
    return rows


async def evaluate(cases: Sequence[dict[str, str]], *, use_llm: bool) -> tuple[list[dict[str, Any]], float]:
    """逐题跑分层路由，返回逐题记录与分类器冷启动训练耗时（毫秒）。

    :param cases: load_eval_cases 的输出。
    :param use_llm: False 时只评测 L1+L2（不碰网络）；True 时允许 L3 兜底。
    """
    # 预热分类器：首次调用会读训练集并训练，单独计时，避免污染逐题耗时
    train_start = time.perf_counter()
    router.classify_by_classifier("预热")
    train_ms = (time.perf_counter() - train_start) * 1000.0

    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        result = await router.classify_intent(case["question"], use_llm=use_llm)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "intent": case["intent"],
                "predicted": result.label,
                "confidence": round(float(result.confidence), 4),
                "layer": result.layer,
                "needs_clarification": bool(result.needs_clarification),
                "matched": list(result.matched),
                "suggested_subagents": list(result.suggested_subagents),
                "correct": result.label == case["intent"],
                "latency_ms": round(elapsed_ms, 3),
            }
        )
    return rows, train_ms


def summarize(rows: Sequence[dict[str, Any]], train_ms: float) -> dict[str, Any]:
    """汇总准确率、macro F1、分层覆盖与耗时。"""
    y_true = [row["intent"] for row in rows]
    y_pred = [row["predicted"] for row in rows]
    metrics = router.classification_metrics(y_true, y_pred)
    total = len(rows)
    layer_counts = {layer: sum(1 for row in rows if row["layer"] == layer) for layer in LAYER_ORDER}
    fallback = layer_counts["L4-clarify"]
    routed = total - fallback
    latencies = [float(row["latency_ms"]) for row in rows]

    accuracy = float(metrics["accuracy"])
    return {
        "total": total,
        "accuracy": accuracy,
        "macro_f1": metrics["macro_f1"],
        "target_accuracy": TARGET_ACCURACY,
        "target_met": accuracy >= TARGET_ACCURACY,
        "layer_counts": layer_counts,
        "l1_hits": layer_counts["L1-rule"],
        "l2_hits": layer_counts["L2-classifier"],
        "l3_hits": layer_counts["L3-llm"],
        "fallback": fallback,
        "routing_coverage": round(routed / total, 4) if total else 0.0,
        "latency_ms": {
            "avg": round(sum(latencies) / total, 3) if latencies else 0.0,
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "classifier_train_ms": round(train_ms, 3),
    }


def _display_path(path: Path) -> str:
    """报告里用相对项目根的路径，方便不同机器复现。"""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_config(*, use_llm: bool, dataset_path: Path) -> dict[str, Any]:
    """记录本次评测的配置，保证报告可复现。"""
    classifier = router.NaiveBayesIntentClassifier()
    training = router.load_training_data()
    return {
        "dataset": _display_path(dataset_path),
        "train_data": _display_path(router.DEFAULT_TRAINING_PATH),
        "train_samples": len(training),
        "use_llm": use_llm,
        "ngram_range": list(classifier.ngram_range),
        "alpha": classifier.alpha,
        "l2_min_confidence": router.L2_MIN_CONFIDENCE,
        "l2_min_margin": router.L2_MIN_MARGIN,
        "target_accuracy": TARGET_ACCURACY,
    }


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _md_cell(text: Any) -> str:
    """转义 markdown 表格里的竖线与换行。"""
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(payload: dict[str, Any]) -> str:
    """把评测结果渲染成 markdown 报告。"""
    summary = payload["summary"]
    config = payload["config"]
    rows = payload["cases"]
    matrix = payload["confusion_matrix"]
    labels: list[str] = list(matrix["labels"])
    values: list[list[int]] = matrix["matrix"]
    total = summary["total"]
    accuracy = float(summary["accuracy"])
    correct = round(accuracy * total)

    lines: list[str] = [
        "# 意图识别离线评测报告",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 评测集：`{config['dataset']}`（{total} 题）",
        f"- 训练集：`{config['train_data']}`（{config['train_samples']} 条）",
        f"- 路由模式：{'L1+L2+L3（--llm 开启，结果不完全可复现）' if config['use_llm'] else 'L1+L2（确定性离线路径）'}",
        "",
        "## 结论",
        "",
        f"- 准确率：**{_percent(accuracy)}**（{correct}/{total}）；"
        f"目标 ≥ {_percent(float(config['target_accuracy']))}：**{'达标 ✅' if summary['target_met'] else '未达标 ❌'}**",
        f"- macro F1：{float(summary['macro_f1']):.4f}",
        f"- 分层命中：L1 {summary['l1_hits']} 题 / L2 {summary['l2_hits']} 题 / "
        f"L3 {summary['l3_hits']} 题 / L4 澄清 {summary['fallback']} 题",
        f"- 路由覆盖率：{_percent(float(summary['routing_coverage']))}"
        f"（{total - summary['fallback']}/{total} 题被 L1/L2/L3 成功分流）",
        f"- 分类耗时：平均 {float(summary['latency_ms']['avg']):.3f} ms / "
        f"最大 {float(summary['latency_ms']['max']):.3f} ms"
        f"（另有 L2 首次冷启动训练 {float(summary['classifier_train_ms']):.1f} ms，不计入逐题均值）",
        "",
        "## 分层命中与覆盖率",
        "",
        "| 层 | 说明 | 题数 | 占比 |",
        "| --- | --- | --- | --- |",
    ]
    for layer in LAYER_ORDER:
        count = int(summary["layer_counts"][layer])
        lines.append(f"| {layer} | {LAYER_LABELS[layer]} | {count} | {_percent(count / total)} |")
    lines.extend(
        [
            "",
            "| 指标 | 数值 |",
            "| --- | --- |",
            f"| L1 规则命中 | {summary['l1_hits']} |",
            f"| L2 分类器命中 | {summary['l2_hits']} |",
            f"| L3 LLM 兜底 | {summary['l3_hits']} |",
            f"| L4 兜底（需澄清） | {summary['fallback']} |",
            "",
            "## 混淆矩阵（行 = 真实，列 = 预测）",
            "",
            "| 真实 \\ 预测 | "
            + " | ".join(f"{label}（{router.INTENT_LABELS_ZH[label]}）" for label in labels)
            + " | 合计 |",
            "| " + " | ".join(["---"] * (len(labels) + 2)) + " |",
        ]
    )
    for row_index, label in enumerate(labels):
        row_values = values[row_index]
        row_total = sum(row_values)
        lines.append("| " + label + " | " + " | ".join(str(value) for value in row_values) + f" | {row_total} |")
    column_totals = [sum(values[row_index][col] for row_index in range(len(labels))) for col in range(len(labels))]
    lines.append("| 合计 | " + " | ".join(str(value) for value in column_totals) + f" | {total} |")
    lines.extend(
        [
            "",
            "## 每类指标",
            "",
            "| 意图 | 中文名 | precision | recall | f1 | support |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for label in labels:
        stats = payload["per_class"][label]
        lines.append(
            f"| {label} | {router.INTENT_LABELS_ZH[label]} | {float(stats['precision']):.4f} | "
            f"{float(stats['recall']):.4f} | {float(stats['f1']):.4f} | {int(stats['support'])} |"
        )
    lines.extend(
        [
            "",
            "## 逐题明细",
            "",
            "| 编号 | 意图 | 预测 | 层 | 置信度 | 耗时(ms) | 结果 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['intent']} | {row['predicted']} | {row['layer']} | "
            f"{float(row['confidence']):.4f} | {float(row['latency_ms']):.3f} | "
            f"{'✅' if row['correct'] else '❌'} |"
        )

    problems = [row for row in rows if not row["correct"] or row["needs_clarification"]]
    lines.extend(["", "## 误判 / 需澄清", ""])
    if not problems:
        lines.append("- 无")
    else:
        lines.extend(
            [
                "| 编号 | 问题 | 预期 | 预测 | 层 | 置信度 | 说明 |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in problems:
            reason = "需澄清" if row["needs_clarification"] else "误判"
            lines.append(
                f"| {row['id']} | {_md_cell(row['question'])} | {row['intent']} | {row['predicted']} | "
                f"{row['layer']} | {float(row['confidence']):.4f} | {reason} |"
            )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            f"- L2 为纯 Python 字符 {config['ngram_range'][0]}-{config['ngram_range'][1]} gram 朴素贝叶斯，"
            f"拉普拉斯平滑 alpha={config['alpha']}；接受条件为 "
            f"confidence ≥ {config['l2_min_confidence']} 且 margin ≥ {config['l2_min_margin']}。",
            "- 训练集与评测集无重叠原句，但覆盖同一领域词汇；短文本下朴素贝叶斯后验容易饱和，"
            "因此 L3/L4 主要作用于词表外或表述模糊的问句。",
            "- mix-* 题的 expected_subagents 是按题面实际需要标注的子集，本报告只评 intent 标签本身；"
            "子智能体召回率由 evals/run_eval.py 的 routing 指标覆盖。",
            "",
        ]
    )
    return "\n".join(lines)


def build_payload(
    *,
    rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    train_ms: float,
) -> dict[str, Any]:
    """组装写入 intent_report.json 的完整结果。"""
    y_true = [row["intent"] for row in rows]
    y_pred = [row["predicted"] for row in rows]
    metrics = router.classification_metrics(y_true, y_pred)
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "summary": summarize(rows, train_ms),
        "per_class": metrics["per_class"],
        "confusion_matrix": {"labels": list(router.INTENTS), "matrix": router.confusion_matrix(y_true, y_pred)},
        "cases": list(rows),
        "misclassified": [row for row in rows if not row["correct"] or row["needs_clarification"]],
    }


def write_reports(payload: dict[str, Any], md_path: Path, json_path: Path) -> None:
    """报告落盘（UTF-8，目录不存在时自动创建）。"""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_summary(payload: dict[str, Any], md_path: Path, json_path: Path) -> None:
    """在控制台打印关键结论与逐题误判。"""
    summary = payload["summary"]
    total = int(summary["total"])
    correct = round(float(summary["accuracy"]) * total)
    print()
    print("意图识别离线评测（L1+L2）" + (" + L3 抽查" if payload["config"]["use_llm"] else ""))
    print(
        f"准确率：{float(summary['accuracy']):.2%}（{correct}/{total}），"
        f"macro F1 {float(summary['macro_f1']):.4f}，"
        f"目标 ≥{_percent(float(summary['target_accuracy']))}："
        f"{'达标' if summary['target_met'] else '未达标'}"
    )
    print(
        f"分层命中：L1 {summary['l1_hits']} / L2 {summary['l2_hits']} / L3 {summary['l3_hits']} / "
        f"L4 澄清 {summary['fallback']}；路由覆盖 {_percent(float(summary['routing_coverage']))}"
    )
    print(
        f"分类耗时：平均 {float(summary['latency_ms']['avg']):.3f} ms，"
        f"最大 {float(summary['latency_ms']['max']):.3f} ms"
        f"（冷启动训练 {float(summary['classifier_train_ms']):.1f} ms）"
    )
    for row in payload["misclassified"]:
        reason = "需澄清" if row["needs_clarification"] else "误判"
        print(f"  ❌ {row['id']} 预期 {row['intent']} → 预测 {row['predicted']}（{row['layer']}，{reason}）")
    print(f"报告已写入：{_display_path(md_path)} / {_display_path(json_path)}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="意图识别离线评测（L1 规则 + L2 字符 n-gram 朴素贝叶斯）")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="允许 L3 LLM 兜底抽查（默认关闭；需要 .env 里的模型配置，且结果不完全可复现）",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="评测集路径，默认 evals/dataset.json")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_REPORT_MD, help="markdown 报告输出路径")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_REPORT_JSON, help="json 报告输出路径")
    parser.add_argument("--no-write", action="store_true", help="只打印结果，不写报告文件")
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    cases = load_eval_cases(args.dataset)
    rows, train_ms = await evaluate(cases, use_llm=args.llm)
    config = build_config(use_llm=args.llm, dataset_path=args.dataset)
    payload = build_payload(rows=rows, config=config, train_ms=train_ms)
    if not args.no_write:
        write_reports(payload, args.out_md, args.out_json)
    print_summary(payload, args.out_md, args.out_json)
    return 0 if payload["summary"]["target_met"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
