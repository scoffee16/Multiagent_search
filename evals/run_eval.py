"""端到端评测脚本：对运行中的 API 跑一遍 dataset.json，产出成功率 / 延迟 / 路由 / 引用覆盖。

跑法：
    python api/server.py                # 另开一个窗口把服务起起来
    python evals/run_eval.py            # 全量 20 题
    python evals/run_eval.py --category 查库 --limit 3
    python evals/run_eval.py --base http://127.0.0.1:8010

设计要点：
* 走真实链路（HTTP 提交 + WebSocket 收事件），不 mock，测的就是用户看到的那条路。
* thread_id 由客户端先生成，先把 WS 连上再提交任务，避免漏掉开头的事件。
* 判分只做确定性匹配（关键词组），不做 LLM 裁判——可复现、不花钱、争议小。
* P0/P1 新增三个指标（对应需求 §3.1 / §3.5 / §3.9）：
  1. 首字延迟与阶段耗时：由服务端 timing 事件给出（total/first_token/retrieve/llm/tool/subagent）；
  2. 路由准确率：dataset 每题标注 expected_subagents，统计 assistant_call 事件匹配情况
     （召回率 = 命中数/应调用数；完全匹配 = 实际调用集合与应调用集合一致）；
  3. 引用覆盖率：答案中带数字的句子，有多少带来源标记或数字能回溯到工具输出
     （tool_result 事件里的输出文本）。
"""

import argparse
import asyncio
import json
import re
import statistics
import time
import urllib.request
import uuid
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(__file__).with_name("dataset.json")
RESULT_JSON = Path(__file__).with_name("report.json")
RESULT_MD = Path(__file__).with_name("report.md")

# 中文大约 1.5 字符 1 token，仅用于给成本一个量级，正式结论以 provider 的 usage 为准
CHARS_PER_TOKEN = 1.5

# 数值结论的“来源标记”：出现任意一个就算这句话标注了来源
SOURCE_MARKERS = (
    "数据来源",
    "知识库",
    "联网搜索",
    "来源：",
    "来源:",
    "Tavily",
    "orders 表",
    "SOP",
)

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
SENTENCE_SPLIT_RE = re.compile(r"[。；;！!？?\n]+")

# task_result 之后服务端还会补发 timing / memory 事件，短等待收集，避免指标缺失
DRAIN_SECONDS = 5.0


def load_cases() -> list[dict]:
    return json.loads(DATASET.read_text(encoding="utf-8"))["cases"]


def wait_server(base: str, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/api/files", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise SystemExit(f"服务没起来：{base}（先跑 python api/server.py）")


def submit(base: str, query: str, thread_id: str) -> dict:
    req = urllib.request.Request(
        base + "/api/task",
        data=json.dumps({"query": query, "thread_id": thread_id}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def score(text: str, groups: list[list[str]]) -> tuple[bool, list[str]]:
    low = (text or "").casefold()
    missing = [alts for alts in groups if not any(a.casefold() in low for a in alts)]
    return (not missing), missing


def percentile(values: list[float], p: float) -> float:
    """线性插值分位数（与 agent/timing.py 的口径一致，保持独立实现避免脚本依赖 agent）。"""
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (max(0.0, min(100.0, p)) / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _numbers(text: str) -> list[str]:
    """抽出文本中的数字（去掉千分位逗号），用于判断答案数字能否回溯到工具输出。"""
    return NUMBER_RE.findall((text or "").replace(",", ""))


def extract_numeric_sentences(text: str) -> list[str]:
    """把答案切成句子，只保留含数字的句子（数值类结论）。"""
    sentences = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(text or "")]
    return [sentence for sentence in sentences if sentence and NUMBER_RE.search(sentence)]


def citation_stats(result: str, tool_output: str) -> dict:
    """引用覆盖率：数值句带来源标记，或句中任一数字能在工具输出里找到。"""
    sentences = extract_numeric_sentences(result)
    if not sentences:
        return {"total": 0, "cited": 0, "coverage": None}
    tool_numbers = set(_numbers(tool_output))
    cited = 0
    for sentence in sentences:
        if any(marker in sentence for marker in SOURCE_MARKERS):
            cited += 1
            continue
        if any(number in tool_numbers for number in _numbers(sentence)):
            cited += 1
    return {"total": len(sentences), "cited": cited, "coverage": round(cited / len(sentences), 3)}


def routing_stats(expected: list[str], actual_names: list[str]) -> dict | None:
    """路由准确率：expected_subagents（应调用）vs assistant_call 事件（实际调用）。"""
    expected_set = {name for name in (expected or []) if name}
    if not expected_set:
        return None
    actual_set = {name for name in (actual_names or []) if name}
    recall = round(len(expected_set & actual_set) / len(expected_set), 3)
    return {
        "expected": sorted(expected_set),
        "actual": sorted(actual_set),
        "recall": recall,
        "exact": expected_set == actual_set,
    }


async def _connect_ws(url: str, attempts: int = 3):
    """建立 WebSocket 连接，失败时重试。

    uvicorn/websockets 层偶发瞬态拒绝（HTTP 403），不代表任务失败，
    重试几次通常即可恢复，避免单题连接抖动影响整轮评测。
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await websockets.connect(url, ping_interval=None, open_timeout=15)
        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                raise
            await asyncio.sleep(0.5 * attempt)
    raise last_exc  # pragma: no cover - 循环保证要么 return 要么 raise


async def run_case(case: dict, base: str, timeout: float) -> dict:
    ws_base = base.replace("http://", "ws://").replace("https://", "wss://")
    thread_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    events: list[dict] = []
    result = ""
    error = ""
    first_token_s: float | None = None

    try:
        ws = await _connect_ws(f"{ws_base}/ws/{thread_id}")
        async with ws:
            submit(base, case["question"], thread_id)
            while True:
                remaining = timeout - (time.perf_counter() - t0)
                if remaining <= 0:
                    error = f"超时 {timeout:.0f}s"
                    break
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                payload = json.loads(raw)
                if payload.get("type") != "monitor_event":
                    continue
                events.append(payload)
                if payload.get("event") == "token" and first_token_s is None:
                    first_token_s = time.perf_counter() - t0
                if payload.get("event") == "task_result":
                    result = (payload.get("data") or {}).get("result") or ""
                    break
                if payload.get("event") == "error":
                    break

            # task_result 之后补收 timing / memory 事件（服务端在任务收尾时才发）
            drain_deadline = time.perf_counter() + DRAIN_SECONDS
            while True:
                drain_remaining = drain_deadline - time.perf_counter()
                if drain_remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=drain_remaining)
                except TimeoutError:
                    break
                payload = json.loads(raw)
                if payload.get("type") != "monitor_event":
                    continue
                events.append(payload)
                if payload.get("event") in ("timing", "error"):
                    break
    except TimeoutError:
        error = f"超时 {timeout:.0f}s"
    except Exception as exc:  # 单题失败不影响整轮
        error = f"{type(exc).__name__}: {exc}"

    latency = time.perf_counter() - t0
    passed, missing = score(result, case["expect"])
    if error:
        passed = False

    timing_data = next((e.get("data") or {} for e in events if e.get("event") == "timing"), {})
    timings = timing_data.get("timings") or {}
    intent_data = next((e.get("data") or {} for e in events if e.get("event") == "intent"), {})
    assistant_names = [
        str((e.get("data") or {}).get("assistant_name") or "") for e in events if e.get("event") == "assistant_call"
    ]
    tool_output = "\n".join(
        str((e.get("data") or {}).get("output") or "") for e in events if e.get("event") == "tool_result"
    )

    routing = routing_stats(case.get("expected_subagents") or [], assistant_names)
    citation = citation_stats(result, tool_output)
    expected_intent = case.get("intent")
    intent_ok = None
    if expected_intent:
        intent_ok = intent_data.get("label") == expected_intent

    first_token_ms = timings.get("first_token_ms")
    if first_token_ms is None and first_token_s is not None:
        first_token_ms = round(first_token_s * 1000.0, 1)

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "passed": passed,
        "latency_s": round(latency, 2),
        "first_token_ms": first_token_ms,
        "tool_calls": sum(1 for e in events if e.get("event") == "tool_start"),
        "assistant_calls": sum(1 for e in events if e.get("event") == "assistant_call"),
        "assistant_names": assistant_names,
        "chars": len(result),
        "est_tokens": round(len(result) / CHARS_PER_TOKEN),
        "timings": timings,
        "intent": intent_data.get("label"),
        "intent_ok": intent_ok,
        "routing": routing,
        "citation": citation,
        "missing": missing,
        "error": error,
        "result": result,
    }


def summarize(rows: list[dict]) -> str:
    total = len(rows)
    passed = sum(r["passed"] for r in rows)
    ok_rows = [r for r in rows if not r["error"]]
    lat = [r["latency_s"] for r in ok_rows] or [0.0]
    chars = [r["chars"] for r in ok_rows] or [0]
    tools = [r["tool_calls"] for r in rows] or [0]
    first_tokens = [r["first_token_ms"] for r in rows if r.get("first_token_ms") is not None]
    p50 = statistics.median(lat)
    p95 = percentile(lat, 95) if len(lat) >= 2 else max(lat)

    lines = [
        f"整体：{passed}/{total} 通过（{passed / total:.0%}）",
        f"延迟：P50 {p50:.1f}s，P95 {p95:.1f}s，最大 {max(lat):.1f}s",
    ]
    if first_tokens:
        lines.append(
            f"首字延迟：P50 {percentile(first_tokens, 50):.0f}ms，P95 {percentile(first_tokens, 95):.0f}ms"
            f"（{len(first_tokens)} 题有流式首字）"
        )
    stage_lines = []
    for stage in ("retrieve_ms", "llm_ms", "tool_ms"):
        values = [r["timings"].get(stage) for r in rows if r.get("timings") and r["timings"].get(stage) is not None]
        if values:
            stage_lines.append(
                f"{stage.replace('_ms', '')} P50 {percentile(values, 50):.0f}ms/P95 {percentile(values, 95):.0f}ms"
            )
    if stage_lines:
        lines.append("阶段耗时：" + "，".join(stage_lines))

    routing_rows = [r for r in rows if r.get("routing")]
    if routing_rows:
        exact = sum(1 for r in routing_rows if r["routing"]["exact"])
        recall = statistics.mean(r["routing"]["recall"] for r in routing_rows)
        lines.append(
            f"路由准确率：完全匹配 {exact}/{len(routing_rows)}（{exact / len(routing_rows):.0%}），平均召回率 {recall:.0%}"
        )

    intent_rows = [r for r in rows if r.get("intent_ok") is not None]
    if intent_rows:
        intent_ok = sum(1 for r in intent_rows if r["intent_ok"])
        lines.append(f"意图路由命中：{intent_ok}/{len(intent_rows)}（{intent_ok / len(intent_rows):.0%}）")

    citation_rows = [r for r in rows if r.get("citation") and r["citation"]["coverage"] is not None]
    if citation_rows:
        coverage = statistics.mean(r["citation"]["coverage"] for r in citation_rows)
        total_sentences = sum(r["citation"]["total"] for r in citation_rows)
        cited_sentences = sum(r["citation"]["cited"] for r in citation_rows)
        lines.append(
            f"引用覆盖率：{coverage:.0%}（{cited_sentences}/{total_sentences} 个数值句可回溯来源，"
            f"{len(citation_rows)} 题含数值结论）"
        )

    lines.append(f"工具调用：平均 {statistics.mean(tools):.1f} 次/题，最多 {max(tools)} 次")
    lines.append(
        f"输出规模：平均 {statistics.mean(chars):.0f} 字（≈{statistics.mean(chars) / CHARS_PER_TOKEN:.0f} tokens）"
    )
    for category in dict.fromkeys(r["category"] for r in rows):
        sub = [r for r in rows if r["category"] == category]
        ok = sum(r["passed"] for r in sub)
        lines.append(f"  · {category}：{ok}/{len(sub)}")
    return "\n".join(lines)


def write_markdown(rows: list[dict], summary: str, base: str) -> None:
    header = (
        "| 编号 | 类别 | 结果 | 延迟(s) | 首字(ms) | 路由 | 引用覆盖 | 工具 | 输出字数 | 未命中 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    body_lines = []
    for r in rows:
        routing = r.get("routing")
        if routing:
            routing_text = "✅" if routing["exact"] else f"召回 {routing['recall']:.0%}"
        else:
            routing_text = "-"
        citation = r.get("citation") or {}
        citation_text = "-" if citation.get("coverage") is None else f"{citation['coverage']:.0%}"
        first_token = "-" if r.get("first_token_ms") is None else f"{r['first_token_ms']:.0f}"
        body_lines.append(
            "| {id} | {cat} | {res} | {lat:.1f} | {ft} | {routing} | {citation} | {tools} | {chars} | {miss} |".format(
                id=r["id"],
                cat=r["category"],
                res="✅" if r["passed"] else "❌",
                lat=r["latency_s"],
                ft=first_token,
                routing=routing_text,
                citation=citation_text,
                tools=r["tool_calls"],
                chars=r["chars"],
                miss=("；".join(" / ".join(g) for g in r["missing"]) or "-").replace("|", "/"),
            )
        )
    md = (
        "# 评测报告\n\n"
        f"- 运行时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- 服务地址：{base}\n"
        f"- 题量：{len(rows)}\n\n"
        "## 汇总\n\n"
        "```\n" + summary + "\n```\n\n"
        "## 明细\n\n" + header + "\n" + "\n".join(body_lines) + "\n"
    )
    RESULT_MD.write_text(md, encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(description="MultiAgent-Search 端到端评测")
    parser.add_argument("--base", default="http://127.0.0.1:8010")
    parser.add_argument("--category", default=None, help="只跑某个类别：查库/查文档/联网/综合")
    parser.add_argument("--only", default=None, help="只跑某题，如 db-01")
    parser.add_argument("--limit", type=int, default=0, help="最多跑几题（0 = 全部）")
    parser.add_argument("--timeout", type=float, default=240.0, help="单题超时秒数")
    parser.add_argument("--skip-web", action="store_true", help="跳过联网类（没有 TAVILY_API_KEY 时）")
    args = parser.parse_args()

    cases = load_cases()
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
    if args.skip_web:
        cases = [c for c in cases if c["category"] != "联网"]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("没有匹配的题目")

    wait_server(args.base)
    rows = []
    for idx, case in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] {case['id']} {case['question'][:34]} ...", flush=True)
        row = await run_case(case, args.base, args.timeout)
        flag = "PASS" if row["passed"] else "FAIL"
        print(
            f"    {flag} {row['latency_s']:.1f}s tools={row['tool_calls']} chars={row['chars']} "
            f"first={row['first_token_ms']} {row['error'] or ''}",
            flush=True,
        )
        rows.append(row)

    summary = summarize(rows)
    print("\n" + summary)
    RESULT_JSON.write_text(
        json.dumps({"base": args.base, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(rows, summary, args.base)
    print(f"\n报告已写入：{RESULT_MD.name} / {RESULT_JSON.name}")


if __name__ == "__main__":
    asyncio.run(main())
