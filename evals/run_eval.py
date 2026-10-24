"""端到端评测脚本：对运行中的 API 跑一遍 dataset.json，产出成功率 / 延迟 / 工具调用。

跑法：
    python api/server.py                # 另开一个窗口把服务起起来
    python evals/run_eval.py            # 全量 20 题
    python evals/run_eval.py --category 查库 --limit 3
    python evals/run_eval.py --base http://127.0.0.1:8010

设计要点：
* 走真实链路（HTTP 提交 + WebSocket 收事件），不 mock，测的就是用户看到的那条路。
* thread_id 由客户端先生成，先把 WS 连上再提交任务，避免漏掉开头的事件。
* 判分只做确定性匹配（关键词组），不做 LLM 裁判——可复现、不花钱、争议小。
"""

import argparse
import asyncio
import json
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


async def run_case(case: dict, base: str, timeout: float) -> dict:
    ws_base = base.replace("http://", "ws://").replace("https://", "wss://")
    thread_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    events: list[dict] = []
    result = ""
    error = ""

    try:
        async with websockets.connect(f"{ws_base}/ws/{thread_id}", ping_interval=None, open_timeout=15) as ws:
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
                if payload.get("event") == "task_result":
                    result = (payload.get("data") or {}).get("result") or ""
                    break
    except TimeoutError:
        error = f"超时 {timeout:.0f}s"
    except Exception as exc:  # 单题失败不影响整轮
        error = f"{type(exc).__name__}: {exc}"

    latency = time.perf_counter() - t0
    passed, missing = score(result, case["expect"])
    if error:
        passed = False

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "passed": passed,
        "latency_s": round(latency, 2),
        "tool_calls": sum(1 for e in events if e.get("event") == "tool_start"),
        "assistant_calls": sum(1 for e in events if e.get("event") == "assistant_call"),
        "chars": len(result),
        "est_tokens": round(len(result) / CHARS_PER_TOKEN),
        "missing": missing,
        "error": error,
        "result": result,
    }


def summarize(rows: list[dict]) -> str:
    total = len(rows)
    passed = sum(r["passed"] for r in rows)
    lat = [r["latency_s"] for r in rows if not r["error"]] or [0]
    chars = [r["chars"] for r in rows if not r["error"]] or [0]
    tools = [r["tool_calls"] for r in rows] or [0]
    p50 = statistics.median(lat)
    p95 = statistics.quantiles(lat, n=20)[18] if len(lat) >= 20 else max(lat)

    lines = [
        f"整体：{passed}/{total} 通过（{passed / total:.0%}）",
        f"延迟：P50 {p50:.1f}s，P95 {p95:.1f}s，最大 {max(lat):.1f}s",
        f"工具调用：平均 {statistics.mean(tools):.1f} 次/题，最多 {max(tools)} 次",
        f"输出规模：平均 {statistics.mean(chars):.0f} 字（≈{statistics.mean(chars) / CHARS_PER_TOKEN:.0f} tokens）",
    ]
    for category in dict.fromkeys(r["category"] for r in rows):
        sub = [r for r in rows if r["category"] == category]
        ok = sum(r["passed"] for r in sub)
        lines.append(f"  · {category}：{ok}/{len(sub)}")
    return "\n".join(lines)


def write_markdown(rows: list[dict], summary: str, base: str) -> None:
    header = (
        "| 编号 | 类别 | 结果 | 延迟(s) | 工具调用 | 输出字数 | 未命中 |\n| --- | --- | --- | --- | --- | --- | --- |"
    )
    body = "\n".join(
        "| {id} | {cat} | {res} | {lat:.1f} | {tools} | {chars} | {miss} |".format(
            id=r["id"],
            cat=r["category"],
            res="✅" if r["passed"] else "❌",
            lat=r["latency_s"],
            tools=r["tool_calls"],
            chars=r["chars"],
            miss=("；".join(" / ".join(g) for g in r["missing"]) or "-").replace("|", "/"),
        )
        for r in rows
    )
    md = (
        "# 评测报告\n\n"
        f"- 运行时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- 服务地址：{base}\n"
        f"- 题量：{len(rows)}\n\n"
        "## 汇总\n\n"
        "```\n" + summary + "\n```\n\n"
        "## 明细\n\n" + header + "\n" + body + "\n"
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
            f"    {flag} {row['latency_s']:.1f}s tools={row['tool_calls']} chars={row['chars']} {row['error'] or ''}",
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
