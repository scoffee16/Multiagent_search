"""跨会话长期记忆端到端评测：验证「会话 A 说一遍，换一个会话 B 能不能用上」。

跑法（8010 服务由 Lead 统一启动，本脚本只当客户端）：
    python api/server.py                      # 另开窗口起服务（Lead 负责）
    python evals/memory_eval.py               # 全量 5 组跨会话用例
    python evals/memory_eval.py --only mem-01
    python evals/memory_eval.py --base http://127.0.0.1:8010 --timeout 240

流程（每组用例）：
1. 会话 A（thread_id=uuidA）走 WS+HTTP 提一个「陈述偏好/口径/约束」的问题，等 task_result；
2. 轮询 GET /api/memory?thread_id=uuidA&limit=20，直到出现 ≥1 条记忆或 30s 超时
   （任务结束后主智能体会异步抽取写入，所以必须轮询而不是立刻断言）；
3. 会话 B（全新 thread_id=uuidB）问一个依赖该记忆的问题，等 task_result；
4. 对 B 的回答做确定性判分（关键词组 + 可选正则 + 开头窗口），汇总写 evals/memory_report.md。

判分只做确定性匹配，不做 LLM 裁判：可复现、不花钱、争议小（与 evals/run_eval.py 同思路）。
"""

import argparse
import asyncio
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = Path(__file__).with_name("memory_report.md")

# 任务结束后记忆抽取是异步的：最多等 30s，每 2s 轮询一次 /api/memory
MEMORY_POLL_TIMEOUT = 30.0
MEMORY_POLL_INTERVAL = 2.0

# 「告诉我这个偏好需要被记住」的 5 组跨会话用例。
# expect_groups：关键词组，组内任一命中即可，所有组都要命中；
# expect_patterns：正则列表，全部命中才通过；
# early_groups：在回答开头 400 字内必须出现的关键词组；
# order_groups：顺序约束，[{"first": [...], "then": [...]}]，若 then 词组出现，
#               则 first 词组的首次出现位置必须更靠前（验证「先结论后过程」这类结构偏好）。
CASES = [
    {
        "id": "mem-01",
        "name": "时间范围偏好",
        "session_a": "我们只看2026年Q1的数据",
        "session_b": "那段时间GMV大概多少？",
        "expect_groups": [["2026"], ["q1", "第一季度", "一季度", "1月", "1-3月", "1~3月", "1 月"]],
        "expect_patterns": [],
        "early_groups": [],
        "note": "A 陈述只看 2026Q1，B 追问 GMV 时应沿用该时间范围",
    },
    {
        "id": "mem-02",
        "name": "口径约定：两位小数",
        "session_a": "以后GMV都按SOP口径，金额保留两位小数",
        "session_b": "算一下客单价，记住要保留两位",
        "expect_groups": [["客单价"]],
        "expect_patterns": [r"\d+\.\d{2}(?!\d)"],
        "early_groups": [],
        "note": "A 约定金额保留两位小数，B 的客单价结果必须是两位小数",
    },
    {
        "id": "mem-03",
        "name": "报告结构偏好",
        "session_a": "以后报告先给结论再给过程",
        "session_b": "给我出个品类占比简报",
        "expect_groups": [["品类"]],
        "expect_patterns": [],
        "early_groups": [["结论"]],
        "order_groups": [{"first": ["结论"], "then": ["过程", "步骤", "推导"]}],
        "note": "A 要求先结论后过程，B 的简报开头 400 字内应出现「结论」且早于「过程」",
    },
    {
        "id": "mem-04",
        "name": "关注业务范围",
        "session_a": "我这边主要负责数码电子品类",
        "session_b": "分析一下我们品类的销售情况",
        "expect_groups": [["数码电子", "数码"]],
        "expect_patterns": [],
        "early_groups": [],
        "note": "A 说明负责数码电子品类，B 分析「我们品类」时应聚焦数码电子",
    },
    {
        "id": "mem-05",
        "name": "统计口径约束",
        "session_a": "统计时不要算退款订单",
        "session_b": "算一下总销售额",
        "expect_groups": [
            ["退款"],
            ["排除", "剔除", "不含", "去掉", "扣除", "过滤", "不算", "status", "已支付"],
        ],
        "expect_patterns": [],
        "early_groups": [],
        "note": "A 约束排除退款订单，B 算总销售额时应说明已排除/过滤退款",
    },
]


# --------------------------------------------------------------------------- HTTP / WS 小工具
def wait_server(base: str, timeout: float = 120.0) -> None:
    """等服务起来；超时给出明确提示并非 0 退出（run_eval 同款探活）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/api/files", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise SystemExit(f"服务没起来：{base}（先启动 python api/server.py，或等 Lead 统一启动后再跑）")


def check_memory_api(base: str) -> None:
    """确认 Lead 已集成记忆接口；404 时直接给出可操作提示并非 0 退出。"""
    try:
        with urllib.request.urlopen(base + "/api/memory/stats", timeout=10) as response:
            if response.status == 200:
                return
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise SystemExit(
                "服务在运行，但 GET /api/memory/stats 返回 404：记忆接口尚未集成 "
                "（需要 Lead 在 api/server.py 增加记忆路由后再跑本脚本）"
            ) from exc
        raise SystemExit(f"GET /api/memory/stats 返回 HTTP {exc.code}，无法评测") from exc
    except Exception as exc:
        raise SystemExit(f"GET /api/memory/stats 不可用：{type(exc).__name__}: {exc}") from exc


def http_json(url: str, payload: dict | None = None, timeout: float = 30.0):
    """发一个 JSON 请求并解析 JSON 响应。"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def submit_task(base: str, query: str, thread_id: str) -> dict:
    """POST /api/task 提交任务（接口立即返回 started，结果走 WS）。"""
    return http_json(base + "/api/task", {"query": query, "thread_id": thread_id}, timeout=30)


def _extract_memory_rows(payload) -> list[dict]:
    """兼容 GET /api/memory 的几种常见返回形状：裸数组 / {memories|items|data|...}。"""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("memories", "items", "data", "results", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
            if isinstance(value, dict):
                nested = _extract_memory_rows(value)
                if nested:
                    return nested
    return []


def poll_memories(base: str, thread_id: str, limit: int = 20) -> tuple[list[dict], float, str]:
    """轮询 /api/memory 直到会话 A 出现至少 1 条记忆；返回 (记忆列表, 耗时, 错误说明)。"""
    query = urllib.parse.urlencode({"thread_id": thread_id, "limit": limit})
    url = f"{base}/api/memory?{query}"
    started = time.perf_counter()
    last_error = ""
    while time.perf_counter() - started < MEMORY_POLL_TIMEOUT:
        try:
            rows = _extract_memory_rows(http_json(url, timeout=10))
            if rows:
                return rows, time.perf_counter() - started, ""
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return [], time.perf_counter() - started, "GET /api/memory 返回 404（记忆接口未集成）"
            last_error = f"HTTP {exc.code}"
        except Exception as exc:  # 单次轮询失败继续重试
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(MEMORY_POLL_INTERVAL)
    elapsed = time.perf_counter() - started
    return [], elapsed, last_error or f"轮询 {MEMORY_POLL_TIMEOUT:.0f}s 仍未出现记忆"


async def run_turn(base: str, query: str, thread_id: str, timeout: float) -> dict:
    """跑一轮会话：先连 WS 再提交任务，等 task_result；返回结果文本、耗时、错误。"""
    ws_base = base.replace("http://", "ws://").replace("https://", "wss://")
    started = time.perf_counter()
    result = ""
    error = ""
    memory_events = 0
    try:
        async with websockets.connect(f"{ws_base}/ws/{thread_id}", ping_interval=None, open_timeout=15) as ws:
            # 提交放线程池，避免阻塞当前事件循环的 WS 收包
            await asyncio.to_thread(submit_task, base, query, thread_id)
            while True:
                remaining = timeout - (time.perf_counter() - started)
                if remaining <= 0:
                    error = f"超时 {timeout:.0f}s"
                    break
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                payload = json.loads(raw)
                if payload.get("type") != "monitor_event":
                    continue
                event = payload.get("event") or ""
                if "memory" in str(event):
                    memory_events += 1  # Lead 新增的 memory 事件，有就计数（没有也不影响判分）
                if event == "task_result":
                    result = (payload.get("data") or {}).get("result") or ""
                    break
    except TimeoutError:
        error = f"超时 {timeout:.0f}s"
    except Exception as exc:  # 单轮失败不影响整轮评测
        error = f"{type(exc).__name__}: {exc}"
    return {
        "result": result,
        "error": error,
        "latency_s": round(time.perf_counter() - started, 2),
        "memory_events": memory_events,
    }


# --------------------------------------------------------------------------- 判分 / 报告
def _first_index(lowered: str, alternatives: list[str]) -> int | None:
    """返回任意关键词在文本中的最早出现位置；都没出现返回 None。"""
    positions = [
        position for position in (lowered.find(alternative.casefold()) for alternative in alternatives) if position >= 0
    ]
    return min(positions) if positions else None


def score_answer(text: str, case: dict) -> tuple[bool, list[str]]:
    """确定性判分：关键词组 + 正则 + 开头窗口 + 顺序约束，全部满足才算通过。"""
    answer = text or ""
    lowered = answer.casefold()
    missing: list[str] = []
    for alternatives in case.get("expect_groups") or []:
        if not any(alternative.casefold() in lowered for alternative in alternatives):
            missing.append("关键词：" + " / ".join(alternatives))
    for pattern in case.get("expect_patterns") or []:
        if not re.search(pattern, answer):
            missing.append(f"正则：{pattern}")
    early = answer[:400].casefold()
    for alternatives in case.get("early_groups") or []:
        if not any(alternative.casefold() in early for alternative in alternatives):
            missing.append("开头400字内：" + " / ".join(alternatives))
    for rule in case.get("order_groups") or []:
        first_alternatives = rule.get("first") or []
        then_alternatives = rule.get("then") or []
        first_position = _first_index(lowered, first_alternatives)
        then_position = _first_index(lowered, then_alternatives)
        if first_position is None:
            missing.append("缺少：" + " / ".join(first_alternatives))
        elif then_position is not None and first_position > then_position:
            missing.append(f"顺序：{' / '.join(first_alternatives)}应早于{' / '.join(then_alternatives)}")
    return (not missing), missing


def preview_memories(rows: list[dict], limit: int = 3) -> list[str]:
    """记忆预览，用于报告里证明「A 的话确实写进去了」。"""
    lines = []
    for row in rows[:limit]:
        content = str(row.get("content") or "").replace("\n", " ")
        if len(content) > 60:
            content = content[:60] + "…"
        lines.append(f"[{row.get('type', '?')}] {content}（id={row.get('id', '?')}）")
    return lines


async def run_case(base: str, case: dict, timeout: float) -> dict:
    """跑完一组跨会话用例：A 会话写入 → 轮询记忆 → B 会话验证。"""
    row: dict = {
        "id": case["id"],
        "name": case["name"],
        "note": case["note"],
        "session_a": case["session_a"],
        "session_b": case["session_b"],
        "thread_a": str(uuid.uuid4()),
        "thread_b": str(uuid.uuid4()),
        "passed": False,
        "missing": [],
        "error": "",
        "a_latency_s": 0.0,
        "b_latency_s": 0.0,
        "a_chars": 0,
        "b_chars": 0,
        "memory_count": 0,
        "memory_poll_s": 0.0,
        "memory_error": "",
        "memory_events": 0,
        "memory_preview": [],
        "answer_b": "",
    }
    errors: list[str] = []

    # 1) 会话 A：陈述偏好/约束，等任务结束
    turn_a = await run_turn(base, case["session_a"], row["thread_a"], timeout)
    row["a_latency_s"] = turn_a["latency_s"]
    row["a_chars"] = len(turn_a["result"])
    row["memory_events"] += turn_a["memory_events"]
    if turn_a["error"]:
        errors.append(f"会话A：{turn_a['error']}")

    # 2) 轮询 /api/memory，确认记忆已写入
    memories, poll_seconds, memory_error = await asyncio.to_thread(poll_memories, base, row["thread_a"])
    row["memory_count"] = len(memories)
    row["memory_poll_s"] = round(poll_seconds, 1)
    row["memory_error"] = memory_error
    row["memory_preview"] = preview_memories(memories)

    # 3) 会话 B：全新 thread_id，问依赖记忆的问题
    turn_b = await run_turn(base, case["session_b"], row["thread_b"], timeout)
    row["b_latency_s"] = turn_b["latency_s"]
    row["b_chars"] = len(turn_b["result"])
    row["answer_b"] = turn_b["result"]
    row["memory_events"] += turn_b["memory_events"]
    if turn_b["error"]:
        errors.append(f"会话B：{turn_b['error']}")

    # 4) 判分：B 的回答要体现记住 A 的偏好/约束
    matched, missing = score_answer(turn_b["result"], case)
    row["missing"] = missing
    row["error"] = "；".join(errors)
    row["passed"] = matched and not errors
    return row


def summarize(rows: list[dict], base: str, timeout: float) -> str:
    """控制台汇总（同时写进报告）。"""
    total = len(rows)
    passed = sum(row["passed"] for row in rows)
    memory_ok = sum(1 for row in rows if row["memory_count"] > 0)
    a_latency = [row["a_latency_s"] for row in rows]
    b_latency = [row["b_latency_s"] for row in rows]
    lines = [
        f"整体：{passed}/{total} 通过（{passed / total:.0%}）" if total else "没有可运行的用例",
        f"记忆写入：{memory_ok}/{total} 组在会话 A 后通过 /api/memory 查到记忆",
        f"会话 A 耗时：平均 {sum(a_latency) / total:.1f}s，最长 {max(a_latency):.1f}s",
        f"会话 B 耗时：平均 {sum(b_latency) / total:.1f}s，最长 {max(b_latency):.1f}s",
        f"单轮超时 {timeout:.0f}s，记忆轮询上限 {MEMORY_POLL_TIMEOUT:.0f}s，服务 {base}",
    ]
    return "\n".join(lines)


def write_report(rows: list[dict], summary: str, base: str, timeout: float, out_path: Path) -> None:
    """写 Markdown 报告（含逐用例明细与 B 的回答原文）。"""
    header = (
        "| 用例 | 名称 | 结果 | A耗时(s) | B耗时(s) | 记忆条数 | 记忆轮询(s) | 未命中 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )

    def _cell(value: str) -> str:
        return str(value).replace("|", "/").replace("\n", " ")

    body = "\n".join(
        "| {id} | {name} | {result} | {a:.1f} | {b:.1f} | {memory} | {poll:.1f} | {missing} |".format(
            id=row["id"],
            name=_cell(row["name"]),
            result="✅" if row["passed"] else "❌",
            a=row["a_latency_s"],
            b=row["b_latency_s"],
            memory=row["memory_count"],
            poll=row["memory_poll_s"],
            missing=_cell("；".join(row["missing"]) or row["error"] or "-"),
        )
        for row in rows
    )

    details: list[str] = []
    for row in rows:
        details.append(f"### {row['id']} {row['name']}")
        details.append("")
        details.append(f"- 期望：{row['note']}")
        details.append(f"- 会话 A（`{row['thread_a']}`）：{row['session_a']}")
        details.append(f"- 会话 B（`{row['thread_b']}`）：{row['session_b']}")
        details.append(
            f"- 耗时：A {row['a_latency_s']:.1f}s（{row['a_chars']} 字），B {row['b_latency_s']:.1f}s（{row['b_chars']} 字）"
        )
        details.append(
            f"- 记忆：{row['memory_count']} 条，轮询 {row['memory_poll_s']:.1f}s，"
            f"WS memory 事件 {row['memory_events']} 个"
        )
        if row["memory_preview"]:
            details.extend(f"  - {line}" for line in row["memory_preview"])
        if row["memory_error"]:
            details.append(f"- 记忆错误：{row['memory_error']}")
        if row["missing"]:
            details.append(f"- 未命中：{'；'.join(row['missing'])}")
        if row["error"]:
            details.append(f"- 执行错误：{row['error']}")
        answer = (row["answer_b"] or "").strip()
        if answer:
            details.append("")
            details.append("<details><summary>会话 B 回答原文</summary>")
            details.append("")
            details.append("```text")
            details.append(answer[:1500])
            details.append("```")
            details.append("")
            details.append("</details>")
        details.append("")

    markdown = (
        "# 跨会话长期记忆评测报告\n\n"
        f"- 运行时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- 服务地址：{base}\n"
        f"- 用例数：{len(rows)}（每组 = 会话 A 陈述 + 会话 B 新会话验证）\n"
        f"- 单轮超时：{timeout:.0f}s；记忆轮询上限：{MEMORY_POLL_TIMEOUT:.0f}s\n\n"
        "## 汇总\n\n"
        "```\n" + summary + "\n```\n\n"
        "## 明细\n\n" + header + "\n" + body + "\n\n"
        "## 逐用例说明\n\n" + "\n".join(details)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(description="MultiAgent-Search 跨会话长期记忆评测")
    parser.add_argument("--base", default="http://127.0.0.1:8010", help="服务地址")
    parser.add_argument("--timeout", type=float, default=240.0, help="单轮任务超时秒数")
    parser.add_argument("--only", default=None, help="只跑某个用例，如 mem-01")
    parser.add_argument("--out", default=str(REPORT_MD), help="报告输出路径（默认 evals/memory_report.md）")
    args = parser.parse_args()

    cases = CASES
    if args.only:
        cases = [case for case in CASES if case["id"] == args.only]
        if not cases:
            raise SystemExit(f"没有匹配的用例：{args.only}（可选：{', '.join(case['id'] for case in CASES)}）")

    wait_server(args.base)
    check_memory_api(args.base)

    rows: list[dict] = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['id']} {case['name']} ...", flush=True)
        row = await run_case(args.base, case, args.timeout)
        flag = "PASS" if row["passed"] else "FAIL"
        print(
            f"    {flag} A={row['a_latency_s']:.1f}s B={row['b_latency_s']:.1f}s "
            f"memory={row['memory_count']} poll={row['memory_poll_s']:.1f}s "
            f"{row['error'] or ('未命中：' + '；'.join(row['missing']) if row['missing'] else '')}",
            flush=True,
        )
        rows.append(row)

    summary = summarize(rows, args.base, args.timeout)
    print("\n" + summary)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    write_report(rows, summary, args.base, args.timeout, out_path)
    print(f"\n报告已写入：{out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    asyncio.run(main())
