# -*- coding: utf-8 -*-
"""走真实 HTTP + WebSocket 链路跑一次任务，把 task_result 的 markdown 落盘，供前端渲染验证。"""
import asyncio
import json
import pathlib
import time
import urllib.request

import websockets

BASE = "http://127.0.0.1:8010"
OUT = pathlib.Path(r"C:\Users\sco11\AppData\Local\Temp\task_result.md")
QUERY = "统计 orders 表里各品类的订单数和总金额"


def wait_up(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(BASE + "/api/files", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def submit():
    req = urllib.request.Request(
        BASE + "/api/task",
        data=json.dumps({"query": QUERY}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


async def listen(thread_id, timeout=420):
    url = f"ws://127.0.0.1:8010/ws/{thread_id}"
    t0 = time.time()
    async with websockets.connect(url, ping_interval=None) as ws:
        while time.time() - t0 < timeout:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                break
            d = json.loads(raw)
            if d.get("type") != "monitor_event":
                continue
            ev = d.get("event")
            msg = (d.get("message") or "")[:90]
            print(f"[{ev}] {msg}")
            if ev == "task_result":
                text = (d.get("data") or {}).get("result") or ""
                OUT.write_text(text, encoding="utf-8")
                print("RESULT_LEN=", len(text), "->", OUT)
                return True
            if ev == "error":
                print("ERROR event:", d)
                return False
    return False


if not wait_up():
    raise SystemExit("server not up")
res = submit()
print("task:", res)
asyncio.run(listen(res["thread_id"]))
