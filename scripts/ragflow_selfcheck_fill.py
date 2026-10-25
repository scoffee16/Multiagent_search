# -*- coding: utf-8 -*-
"""自检知识库 + 把项目真实文档（README / study-plan）补进去，然后验证检索。"""
import os
import time
from pathlib import Path

from ragflow_sdk import RAGFlow

from _env import load_env, require

load_env()

KEY = require("RAGFLOW_API_KEY")
BASE = os.environ.get("RAGFLOW_API_URL", "http://127.0.0.1:9380")
DS_NAME = "multiagent_demo"
CHAT_NAME = "multiagent_demo_chat"
ROOT = Path(__file__).resolve().parents[1]

rag = RAGFlow(api_key=KEY, base_url=BASE)
ds = [d for d in rag.list_datasets() if d.name == DS_NAME][0]

print("=== 现状：知识库里有什么 ===")
for d in ds.list_documents():
    print(f"  - {d.name}  run={getattr(d,'run',None)} chunks={getattr(d,'chunk_count',None)}")
    for c in d.list_chunks():
        txt = (getattr(c, "content", "") or "").replace("\n", " ")[:90]
        print(f"      chunk> {txt}")

# 补入项目真实文档
for fn in ("README.md", "study-plan.md"):
    p = ROOT / fn
    if not p.exists():
        print("skip missing", fn)
        continue
    if any(d.name == fn for d in ds.list_documents()):
        print("already in kb:", fn)
        continue
    blob = p.read_bytes()
    ds.upload_documents([{"display_name": fn, "blob": blob}])
    print("uploaded:", fn, len(blob), "bytes")

docs = ds.list_documents()
new_ids = [d.id for d in docs if getattr(d, "run", "1") in ("0", "UNSTART", None)]
if new_ids:
    ds.async_parse_documents(new_ids)
    print("parsing:", len(new_ids))

deadline = time.time() + 240
while time.time() < deadline:
    docs = ds.list_documents()
    states = [(d.name, getattr(d, "run", None), round(float(getattr(d, "progress", 0) or 0), 2)) for d in docs]
    print("states:", states)
    if all(s[1] in ("1", 1) for s in states):
        break
    time.sleep(6)

print("=== 解析完成后的 chunk 数 ===")
for d in ds.list_documents():
    print(f"  - {d.name} run={getattr(d,'run',None)} chunks={getattr(d,'chunk_count',None)}")
