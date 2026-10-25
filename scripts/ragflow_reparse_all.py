# -*- coding: utf-8 -*-
"""用新向量模型重新解析全部文档，并等待完成。"""
import os
import time

from ragflow_sdk import RAGFlow

from _env import load_env, require

load_env()

rag = RAGFlow(
    api_key=require("RAGFLOW_API_KEY"),
    base_url=os.environ.get("RAGFLOW_API_URL", "http://127.0.0.1:9380"),
)
ds = [d for d in rag.list_datasets() if d.name == "multiagent_demo"][0]
print("embd:", getattr(ds, "embedding_model", "?"))

ids = [d.id for d in ds.list_documents()]
print("docs:", len(ids))
ds.async_parse_documents(ids)

deadline = time.time() + 300
while time.time() < deadline:
    docs = ds.list_documents()
    states = [(d.name, getattr(d, "run", None), round(float(getattr(d, "progress", 0) or 0), 2),
               getattr(d, "chunk_count", None)) for d in docs]
    print("states:", states)
    if all(s[1] in ("1", 1) for s in states):
        break
    time.sleep(8)

for d in ds.list_documents():
    if getattr(d, "run", None) == "FAIL":
        print("FAILMSG", d.name, (getattr(d, "progress_msg", "") or "")[-300:])
print("done")
