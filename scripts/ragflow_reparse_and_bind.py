# -*- coding: utf-8 -*-
"""重新解析演示文档 -> 把知识库绑到助手 -> 用项目工具问一个只有文档里才有答案的问题。"""
import os
import time

from ragflow_sdk import RAGFlow

from _env import load_env, require

load_env()

KEY = require("RAGFLOW_API_KEY")
BASE = os.environ.get("RAGFLOW_API_URL", "http://127.0.0.1:9380")
DS_NAME = "multiagent_demo"
CHAT_NAME = "multiagent_demo_chat"

rag = RAGFlow(api_key=KEY, base_url=BASE)
ds = [d for d in rag.list_datasets() if d.name == DS_NAME][0]
print("dataset:", ds.id, "embd:", getattr(ds, "embedding_model", "?"))

docs = ds.list_documents()
doc = docs[0]
print("doc:", doc.id, doc.name, "run=", getattr(doc, "run", None))

if getattr(doc, "run", None) != "1":
    ds.async_parse_documents([doc.id])
    print("parse triggered")

deadline = time.time() + 240
last = None
while time.time() < deadline:
    d2 = [x for x in ds.list_documents() if x.id == doc.id][0]
    state = (getattr(d2, "run", None), getattr(d2, "progress", None), getattr(d2, "chunk_count", None))
    if state != last:
        print("state:", state)
        last = state
    if state[0] in ("1", 1) and float(state[1] or 0) >= 1.0:
        break
    if state[0] == "FAIL":
        print("PARSE_FAIL:", getattr(d2, "progress_msg", "")[:400])
        break
    time.sleep(5)

d3 = [x for x in ds.list_documents() if x.id == doc.id][0]
print("final run=", getattr(d3, "run", None), "chunks=", getattr(d3, "chunk_count", None))
if getattr(d3, "chunk_count", 0) in (0, None):
    raise SystemExit("解析没产出 chunk，先停下")

# 重建助手并绑定知识库
for c in rag.list_chats():
    try:
        rag.delete_chats(ids=[c.id])
    except Exception as e:
        print("del chat fail:", repr(e)[:120])
chat = rag.create_chat(name=CHAT_NAME, dataset_ids=[ds.id])
print("chat:", chat.id, chat.name)

# 用项目工具提问（只有文档里才有这条答案）
from tools.ragflow_tools import create_ask_delete, get_assistant_list  # noqa: E402

print("--- get_assistant_list ---")
print(get_assistant_list.invoke({}))
print("--- create_ask_delete ---")
print(create_ask_delete.invoke({"chat_name": CHAT_NAME, "question": "数码电子类订单有几笔？金额是多少？"}))
