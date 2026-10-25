# -*- coding: utf-8 -*-
"""把 docs/kb/ 下的业务文档灌进 RAGFlow 知识库（上传 → 解析 → 等完成）。

用法：
    python scripts/ragflow_upload_kb_docs.py
    python scripts/ragflow_upload_kb_docs.py --only 订单分析口径SOP.md
"""

import argparse
import os
import time
from pathlib import Path

from ragflow_sdk import RAGFlow

from _env import load_env, require

load_env()

KEY = require("RAGFLOW_API_KEY")
BASE = os.environ.get("RAGFLOW_API_URL", "http://127.0.0.1:9380")
DS_NAME = "multiagent_demo"
ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "docs" / "kb"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="只传某个文件")
    args = parser.parse_args()

    rag = RAGFlow(api_key=KEY, base_url=BASE)
    datasets = [d for d in rag.list_datasets() if d.name == DS_NAME]
    if not datasets:
        raise SystemExit(f"知识库不存在：{DS_NAME}")
    ds = datasets[0]

    files = sorted(p for p in KB_DIR.glob("*.md") if args.only is None or p.name == args.only)
    if not files:
        raise SystemExit(f"{KB_DIR} 下没有待上传的文档")

    existing = {d.name for d in ds.list_documents()}
    uploaded = []
    for path in files:
        if path.name in existing:
            print(f"跳过（已存在）：{path.name}")
            continue
        ds.upload_documents([{"display_name": path.name, "blob": path.read_bytes()}])
        uploaded.append(path.name)
        print(f"已上传：{path.name}（{path.stat().st_size} bytes）")

    if not uploaded:
        print("没有新增文档，直接触发重新解析")
        uploaded = [d.name for d in ds.list_documents()]

    doc_map = {d.name: d.id for d in ds.list_documents()}
    ids = [doc_map[name] for name in uploaded if name in doc_map]
    ds.async_parse_documents(ids)
    print(f"开始解析 {len(ids)} 个文档 ...")

    deadline = time.time() + 300
    while time.time() < deadline:
        docs = ds.list_documents()
        for doc in docs:
            print(f"  {doc.name}: run={getattr(doc, 'run', '?')} chunks={getattr(doc, 'chunk_count', '?')}")
        if all(str(getattr(d, "run", "")) in ("1", "DONE") for d in docs):
            print("全部解析完成")
            break
        time.sleep(8)
    else:
        raise SystemExit("解析超时，检查 RAGFlow 容器日志")


if __name__ == "__main__":
    main()
