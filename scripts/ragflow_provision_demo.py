# -*- coding: utf-8 -*-
"""为 MultiAgent-Search 在本地 RAGFlow 里建一个演示知识库 + 聊天助手。

用项目环境跑： D:\\conda\\envs_dirs\\multiagent\\python.exe provision_demo.py
依赖：项目 .env 里的 RAGFLOW_API_URL / RAGFLOW_API_KEY（或环境变量）。
"""
import os
import sys
import time
from pathlib import Path

from ragflow_sdk import RAGFlow

BASE = os.getenv("RAGFLOW_API_URL", "http://127.0.0.1:9380")
KEY = os.getenv("RAGFLOW_API_KEY", "")

DS_NAME = "multiagent_demo"
CHAT_NAME = "multiagent_demo_chat"
DOC_NAME = "orders-demo.md"

DOC = """# deep_search.orders 演示数据说明

order_id | customer_name | product_name | category | price | order_date | status
--- | --- | --- | --- | --- | --- | ---
{rows}

统计口径：
- 数码电子类订单共 4 笔，金额最高的是 8999 元的笔记本电脑。
- 美妆护肤类订单 1 笔。
- 生活家电类订单 1 笔。
- 运动鞋服类订单 2 笔。

数据来源：项目根目录 sql/init.sql，随 docker compose 的 mysql 容器自动初始化。
"""

ROWS = """ORD2026001 | 张三 | iPhone 16 Pro | 数码电子 | 8999.00 | 2026-01-05 | 已完成
ORD2026002 | 李四 | 小米空气净化器 | 生活家电 | 1299.00 | 2026-01-08 | 已发货
ORD2026003 | 王五 | 雅诗兰黛小棕瓶 | 美妆护肤 | 850.00 | 2026-01-12 | 已完成
ORD2026004 | 赵六 | 耐克跑鞋 | 运动鞋服 | 699.00 | 2026-01-15 | 已完成
ORD2026005 | 孙七 | 华为 MateBook | 数码电子 | 6999.00 | 2026-02-02 | 已完成
ORD2026006 | 周八 | 戴森吹风机 | 生活家电 | 2999.00 | 2026-02-11 | 已发货
ORD2026007 | 吴九 | 阿迪达斯外套 | 运动鞋服 | 599.00 | 2026-02-20 | 已完成
ORD2026008 | 郑十 | 索尼耳机 | 数码电子 | 1999.00 | 2026-03-01 | 已完成"""


def main():
    if not KEY:
        print("ERROR: RAGFLOW_API_KEY 为空", file=sys.stderr)
        return 2
    rag = RAGFlow(api_key=KEY, base_url=BASE)
    print("base_url =", BASE)

    # 1) 知识库（注意：v0.23.1 的 list_datasets(name=...) 在名字不存在时会直接报“lacks permission”，
    #    所以这里先列全部再本地过滤）
    ds_list = [d for d in rag.list_datasets() if d.name == DS_NAME]
    if ds_list:
        ds = ds_list[0]
        print("dataset 已存在:", ds.id, ds.name)
    else:
        try:
            ds = rag.create_dataset(name=DS_NAME, description="MultiAgent-Search 演示知识库")
            print("dataset 已创建:", ds.id, ds.name)
        except Exception as e:
            print("create_dataset 失败:", repr(e)[:400])
            print("-> 通常是因为租户还没配置 embedding 模型（RAGFlow 设置里要先加模型供应商）")
            return 3

    # 2) 文档
    docs = [x for x in ds.list_documents() if x.name == DOC_NAME]
    if docs:
        doc = docs[0]
        print("document 已存在:", doc.id, doc.name)
    else:
        payload = [{"display_name": DOC_NAME, "blob": DOC.format(rows=ROWS).encode("utf-8")}]
        try:
            created = ds.upload_documents(payload)
            doc = created[0]
            print("document 已上传:", doc.id, doc.name)
        except Exception as e:
            print("upload_documents 失败:", repr(e)[:400])
            return 4

    # 3) 解析（不需要等它跑完，这里只触发 + 看状态）
    try:
        ds.async_parse_documents([doc.id])
        print("已触发解析")
    except Exception as e:
        print("async_parse_documents 失败:", repr(e)[:300])

    # 4) 聊天助手
    chats = [c for c in rag.list_chats() if c.name == CHAT_NAME]
    if chats:
        chat = chats[0]
        print("chat 已存在:", chat.id, chat.name)
    else:
        try:
            chat = rag.create_chat(name=CHAT_NAME, dataset_ids=[ds.id])
            print("chat 已创建:", chat.id, chat.name)
        except Exception as e:
            print("create_chat 失败:", repr(e)[:400])
            return 5

    # 5) 复核
    time.sleep(1)
    print("--- 复核 ---")
    print("datasets:", [(d.name, d.id) for d in rag.list_datasets()])
    print("chats:", [(c.name, c.id) for c in rag.list_chats()])
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())


