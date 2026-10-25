# -*- coding: utf-8 -*-
"""在 RAGFlow 容器内直接写入一条 API Token（等价于 POST /v1/system/new_token）。
用法（宿主机）:
    docker cp gen_api_key.py ragflow-ragflow-cpu-1:/tmp/gen_api_key.py
    docker exec ragflow-ragflow-cpu-1 sh -c 'cd /ragflow && .venv/bin/python /tmp/gen_api_key.py'
输出最后一行: API_KEY=ragflow-xxxx
"""
import os
import secrets
import sys
from datetime import datetime

from api.db.services.api_service import APITokenService
from api.db.services.user_service import UserService, UserTenantService

try:
    from api.utils import current_timestamp, datetime_format
except Exception:  # pragma: no cover
    def current_timestamp():
        return int(datetime.now().timestamp() * 1000)

    def datetime_format(dt):
        return dt.strftime("%Y-%m-%d %H:%M:%S")


EMAIL = os.environ.get("RAGFLOW_ADMIN_EMAIL", "admin@multiagent.local")
NAME = "multiagent-search"

users = UserService.query(email=EMAIL)
if not users:
    print("ERROR: user not found", file=sys.stderr)
    sys.exit(1)
user = users[0]

tenants = UserTenantService.query(user_id=user.id)
owners = [t for t in tenants if t.role == "owner"]
if not owners:
    print("ERROR: no owner tenant", file=sys.stderr)
    sys.exit(1)
tenant_id = owners[0].tenant_id

# 已有同名 token 就复用
existing = APITokenService.query(tenant_id=tenant_id)
for row in existing:
    if row.beta and len(row.token or "") > 10:
        print("EXISTING_COUNT=%d" % len(existing))

token = "ragflow-" + secrets.token_hex(16)
beta = secrets.token_hex(16)
obj = {
    "tenant_id": tenant_id,
    "token": token,
    "beta": beta,
    "create_time": current_timestamp(),
    "create_date": datetime_format(datetime.now()),
    "update_time": None,
    "update_date": None,
}
ok = APITokenService.save(**obj)
print("SAVE_OK=", ok)
print("TENANT=", tenant_id)
print("TOTAL_TOKENS=", len(APITokenService.query(tenant_id=tenant_id)))
print("API_KEY=" + token)
