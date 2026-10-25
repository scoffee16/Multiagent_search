# -*- coding: utf-8 -*-
"""脚本共用的小工具：从项目根目录 .env 读取配置，避免把密钥硬编码进仓库。"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    """把项目根 .env 读进 os.environ（已存在的环境变量优先，不覆盖）。"""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require(name: str) -> str:
    """取必填环境变量，缺失时给出可操作的提示（而不是抛 KeyError）。"""
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"缺少环境变量 {name}：请在项目根 .env 里配置，或先 `$env:{name}='...'` 再运行"
        )
    return value


def get(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default
