#!/usr/bin/env python3
"""
OpenAI 相关配置解析工具。

统一管理 API Key 的读取逻辑，避免不同 day 出现多套实现。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_CONFIG_DIR = PROJECT_ROOT_DIR / ".local"
DEFAULT_OPENAI_API_KEY_FILE = LOCAL_CONFIG_DIR / "openai_api_key.txt"


def _read_secret_file(path: Path) -> Optional[str]:
    """读取密钥文件内容，兼容带 BOM 的 UTF-8 文本。"""
    if not path.is_file():
        return None

    value = path.read_text(encoding="utf-8-sig").strip()
    return value or None


def resolve_openai_api_key(explicit_api_key: Optional[str] = None) -> Optional[str]:
    """按统一优先级解析 OpenAI API Key。"""
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()

    env_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_api_key:
        return env_api_key

    custom_file = os.getenv("OPENAI_API_KEY_FILE", "").strip()
    candidate_files = [Path(custom_file).expanduser()] if custom_file else []
    candidate_files.append(DEFAULT_OPENAI_API_KEY_FILE)

    for path in candidate_files:
        api_key = _read_secret_file(path)
        if api_key:
            return api_key

    return None
