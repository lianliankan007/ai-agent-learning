#!/usr/bin/env python3
"""
OpenAI ???????

????????????? `.env`???? `OPENAI_API_KEY`?
??????????????????????????????
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_FILE_PATH = PROJECT_ROOT_DIR / ".env"


def load_project_env() -> None:
    """????????? `.env` ???"""
    if not ENV_FILE_PATH.is_file():
        return

    for raw_line in ENV_FILE_PATH.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        # ?????????????????????????
        if key and key not in os.environ:
            os.environ[key] = value


load_project_env()


def resolve_openai_api_key(explicit_api_key: Optional[str] = None) -> Optional[str]:
    """???????? OpenAI API Key?"""
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()

    env_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_api_key:
        return env_api_key

    return None
