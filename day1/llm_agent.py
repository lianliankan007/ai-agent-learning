#!/usr/bin/env python3
"""
Day 1 LLM agent.

This module wraps a simple chat-completions call and records markdown logs.
The runtime strings use ASCII-first wording so the script prints cleanly in
Windows terminals that still default to GBK.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_markdown_logger import get_default_llm_logger
from utils.openai_config import resolve_openai_api_key

llm_logger = get_default_llm_logger()


class LLMAgent:
    """A minimal teaching-friendly LLM agent."""

    def __init__(
        self,
        name: str = "assistant",
        api_key: Optional[str] = None,
        base_url: str = "https://coding.dashscope.aliyuncs.com/v1",
        model: str = "qwen3.5-plus",
        temperature: float = 1.9,
        max_tokens: int = 1000,
    ):
        self.name = name
        self.api_key = resolve_openai_api_key(api_key)
        if not self.api_key:
            raise ValueError(
                f"[{name}] Missing api_key. Set OPENAI_API_KEY in the project .env file."
            )

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages: List[Dict[str, str]] = []

    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.messages)
        messages.append({"role": "user", "content": message})

        response = self._call_api(messages, stream)
        self.messages.append({"role": "user", "content": message})
        self.messages.append({"role": "assistant", "content": response})
        return response

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

        try:
            start_time = time.perf_counter()
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            response.raise_for_status()

            result = response.json()
            llm_logger.log_exchange(
                provider="dashscope-compatible",
                model=self.model,
                endpoint=url,
                request_payload=payload,
                request_headers=headers,
                response_payload=result,
                status_code=response.status_code,
                duration_ms=elapsed_ms,
                extra={"agent_name": self.name},
            )
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as exc:
            response = getattr(exc, "response", None)
            llm_logger.log_exchange(
                provider="dashscope-compatible",
                model=self.model,
                endpoint=url,
                request_payload=payload,
                request_headers=headers,
                response_payload=self._safe_json(response) if response is not None else None,
                status_code=response.status_code if response is not None else None,
                error=str(exc),
                extra={"agent_name": self.name},
            )
            raise Exception(f"API request failed: {exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            llm_logger.log_exchange(
                provider="dashscope-compatible",
                model=self.model,
                endpoint=url,
                request_payload=payload,
                request_headers=headers,
                error=f"response_parse_error: {exc}",
                extra={"agent_name": self.name},
            )
            raise Exception(f"Response parse failed: {exc}") from exc

    @staticmethod
    def _safe_json(response: Optional[requests.Response]) -> Any:
        if response is None:
            return None
        try:
            return response.json()
        except ValueError:
            return {"raw_text": response.text}

    def clear_history(self) -> None:
        self.messages = []

    def get_history(self) -> List[Dict[str, str]]:
        return self.messages.copy()

    def get_info(self) -> str:
        return f"[{self.name}] model={self.model}"
