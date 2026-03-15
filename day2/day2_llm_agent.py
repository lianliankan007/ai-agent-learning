#!/usr/bin/env python3
"""
Day 2 LLM Agent 模块
在 Day 1 基础上增加参数控制、重试和历史压缩。
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_markdown_logger import get_default_llm_logger

load_dotenv()

llm_logger = get_default_llm_logger()


class LLMAgent:
    """负责调用 API、维护历史，并在失败时按策略重试。"""

    def __init__(
        self,
        name: str = "Day2助手",
        api_key: Optional[str] = None,
        base_url: str = "https://coding.dashscope.aliyuncs.com/v1",
        model: str = "qwen3.5-plus",
        temperature: float = 0.4,
        top_p: float = 1.0,
        max_tokens: int = 800,
        timeout: int = 60,
        max_retries: int = 4,
        max_history_messages: int = 10,
    ):
        self.name = name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(f"[{name}] 请提供 api_key 或设置 OPENAI_API_KEY 环境变量")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_history_messages = max_history_messages
        self.messages: List[Dict[str, str]] = []

    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = self._build_messages(message=message, system_prompt=system_prompt)
        response = self._call_api(
            messages=messages,
            stream=stream,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p if top_p is not None else self.top_p,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )

        self.messages.append({"role": "user", "content": message})
        self.messages.append({"role": "assistant", "content": response})
        self._compress_history_if_needed(system_prompt=system_prompt)
        return response

    def run_parameter_experiment(
        self,
        prompt: str,
        system_prompt: Optional[str] = "你是简洁清晰的学习助教。",
    ) -> List[Dict[str, Any]]:
        settings = [
            {"temperature": 0.2, "top_p": 1.0, "max_tokens": 350},
            {"temperature": 0.5, "top_p": 1.0, "max_tokens": 350},
            {"temperature": 0.9, "top_p": 1.0, "max_tokens": 350},
            {"temperature": 0.5, "top_p": 0.5, "max_tokens": 350},
        ]

        results: List[Dict[str, Any]] = []
        for idx, item in enumerate(settings, start=1):
            answer = self.chat(
                message=prompt,
                system_prompt=system_prompt,
                temperature=item["temperature"],
                top_p=item["top_p"],
                max_tokens=item["max_tokens"],
            )
            results.append(
                {
                    "case": idx,
                    "temperature": item["temperature"],
                    "top_p": item["top_p"],
                    "max_tokens": item["max_tokens"],
                    "answer": answer,
                    "length": len(answer),
                }
            )
        return results

    def _build_messages(self, message: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.messages)
        messages.append({"role": "user", "content": message})
        return messages

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: float = 0.4,
        top_p: float = 1.0,
        max_tokens: int = 800,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.perf_counter()
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                if self._should_retry_status(response.status_code):
                    raise requests.exceptions.HTTPError(
                        f"status={response.status_code}",
                        response=response,
                    )

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
                    extra={"agent_name": self.name, "attempt": attempt},
                )
                return result["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as exc:
                last_error = exc
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
                retryable = self._is_retryable_exception(exc, status_code)
                llm_logger.log_exchange(
                    provider="dashscope-compatible",
                    model=self.model,
                    endpoint=url,
                    request_payload=payload,
                    request_headers=headers,
                    response_payload=self._safe_json(response) if response is not None else None,
                    status_code=status_code,
                    error=str(exc),
                    extra={"agent_name": self.name, "attempt": attempt, "retryable": retryable},
                )

                if (not retryable) or (attempt >= self.max_retries):
                    break

                delay = self._compute_retry_delay(attempt)
                print(
                    f"[{self.name}] 第 {attempt} 次失败，{delay:.2f}s 后重试 "
                    f"(status={status_code}, error={type(exc).__name__})"
                )
                time.sleep(delay)
            except (KeyError, IndexError, ValueError) as exc:
                llm_logger.log_exchange(
                    provider="dashscope-compatible",
                    model=self.model,
                    endpoint=url,
                    request_payload=payload,
                    request_headers=headers,
                    error=f"response_parse_error: {exc}",
                    extra={"agent_name": self.name, "attempt": attempt},
                )
                raise Exception(f"解析响应失败: {exc}") from exc

        raise Exception(f"API 调用失败(重试后): {last_error}") from last_error

    def _is_retryable_exception(
        self,
        error: requests.exceptions.RequestException,
        status_code: Optional[int],
    ) -> bool:
        if isinstance(error, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
            return True
        if status_code is None:
            return False
        return self._should_retry_status(status_code)

    @staticmethod
    def _should_retry_status(status_code: int) -> bool:
        return status_code == 429 or 500 <= status_code < 600

    @staticmethod
    def _compute_retry_delay(
        attempt: int,
        base: float = 1.0,
        jitter: float = 0.3,
        max_delay: float = 12.0,
    ) -> float:
        delay = min(max_delay, base * (2 ** (attempt - 1)))
        return delay + delay * jitter * random.random()

    @staticmethod
    def _safe_json(response: Optional[requests.Response]) -> Any:
        if response is None:
            return None
        try:
            return response.json()
        except ValueError:
            return {"raw_text": response.text}

    def _compress_history_if_needed(self, system_prompt: Optional[str]) -> None:
        if len(self.messages) <= self.max_history_messages:
            return

        keep_count = max(4, self.max_history_messages // 2)
        old_part = self.messages[:-keep_count]
        recent_part = self.messages[-keep_count:]
        plain_text = "\n".join(f"{item['role']}: {item['content']}" for item in old_part)
        summary_prompt = (
            "请把以下历史对话压缩为不超过 6 条要点，保留用户偏好、约束条件、已确定结论：\n\n"
            + plain_text
        )

        try:
            summary = self._call_api(
                messages=[
                    {"role": "system", "content": "你是对话记录压缩器。"},
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.2,
                top_p=1.0,
                max_tokens=300,
            )
        except Exception:
            self.messages = recent_part
            return

        rebuilt: List[Dict[str, str]] = []
        if system_prompt:
            rebuilt.append({"role": "system", "content": system_prompt})
        rebuilt.append({"role": "assistant", "content": "[历史摘要]\n" + summary})
        rebuilt.extend(recent_part)
        self.messages = [item for item in rebuilt if item["role"] in ["user", "assistant"]]

    def clear_history(self) -> None:
        self.messages = []

    def get_history(self) -> List[Dict[str, str]]:
        return self.messages.copy()

    def set_sampling_params(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if max_tokens is not None:
            self.max_tokens = max_tokens

    def get_info(self) -> str:
        return (
            f"[{self.name}] model={self.model}, temp={self.temperature}, "
            f"top_p={self.top_p}, max_tokens={self.max_tokens}"
        )
