#!/usr/bin/env python3
"""
Day 2 LLM Agent 模块
在 Day 1 基础上扩展：
1) 参数控制（temperature / top_p / max_tokens）
2) 多轮上下文管理（窗口 + 摘要）
3) 错误处理与重试（网络异常、限流、5xx）
"""

import os
import random
import time
from typing import Optional, List, Dict, Any

import requests


class LLMAgent:
    """LLM Agent 类，负责调用 API、维护历史、处理重试。"""

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
        """发送消息并返回回复。"""
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

        # 对话过长时压缩
        self._compress_history_if_needed(system_prompt=system_prompt)
        return response

    def run_parameter_experiment(
        self,
        prompt: str,
        system_prompt: Optional[str] = "你是简洁清晰的学习助教。",
    ) -> List[Dict[str, Any]]:
        """固定同一 prompt，比较不同参数效果。"""
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
        """构建发送给模型的消息列表。"""
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
        """调用 API，带重试。"""
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
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if self._should_retry_status(response.status_code):
                    raise requests.exceptions.HTTPError(
                        f"status={response.status_code}",
                        response=response,
                    )

                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]

            except requests.exceptions.RequestException as e:
                last_error = e
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                retryable = self._is_retryable_exception(e, status_code)

                if (not retryable) or (attempt >= self.max_retries):
                    break

                delay = self._compute_retry_delay(attempt)
                print(
                    f"[{self.name}] 第 {attempt} 次失败，{delay:.2f}s 后重试 "
                    f"(status={status_code}, error={type(e).__name__})"
                )
                time.sleep(delay)

            except (KeyError, IndexError, ValueError) as e:
                raise Exception(f"解析响应失败: {e}")

        raise Exception(f"API 调用失败(重试后): {last_error}")

    def _is_retryable_exception(
        self,
        error: requests.exceptions.RequestException,
        status_code: Optional[int],
    ) -> bool:
        """判断异常是否可重试。"""
        if isinstance(
            error,
            (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ),
        ):
            return True

        if status_code is None:
            return False

        return self._should_retry_status(status_code)

    @staticmethod
    def _should_retry_status(status_code: int) -> bool:
        """429 和 5xx 可重试。"""
        return status_code == 429 or 500 <= status_code < 600

    @staticmethod
    def _compute_retry_delay(attempt: int, base: float = 1.0, jitter: float = 0.3, max_delay: float = 12.0) -> float:
        """指数退避 + jitter。"""
        delay = min(max_delay, base * (2 ** (attempt - 1)))
        return delay + delay * jitter * random.random()

    def _compress_history_if_needed(self, system_prompt: Optional[str]) -> None:
        """历史超过阈值时，压缩早期消息。"""
        if len(self.messages) <= self.max_history_messages:
            return

        keep_count = max(4, self.max_history_messages // 2)
        old_part = self.messages[:-keep_count]
        recent_part = self.messages[-keep_count:]

        plain_text = "\n".join([f"{m['role']}: {m['content']}" for m in old_part])
        summary_prompt = (
            "请把以下历史对话压缩为不超过 6 条要点，保留用户偏好、约束条件、已确定结论：\n\n"
            + plain_text
        )

        try:
            summary = self._call_api(
                messages=[
                    {
                        "role": "system",
                        "content": "你是对话记录压缩器。",
                    },
                    {
                        "role": "user",
                        "content": summary_prompt,
                    },
                ],
                temperature=0.2,
                top_p=1.0,
                max_tokens=300,
            )
        except Exception:
            # 摘要失败时不阻塞主流程，直接截断
            self.messages = recent_part
            return

        memory_msg = {
            "role": "assistant",
            "content": "[历史摘要]\n" + summary,
        }

        rebuilt: List[Dict[str, str]] = []
        if system_prompt:
            rebuilt.append({"role": "system", "content": system_prompt})
        rebuilt.append(memory_msg)
        rebuilt.extend(recent_part)

        # self.messages 只保存 user/assistant，避免 system 重复，过滤掉可能插入的 system
        self.messages = [m for m in rebuilt if m["role"] in ["user", "assistant"]]

    def clear_history(self) -> None:
        self.messages = []

    def get_history(self) -> List[Dict[str, str]]:
        return self.messages.copy()

    def set_sampling_params(self, temperature: Optional[float] = None, top_p: Optional[float] = None, max_tokens: Optional[int] = None) -> None:
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
