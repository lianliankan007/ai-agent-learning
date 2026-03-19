#!/usr/bin/env python3
"""Day 2 LLM lab: parameter experiments + chat context + retry wrapper."""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)


@dataclass
class RetryConfig:
    max_retries: int = 4
    base_delay: float = 1.0
    max_delay: float = 12.0
    jitter_ratio: float = 0.3
    timeout: float = 30.0


class LLMClient:
    def __init__(self, model: str, retry: RetryConfig):
        self.client = OpenAI(timeout=retry.timeout)
        self.model = model
        self.retry = retry

    def _calc_delay(self, attempt: int) -> float:
        exp = min(self.retry.max_delay, self.retry.base_delay * (2 ** (attempt - 1)))
        jitter = exp * self.retry.jitter_ratio * random.random()
        return exp + jitter

    def _is_retryable_status(self, status_code: int | None) -> bool:
        if status_code is None:
            return False
        return status_code == 429 or 500 <= status_code < 600

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        last_error: Exception | None = None

        for attempt in range(1, self.retry.max_retries + 1):
            start = time.perf_counter()
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_tokens,
                )
                elapsed = (time.perf_counter() - start) * 1000
                logging.info("request_ok attempt=%s elapsed_ms=%.1f", attempt, elapsed)
                return resp.output_text.strip()
            except (APIConnectionError, APITimeoutError, RateLimitError) as err:
                last_error = err
                retryable = True
                status_code = getattr(err, "status_code", None)
                err_name = err.__class__.__name__
            except APIStatusError as err:
                last_error = err
                status_code = err.status_code
                retryable = self._is_retryable_status(status_code)
                err_name = err.__class__.__name__
            except Exception as err:  # unknown error, fail fast
                raise RuntimeError(f"Unexpected error: {err}") from err

            elapsed = (time.perf_counter() - start) * 1000
            logging.warning(
                "request_fail attempt=%s retryable=%s status=%s error=%s elapsed_ms=%.1f",
                attempt,
                retryable,
                status_code,
                err_name,
                elapsed,
            )

            if not retryable or attempt == self.retry.max_retries:
                break

            delay = self._calc_delay(attempt)
            logging.info("retry_wait attempt=%s sleep_s=%.2f", attempt, delay)
            time.sleep(delay)

        raise RuntimeError(f"Request failed after retries: {last_error}") from last_error


def maybe_summarize_history(
    llm: LLMClient,
    history: list[dict[str, str]],
    max_messages: int,
) -> list[dict[str, str]]:
    if len(history) <= max_messages:
        return history

    # Keep system + latest turns; summarize older turns into one memory block.
    system_msgs = [m for m in history if m["role"] == "system"]
    non_system = [m for m in history if m["role"] != "system"]
    recent = non_system[-(max_messages - 2) :]
    old = non_system[: -(max_messages - 2)]

    if not old:
        return history

    summary_prompt = [
        {
            "role": "system",
            "content": "你是对话记录整理器。请把历史对话压缩为 6 条以内要点，保留用户偏好、约束、已确认结论。",
        },
        {
            "role": "user",
            "content": "请总结这段历史对话：\n\n" + "\n".join(
                f"{item['role']}: {item['content']}" for item in old
            ),
        },
    ]

    summary = llm.chat_completion(
        summary_prompt,
        temperature=0.2,
        top_p=1.0,
        max_tokens=300,
    )

    memory = {
        "role": "system",
        "content": "历史摘要记忆（由系统生成）:\n" + summary,
    }
    return system_msgs + [memory] + recent


def run_experiment(llm: LLMClient, prompt: str, max_tokens: int) -> None:
    settings: list[tuple[float, float, int]] = [
        (0.2, 1.0, max_tokens),
        (0.5, 1.0, max_tokens),
        (0.9, 1.0, max_tokens),
        (0.5, 0.5, max_tokens),
    ]

    print("\n=== 参数实验结果 ===")
    for idx, (temp, top_p, out_max) in enumerate(settings, start=1):
        messages = [
            {
                "role": "system",
                "content": "你是一个简洁的学习助教。",
            },
            {"role": "user", "content": prompt},
        ]
        answer = llm.chat_completion(
            messages=messages,
            temperature=temp,
            top_p=top_p,
            max_tokens=out_max,
        )
        print(f"\n--- Case {idx} | temperature={temp}, top_p={top_p}, max_tokens={out_max} ---")
        print(answer)
        print(f"[length={len(answer)} chars]")


def run_chat(llm: LLMClient, max_history: int, temperature: float, top_p: float, max_tokens: int) -> None:
    history: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "你是学习助手。回答要简洁、结构化，优先给可执行建议。",
        }
    ]

    print("进入多轮聊天，输入 /exit 退出，输入 /history 查看当前上下文长度。\n")

    while True:
        user_input = input("You> ").strip()
        if not user_input:
            continue
        if user_input == "/exit":
            print("Bye")
            return
        if user_input == "/history":
            print(f"当前消息条数: {len(history)}")
            continue

        history.append({"role": "user", "content": user_input})
        history = maybe_summarize_history(llm, history, max_messages=max_history)

        answer = llm.chat_completion(
            messages=history,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        print(f"Assistant> {answer}\n")
        history.append({"role": "assistant", "content": answer})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 2 LLM practice template")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")

    sub = parser.add_subparsers(dest="cmd", required=True)

    exp = sub.add_parser("experiment", help="Run parameter experiments")
    exp.add_argument(
        "--prompt",
        default="请分别用 3 个层次解释什么是反向传播：小白版、工程师版、数学版。",
        help="Prompt to test under different parameters",
    )

    sub.add_parser("chat", help="Run multi-turn chat")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    retry = RetryConfig(
        max_retries=args.max_retries,
        timeout=args.timeout,
    )
    llm = LLMClient(model=args.model, retry=retry)

    if args.cmd == "experiment":
        run_experiment(llm, prompt=args.prompt, max_tokens=args.max_tokens)
    elif args.cmd == "chat":
        run_chat(
            llm,
            max_history=args.max_history,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
