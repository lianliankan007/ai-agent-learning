"""Shared Markdown logger for all LLM requests and responses."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


# 当日志里出现敏感信息时，用这个占位字符串替换，避免把真实密钥写入本地日志。
REDACTED = "***REDACTED***"
SENSITIVE_KEYS = {
    # 这些字段名一旦出现在请求头或请求体里，就会被自动脱敏。
    "api_key",
    "authorization",
    "x-api-key",
    "openai_api_key",
}


class MarkdownLLMLogger:
    """Append LLM request/response records to a local Markdown file."""

    def __init__(self, log_dir: str | os.PathLike[str] | None = None):
        # 默认把日志写到项目根目录下的 logs/llm/ 目录。
        # 这里用 Path 来处理路径，兼容性和可读性都更好。
        default_dir = Path(__file__).resolve().parents[1] / "logs" / "llm"
        root_dir = Path(log_dir) if log_dir else default_dir
        self.log_dir = root_dir
        # 如果目录不存在就自动创建，这样外部调用时不用手动建目录。
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_exchange(
        self,
        *,
        provider: str,
        model: str,
        request_payload: Any,
        response_payload: Any | None = None,
        endpoint: str | None = None,
        request_headers: dict[str, Any] | None = None,
        status_code: int | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Write one complete request/response exchange to today's Markdown log."""
        # 每次记录都按“当天日期”写入同一个 Markdown 文件，方便每天集中查看。
        now = datetime.now()
        log_path = self.log_dir / f"{now:%Y-%m-%d}.md"

        # summary 是这次调用的概要信息，适合放在 Markdown 顶部快速浏览。
        summary = {
            "time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "provider": provider,
            "model": model,
        }
        if endpoint:
            summary["endpoint"] = endpoint
        if status_code is not None:
            summary["status_code"] = status_code
        if duration_ms is not None:
            summary["duration_ms"] = round(duration_ms, 2)
        if error:
            summary["error"] = error
        if extra:
            # extra 允许业务代码额外补充信息。
            # 这里也会先做一次脱敏，避免自定义字段里带出敏感内容。
            summary.update(self._sanitize(extra))

        # parts 是最终要写入 Markdown 文件的各个片段。
        # 先组织成列表，最后统一 join，代码会更清晰。
        parts = [
            f"## {summary['time']}",
            "",
            self._format_summary(summary),
            "",
            "### Request",
            self._format_code_block(self._sanitize(request_payload)),
        ]

        if request_headers:
            # 请求头单独记录出来，调试鉴权、追踪 header 问题时很有帮助。
            parts.extend(
                [
                    "",
                    "### Request Headers",
                    self._format_code_block(self._sanitize(request_headers)),
                ]
            )

        if response_payload is not None:
            # 只有真的拿到了响应，才写入 Response 区块。
            parts.extend(
                [
                    "",
                    "### Response",
                    self._format_code_block(self._sanitize(response_payload)),
                ]
            )

        if error:
            # 如果调用失败，除了状态码，也会额外记录错误文本。
            parts.extend(["", "### Error", error])

        parts.append("\n---\n")

        # 使用追加模式 a，表示保留历史记录，在文件末尾继续写。
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(parts))

        return log_path

    def _format_summary(self, summary: dict[str, Any]) -> str:
        # 把概要信息格式化成 Markdown 列表，便于阅读。
        return "\n".join(f"- {key}: {value}" for key, value in summary.items())

    def _format_code_block(self, payload: Any) -> str:
        # 把 Python 对象转成漂亮的 JSON 字符串，再包成 Markdown 代码块。
        # ensure_ascii=False 可以让中文正常显示，不会变成 \uXXXX。
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        return f"```json\n{text}\n```"

    def _sanitize(self, payload: Any) -> Any:
        # 深拷贝一份数据，避免“为了写日志做脱敏”而影响原始业务数据。
        copied = deepcopy(payload)
        return self._sanitize_value(copied)

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            # 递归处理字典：遇到敏感字段就替换，普通字段继续向下检查。
            sanitized: dict[str, Any] = {}
            for key, inner_value in value.items():
                if key.lower() in SENSITIVE_KEYS:
                    sanitized[key] = REDACTED
                else:
                    sanitized[key] = self._sanitize_value(inner_value)
            return sanitized

        if isinstance(value, list):
            # 列表中的每个元素也可能是嵌套结构，所以要递归处理。
            return [self._sanitize_value(item) for item in value]

        if isinstance(value, tuple):
            # tuple 这里统一转成 list 处理，方便 JSON 序列化输出。
            return [self._sanitize_value(item) for item in value]

        # 基础类型直接返回，例如字符串、数字、布尔值等。
        return value


_DEFAULT_LOGGER: MarkdownLLMLogger | None = None


def get_default_llm_logger() -> MarkdownLLMLogger:
    """Return a shared logger instance for the whole project."""
    global _DEFAULT_LOGGER
    if _DEFAULT_LOGGER is None:
        # 用单例的方式复用 logger，避免项目里到处重复创建实例。
        _DEFAULT_LOGGER = MarkdownLLMLogger()
    return _DEFAULT_LOGGER
