"""Shared Markdown logger for all LLM requests and responses."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


REDACTED = "***REDACTED***"
SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "x-api-key",
    "openai_api_key",
}


class MarkdownLLMLogger:
    """Append LLM request/response records to a local Markdown file."""

    def __init__(self, log_dir: str | os.PathLike[str] | None = None):
        default_dir = Path(__file__).resolve().parents[1] / "logs" / "llm"
        root_dir = Path(log_dir) if log_dir else default_dir
        self.log_dir = root_dir
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
        now = datetime.now()
        log_path = self.log_dir / f"{now:%Y-%m-%d}.md"

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
            summary.update(self._sanitize(extra))

        parts = [
            f"## {summary['time']}",
            "",
            self._format_summary(summary),
            "",
            "### Request",
            self._format_code_block(self._sanitize(request_payload)),
        ]

        if request_headers:
            parts.extend(
                [
                    "",
                    "### Request Headers",
                    self._format_code_block(self._sanitize(request_headers)),
                ]
            )

        if response_payload is not None:
            parts.extend(
                [
                    "",
                    "### Response",
                    self._format_code_block(self._sanitize(response_payload)),
                ]
            )

        if error:
            parts.extend(["", "### Error", error])

        parts.append("\n---\n")

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(parts))

        return log_path

    def _format_summary(self, summary: dict[str, Any]) -> str:
        return "\n".join(f"- {key}: {value}" for key, value in summary.items())

    def _format_code_block(self, payload: Any) -> str:
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        return f"```json\n{text}\n```"

    def _sanitize(self, payload: Any) -> Any:
        copied = deepcopy(payload)
        return self._sanitize_value(copied)

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, inner_value in value.items():
                if key.lower() in SENSITIVE_KEYS:
                    sanitized[key] = REDACTED
                else:
                    sanitized[key] = self._sanitize_value(inner_value)
            return sanitized

        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value]

        if isinstance(value, tuple):
            return [self._sanitize_value(item) for item in value]

        return value


_DEFAULT_LOGGER: MarkdownLLMLogger | None = None


def get_default_llm_logger() -> MarkdownLLMLogger:
    """Return a shared logger instance for the whole project."""
    global _DEFAULT_LOGGER
    if _DEFAULT_LOGGER is None:
        _DEFAULT_LOGGER = MarkdownLLMLogger()
    return _DEFAULT_LOGGER
