#!/usr/bin/env python3
"""
day3: 基于 day1 风格的 function calling 示例

运行前请先设置环境变量:
  set OPENAI_API_KEY=your_api_key
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_markdown_logger import get_default_llm_logger
from utils.openai_config import resolve_openai_api_key

load_dotenv()

llm_logger = get_default_llm_logger()


class FunctionCallingAgent:
    """使用兼容 OpenAI 的接口演示 function calling。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://coding.dashscope.aliyuncs.com/v1",
        model: str = "qwen3.5-plus",
        temperature: float = 0.2,
        max_tokens: int = 100,
    ):
        self.api_key = resolve_openai_api_key(api_key)
        if not self.api_key:
            raise ValueError("请提供 api_key 或设置 OPENAI_API_KEY 环境变量")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages: List[Dict[str, Any]] = []
        self.tool_registry: Dict[str, Callable[..., str]] = {}
        self.tool_definitions: List[Dict[str, Any]] = []

    def register_tool(
        self, name: str, func: Callable[..., str], tool_definition: Dict[str, Any]
    ) -> None:
        self.tool_registry[name] = func
        self.tool_definitions.append(tool_definition)

    def clear_history(self) -> None:
        self.messages = []

    def list_tools(self) -> None:
        print("\n可用 Functions:")
        for tool in self.tool_definitions:
            function_info = tool["function"]
            print(f"  - {function_info['name']}: {function_info['description']}")
        print()

    def chat_with_functions(
        self,
        user_message: str,
        system_prompt: str = "你是一个会在必要时主动调用工具解决问题的助手。",
    ) -> str:
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.messages)
        messages.append({"role": "user", "content": user_message})

        while True:
            response_message = self._call_api(messages, tools=self.tool_definitions)
            messages.append(response_message)

            tool_calls = response_message.get("tool_calls") or []
            if not tool_calls:
                final_content = response_message.get("content", "")
                self.messages.append({"role": "user", "content": user_message})
                self.messages.append({"role": "assistant", "content": final_content})
                return final_content

            for tool_call in tool_calls:
                function_info = tool_call["function"]
                function_name = function_info["name"]
                arguments_text = function_info.get("arguments", "{}")
                arguments = json.loads(arguments_text)
                print(f"调用函数: {function_name}({arguments})")

                if function_name not in self.tool_registry:
                    tool_result = f"未找到函数: {function_name}"
                else:
                    tool_result = self.tool_registry[function_name](**arguments)
                print(f"函数结果: {tool_result}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": tool_result,
                    }
                )

    def _call_api(
        self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

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
                extra={"agent_name": "FunctionCallingAgent"},
            )
            return result["choices"][0]["message"]
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
                extra={"agent_name": "FunctionCallingAgent"},
            )
            raise Exception(f"API 调用失败: {exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            llm_logger.log_exchange(
                provider="dashscope-compatible",
                model=self.model,
                endpoint=url,
                request_payload=payload,
                request_headers=headers,
                error=f"response_parse_error: {exc}",
                extra={"agent_name": "FunctionCallingAgent"},
            )
            raise Exception(f"解析响应失败: {exc}") from exc

    @staticmethod
    def _safe_json(response: Optional[requests.Response]) -> Any:
        if response is None:
            return None
        try:
            return response.json()
        except ValueError:
            return {"raw_text": response.text}


def get_weather(city: str) -> str:
    weather_map = {
        "北京": "晴，18°C，东北风 2 级",
        "上海": "多云，22°C，东南风 3 级",
        "杭州": "小雨，20°C，东北风 2 级",
        "深圳": "晴，27°C，南风 3 级",
    }
    return weather_map.get(city, f"{city} 暂无内置天气数据，你可以返回“未查询到天气”。")


def get_current_time(city: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{city} 当前时间: {now}"


def build_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "需要查询天气的城市名，比如北京、上海",
                        }
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "查询指定城市当前时间",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "需要查询时间的城市名，比如北京、上海",
                        }
                    },
                    "required": ["city"],
                },
            },
        },
    ]


class FunctionCallingRunner:
    def __init__(self, agent: FunctionCallingAgent):
        self.agent = agent
        self.system_prompt = "你是一个会在必要时主动调用工具解决问题的助手。"

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
        print("system prompt 已更新")

    def run(self) -> None:
        print("=" * 60)
        print("Day3 Function Calling 对话系统")
        print("=" * 60)
        print("\n可用命令:")
        print("  list-tools         - 列出所有可用函数")
        print("  clear              - 清空对话历史")
        print("  prompt <text>      - 设置系统提示词")
        print("  temperature <num>  - 设置模型 temperature")
        print("  quit/exit          - 退出程序")
        print("  <任意文字>         - 与当前 Agent 对话\n")

        while True:
            try:
                user_input = input("[function-agent]> ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit"]:
                    print("\n再见!")
                    break

                if user_input.lower() == "list-tools":
                    self.agent.list_tools()
                    continue

                if user_input.lower() == "clear":
                    self.agent.clear_history()
                    print("对话历史已清空")
                    continue

                if user_input.lower().startswith("prompt "):
                    self.set_system_prompt(user_input[7:].strip())
                    continue

                if user_input.lower().startswith("temperature "):
                    new_temperature = float(user_input[12:].strip())
                    self.agent.temperature = new_temperature
                    print(f"temperature: {new_temperature}")
                    continue

                print("\n[function-agent] 思考中...")
                answer = self.agent.chat_with_functions(
                    user_input,
                    system_prompt=self.system_prompt,
                )
                print(f"AI: {answer}\n")
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as exc:
                print(f"\n错误: {exc}\n")


def main() -> None:
    agent = FunctionCallingAgent()
    for tool in build_tools():
        function_name = tool["function"]["name"]
        if function_name == "get_weather":
            agent.register_tool(function_name, get_weather, tool)
        elif function_name == "get_current_time":
            agent.register_tool(function_name, get_current_time, tool)

    runner = FunctionCallingRunner(agent)
    runner.run()


if __name__ == "__main__":
    main()
