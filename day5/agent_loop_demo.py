#!/usr/bin/env python3
"""
Day5: Agent Loop 入门示例

这个示例的重点不是“做出最强 Agent”，而是帮助你看懂：
1. LLM 如何决定下一步动作
2. 程序如何驱动工具调用循环
3. 为什么要设置最大步数，防止死循环
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import requests
from dotenv import load_dotenv

from utils.openai_config import resolve_openai_api_key

load_dotenv()


def get_current_time(city: str) -> str:
    """返回当前时间。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{city}当前时间是 {now}"


def get_weather(city: str) -> str:
    """返回内置天气假数据，方便先理解 Loop 流程。"""
    weather_map = {
        "北京": "北京天气晴，22度，微风",
        "上海": "上海天气多云，24度，东南风",
        "杭州": "杭州天气小雨，19度，东北风",
        "深圳": "深圳天气晴，28度，南风",
    }
    return weather_map.get(city, f"{city}暂无内置天气数据，请告诉用户当前无法查询该城市天气")


def build_tool_definitions() -> List[Dict[str, Any]]:
    """按照 OpenAI 兼容接口要求，定义可调用工具。"""
    return [
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
                            "description": "需要查询时间的城市，例如北京、上海",
                        }
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询指定城市当前天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "需要查询天气的城市，例如北京、上海",
                        }
                    },
                    "required": ["city"],
                },
            },
        },
    ]


class AgentLoopDemo:
    """一个教学版的最小 Agent Loop。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://coding.dashscope.aliyuncs.com/v1",
        model: str = "qwen3.5-plus",
        temperature: float = 0.2,
        max_tokens: int = 500,
        max_steps: int = 3,
    ):
        self.api_key = resolve_openai_api_key(api_key)
        if not self.api_key:
            raise ValueError(
                "请提供 api_key、设置 OPENAI_API_KEY，"
                "或在 .local/openai_api_key.txt 中写入 API Key"
            )

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_steps = max_steps

        self.messages: List[Dict[str, Any]] = []
        self.tool_definitions = build_tool_definitions()
        self.tool_registry: Dict[str, Callable[..., str]] = {
            "get_current_time": get_current_time,
            "get_weather": get_weather,
        }

        self.system_prompt = (
            "你是一个会使用工具解决问题的 AI Agent。"
            "当用户问题需要时间或天气信息时，请优先调用工具，而不是自己编造。"
            "你可以按需要连续调用多个工具。"
            "当信息足够时，再输出最终答案。"
        )

    def clear_history(self) -> None:
        self.messages = []

    def run(self, user_message: str) -> str:
        """执行一次完整的 Agent Loop。"""
        working_messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        working_messages.extend(self.messages)
        working_messages.append({"role": "user", "content": user_message})

        print(f"\n[开始任务] 用户问题: {user_message}")

        for step in range(1, self.max_steps + 1):
            print(f"\n[Step {step}] 正在请求模型判断下一步动作...")
            assistant_message = self._call_api(working_messages, self.tool_definitions)
            working_messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                final_answer = assistant_message.get("content", "").strip()
                print(f"[Step {step}] 模型决策: 输出最终答案")
                print(f"[Step {step}] 最终答案: {final_answer}")

                self.messages.append({"role": "user", "content": user_message})
                self.messages.append({"role": "assistant", "content": final_answer})
                return final_answer

            for tool_call in tool_calls:
                function_info = tool_call["function"]
                function_name = function_info["name"]
                arguments_text = function_info.get("arguments", "{}")
                arguments = json.loads(arguments_text)

                print(f"[Step {step}] 模型决策: 调用 {function_name}")
                print(f"[Step {step}] 工具参数: {arguments}")

                tool_result = self._execute_tool(function_name, arguments)
                print(f"[Step {step}] 工具结果: {tool_result}")

                working_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": tool_result,
                    }
                )

        fail_message = (
            f"已达到最大执行步数 {self.max_steps}，为避免死循环，Agent 已停止。"
            "请检查 prompt、工具设计或增加退出条件。"
        )
        print(f"\n[停止执行] {fail_message}")
        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": fail_message})
        return fail_message

    def _execute_tool(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """执行本地工具。"""
        tool_func = self.tool_registry.get(function_name)
        if tool_func is None:
            return f"未找到工具: {function_name}"

        try:
            return tool_func(**arguments)
        except TypeError as exc:
            return f"工具参数错误: {exc}"
        except Exception as exc:  # pragma: no cover
            return f"工具执行失败: {exc}"

    def _call_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """调用 OpenAI 兼容接口，让模型决定下一步。"""
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

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]


class AgentLoopRunner:
    """命令行交互入口。"""

    def __init__(self, agent: AgentLoopDemo):
        self.agent = agent

    def run(self) -> None:
        print("=" * 64)
        print("Day5 Agent Loop 演示")
        print("=" * 64)
        print("\n可用命令:")
        print("  clear              - 清空对话历史")
        print("  max-steps <num>    - 设置最大执行步数")
        print("  quit/exit          - 退出程序")
        print("  <任意文字>         - 让 Agent 开始执行任务\n")

        while True:
            try:
                user_input = input("[agent-loop]> ").strip()
                if not user_input:
                    continue

                lower_text = user_input.lower()
                if lower_text in {"quit", "exit"}:
                    print("\n再见!")
                    break

                if lower_text == "clear":
                    self.agent.clear_history()
                    print("对话历史已清空\n")
                    continue

                if lower_text.startswith("max-steps "):
                    new_value = int(user_input.split(" ", 1)[1].strip())
                    self.agent.max_steps = new_value
                    print(f"最大执行步数已更新为: {new_value}\n")
                    continue

                answer = self.agent.run(user_input)
                print(f"\nAI: {answer}\n")
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as exc:
                print(f"\n错误: {exc}\n")


def main() -> None:
    agent = AgentLoopDemo(
        api_key=None,
        base_url=os.getenv("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"),
        model=os.getenv("OPENAI_MODEL", "qwen3.5-plus"),
    )
    runner = AgentLoopRunner(agent)
    runner.run()


if __name__ == "__main__":
    main()
