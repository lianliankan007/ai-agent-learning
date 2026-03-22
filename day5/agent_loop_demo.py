#!/usr/bin/env python3
"""
Day5: Agent Loop teaching demo.

This version includes:
1. A longer Agent Loop with 6 chained function calls
2. A fixed workflow demo for comparison
3. LLM markdown logging for each API call
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_markdown_logger import get_default_llm_logger
from utils.openai_config import resolve_openai_api_key

llm_logger = get_default_llm_logger()


def get_current_location(user_name: str) -> str:
    """Return a built-in location for the demo user."""
    location_map = {
        "小明": "北京",
        "小红": "上海",
        "alice": "杭州",
        "bob": "深圳",
    }
    city = location_map.get(user_name.strip().lower())
    if city is None:
        city = location_map.get(user_name.strip(), "北京")
    return f"{user_name}当前所在城市是 {city}"


def get_current_time(city: str) -> str:
    """Return current local time text for a city."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{city}当前时间是 {now}"


def get_weather_by_context(city: str, current_time: str) -> str:
    """Return built-in weather using city and time context."""
    weather_map = {
        "北京": "晴，22度，微风",
        "上海": "多云，24度，东南风",
        "杭州": "小雨，19度，东北风",
        "深圳": "晴，28度，南风",
    }
    weather = weather_map.get(city, "阴，20度，微风")
    return f"基于城市 {city} 和时间 {current_time}，当前天气是 {weather}"


def get_clothing_advice(weather: str, current_time: str) -> str:
    """Return clothing suggestion based on weather and time."""
    if "小雨" in weather:
        return f"根据天气 {weather} 和时间 {current_time}，建议穿薄外套、长裤，并带伞"
    if "28度" in weather or ("晴" in weather and "深圳" in weather):
        return f"根据天气 {weather} 和时间 {current_time}，建议穿短袖、薄裤，注意防晒"
    if "24度" in weather:
        return f"根据天气 {weather} 和时间 {current_time}，建议穿短袖加轻薄外套"
    return f"根据天气 {weather} 和时间 {current_time}，建议穿短袖、长裤，早晚可加一件薄外套"


def estimate_clothing_layers(clothing_advice: str) -> str:
    """Estimate number of clothing layers from advice."""
    if "薄外套" in clothing_advice and "短袖" in clothing_advice:
        layers = 2
    elif "薄外套" in clothing_advice:
        layers = 2
    else:
        layers = 1
    return f"根据穿衣建议“{clothing_advice}”，预计今天需要 {layers} 层衣物"


def get_clothing_weight(clothing_advice: str, layer_count: str) -> str:
    """Estimate clothing weight from advice and layers."""
    weight = 350
    if "薄外套" in clothing_advice:
        weight += 280
    if "长裤" in clothing_advice:
        weight += 220
    if "带伞" in clothing_advice:
        weight += 300
    if "短袖" in clothing_advice:
        weight += 120
    if "薄裤" in clothing_advice:
        weight += 180
    if "2 层" in layer_count:
        weight += 80
    return f"根据穿衣建议和层数估算，今天整套衣物重量约为 {weight} 克"


def build_tool_definitions() -> List[Dict[str, Any]]:
    """Define tools in OpenAI-compatible format."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_current_location",
                "description": "先查询用户当前所在城市，适合作为整个链式任务的第一步",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_name": {
                            "type": "string",
                            "description": "用户名字，例如小明、小红、alice",
                        }
                    },
                    "required": ["user_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "根据城市查询当前时间，通常在拿到位置后调用",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名，例如北京、上海",
                        }
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather_by_context",
                "description": "根据城市和当前时间查询天气，应该在位置和时间之后调用",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名",
                        },
                        "current_time": {
                            "type": "string",
                            "description": "上一步工具返回的当前时间文本",
                        },
                    },
                    "required": ["city", "current_time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_clothing_advice",
                "description": "根据天气和时间给出穿衣建议，应该在天气之后调用",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "weather": {
                            "type": "string",
                            "description": "上一步工具返回的天气文本",
                        },
                        "current_time": {
                            "type": "string",
                            "description": "当前时间文本",
                        },
                    },
                    "required": ["weather", "current_time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "estimate_clothing_layers",
                "description": "根据穿衣建议估计衣物层数，用于增加一个中间步骤，展示更长的 Agent Loop",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "clothing_advice": {
                            "type": "string",
                            "description": "穿衣建议文本",
                        }
                    },
                    "required": ["clothing_advice"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_clothing_weight",
                "description": "根据穿衣建议和层数估计今天衣服总重量，应该在穿衣建议和层数之后调用",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "clothing_advice": {
                            "type": "string",
                            "description": "穿衣建议文本",
                        },
                        "layer_count": {
                            "type": "string",
                            "description": "衣物层数文本",
                        },
                    },
                    "required": ["clothing_advice", "layer_count"],
                },
            },
        },
    ]


def build_demo_chain_prompt() -> str:
    """Return the recommended demo prompt."""
    return (
        "请帮我完成一个完整的链式任务："
        "先查询小明当前所在位置，"
        "再查询该位置当前时间，"
        "然后根据位置和时间查询天气，"
        "再根据天气和时间给出今天穿什么衣服，"
        "接着估算今天衣服有几层，"
        "最后计算今天衣服总重量。"
        "请严格按工具顺序一步步调用，全部完成后再总结。"
    )


class AgentLoopDemo:
    """A teaching-friendly agent loop with a longer tool chain."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://coding.dashscope.aliyuncs.com/v1",
        model: str = "qwen3.5-plus",
        temperature: float = 0.2,
        max_tokens: int = 800,
        max_steps: int = 8,
    ):
        self.api_key = resolve_openai_api_key(api_key)
        if not self.api_key:
            raise ValueError(
                "请提供 api_key，"
                "或在项目根目录的 .env 中配置 OPENAI_API_KEY"
            )

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_steps = max_steps

        self.messages: List[Dict[str, Any]] = []
        self.tool_definitions = build_tool_definitions()
        self.tool_registry: Dict[str, Callable[..., str]] = {
            "get_current_location": get_current_location,
            "get_current_time": get_current_time,
            "get_weather_by_context": get_weather_by_context,
            "get_clothing_advice": get_clothing_advice,
            "estimate_clothing_layers": estimate_clothing_layers,
            "get_clothing_weight": get_clothing_weight,
        }

        self.system_prompt = (
            "你是一个会使用工具解决问题的 AI Agent。"
            "如果用户的问题包含位置、时间、天气、穿衣建议、衣服重量这条链路，"
            "你必须严格按顺序调用多个工具，而不是跳步。"
            "推荐顺序是："
            "1.get_current_location "
            "2.get_current_time "
            "3.get_weather_by_context "
            "4.get_clothing_advice "
            "5.estimate_clothing_layers "
            "6.get_clothing_weight。"
            "只有在完成这条链后，才能输出最终答案。"
        )

    def clear_history(self) -> None:
        self.messages = []

    def run(self, user_message: str) -> str:
        """Run one complete agent loop."""
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
        """Execute one local tool."""
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
        """Call the OpenAI-compatible API and log the exchange."""
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

        start_time = time.perf_counter()
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        response.raise_for_status()
        data = response.json()
        llm_logger.log_exchange(
            provider="dashscope-compatible",
            model=self.model,
            endpoint=url,
            request_payload=payload,
            request_headers=headers,
            response_payload=data,
            status_code=response.status_code,
            duration_ms=elapsed_ms,
            extra={"agent_name": "AgentLoopDemo"},
        )
        return data["choices"][0]["message"]


class FixedWorkflowDemo:
    """A fixed workflow version for comparison with Agent Loop."""

    @staticmethod
    def run(user_name: str = "小明") -> str:
        print("\n[Workflow] 开始固定流程演示")

        step1 = get_current_location(user_name)
        print(f"[Workflow Step 1] 位置: {step1}")
        city = step1.split("是", 1)[-1].strip()

        step2 = get_current_time(city)
        print(f"[Workflow Step 2] 时间: {step2}")

        step3 = get_weather_by_context(city, step2)
        print(f"[Workflow Step 3] 天气: {step3}")

        step4 = get_clothing_advice(step3, step2)
        print(f"[Workflow Step 4] 穿衣建议: {step4}")

        step5 = estimate_clothing_layers(step4)
        print(f"[Workflow Step 5] 衣物层数: {step5}")

        step6 = get_clothing_weight(step4, step5)
        print(f"[Workflow Step 6] 衣服重量: {step6}")

        final_answer = (
            f"固定工作流执行完成。\n"
            f"位置: {step1}\n"
            f"时间: {step2}\n"
            f"天气: {step3}\n"
            f"穿衣建议: {step4}\n"
            f"衣物层数: {step5}\n"
            f"衣服重量: {step6}"
        )
        return final_answer


class AgentLoopRunner:
    """CLI entry point."""

    def __init__(self, agent: AgentLoopDemo):
        self.agent = agent

    def run(self) -> None:
        print("=" * 64)
        print("Day5 Agent Loop 演示")
        print("=" * 64)
        print("\n可用命令:")
        print("  clear              - 清空对话历史")
        print("  max-steps <num>    - 设置最大执行步数")
        print("  demo-chain         - 运行 6 步链式 Agent Loop 示例")
        print("  demo-workflow      - 运行固定 6 步 Workflow 示例")
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

                if lower_text == "demo-chain":
                    user_input = build_demo_chain_prompt()

                if lower_text == "demo-workflow":
                    answer = FixedWorkflowDemo.run("小明")
                    print(f"\nWorkflow Summary:\n{answer}\n")
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
