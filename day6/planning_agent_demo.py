#!/usr/bin/env python3
"""
Day6: Planning Agent teaching demo.

This version extends the tool chain based on Day5, but keeps the
teaching focus on Planning / Action / Observation.
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
    """返回演示用户所在城市，帮助模型先补齐上下文。"""
    location_map = {
        "小明": "北京",
        "小红": "上海",
        "alice": "杭州",
        "bob": "深圳",
    }
    city = location_map.get(user_name.strip().lower())
    if city is None:
        city = location_map.get(user_name.strip(), "北京")
    return f"{user_name} 当前所在城市是 {city}"


def get_current_time(city: str) -> str:
    """返回城市当前时间。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{city} 当前时间是 {now}"


def get_weather_by_context(city: str, current_time: str) -> str:
    """根据城市和时间返回天气。"""
    weather_map = {
        "北京": "晴，22度，微风",
        "上海": "多云，24度，东南风",
        "杭州": "小雨，19度，东北风",
        "深圳": "晴，28度，南风",
    }
    weather = weather_map.get(city, "阴，20度，微风")
    return f"基于城市 {city} 和时间 {current_time}，当前天气是 {weather}"


def get_clothing_advice(weather: str, current_time: str) -> str:
    """根据天气和时间给出穿衣建议。"""
    if "小雨" in weather:
        return f"根据天气 {weather} 和时间 {current_time}，建议穿薄外套、长裤，并带伞"
    if "28度" in weather:
        return f"根据天气 {weather} 和时间 {current_time}，建议穿短袖、薄裤，注意防晒"
    if "24度" in weather:
        return f"根据天气 {weather} 和时间 {current_time}，建议穿短袖加轻薄外套"
    return f"根据天气 {weather} 和时间 {current_time}，建议穿短袖、长裤，早晚可加一件薄外套"


def estimate_clothing_layers(clothing_advice: str) -> str:
    """估算今天大概需要几层衣物。"""
    if "薄外套" in clothing_advice and "短袖" in clothing_advice:
        layers = 2
    elif "薄外套" in clothing_advice:
        layers = 2
    else:
        layers = 1
    return f"根据穿衣建议“{clothing_advice}”，预计今天需要 {layers} 层衣物"


def get_clothing_weight(clothing_advice: str, layer_count: str) -> str:
    """粗略估算整套衣物重量。"""
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
    """定义给模型使用的长链工具。"""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_current_location",
                "description": "当用户只给出人名，没有给出城市时，先查询该用户当前所在城市。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_name": {
                            "type": "string",
                            "description": "用户名，例如 小明、小红、alice",
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
                "description": "当已经知道城市，且用户询问当前时间时调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名，例如 北京、上海",
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
                "description": "当已经知道城市和时间，且用户需要天气信息时调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名",
                        },
                        "current_time": {
                            "type": "string",
                            "description": "前一步工具返回的时间文本",
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
                "description": "当用户进一步询问穿衣建议时，在拿到天气和时间后调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "weather": {
                            "type": "string",
                            "description": "天气文本",
                        },
                        "current_time": {
                            "type": "string",
                            "description": "时间文本",
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
                "description": "当用户还想知道大概穿几层时，根据穿衣建议估算层数。",
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
                "description": "当用户想知道衣物大概有多重时，根据穿衣建议和层数估算总重量。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "clothing_advice": {
                            "type": "string",
                            "description": "穿衣建议文本",
                        },
                        "layer_count": {
                            "type": "string",
                            "description": "层数文本",
                        },
                    },
                    "required": ["clothing_advice", "layer_count"],
                },
            },
        },
    ]


def build_demo_chain_prompt() -> str:
    """构造一个适合观察长链 planning 的问题。"""
    return (
        "请你帮我完成一个完整的多步任务："
        "先查询小明当前所在位置，"
        "再查询该位置当前时间，"
        "然后根据位置和时间查询天气，"
        "再根据天气和时间给出今天穿什么衣服，"
        "接着估算今天衣服有几层，"
        "最后估算整套衣服的重量。"
        "请每一步都先判断是否真的需要调用下一个工具，"
        "全部信息足够后再输出最终总结。"
    )


class PlanningAgentDemo:
    """教学版 Planning Agent。"""

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
                "请提供 api_key，或在项目根目录的 .env 中配置 OPENAI_API_KEY"
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
            "你是一个教学型 AI Agent。"
            "每一步都要先判断：用户当前的问题是否真的需要继续调用工具。"
            "如果是常识解释、自我介绍、概念说明等普通问题，直接回答，不要调用工具。"
            "如果问题涉及位置、时间、天气、穿衣建议、衣物层数、衣物重量这条链路，"
            "你要根据当前已知信息决定下一步最合适的工具。"
            "推荐顺序通常是："
            "1.get_current_location "
            "2.get_current_time "
            "3.get_weather_by_context "
            "4.get_clothing_advice "
            "5.estimate_clothing_layers "
            "6.get_clothing_weight。"
            "但你不能为了凑步骤而调用工具，只有在当前信息不足时才继续。"
            "拿到工具结果后，要基于 observation 再决定下一步。"
            "当信息已经足够时，立刻输出最终答案，不要重复调用工具。"
        )

    def clear_history(self) -> None:
        self.messages = []

    def run(self, user_message: str) -> str:
        """运行一轮完整的 Planning Agent。"""
        working_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        working_messages.extend(self.messages)
        working_messages.append({"role": "user", "content": user_message})

        print("\n" + "=" * 64)
        print(f"[Task] 用户问题: {user_message}")
        print(f"[Config] max_steps = {self.max_steps}")
        print("=" * 64)

        for step in range(1, self.max_steps + 1):
            print(f"\n[Step {step}] Planning: 正在判断下一步该做什么...")
            assistant_message = self._call_api(working_messages, self.tool_definitions)
            working_messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                final_answer = (assistant_message.get("content") or "").strip()
                print(f"[Step {step}] Action: 直接回答，不调用工具")
                print(f"[Step {step}] Final Answer: {final_answer}")

                self.messages.append({"role": "user", "content": user_message})
                self.messages.append({"role": "assistant", "content": final_answer})
                return final_answer

            for tool_call in tool_calls:
                function_info = tool_call["function"]
                function_name = function_info["name"]
                arguments_text = function_info.get("arguments", "{}")

                try:
                    arguments = json.loads(arguments_text)
                except json.JSONDecodeError:
                    arguments = {}

                print(f"[Step {step}] Action: 调用工具 {function_name}")
                print(f"[Step {step}] Tool Args: {arguments}")

                observation = self._execute_tool(function_name, arguments)
                print(f"[Step {step}] Observation: {observation}")

                working_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": observation,
                    }
                )

        fail_message = (
            f"已达到最大执行步数 {self.max_steps}。"
            "本次任务停止，以避免 Agent 陷入无意义循环。"
        )
        print(f"\n[Stop] {fail_message}")

        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": fail_message})
        return fail_message

    def _execute_tool(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """执行本地工具，并把结果作为 observation 返回。"""
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
        """调用兼容 OpenAI 的聊天接口，并记录日志。"""
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
                extra={"agent_name": "PlanningAgentDemo"},
            )
            return data["choices"][0]["message"]
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
                extra={"agent_name": "PlanningAgentDemo"},
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
                extra={"agent_name": "PlanningAgentDemo"},
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


class PlanningAgentRunner:
    """命令行入口，方便反复实验。"""

    def __init__(self, agent: PlanningAgentDemo):
        self.agent = agent

    def run(self) -> None:
        print("=" * 64)
        print("Day6 Planning Agent 演示")
        print("=" * 64)
        print("\n可用命令:")
        print("  clear              - 清空对话历史")
        print("  max-steps <num>    - 设置最大执行步数")
        print("  demo-chat          - 测试一个不需要工具的问题")
        print("  demo-time          - 测试一个单工具问题")
        print("  demo-both          - 测试一个双工具问题")
        print("  demo-chain         - 测试一个长链 planning 问题")
        print("  quit/exit          - 退出程序")
        print("  <任意文字>         - 直接开始对话\n")

        while True:
            try:
                user_input = input("[day6-agent]> ").strip()
                if not user_input:
                    continue

                lower_text = user_input.lower()
                if lower_text in {"quit", "exit"}:
                    print("\n再见!")
                    break

                if lower_text == "clear":
                    self.agent.clear_history()
                    print("对话历史已清空。\n")
                    continue

                if lower_text.startswith("max-steps "):
                    new_value = int(user_input.split(" ", 1)[1].strip())
                    self.agent.max_steps = new_value
                    print(f"最大执行步数已更新为 {new_value}\n")
                    continue

                if lower_text == "demo-chat":
                    user_input = "请解释什么是 Observation"
                elif lower_text == "demo-time":
                    user_input = "北京现在几点？"
                elif lower_text == "demo-both":
                    user_input = "北京现在几点，并告诉我天气"
                elif lower_text == "demo-chain":
                    user_input = build_demo_chain_prompt()

                answer = self.agent.run(user_input)
                print(f"\nAI: {answer}\n")
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as exc:
                print(f"\n错误: {exc}\n")


def main() -> None:
    agent = PlanningAgentDemo(
        api_key=None,
        base_url=os.getenv("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"),
        model=os.getenv("OPENAI_MODEL", "qwen3.5-plus"),
    )
    runner = PlanningAgentRunner(agent)
    runner.run()


if __name__ == "__main__":
    main()
