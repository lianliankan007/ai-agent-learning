#!/usr/bin/env python3
"""
Day20: 多工具选择与工具路由教学 Demo。

这个脚本帮助你理解：
1. 工具数量一旦增加，真正困难的地方往往不是“能不能调用”，而是“该选哪个”
2. Tool Routing 的核心是：减少误调用、减少漏调用、减少多余调用
3. 程序层需要对模型选出的工具做最基本的校验和约束
4. 一个更稳的 Agent，不只是会调工具，还要会在多个工具之间做合理路由
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_markdown_logger import get_default_llm_logger
from utils.openai_config import resolve_openai_api_key

llm_logger = get_default_llm_logger()


@dataclass
class ToolSpec:
    """表示一个工具定义。"""

    name: str
    description: str
    parameters: Dict[str, str]
    routing_hint: str

    def to_prompt_block(self) -> str:
        """把工具描述整理成提示词文本。"""
        parameter_lines = [f"- {key}: {value}" for key, value in self.parameters.items()]
        parameters_text = "\n".join(parameter_lines) if parameter_lines else "- 无参数"
        return (
            f"工具名: {self.name}\n"
            f"作用: {self.description}\n"
            f"适用提示: {self.routing_hint}\n"
            f"参数:\n{parameters_text}"
        )


@dataclass
class FunctionCall:
    """表示一次结构化工具调用。"""

    tool_name: str
    arguments: Dict[str, str]
    reason: str


@dataclass
class DecisionResult:
    """表示一次决策结果。"""

    mode: str
    final_answer: Optional[str] = None
    function_call: Optional[FunctionCall] = None


class RoutingToolbox:
    """教学型多工具箱。"""

    def __init__(self) -> None:
        self.docs = {
            "rag": "RAG 是先检索资料，再把资料注入 Prompt，最后让模型基于资料回答。",
            "tool routing": "Tool Routing 是在多个工具之间做选择，目标是让正确问题进入正确工具。",
            "fallback": "Fallback 是主工具失败后，切换到更稳但能力更弱的替代路径。",
            "agent loop": "Agent Loop 是模型循环执行决策、行动、观察，直到得到最终答案。",
        }
        self.memories = {
            "goal": "用户当前目标是在 90 天内从 Java 后端转型为 AI Agent 应用工程师。",
            "preference": "用户偏好中文、结构化、教学型讲解。",
        }
        self.weather_data = {
            "上海": "上海：晴转多云，23 摄氏度，东风 3 级。",
            "北京": "北京：多云，19 摄氏度，北风 2 级。",
            "广州": "广州：阵雨，27 摄氏度，湿度较高。",
        }
        self.calendar_data = {
            "今天": "今天晚上 20:00 有一次 Agent 学习复盘。",
            "明天": "明天晚上 19:30 安排了 RAG 代码练习。",
        }
        self.tools: Dict[str, Callable[..., str]] = {
            "calculator": self.calculator,
            "weather_lookup": self.weather_lookup,
            "calendar_lookup": self.calendar_lookup,
            "doc_search": self.doc_search,
            "memory_lookup": self.memory_lookup,
        }
        self.tool_specs = [
            ToolSpec(
                name="calculator",
                description="用于精确数学计算。",
                routing_hint="当问题需要算数、表达式求值、数值比较时优先使用。",
                parameters={"expression": "要计算的数学表达式，例如 18 * 6 + 3"},
            ),
            ToolSpec(
                name="weather_lookup",
                description="用于查询指定城市天气。",
                routing_hint="当问题在询问城市天气、温度、风力等实时环境信息时使用。",
                parameters={"location": "要查询天气的城市，例如 上海、北京"},
            ),
            ToolSpec(
                name="calendar_lookup",
                description="用于查询学习日程安排。",
                routing_hint="当问题在询问今天、明天、本周的学习安排或计划时使用。",
                parameters={"date_label": "日期标签，例如 今天、明天"},
            ),
            ToolSpec(
                name="doc_search",
                description="用于查询概念知识。",
                routing_hint="当问题需要解释 RAG、Tool Routing、Fallback、Agent Loop 等概念时使用。",
                parameters={"query": "用户的知识查询语句"},
            ),
            ToolSpec(
                name="memory_lookup",
                description="用于查询长期记忆。",
                routing_hint="当问题在询问用户目标、偏好、已知长期信息时使用。",
                parameters={"query": "用户的记忆查询语句"},
            ),
        ]

    def get_tool_specs_text(self) -> str:
        return "\n\n".join(spec.to_prompt_block() for spec in self.tool_specs)

    def calculator(self, expression: str) -> str:
        safe_expression = expression.strip()
        if not re.fullmatch(r"[\d\.\+\-\*\/\(\)\s]+", safe_expression):
            return "计算失败：表达式不合法。"
        try:
            result = eval(safe_expression, {"__builtins__": {}}, {})
            return f"计算结果: {result}"
        except Exception as exc:
            return f"计算失败: {exc}"

    def weather_lookup(self, location: str) -> str:
        normalized_location = location.strip()
        if normalized_location in self.weather_data:
            return self.weather_data[normalized_location]
        return f"没有找到 {normalized_location} 的天气数据。"

    def calendar_lookup(self, date_label: str) -> str:
        normalized_label = date_label.strip()
        if normalized_label in self.calendar_data:
            return self.calendar_data[normalized_label]
        return f"没有找到 {normalized_label} 的学习安排。"

    def doc_search(self, query: str) -> str:
        lowered = query.lower()
        hits = []
        for key, value in self.docs.items():
            if key in lowered or key.replace(" ", "") in lowered:
                hits.append(f"{key}: {value}")
        if "路由" in query:
            hits.append(f"tool routing: {self.docs['tool routing']}")
        if "降级" in query:
            hits.append(f"fallback: {self.docs['fallback']}")
        if not hits:
            return "知识库中没有直接命中的内容。"
        return " | ".join(dict.fromkeys(hits))

    def memory_lookup(self, query: str) -> str:
        lowered = query.lower()
        hits = []
        if "目标" in query or "goal" in lowered:
            hits.append(self.memories["goal"])
        if "偏好" in query or "preference" in lowered:
            hits.append(self.memories["preference"])
        if not hits:
            return "没有命中相关长期记忆。"
        return " | ".join(hits)


class RoutingLLMModel:
    """使用真实 LLM 进行多工具路由决策。"""

    def __init__(
        self,
        toolbox: RoutingToolbox,
        api_key: Optional[str] = None,
        base_url: str = "https://coding.dashscope.aliyuncs.com/v1",
        model: str = "qwen3.5-plus",
        temperature: float = 0.1,
        max_tokens: int = 900,
    ) -> None:
        self.toolbox = toolbox
        self.api_key = resolve_openai_api_key(api_key)
        if not self.api_key:
            raise ValueError(
                "[Day20ToolRoutingAgent] 请提供 api_key，"
                "或在项目根目录的 .env 中配置 OPENAI_API_KEY"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def decide(self, user_input: str, scratchpad: List[Dict[str, str]], step_index: int) -> DecisionResult:
        scratchpad_text = self._format_scratchpad(scratchpad)
        system_prompt = (
            "你是一个教学型 Tool Routing 决策器。\n"
            "你的重点任务不是盲目多调工具，而是在多个工具之间做正确选择。\n"
            "你每一步只能做两种选择之一：\n"
            "1. 调用一个工具\n"
            "2. 输出最终答案\n\n"
            "路由要求：\n"
            "- 优先选择最直接、最匹配当前问题的工具\n"
            "- 如果当前 scratchpad 已经足够回答，就直接结束，不要为了多调用而多调用\n"
            "- 不要把概念问题错误路由到天气、日程、计算工具\n"
            "- 不要把查询日程的问题错误路由到长期记忆\n"
            "- 只能从给定工具中选择，不能编造新工具\n\n"
            f"【工具描述】\n{self.toolbox.get_tool_specs_text()}\n\n"
            "输出 JSON 格式要求如下：\n"
            "1. 如果继续调用工具：\n"
            '{"mode":"function_call","tool_name":"工具名","arguments":{"参数名":"参数值"},"reason":"原因"}\n'
            "2. 如果已经足够回答：\n"
            '{"mode":"final_answer","final_answer":"给用户的最终回答"}\n'
            "不要输出 Markdown 代码块，不要输出额外解释。"
        )
        user_prompt = (
            f"当前步数: {step_index}\n"
            f"用户问题: {user_input}\n"
            f"当前 scratchpad:\n{scratchpad_text}"
        )
        content = self._call_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stage=f"loop_step_{step_index}",
        )
        data = self._parse_json_object(content)
        mode = str(data.get("mode", "")).strip()

        if mode == "final_answer":
            return DecisionResult(
                mode="final_answer",
                final_answer=str(data.get("final_answer", "")).strip(),
            )

        if mode == "function_call":
            tool_name = str(data.get("tool_name", "")).strip()
            arguments = data.get("arguments", {})
            reason = str(data.get("reason", "")).strip() or "模型判断需要调用工具。"
            if tool_name not in self.toolbox.tools:
                raise ValueError(f"LLM 返回了未知工具: {tool_name}")
            if not isinstance(arguments, dict):
                raise ValueError("LLM 返回的 arguments 不是对象。")
            normalized_arguments = {
                str(key): str(value)
                for key, value in arguments.items()
                if value is not None
            }
            return DecisionResult(
                mode="function_call",
                function_call=FunctionCall(
                    tool_name=tool_name,
                    arguments=normalized_arguments,
                    reason=reason,
                ),
            )

        raise ValueError(f"LLM 返回了不支持的 mode: {mode}")

    def _call_chat(self, messages: list[dict[str, str]], stage: str) -> str:
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
                extra={"agent_name": "Day20ToolRoutingAgent", "stage": stage},
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
                extra={"agent_name": "Day20ToolRoutingAgent", "stage": stage},
            )
            raise RuntimeError(f"LLM 调用失败: {exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            llm_logger.log_exchange(
                provider="dashscope-compatible",
                model=self.model,
                endpoint=url,
                request_payload=payload,
                request_headers=headers,
                error=f"response_parse_error: {exc}",
                extra={"agent_name": "Day20ToolRoutingAgent", "stage": stage},
            )
            raise RuntimeError(f"LLM 响应解析失败: {exc}") from exc

    @staticmethod
    def _format_scratchpad(scratchpad: List[Dict[str, str]]) -> str:
        if not scratchpad:
            return "暂无 observation。"
        lines = []
        for index, item in enumerate(scratchpad, start=1):
            lines.append(
                f"{index}. tool={item['tool_name']}, input={item['tool_input']}, observation={item['observation']}"
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_json_object(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.S)
            if not match:
                raise
            return json.loads(match.group(0))

    @staticmethod
    def _safe_json(response: Optional[requests.Response]) -> Any:
        if response is None:
            return None
        try:
            return response.json()
        except ValueError:
            return {"raw_text": response.text}


class ToolRoutingAgent:
    """最小多工具路由 Agent。"""

    def __init__(self, toolbox: RoutingToolbox, llm_model: RoutingLLMModel, max_steps: int = 4) -> None:
        self.toolbox = toolbox
        self.llm_model = llm_model
        self.max_steps = max_steps
        self.tool_argument_map = {
            "calculator": "expression",
            "weather_lookup": "location",
            "calendar_lookup": "date_label",
            "doc_search": "query",
            "memory_lookup": "query",
        }

    def handle(self, user_input: str) -> Dict[str, Any]:
        scratchpad: List[Dict[str, str]] = []
        trace: List[Dict[str, Any]] = []

        for step_index in range(1, self.max_steps + 1):
            decision = self.llm_model.decide(user_input=user_input, scratchpad=scratchpad, step_index=step_index)

            if decision.mode == "final_answer":
                trace.append(
                    {
                        "step": step_index,
                        "mode": "final_answer",
                        "final_answer": decision.final_answer or "",
                    }
                )
                return {
                    "status": "completed",
                    "trace": trace,
                    "final_answer": decision.final_answer or "",
                }

            assert decision.function_call is not None
            normalized_tool_input = self._normalize_tool_input(decision.function_call)
            tool = self.toolbox.tools[decision.function_call.tool_name]
            observation = tool(normalized_tool_input)
            scratchpad.append(
                {
                    "tool_name": decision.function_call.tool_name,
                    "tool_input": normalized_tool_input,
                    "observation": observation,
                }
            )
            trace.append(
                {
                    "step": step_index,
                    "mode": "function_call",
                    "tool_name": decision.function_call.tool_name,
                    "arguments": decision.function_call.arguments,
                    "normalized_input": normalized_tool_input,
                    "reason": decision.function_call.reason,
                    "observation": observation,
                }
            )

        return {
            "status": "max_steps_reached",
            "trace": trace,
            "final_answer": f"已达到最大步数 {self.max_steps}，系统被强制停止。",
        }

    def _normalize_tool_input(self, function_call: FunctionCall) -> str:
        """程序层对工具参数做一个最小校验。

        这里的目标不是替模型完全决策，而是避免：
        - 参数名错了但程序直接崩溃
        - 多工具场景下 argument key 不一致导致执行层异常
        """
        expected_key = self.tool_argument_map.get(function_call.tool_name)
        if expected_key and expected_key in function_call.arguments:
            return function_call.arguments[expected_key]
        if function_call.arguments:
            return next(iter(function_call.arguments.values()))
        return ""


class ToolRoutingRunner:
    """Day20 命令行入口。"""

    def __init__(self, agent: ToolRoutingAgent, toolbox: RoutingToolbox) -> None:
        self.agent = agent
        self.toolbox = toolbox

    def run(self) -> None:
        print("=" * 72)
        print("Day20 Tool Routing Agent Demo")
        print("=" * 72)
        print("\n可用命令:")
        print("  status                    - 查看当前配置")
        print("  list-tools                - 查看工具描述")
        print("  ask <问题>                - 运行一次多工具路由 Agent")
        print("  demo-weather              - 演示天气查询")
        print("  demo-calendar             - 演示日程查询")
        print("  demo-memory-doc           - 演示先查长期目标，再解释概念")
        print("  demo-calc                 - 演示计算问题")
        print("  demo-direct               - 演示直接结束的问题")
        print("  review-all                - 依次查看全部案例")
        print("  quit/exit                 - 退出")
        print()

        while True:
            try:
                user_input = input("[day20-routing]> ").strip()
                if not user_input:
                    continue

                lower_text = user_input.lower()
                if lower_text in {"quit", "exit"}:
                    print("\n再见!")
                    break

                if lower_text == "status":
                    self._print_status()
                    continue

                if lower_text == "list-tools":
                    self._print_tools()
                    continue

                if lower_text == "demo-weather":
                    self._run_case("请查询上海天气，并用一句话总结。")
                    continue

                if lower_text == "demo-calendar":
                    self._run_case("帮我看一下明天的学习安排。")
                    continue

                if lower_text == "demo-memory-doc":
                    self._run_case("先查一下我的学习目标，再解释一下 Tool Routing 是什么。")
                    continue

                if lower_text == "demo-calc":
                    self._run_case("请帮我计算 18 * 6 + 3。")
                    continue

                if lower_text == "demo-direct":
                    self._run_case("为什么多工具场景下更容易误调工具？")
                    continue

                if lower_text == "review-all":
                    self._run_case("请查询上海天气，并用一句话总结。")
                    self._run_case("帮我看一下明天的学习安排。")
                    self._run_case("先查一下我的学习目标，再解释一下 Tool Routing 是什么。")
                    self._run_case("请帮我计算 18 * 6 + 3。")
                    self._run_case("为什么多工具场景下更容易误调工具？")
                    continue

                if lower_text.startswith("ask "):
                    self._run_case(user_input[4:].strip())
                    continue

                print("不支持的命令，请输入 status / list-tools / ask / review-all / quit\n")
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except EOFError:
                print("\n\n检测到输入结束(EOF)，退出。")
                break
            except Exception as exc:
                print(f"\n错误: {exc}\n")

    def _run_case(self, question: str) -> None:
        result = self.agent.handle(question)
        print("\n" + "=" * 72)
        print(f"[Question] {question}")
        print("=" * 72)
        print(f"状态: {result['status']}")
        print("\n执行轨迹:")
        for item in result["trace"]:
            print(f"- step={item['step']} mode={item['mode']}")
            if item["mode"] == "function_call":
                print(f"  tool_name={item['tool_name']}")
                print(f"  arguments={json.dumps(item['arguments'], ensure_ascii=False)}")
                print(f"  normalized_input={item['normalized_input']}")
                print(f"  reason={item['reason']}")
                print(f"  observation={item['observation']}")
            else:
                print(f"  final_answer={item['final_answer']}")
        print("\n最终回答:")
        print(result["final_answer"])
        print()

    def _print_tools(self) -> None:
        print("\n当前工具描述:")
        for spec in self.toolbox.tool_specs:
            print(f"- 工具名: {spec.name}")
            print(f"  作用: {spec.description}")
            print(f"  适用提示: {spec.routing_hint}")
            print(f"  参数: {json.dumps(spec.parameters, ensure_ascii=False)}")
        print()

    def _print_status(self) -> None:
        print("\n当前配置:")
        print("  - 当前仅支持真实 LLM 路由决策链路")
        print(f"  - 模型: {self.agent.llm_model.model}")
        print(f"  - Base URL: {self.agent.llm_model.base_url}")
        print(f"  - max_steps: {self.agent.max_steps}")
        print("  - 当前重点观察：多工具选择是否合理")
        print()


def main() -> None:
    toolbox = RoutingToolbox()
    llm_model = RoutingLLMModel(
        toolbox=toolbox,
        api_key=None,
        base_url=os.getenv("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"),
        model=os.getenv("OPENAI_MODEL", "qwen3.5-plus"),
    )
    agent = ToolRoutingAgent(
        toolbox=toolbox,
        llm_model=llm_model,
        max_steps=int(os.getenv("DAY20_MAX_STEPS", "4")),
    )
    runner = ToolRoutingRunner(agent, toolbox)
    runner.run()


if __name__ == "__main__":
    main()
