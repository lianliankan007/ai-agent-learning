#!/usr/bin/env python3
"""
Day17: Function Calling 教学 Demo。

这个脚本不接真实大模型 API，而是用教学型“模型决策器”模拟：
1. 返回普通文本回答
2. 返回结构化 tool call

目的是先看清 Function Calling 的协议形态，而不是先引入外部依赖。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class ToolSpec:
    """表示一个工具定义。"""

    name: str
    description: str
    parameters: Dict[str, str]


@dataclass
class FunctionCall:
    """表示一次结构化工具调用请求。"""

    tool_name: str
    arguments: Dict[str, str]
    reason: str


@dataclass
class ModelDecision:
    """表示模型阶段的输出。"""

    mode: str
    text_answer: Optional[str] = None
    function_call: Optional[FunctionCall] = None


class DemoToolbox:
    """教学型工具箱。"""

    def __init__(self) -> None:
        self.docs = {
            "rag": "RAG 是先检索相关资料，再把资料放进 Prompt，让模型基于资料生成答案。",
            "tool use": "Tool Use 是让 Agent 按需调用外部工具，并结合工具结果完成任务。",
            "memory": "Memory 负责保存长期偏好、历史事实和任务状态。",
        }
        self.memories = {
            "goal": "用户当前目标是在 90 天内转型为 AI Agent 应用工程师。",
            "preference": "用户偏好中文、结构化、教学型说明。",
        }
        self.tools: Dict[str, Callable[..., str]] = {
            "calculator": self.calculator,
            "doc_search": self.doc_search,
            "memory_lookup": self.memory_lookup,
        }
        self.tool_specs = [
            ToolSpec(
                name="calculator",
                description="用于执行精确数学计算。适用于表达式求值。",
                parameters={"expression": "要计算的数学表达式，例如 18 * 6 + 3"},
            ),
            ToolSpec(
                name="doc_search",
                description="用于查询内置知识库。适用于概念解释、资料查询。",
                parameters={"query": "用户的知识查询语句"},
            ),
            ToolSpec(
                name="memory_lookup",
                description="用于查询长期记忆。适用于用户偏好、学习目标等历史信息。",
                parameters={"query": "用户的记忆查询语句"},
            ),
        ]

    def calculator(self, expression: str) -> str:
        safe_expression = expression.strip()
        if not re.fullmatch(r"[\d\.\+\-\*\/\(\)\s]+", safe_expression):
            return "计算失败：表达式不合法。"
        try:
            result = eval(safe_expression, {"__builtins__": {}}, {})
            return f"计算结果: {result}"
        except Exception as exc:
            return f"计算失败: {exc}"

    def doc_search(self, query: str) -> str:
        lowered = query.lower()
        hits = []
        for key, value in self.docs.items():
            if key in lowered:
                hits.append(f"{key}: {value}")
        if not hits:
            return "知识库中没有直接命中的内容。"
        return " | ".join(hits)

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


class FunctionCallingModelSimulator:
    """教学型“模型决策器”。

    它模拟真实模型根据工具描述做出两种输出：
    - 直接文本回答
    - 结构化 function call
    """

    def __init__(self, toolbox: DemoToolbox) -> None:
        self.toolbox = toolbox

    def decide(self, user_input: str) -> ModelDecision:
        text = user_input.strip()
        lowered = text.lower()

        if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", text):
            expression = self._extract_expression(text)
            return ModelDecision(
                mode="function_call",
                function_call=FunctionCall(
                    tool_name="calculator",
                    arguments={"expression": expression},
                    reason="问题包含精确计算需求，使用 calculator 更稳定。",
                ),
            )

        if any(keyword in lowered for keyword in ["rag", "tool use", "memory", "agent loop"]):
            return ModelDecision(
                mode="function_call",
                function_call=FunctionCall(
                    tool_name="doc_search",
                    arguments={"query": text},
                    reason="问题更适合先查资料，再基于资料回答。",
                ),
            )

        if any(keyword in text for keyword in ["目标", "偏好"]) or any(
            keyword in lowered for keyword in ["goal", "preference"]
        ):
            return ModelDecision(
                mode="function_call",
                function_call=FunctionCall(
                    tool_name="memory_lookup",
                    arguments={"query": text},
                    reason="问题在询问长期记忆信息，更适合调用 memory_lookup。",
                ),
            )

        return ModelDecision(
            mode="text_answer",
            text_answer=(
                "这是一个可以直接解释的问题，不依赖外部工具。"
                "系统选择直接回答，避免增加额外调用成本。"
            ),
        )

    @staticmethod
    def _extract_expression(text: str) -> str:
        match = re.search(r"(\d+(?:\.\d+)?(?:\s*[\+\-\*\/]\s*\d+(?:\.\d+)?)+)", text)
        if not match:
            return text
        return match.group(1)


class FunctionCallingAgent:
    """最小 Function Calling Agent。"""

    def __init__(self, toolbox: DemoToolbox, model: FunctionCallingModelSimulator) -> None:
        self.toolbox = toolbox
        self.model = model

    def handle(self, user_input: str) -> Dict[str, str]:
        decision = self.model.decide(user_input)
        if decision.mode == "text_answer":
            return {
                "mode": "text_answer",
                "model_output": decision.text_answer or "",
                "final_answer": decision.text_answer or "",
            }

        assert decision.function_call is not None
        function_call = decision.function_call
        tool = self.toolbox.tools[function_call.tool_name]
        tool_value = next(iter(function_call.arguments.values()))
        observation = tool(tool_value)
        final_answer = (
            f"系统先执行了 {function_call.tool_name}，"
            f"拿到 observation 后，再组织成最终回答。\n"
            f"observation: {observation}"
        )
        return {
            "mode": "function_call",
            "model_output": json.dumps(
                {
                    "tool_name": function_call.tool_name,
                    "arguments": function_call.arguments,
                    "reason": function_call.reason,
                },
                ensure_ascii=False,
                indent=2,
            ),
            "observation": observation,
            "final_answer": final_answer,
        }


class FunctionCallingRunner:
    """Day17 命令行入口。"""

    def __init__(self, agent: FunctionCallingAgent, toolbox: DemoToolbox) -> None:
        self.agent = agent
        self.toolbox = toolbox

    def run(self) -> None:
        print("=" * 72)
        print("Day17 Function Calling Agent Demo")
        print("=" * 72)
        print("\n可用命令:")
        print("  list-tools                - 查看工具描述")
        print("  ask <问题>                - 运行一次 Function Calling")
        print("  demo-calc                 - 演示计算类调用")
        print("  demo-doc                  - 演示知识检索类调用")
        print("  demo-memory               - 演示长期记忆类调用")
        print("  demo-chat                 - 演示直接文本回答")
        print("  review-all                - 依次查看全部案例")
        print("  quit/exit                 - 退出")
        print()

        while True:
            try:
                user_input = input("[day17-function-calling]> ").strip()
                if not user_input:
                    continue

                lower_text = user_input.lower()
                if lower_text in {"quit", "exit"}:
                    print("\n再见!")
                    break

                if lower_text == "list-tools":
                    self._print_tools()
                    continue

                if lower_text == "demo-calc":
                    self._run_case("请帮我计算 25 * 4 + 6")
                    continue

                if lower_text == "demo-doc":
                    self._run_case("请解释一下 RAG 和 Tool Use 的区别")
                    continue

                if lower_text == "demo-memory":
                    self._run_case("我的学习目标是什么？")
                    continue

                if lower_text == "demo-chat":
                    self._run_case("为什么结构化输出更稳定？")
                    continue

                if lower_text == "review-all":
                    self._run_case("请帮我计算 25 * 4 + 6")
                    self._run_case("请解释一下 RAG 和 Tool Use 的区别")
                    self._run_case("我的学习目标是什么？")
                    self._run_case("为什么结构化输出更稳定？")
                    continue

                if lower_text.startswith("ask "):
                    self._run_case(user_input[4:].strip())
                    continue

                print("不支持的命令，请输入 list-tools / ask / review-all / quit\n")
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as exc:
                print(f"\n错误: {exc}\n")

    def _run_case(self, question: str) -> None:
        result = self.agent.handle(question)
        print("\n" + "=" * 72)
        print(f"[Question] {question}")
        print("=" * 72)
        print(f"模式: {result['mode']}")
        print("\n模型输出:")
        print(result["model_output"])
        if "observation" in result:
            print("\n工具 observation:")
            print(result["observation"])
        print("\n最终回答:")
        print(result["final_answer"])
        print()

    def _print_tools(self) -> None:
        print("\n当前工具描述:")
        for spec in self.toolbox.tool_specs:
            print(f"- 工具名: {spec.name}")
            print(f"  作用: {spec.description}")
            print(f"  参数: {json.dumps(spec.parameters, ensure_ascii=False)}")
        print()


def main() -> None:
    toolbox = DemoToolbox()
    model = FunctionCallingModelSimulator(toolbox)
    agent = FunctionCallingAgent(toolbox, model)
    runner = FunctionCallingRunner(agent, toolbox)
    runner.run()


if __name__ == "__main__":
    main()
