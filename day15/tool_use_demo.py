#!/usr/bin/env python3
"""
Day15: Tool Use 教学 Demo。

这个脚本专门帮助你理解：
1. 什么情况下应该调用工具
2. 什么情况下不应该调用工具
3. 一个最小 Tool Use 链路是什么样

这里不接真实 LLM Function Calling，而是用教学型规则路由，
让你先把“工具选择 -> 执行 -> observation -> 回答”这条链路看清楚。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class ToolCall:
    """表示一次工具调用决策。"""

    need_tool: bool
    tool_name: Optional[str]
    tool_input: Optional[str]
    reason: str


@dataclass
class ToolResult:
    """表示一次工具执行结果。"""

    tool_name: str
    tool_input: str
    observation: str


class MiniToolbox:
    """教学型工具箱。"""

    def __init__(self) -> None:
        self.docs = {
            "rag": "RAG 是先检索资料，再把资料放进 Prompt，最后让模型基于资料生成答案。",
            "tool use": "Tool Use 是让 Agent 根据问题决定是否调用外部工具，并基于工具结果继续完成任务。",
            "agent loop": "Agent Loop 是模型循环执行思考、行动、观察，直到产出最终答案。",
        }
        self.memory = {
            "user_preference": "用户偏好结构化、简洁、中文回答。",
            "current_goal": "用户当前目标是在 90 天内从 Java 后端转型为 AI Agent 应用工程师。",
        }
        self.tools: Dict[str, Callable[[str], str]] = {
            "calculator": self.calculator,
            "doc_search": self.doc_search,
            "memory_lookup": self.memory_lookup,
        }

    def calculator(self, expression: str) -> str:
        """执行教学版计算。

        这里只允许数字和基础运算符，避免引入任意执行风险。
        """
        safe_expression = expression.strip()
        if not re.fullmatch(r"[\d\.\+\-\*\/\(\)\s]+", safe_expression):
            return "计算工具拒绝执行：表达式包含不支持的字符。"
        try:
            result = eval(safe_expression, {"__builtins__": {}}, {"math": math})
            return f"计算结果: {result}"
        except Exception as exc:
            return f"计算失败: {exc}"

    def doc_search(self, query: str) -> str:
        """在内置知识库中做一个极简查询。"""
        lowered_query = query.lower()
        hits = []
        for key, value in self.docs.items():
            if key in lowered_query:
                hits.append(f"{key}: {value}")
        if not hits:
            return "知识库中没有直接命中的资料。"
        return " | ".join(hits)

    def memory_lookup(self, query: str) -> str:
        """在内置长期记忆中查询。"""
        lowered_query = query.lower()
        hits = []
        if "偏好" in query or "preference" in lowered_query:
            hits.append(self.memory["user_preference"])
        if "目标" in query or "goal" in lowered_query:
            hits.append(self.memory["current_goal"])
        if not hits:
            return "没有检索到相关长期记忆。"
        return " | ".join(hits)


class ToolUseTeacher:
    """教学型 Tool Use 路由器。"""

    def __init__(self, toolbox: MiniToolbox) -> None:
        self.toolbox = toolbox

    def decide(self, user_input: str) -> ToolCall:
        """判断当前问题是否需要工具。"""
        text = user_input.strip()
        lowered = text.lower()

        # 规则1：出现明显的计算表达式时，优先走计算工具。
        if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", text):
            return ToolCall(
                need_tool=True,
                tool_name="calculator",
                tool_input=self._extract_expression(text),
                reason="问题包含明确计算表达式，直接调用计算工具更稳定。",
            )

        # 规则2：出现“查资料/文档/RAG/Agent Loop”等关键词时，走知识检索工具。
        if any(keyword in lowered for keyword in ["rag", "agent loop", "tool use", "查资料", "文档"]):
            return ToolCall(
                need_tool=True,
                tool_name="doc_search",
                tool_input=text,
                reason="问题更适合先查知识资料，再基于资料回答。",
            )

        # 规则3：出现“偏好/目标/记住”时，走长期记忆检索。
        if any(keyword in text for keyword in ["偏好", "目标", "记住"]) or any(
            keyword in lowered for keyword in ["preference", "goal", "memory"]
        ):
            return ToolCall(
                need_tool=True,
                tool_name="memory_lookup",
                tool_input=text,
                reason="问题在询问长期记忆或用户信息，更适合调用记忆工具。",
            )

        # 规则4：其余常规解释类问题，直接回答即可。
        return ToolCall(
            need_tool=False,
            tool_name=None,
            tool_input=None,
            reason="这是常规解释型问题，不依赖外部数据或执行动作，可以直接回答。",
        )

    def act(self, decision: ToolCall) -> Optional[ToolResult]:
        """执行工具。"""
        if not decision.need_tool or not decision.tool_name or decision.tool_input is None:
            return None
        tool = self.toolbox.tools[decision.tool_name]
        observation = tool(decision.tool_input)
        return ToolResult(
            tool_name=decision.tool_name,
            tool_input=decision.tool_input,
            observation=observation,
        )

    def answer(self, user_input: str, decision: ToolCall, observation: Optional[ToolResult]) -> str:
        """根据是否调用工具，生成最终回答。"""
        if not decision.need_tool:
            return (
                "直接回答: 这个问题不需要外部工具。\n"
                "原因: 系统判断它属于常规解释型问题。\n"
                f"示例回答: {self._direct_reply(user_input)}"
            )

        assert observation is not None
        return (
            "工具调用回答:\n"
            f"- 选择工具: {observation.tool_name}\n"
            f"- 工具输入: {observation.tool_input}\n"
            f"- observation: {observation.observation}\n"
            f"- 最终说明: 系统先用工具拿到更可靠的信息，再基于 observation 组织回答。"
        )

    @staticmethod
    def _extract_expression(text: str) -> str:
        """提取最小计算表达式。"""
        match = re.search(r"(\d+(?:\.\d+)?(?:\s*[\+\-\*\/]\s*\d+(?:\.\d+)?)+)", text)
        if not match:
            return text
        return match.group(1)

    @staticmethod
    def _direct_reply(user_input: str) -> str:
        """教学版直接回答。"""
        return f"针对问题“{user_input}”，系统认为可以直接给出概念解释，不必增加工具调用成本。"


class ToolUseRunner:
    """Day15 命令行入口。"""

    def __init__(self, teacher: ToolUseTeacher):
        self.teacher = teacher

    def run(self) -> None:
        print("=" * 72)
        print("Day15 Tool Use Demo")
        print("=" * 72)
        print("\n可用命令:")
        print("  list-tools                - 查看当前工具")
        print("  ask <问题>                - 询问一个问题")
        print("  demo-calc                 - 演示计算类问题")
        print("  demo-doc                  - 演示知识检索类问题")
        print("  demo-memory               - 演示记忆检索类问题")
        print("  demo-chat                 - 演示直接回答类问题")
        print("  review-all                - 依次查看全部案例")
        print("  quit/exit                 - 退出")
        print()

        while True:
            try:
                user_input = input("[day15-tool-use]> ").strip()
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
                    self._run_case("23 * 7 + 5")
                    continue

                if lower_text == "demo-doc":
                    self._run_case("请帮我查一下 RAG 是什么意思")
                    continue

                if lower_text == "demo-memory":
                    self._run_case("我的学习目标是什么？")
                    continue

                if lower_text == "demo-chat":
                    self._run_case("请解释一下为什么 Tool Use 很重要")
                    continue

                if lower_text == "review-all":
                    self._run_case("23 * 7 + 5")
                    self._run_case("请帮我查一下 RAG 是什么意思")
                    self._run_case("我的学习目标是什么？")
                    self._run_case("请解释一下为什么 Tool Use 很重要")
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
        decision = self.teacher.decide(question)
        observation = self.teacher.act(decision)
        final_answer = self.teacher.answer(question, decision, observation)

        print("\n" + "=" * 72)
        print(f"[Question] {question}")
        print("=" * 72)
        print(f"是否需要工具: {decision.need_tool}")
        print(f"判断原因: {decision.reason}")
        if observation is None:
            print("工具执行: 未调用工具")
        else:
            print(f"工具执行: {observation.tool_name}")
            print(f"工具输入: {observation.tool_input}")
            print(f"observation: {observation.observation}")
        print("\n最终回答:")
        print(final_answer)
        print()

    def _print_tools(self) -> None:
        print("\n当前工具:")
        print("  - calculator: 处理精确计算")
        print("  - doc_search: 查询内置知识库")
        print("  - memory_lookup: 查询长期记忆")
        print()


def main() -> None:
    toolbox = MiniToolbox()
    teacher = ToolUseTeacher(toolbox)
    runner = ToolUseRunner(teacher)
    runner.run()


if __name__ == "__main__":
    main()
