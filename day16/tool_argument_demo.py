#!/usr/bin/env python3
"""
Day16: Tool 参数与 observation 教学 Demo。

这个脚本帮助你理解：
1. 选对工具之后，参数为什么仍然很关键
2. observation 为什么只是中间结果
3. 参数错了时，最终回答会如何偏掉
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class ToolDecision:
    """表示一次工具决策。"""

    tool_name: str
    tool_argument: str
    reason: str


@dataclass
class ToolObservation:
    """表示工具执行后的中间结果。"""

    tool_name: str
    tool_argument: str
    raw_result: str


class ToolArgumentBox:
    """教学型工具箱。"""

    def __init__(self) -> None:
        self.docs = {
            "rag": "RAG 是先检索，再生成。",
            "tool use": "Tool Use 是先判断是否要调用外部工具，再基于工具结果继续回答。",
            "memory": "Memory 负责保存用户偏好、长期目标和历史事实。",
        }
        self.memory = {
            "learning_goal": "当前目标是在 90 天内完成从 Java 后端到 AI Agent 工程师的转型。",
            "answer_preference": "用户偏好中文、结构化、易理解的讲解。",
        }
        self.tools: Dict[str, Callable[[str], str]] = {
            "calculator": self.calculator,
            "doc_search": self.doc_search,
            "memory_lookup": self.memory_lookup,
        }

    def calculator(self, expression: str) -> str:
        safe_expression = expression.strip()
        if not re.fullmatch(r"[\d\.\+\-\*\/\(\)\s]+", safe_expression):
            return "计算工具拒绝执行：表达式不合法。"
        try:
            result = eval(safe_expression, {"__builtins__": {}}, {})
            return f"计算结果={result}"
        except Exception as exc:
            return f"计算失败={exc}"

    def doc_search(self, query: str) -> str:
        lowered = query.lower()
        hits = []
        for key, value in self.docs.items():
            if key in lowered:
                hits.append(f"{key}: {value}")
        if not hits:
            return "未命中任何知识"
        return " || ".join(hits)

    def memory_lookup(self, query: str) -> str:
        lowered = query.lower()
        hits = []
        if "目标" in query or "goal" in lowered:
            hits.append(self.memory["learning_goal"])
        if "偏好" in query or "preference" in lowered:
            hits.append(self.memory["answer_preference"])
        if not hits:
            return "未命中任何长期记忆"
        return " || ".join(hits)


class ToolArgumentTeacher:
    """教学型参数提取与 observation 解释器。"""

    def __init__(self, tool_box: ToolArgumentBox) -> None:
        self.tool_box = tool_box

    def decide(self, user_input: str) -> Optional[ToolDecision]:
        text = user_input.strip()
        lowered = text.lower()

        if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", text):
            return ToolDecision(
                tool_name="calculator",
                tool_argument=self._extract_expression(text),
                reason="检测到计算表达式，适合调用 calculator。",
            )

        if any(keyword in lowered for keyword in ["rag", "tool use", "memory", "agent loop"]):
            return ToolDecision(
                tool_name="doc_search",
                tool_argument=text,
                reason="检测到知识查询意图，适合调用 doc_search。",
            )

        if any(keyword in text for keyword in ["目标", "偏好"]) or any(
            keyword in lowered for keyword in ["goal", "preference"]
        ):
            return ToolDecision(
                tool_name="memory_lookup",
                tool_argument=text,
                reason="检测到长期记忆查询意图，适合调用 memory_lookup。",
            )

        return None

    def act(self, decision: ToolDecision) -> ToolObservation:
        tool = self.tool_box.tools[decision.tool_name]
        raw_result = tool(decision.tool_argument)
        return ToolObservation(
            tool_name=decision.tool_name,
            tool_argument=decision.tool_argument,
            raw_result=raw_result,
        )

    def answer(self, user_input: str, decision: Optional[ToolDecision], observation: Optional[ToolObservation]) -> str:
        if decision is None or observation is None:
            return (
                "系统判断当前问题不依赖外部工具，所以直接回答。\n"
                f"示例说明: 对“{user_input}”这种问题，可以优先给出概念解释。"
            )

        return (
            f"工具选择: {decision.tool_name}\n"
            f"参数提取: {decision.tool_argument}\n"
            f"observation: {observation.raw_result}\n"
            "最终理解: observation 是工具返回的中间结果，系统需要基于它整理成适合用户的回答。"
        )

    @staticmethod
    def _extract_expression(text: str) -> str:
        match = re.search(r"(\d+(?:\.\d+)?(?:\s*[\+\-\*\/]\s*\d+(?:\.\d+)?)+)", text)
        if not match:
            return text
        return match.group(1)


class ToolArgumentRunner:
    """Day16 命令行入口。"""

    def __init__(self, teacher: ToolArgumentTeacher) -> None:
        self.teacher = teacher

    def run(self) -> None:
        print("=" * 72)
        print("Day16 Tool Argument Demo")
        print("=" * 72)
        print("\n可用命令:")
        print("  list-tools                - 查看当前工具")
        print("  ask <问题>                - 运行一次 Tool Use")
        print("  demo-calc                 - 演示正确计算参数")
        print("  demo-search               - 演示知识查询")
        print("  demo-memory               - 演示长期记忆查询")
        print("  demo-bad-param            - 演示参数错误案例")
        print("  review-all                - 依次查看全部案例")
        print("  quit/exit                 - 退出")
        print()

        while True:
            try:
                user_input = input("[day16-tool-arg]> ").strip()
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
                    self._run_case("请帮我计算 18 * 6 + 3")
                    continue

                if lower_text == "demo-search":
                    self._run_case("请解释一下 RAG 和 Tool Use 的区别")
                    continue

                if lower_text == "demo-memory":
                    self._run_case("我的学习目标是什么？")
                    continue

                if lower_text == "demo-bad-param":
                    self._run_bad_param_case()
                    continue

                if lower_text == "review-all":
                    self._run_case("请帮我计算 18 * 6 + 3")
                    self._run_case("请解释一下 RAG 和 Tool Use 的区别")
                    self._run_case("我的学习目标是什么？")
                    self._run_bad_param_case()
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
        observation = self.teacher.act(decision) if decision is not None else None
        answer = self.teacher.answer(question, decision, observation)

        print("\n" + "=" * 72)
        print(f"[Question] {question}")
        print("=" * 72)
        if decision is None:
            print("是否调用工具: 否")
            print("工具选择: 无")
            print("参数提取: 无")
            print("observation: 无")
        else:
            print("是否调用工具: 是")
            print(f"工具选择: {decision.tool_name}")
            print(f"参数提取: {decision.tool_argument}")
            print(f"判断原因: {decision.reason}")
            print(f"observation: {observation.raw_result if observation else '无'}")
        print("\n最终回答:")
        print(answer)
        print()

    def _run_bad_param_case(self) -> None:
        """构造一个工具选对但参数故意抽错的案例。"""
        decision = ToolDecision(
            tool_name="calculator",
            tool_argument="18 *",
            reason="工具选对了，但参数抽取不完整。",
        )
        observation = self.teacher.act(decision)
        answer = self.teacher.answer("请帮我计算 18 * 6 + 3", decision, observation)

        print("\n" + "=" * 72)
        print("[Bad Param Case]")
        print("=" * 72)
        print("用户问题: 请帮我计算 18 * 6 + 3")
        print(f"工具选择: {decision.tool_name}")
        print(f"错误参数: {decision.tool_argument}")
        print(f"observation: {observation.raw_result}")
        print("\n分析:")
        print("这个案例里，系统选对了 calculator，但参数只抽到了“18 *”，所以工具无法得到正确结果。")
        print("这说明 Tool Use 不只是决定要不要调工具，更关键的是参数抽取是否正确。")
        print("\n最终回答:")
        print(answer)
        print()

    @staticmethod
    def _print_tools() -> None:
        print("\n当前工具:")
        print("  - calculator: 接收表达式并返回计算结果")
        print("  - doc_search: 接收查询语句并返回知识命中结果")
        print("  - memory_lookup: 接收记忆查询语句并返回长期记忆")
        print()


def main() -> None:
    tool_box = ToolArgumentBox()
    teacher = ToolArgumentTeacher(tool_box)
    runner = ToolArgumentRunner(teacher)
    runner.run()


if __name__ == "__main__":
    main()
