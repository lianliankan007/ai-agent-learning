#!/usr/bin/env python3
"""
Day18: 多轮 Function Calling 与最小 Agent Loop 教学 Demo。

这个脚本只使用真实 LLM 参与决策，并把“思考 -> 调工具 -> observation -> 再决策”
串成一个最小可观察循环。
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

    def to_prompt_block(self) -> str:
        parameter_lines = [f"- {key}: {value}" for key, value in self.parameters.items()]
        parameters_text = "\n".join(parameter_lines) if parameter_lines else "- 无参数"
        return (
            f"工具名: {self.name}\n"
            f"作用: {self.description}\n"
            f"参数:\n{parameters_text}"
        )


@dataclass
class FunctionCall:
    """表示一次结构化工具调用请求。"""

    tool_name: str
    arguments: Dict[str, str]
    reason: str


@dataclass
class LoopDecision:
    """表示模型当前一步的决策。"""

    mode: str
    final_answer: Optional[str] = None
    function_call: Optional[FunctionCall] = None


class DemoToolbox:
    """教学型工具箱。"""

    def __init__(self) -> None:
        self.docs = {
            "rag": "RAG 是先检索资料，再把资料放进 Prompt，让模型基于资料回答。",
            "tool use": "Tool Use 是让 Agent 按需调用外部工具，并基于工具结果继续工作。",
            "agent loop": "Agent Loop 是模型循环执行决策、行动、观察，直到得到最终答案。",
        }
        self.memories = {
            "goal": "用户当前目标是在 90 天内从 Java 后端转型为 AI Agent 应用工程师。",
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
                description="用于精确数学计算。适合表达式求值。",
                parameters={"expression": "要计算的数学表达式，例如 18 * 6 + 3"},
            ),
            ToolSpec(
                name="doc_search",
                description="用于查询内置知识库。适合概念解释和资料查询。",
                parameters={"query": "用户想查询的知识问题"},
            ),
            ToolSpec(
                name="memory_lookup",
                description="用于查询长期记忆。适合用户目标、偏好等历史信息。",
                parameters={"query": "用户想查询的记忆问题"},
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


class LoopLLMModel:
    """使用真实 LLM 进行多轮决策。"""

    def __init__(
        self,
        toolbox: DemoToolbox,
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
                "[Day18AgentLoopDemo] 请提供 api_key，"
                "或在项目根目录的 .env 中配置 OPENAI_API_KEY"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def decide(self, user_input: str, scratchpad: List[Dict[str, str]], step_index: int) -> LoopDecision:
        """基于当前 scratchpad 决定下一步动作。"""
        scratchpad_text = self._format_scratchpad(scratchpad)
        system_prompt = (
            "你是一个教学型 Agent Loop 决策器。\n"
            "你需要根据用户问题、已有 observation 和工具描述，决定下一步动作。\n"
            "你每一步只能做两种选择之一：\n"
            "1. 调用一个工具\n"
            "2. 输出最终答案\n\n"
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
            return LoopDecision(
                mode="final_answer",
                final_answer=str(data.get("final_answer", "")).strip(),
            )

        if mode == "function_call":
            tool_name = str(data.get("tool_name", "")).strip()
            arguments = data.get("arguments", {})
            reason = str(data.get("reason", "")).strip() or "模型判断需要继续调用工具。"
            if tool_name not in self.toolbox.tools:
                raise ValueError(f"LLM 返回了未知工具: {tool_name}")
            if not isinstance(arguments, dict):
                raise ValueError("LLM 返回的 arguments 不是对象。")
            normalized_arguments = {
                str(key): str(value)
                for key, value in arguments.items()
                if value is not None
            }
            return LoopDecision(
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
                extra={"agent_name": "Day18AgentLoopDemo", "stage": stage},
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
                extra={"agent_name": "Day18AgentLoopDemo", "stage": stage},
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
                extra={"agent_name": "Day18AgentLoopDemo", "stage": stage},
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


class AgentLoopFunctionCallingAgent:
    """最小多轮 Function Calling Agent。"""

    def __init__(self, toolbox: DemoToolbox, llm_model: LoopLLMModel, max_steps: int = 4) -> None:
        self.toolbox = toolbox
        self.llm_model = llm_model
        self.max_steps = max_steps

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
            function_call = decision.function_call
            tool = self.toolbox.tools[function_call.tool_name]
            tool_input = next(iter(function_call.arguments.values()), "")
            observation = tool(tool_input)
            scratchpad.append(
                {
                    "tool_name": function_call.tool_name,
                    "tool_input": tool_input,
                    "observation": observation,
                }
            )
            trace.append(
                {
                    "step": step_index,
                    "mode": "function_call",
                    "tool_name": function_call.tool_name,
                    "arguments": function_call.arguments,
                    "reason": function_call.reason,
                    "observation": observation,
                }
            )

        return {
            "status": "max_steps_reached",
            "trace": trace,
            "final_answer": f"已达到最大步数 {self.max_steps}，系统被强制停止。",
        }


class AgentLoopRunner:
    """Day18 命令行入口。"""

    def __init__(self, agent: AgentLoopFunctionCallingAgent, toolbox: DemoToolbox) -> None:
        self.agent = agent
        self.toolbox = toolbox

    def run(self) -> None:
        print("=" * 72)
        print("Day18 Agent Loop Function Calling Demo")
        print("=" * 72)
        print("\n可用命令:")
        print("  status                    - 查看当前配置")
        print("  list-tools                - 查看工具描述")
        print("  ask <问题>                - 运行一次最小 Agent Loop")
        print("  demo-memory-doc           - 演示先查记忆再继续决策")
        print("  demo-calc                 - 演示计算类问题")
        print("  demo-direct               - 演示直接结束的问题")
        print("  review-all                - 依次查看全部案例")
        print("  quit/exit                 - 退出")
        print()

        while True:
            try:
                user_input = input("[day18-agent-loop]> ").strip()
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

                if lower_text == "demo-memory-doc":
                    self._run_case("先查一下我的学习目标，再解释一下 Tool Use 是什么。")
                    continue

                if lower_text == "demo-calc":
                    self._run_case("请帮我计算 25 * 4 + 6，并说明结果。")
                    continue

                if lower_text == "demo-direct":
                    self._run_case("为什么 max_steps 很重要？")
                    continue

                if lower_text == "review-all":
                    self._run_case("先查一下我的学习目标，再解释一下 Tool Use 是什么。")
                    self._run_case("请帮我计算 25 * 4 + 6，并说明结果。")
                    self._run_case("为什么 max_steps 很重要？")
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
            print(f"  参数: {json.dumps(spec.parameters, ensure_ascii=False)}")
        print()

    def _print_status(self) -> None:
        print("\n当前配置:")
        print("  - 当前仅支持真实 LLM Agent Loop 链路")
        print(f"  - 模型: {self.agent.llm_model.model}")
        print(f"  - Base URL: {self.agent.llm_model.base_url}")
        print(f"  - max_steps: {self.agent.max_steps}")
        print()


def main() -> None:
    toolbox = DemoToolbox()
    llm_model = LoopLLMModel(
        toolbox=toolbox,
        api_key=None,
        base_url=os.getenv("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"),
        model=os.getenv("OPENAI_MODEL", "qwen3.5-plus"),
    )
    agent = AgentLoopFunctionCallingAgent(
        toolbox=toolbox,
        llm_model=llm_model,
        max_steps=int(os.getenv("DAY18_MAX_STEPS", "4")),
    )
    runner = AgentLoopRunner(agent, toolbox)
    runner.run()


if __name__ == "__main__":
    main()
