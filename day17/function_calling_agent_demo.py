#!/usr/bin/env python3
"""
Day17: Function Calling 教学 Demo。

这个脚本只使用真实 LLM 生成结构化 tool call。
获取 API Key 的方式与 Day1、Day2 保持一致：
- 优先使用显式传入的 api_key
- 否则通过 utils.openai_config.resolve_openai_api_key 从项目根目录 .env
  或环境变量中读取 OPENAI_API_KEY
- 如果没有读取到，就在初始化阶段直接报错
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
        """把工具定义整理成适合放进提示词的文本。"""
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
class ModelDecision:
    """表示模型阶段的输出。"""

    mode: str
    text_answer: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    source: str = "llm"


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

    def get_tool_specs_text(self) -> str:
        """把所有工具描述拼成一段提示词。"""
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


class LLMFunctionCallingModel:
    """使用真实 LLM 生成结构化工具调用。"""

    def __init__(
        self,
        toolbox: DemoToolbox,
        api_key: Optional[str] = None,
        base_url: str = "https://coding.dashscope.aliyuncs.com/v1",
        model: str = "qwen3.5-plus",
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> None:
        self.toolbox = toolbox
        self.api_key = resolve_openai_api_key(api_key)
        if not self.api_key:
            raise ValueError(
                "[Day17FunctionCallingAgent] 请提供 api_key，"
                "或在项目根目录的 .env 中配置 OPENAI_API_KEY"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def is_available(self) -> bool:
        """是否已配置可用的 LLM。"""
        return bool(self.api_key)

    def decide(self, user_input: str) -> ModelDecision:
        """让 LLM 决定是直接回答，还是输出结构化 tool call。"""
        if not self.is_available:
            raise RuntimeError("未配置 OPENAI_API_KEY，无法使用真实 LLM 决策。")

        system_prompt = (
            "你是一个教学型 Function Calling 决策器。\n"
            "你需要根据用户问题，在“直接回答”与“调用工具”之间做选择。\n"
            "如果需要调用工具，必须输出结构化 JSON；如果不需要，也必须输出结构化 JSON。\n"
            "你只能从下面给定的工具里选择，不能编造不存在的工具。\n\n"
            f"【工具描述】\n{self.toolbox.get_tool_specs_text()}\n\n"
            "输出 JSON 格式要求如下：\n"
            "1. 如果需要调用工具：\n"
            '{"mode":"function_call","tool_name":"工具名","arguments":{"参数名":"参数值"},"reason":"原因"}\n'
            "2. 如果不需要调用工具：\n"
            '{"mode":"text_answer","text_answer":"给用户的直接回答"}\n'
            "不要输出 Markdown 代码块，不要添加额外解释。"
        )
        content = self._call_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            stage="decision",
        )
        data = self._parse_json_object(content)
        mode = str(data.get("mode", "")).strip()

        if mode == "text_answer":
            return ModelDecision(
                mode="text_answer",
                text_answer=str(data.get("text_answer", "")).strip(),
                source="llm",
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
            return ModelDecision(
                mode="function_call",
                function_call=FunctionCall(
                    tool_name=tool_name,
                    arguments=normalized_arguments,
                    reason=reason,
                ),
                source="llm",
            )

        raise ValueError(f"LLM 返回了不支持的 mode: {mode}")

    def compose_final_answer(
        self,
        user_input: str,
        function_call: FunctionCall,
        observation: str,
    ) -> str:
        """让 LLM 基于 observation 组织更自然的最终回答。"""
        if not self.is_available:
            raise RuntimeError("未配置 OPENAI_API_KEY，无法使用真实 LLM 组织最终回答。")

        system_prompt = (
            "你是一个教学型 AI 助手。\n"
            "你已经拿到了工具调用结果 observation。\n"
            "请基于 observation 回答用户问题，并清楚说明：\n"
            "1. 系统调用了什么工具\n"
            "2. observation 提供了什么信息\n"
            "3. 最终结论是什么\n"
            "请保持中文、结构清晰、适合初学者阅读。"
        )
        user_prompt = (
            f"用户问题: {user_input}\n"
            f"工具名: {function_call.tool_name}\n"
            f"工具参数: {json.dumps(function_call.arguments, ensure_ascii=False)}\n"
            f"observation: {observation}"
        )
        return self._call_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stage="final_answer",
        )

    def _call_chat(self, messages: list[dict[str, str]], stage: str) -> str:
        """调用兼容 OpenAI 的聊天接口，并记录日志。"""
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
                extra={"agent_name": "Day17FunctionCallingAgent", "stage": stage},
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
                extra={"agent_name": "Day17FunctionCallingAgent", "stage": stage},
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
                extra={"agent_name": "Day17FunctionCallingAgent", "stage": stage},
            )
            raise RuntimeError(f"LLM 响应解析失败: {exc}") from exc

    @staticmethod
    def _parse_json_object(text: str) -> Dict[str, Any]:
        """从 LLM 输出中提取 JSON 对象。"""
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


class FunctionCallingAgent:
    """最小 Function Calling Agent。"""

    def __init__(
        self,
        toolbox: DemoToolbox,
        llm_model: LLMFunctionCallingModel,
    ) -> None:
        self.toolbox = toolbox
        self.llm_model = llm_model

    def handle(self, user_input: str) -> Dict[str, str]:
        decision = self.llm_model.decide(user_input)

        if decision.mode == "text_answer":
            return {
                "mode": "text_answer",
                "decision_source": decision.source,
                "model_output": decision.text_answer or "",
                "final_answer": decision.text_answer or "",
            }

        assert decision.function_call is not None
        function_call = decision.function_call
        if function_call.tool_name not in self.toolbox.tools:
            raise ValueError(f"工具不存在: {function_call.tool_name}")
        tool = self.toolbox.tools[function_call.tool_name]
        tool_value = next(iter(function_call.arguments.values()), "")
        observation = tool(tool_value)
        final_answer = self.llm_model.compose_final_answer(user_input, function_call, observation)
        return {
            "mode": "function_call",
            "decision_source": decision.source,
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
        print("  status                    - 查看当前是否启用真实 LLM")
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

                if lower_text == "status":
                    self._print_status()
                    continue

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
        print(f"决策来源: {result.get('decision_source', 'unknown')}")
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

    def _print_status(self) -> None:
        llm_enabled = self.agent.llm_model.is_available
        print("\n当前模式:")
        print("  - 当前仅支持真实 LLM Function Calling 链路")
        print(f"  - 当前真实 LLM 可用: {llm_enabled}")
        print(f"  - 模型: {self.agent.llm_model.model}")
        print(f"  - Base URL: {self.agent.llm_model.base_url}")
        print()


def main() -> None:
    toolbox = DemoToolbox()
    llm_model = LLMFunctionCallingModel(
        toolbox=toolbox,
        api_key=None,
        base_url=os.getenv("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"),
        model=os.getenv("OPENAI_MODEL", "qwen3.5-plus"),
    )
    agent = FunctionCallingAgent(toolbox, llm_model=llm_model)
    runner = FunctionCallingRunner(agent, toolbox)
    runner.run()


if __name__ == "__main__":
    main()
