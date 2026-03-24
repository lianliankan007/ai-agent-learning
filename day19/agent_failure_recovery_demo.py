#!/usr/bin/env python3
"""
Day19: 工具失败处理、重试与降级教学 Demo。

这个脚本帮助你理解：
1. 工具调用不是“成功”这一种结果，还可能失败
2. 失败要区分“可重试”和“不可重试”
3. Agent 运行时常见恢复策略包括：重试、降级、最终解释失败原因
4. 真正可靠的 Agent，不只是会调工具，还要能在工具不稳定时继续工作
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
        """把工具定义转换成提示词中的说明文本。"""
        parameter_lines = [f"- {key}: {value}" for key, value in self.parameters.items()]
        parameters_text = "\n".join(parameter_lines) if parameter_lines else "- 无参数"
        return (
            f"工具名: {self.name}\n"
            f"作用: {self.description}\n"
            f"参数:\n{parameters_text}"
        )


@dataclass
class FunctionCall:
    """表示模型输出的一次结构化工具调用。"""

    tool_name: str
    arguments: Dict[str, str]
    reason: str


@dataclass
class LoopDecision:
    """表示当前一步的决策结果。"""

    mode: str
    final_answer: Optional[str] = None
    function_call: Optional[FunctionCall] = None


@dataclass
class ToolAttempt:
    """表示一次工具执行尝试。"""

    tool_name: str
    tool_input: str
    status: str
    observation: str
    attempt_index: int


@dataclass
class ToolExecutionSummary:
    """表示一次工具执行后的整体结果。"""

    final_status: str
    observation: str
    attempts: List[ToolAttempt]
    recovery_action: str


class ResilientToolbox:
    """教学型工具箱。

    这里特意加入“会失败的天气工具”，用来模拟真实系统里的外部 API 不稳定。
    """

    def __init__(self) -> None:
        self.docs = {
            "retry": "重试适合处理临时性错误，例如限流、超时、短暂网络抖动。",
            "fallback": "降级是在主工具失败后，改用缓存、默认答案或替代工具继续完成任务。",
            "tool failure": "工具失败需要区分可重试错误和不可重试错误，不能一律无限重试。",
            "agent loop": "Agent Loop 让模型可以在 observation 变化后继续决策，而不是只执行一次动作。",
        }
        self.memories = {
            "goal": "用户当前目标是在 90 天内从 Java 后端转型为 AI Agent 应用工程师。",
            "preference": "用户偏好中文、结构化、教学型解释。",
        }
        self.weather_cache = {
            "上海": "缓存天气：上海，晴转多云，22 摄氏度，东风 3 级。",
            "北京": "缓存天气：北京，多云，18 摄氏度，北风 2 级。",
            "广州": "缓存天气：广州，小雨，26 摄氏度，湿度较高。",
        }
        self.weather_attempt_counter: Dict[str, int] = {}
        self.tools: Dict[str, Callable[[str], tuple[str, str]]] = {
            "weather_api": self.weather_api,
            "weather_cache_lookup": self.weather_cache_lookup,
            "doc_search": self.doc_search,
            "memory_lookup": self.memory_lookup,
            "calculator": self.calculator,
        }
        self.tool_specs = [
            ToolSpec(
                name="weather_api",
                description="查询实时天气。可能出现限流、超时或城市不支持等失败。",
                parameters={"location": "要查询的城市，例如 上海、北京"},
            ),
            ToolSpec(
                name="weather_cache_lookup",
                description="查询缓存天气。适合 weather_api 失败后的降级兜底。",
                parameters={"location": "要查询的城市，例如 上海、北京"},
            ),
            ToolSpec(
                name="doc_search",
                description="查询内置知识库。适合解释 retry、fallback、Agent 等概念。",
                parameters={"query": "用户的知识查询语句"},
            ),
            ToolSpec(
                name="memory_lookup",
                description="查询长期记忆。适合查询用户目标或偏好。",
                parameters={"query": "用户的记忆查询语句"},
            ),
            ToolSpec(
                name="calculator",
                description="执行精确数学计算。",
                parameters={"expression": "要计算的数学表达式，例如 18 * 6 + 3"},
            ),
        ]

    def get_tool_specs_text(self) -> str:
        return "\n\n".join(spec.to_prompt_block() for spec in self.tool_specs)

    def weather_api(self, location: str) -> tuple[str, str]:
        """模拟一个不稳定的天气 API。

        返回值是二元组：
        - status: success / retryable_error / fatal_error
        - observation: 给 Agent 的 observation 文本
        """
        normalized_location = location.strip()
        current_attempt = self.weather_attempt_counter.get(normalized_location, 0) + 1
        self.weather_attempt_counter[normalized_location] = current_attempt

        if not normalized_location:
            return "fatal_error", "天气工具失败：location 不能为空。"

        if normalized_location == "上海":
            if current_attempt == 1:
                return "retryable_error", "天气 API 临时限流（模拟 429），建议稍后重试。"
            return "success", "实时天气：上海，晴转多云，23 摄氏度，空气质量良。"

        if normalized_location == "广州":
            if current_attempt <= 2:
                return "retryable_error", "天气 API 请求超时（模拟 timeout），本次结果不可靠。"
            return "success", "实时天气：广州，中雨，27 摄氏度，湿度 84%。"

        if normalized_location == "北京":
            return "success", "实时天气：北京，多云，19 摄氏度，风力较小。"

        return "fatal_error", f"天气 API 不支持城市: {normalized_location}。"

    def weather_cache_lookup(self, location: str) -> tuple[str, str]:
        normalized_location = location.strip()
        if normalized_location in self.weather_cache:
            return "success", self.weather_cache[normalized_location]
        return "fatal_error", f"缓存中没有 {normalized_location} 的天气数据。"

    def doc_search(self, query: str) -> tuple[str, str]:
        lowered = query.lower()
        hits = []
        for key, value in self.docs.items():
            if key in lowered or key.replace(" ", "") in lowered:
                hits.append(f"{key}: {value}")
        if "重试" in query:
            hits.append(f"retry: {self.docs['retry']}")
        if "降级" in query or "兜底" in query:
            hits.append(f"fallback: {self.docs['fallback']}")
        if "失败" in query:
            hits.append(f"tool failure: {self.docs['tool failure']}")
        if not hits:
            return "success", "知识库中没有直接命中的内容。"
        return "success", " | ".join(dict.fromkeys(hits))

    def memory_lookup(self, query: str) -> tuple[str, str]:
        lowered = query.lower()
        hits = []
        if "目标" in query or "goal" in lowered:
            hits.append(self.memories["goal"])
        if "偏好" in query or "preference" in lowered:
            hits.append(self.memories["preference"])
        if not hits:
            return "success", "没有命中相关长期记忆。"
        return "success", " | ".join(hits)

    def calculator(self, expression: str) -> tuple[str, str]:
        safe_expression = expression.strip()
        if not re.fullmatch(r"[\d\.\+\-\*\/\(\)\s]+", safe_expression):
            return "fatal_error", "计算失败：表达式不合法。"
        try:
            result = eval(safe_expression, {"__builtins__": {}}, {})
            return "success", f"计算结果: {result}"
        except Exception as exc:
            return "fatal_error", f"计算失败: {exc}"


class RecoveryAwareLLMModel:
    """使用真实 LLM 决定下一步动作。"""

    def __init__(
        self,
        toolbox: ResilientToolbox,
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
                "[Day19RecoveryAgent] 请提供 api_key，"
                "或在项目根目录的 .env 中配置 OPENAI_API_KEY"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def decide(self, user_input: str, scratchpad: List[Dict[str, str]], step_index: int) -> LoopDecision:
        scratchpad_text = self._format_scratchpad(scratchpad)
        system_prompt = (
            "你是一个教学型 Agent 决策器，重点处理工具失败后的恢复。\n"
            "你需要根据用户问题、已有 observation 和工具描述，决定下一步动作。\n"
            "你每一步只能做两种选择之一：\n"
            "1. 调用一个工具\n"
            "2. 输出最终答案\n\n"
            "决策要求：\n"
            "- 如果问题需要外部数据，优先选择合适工具\n"
            "- 如果 scratchpad 里已经有足够 observation，可以直接给最终答案\n"
            "- 如果工具失败了，要结合失败信息说明是否还能回答，以及回答的可靠性\n"
            "- 只能从给定工具中选择，不能编造新工具\n\n"
            f"【工具描述】\n{self.toolbox.get_tool_specs_text()}\n\n"
            "输出 JSON 格式要求如下：\n"
            "1. 如果继续调用工具：\n"
            '{"mode":"function_call","tool_name":"工具名","arguments":{"参数名":"参数值"},"reason":"原因"}\n'
            "2. 如果已经足够回答：\n"
            '{"mode":"final_answer","final_answer":"给用户的最终回答"}\n'
            "不要输出 Markdown 代码块，不要添加额外解释。"
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
                extra={"agent_name": "Day19RecoveryAgent", "stage": stage},
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
                extra={"agent_name": "Day19RecoveryAgent", "stage": stage},
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
                extra={"agent_name": "Day19RecoveryAgent", "stage": stage},
            )
            raise RuntimeError(f"LLM 响应解析失败: {exc}") from exc

    @staticmethod
    def _format_scratchpad(scratchpad: List[Dict[str, str]]) -> str:
        if not scratchpad:
            return "暂无 observation。"
        lines = []
        for index, item in enumerate(scratchpad, start=1):
            lines.append(
                f"{index}. tool={item['tool_name']}, input={item['tool_input']}, status={item['status']}, "
                f"recovery={item['recovery_action']}, observation={item['observation']}"
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


class RecoveryAwareAgent:
    """带有失败恢复能力的最小 Agent。"""

    def __init__(
        self,
        toolbox: ResilientToolbox,
        llm_model: RecoveryAwareLLMModel,
        max_steps: int = 4,
        max_retry_per_tool: int = 2,
    ) -> None:
        self.toolbox = toolbox
        self.llm_model = llm_model
        self.max_steps = max_steps
        self.max_retry_per_tool = max_retry_per_tool
        self.fallback_tool_map = {
            "weather_api": "weather_cache_lookup",
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
            execution = self._execute_with_recovery(decision.function_call)
            tool_input = next(iter(decision.function_call.arguments.values()), "")
            scratchpad.append(
                {
                    "tool_name": decision.function_call.tool_name,
                    "tool_input": tool_input,
                    "status": execution.final_status,
                    "recovery_action": execution.recovery_action,
                    "observation": execution.observation,
                }
            )
            trace.append(
                {
                    "step": step_index,
                    "mode": "function_call",
                    "tool_name": decision.function_call.tool_name,
                    "arguments": decision.function_call.arguments,
                    "reason": decision.function_call.reason,
                    "final_status": execution.final_status,
                    "recovery_action": execution.recovery_action,
                    "attempts": [attempt.__dict__ for attempt in execution.attempts],
                    "observation": execution.observation,
                }
            )

        return {
            "status": "max_steps_reached",
            "trace": trace,
            "final_answer": f"已达到最大步数 {self.max_steps}，系统被强制停止。",
        }

    def _execute_with_recovery(self, function_call: FunctionCall) -> ToolExecutionSummary:
        """执行工具，并在需要时执行重试或降级。

        这里故意把恢复逻辑放在程序层，而不是完全交给 LLM，
        是为了让你看到：很多鲁棒性策略属于工程控制，而不是语言模型自由发挥。
        """
        tool_name = function_call.tool_name
        tool_input = next(iter(function_call.arguments.values()), "")
        attempts: List[ToolAttempt] = []
        tool = self.toolbox.tools[tool_name]

        for attempt_index in range(1, self.max_retry_per_tool + 2):
            status, observation = tool(tool_input)
            attempts.append(
                ToolAttempt(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    status=status,
                    observation=observation,
                    attempt_index=attempt_index,
                )
            )

            if status == "success":
                return ToolExecutionSummary(
                    final_status="success",
                    observation=observation,
                    attempts=attempts,
                    recovery_action="主工具调用成功，无需恢复。",
                )

            if status == "retryable_error" and attempt_index <= self.max_retry_per_tool:
                continue

            if status == "retryable_error":
                fallback_summary = self._try_fallback(tool_name=tool_name, tool_input=tool_input, attempts=attempts)
                if fallback_summary is not None:
                    return fallback_summary
                return ToolExecutionSummary(
                    final_status="retry_exhausted",
                    observation=f"主工具多次重试后仍失败。最后一次错误: {observation}",
                    attempts=attempts,
                    recovery_action="已达到最大重试次数，且没有可用降级结果。",
                )

            if status == "fatal_error":
                fallback_summary = self._try_fallback(tool_name=tool_name, tool_input=tool_input, attempts=attempts)
                if fallback_summary is not None:
                    return fallback_summary
                return ToolExecutionSummary(
                    final_status="fatal_error",
                    observation=observation,
                    attempts=attempts,
                    recovery_action="错误不可重试，且没有可用降级结果。",
                )

        return ToolExecutionSummary(
            final_status="fatal_error",
            observation="工具执行进入了未覆盖分支。",
            attempts=attempts,
            recovery_action="请检查恢复逻辑实现。",
        )

    def _try_fallback(
        self,
        tool_name: str,
        tool_input: str,
        attempts: List[ToolAttempt],
    ) -> Optional[ToolExecutionSummary]:
        fallback_tool_name = self.fallback_tool_map.get(tool_name)
        if not fallback_tool_name:
            return None

        fallback_tool = self.toolbox.tools[fallback_tool_name]
        fallback_status, fallback_observation = fallback_tool(tool_input)
        attempts.append(
            ToolAttempt(
                tool_name=fallback_tool_name,
                tool_input=tool_input,
                status=fallback_status,
                observation=fallback_observation,
                attempt_index=1,
            )
        )
        if fallback_status == "success":
            return ToolExecutionSummary(
                final_status="fallback_success",
                observation=fallback_observation,
                attempts=attempts,
                recovery_action=f"主工具失败后，已降级到 {fallback_tool_name}。",
            )
        return ToolExecutionSummary(
            final_status="fallback_failed",
            observation=f"降级工具也失败了: {fallback_observation}",
            attempts=attempts,
            recovery_action=f"尝试降级到 {fallback_tool_name}，但兜底失败。",
        )


class RecoveryRunner:
    """Day19 命令行入口。"""

    def __init__(self, agent: RecoveryAwareAgent, toolbox: ResilientToolbox) -> None:
        self.agent = agent
        self.toolbox = toolbox

    def run(self) -> None:
        print("=" * 72)
        print("Day19 Agent Failure Recovery Demo")
        print("=" * 72)
        print("\n可用命令:")
        print("  status                    - 查看当前配置")
        print("  list-tools                - 查看工具描述")
        print("  ask <问题>                - 运行一次带恢复能力的 Agent")
        print("  demo-retry-success        - 演示重试后成功")
        print("  demo-fallback-success     - 演示主工具失败后降级成功")
        print("  demo-fallback-failed      - 演示主工具和降级都失败")
        print("  demo-direct               - 演示无需工具的直接回答")
        print("  review-all                - 依次查看全部案例")
        print("  quit/exit                 - 退出")
        print()

        while True:
            try:
                user_input = input("[day19-recovery]> ").strip()
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

                if lower_text == "demo-retry-success":
                    self._run_case("请查询上海天气，并说明这次 Agent 是如何处理临时失败的。")
                    continue

                if lower_text == "demo-fallback-success":
                    self._run_case("请查询广州天气。如果实时工具失败，请继续给我一个可靠但有边界的回答。")
                    continue

                if lower_text == "demo-fallback-failed":
                    self._run_case("请查询火星天气。如果失败，请明确说明系统为什么没法继续提供真实结果。")
                    continue

                if lower_text == "demo-direct":
                    self._run_case("为什么 Agent 里的重试和降级不能无限执行？")
                    continue

                if lower_text == "review-all":
                    self._run_case("请查询上海天气，并说明这次 Agent 是如何处理临时失败的。")
                    self._run_case("请查询广州天气。如果实时工具失败，请继续给我一个可靠但有边界的回答。")
                    self._run_case("请查询火星天气。如果失败，请明确说明系统为什么没法继续提供真实结果。")
                    self._run_case("为什么 Agent 里的重试和降级不能无限执行？")
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
                print(f"  final_status={item['final_status']}")
                print(f"  recovery_action={item['recovery_action']}")
                print("  attempts:")
                for attempt in item["attempts"]:
                    print(
                        "    "
                        f"- try={attempt['attempt_index']} tool={attempt['tool_name']} "
                        f"status={attempt['status']} observation={attempt['observation']}"
                    )
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
        print("  - 当前仅支持真实 LLM 决策链路")
        print(f"  - 模型: {self.agent.llm_model.model}")
        print(f"  - Base URL: {self.agent.llm_model.base_url}")
        print(f"  - max_steps: {self.agent.max_steps}")
        print(f"  - max_retry_per_tool: {self.agent.max_retry_per_tool}")
        print("  - 主工具失败后会尝试程序层重试与降级")
        print()


def main() -> None:
    toolbox = ResilientToolbox()
    llm_model = RecoveryAwareLLMModel(
        toolbox=toolbox,
        api_key=None,
        base_url=os.getenv("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"),
        model=os.getenv("OPENAI_MODEL", "qwen3.5-plus"),
    )
    agent = RecoveryAwareAgent(
        toolbox=toolbox,
        llm_model=llm_model,
        max_steps=int(os.getenv("DAY19_MAX_STEPS", "4")),
        max_retry_per_tool=int(os.getenv("DAY19_MAX_RETRY", "2")),
    )
    runner = RecoveryRunner(agent, toolbox)
    runner.run()


if __name__ == "__main__":
    main()
