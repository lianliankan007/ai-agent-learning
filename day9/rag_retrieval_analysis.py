#!/usr/bin/env python3
"""
Day9: 使用继承组织检索分析 Demo。

这个脚本的目标是：
1. 对比严格关键词检索和“问法增强”检索的差异
2. 用继承展示如何复用 Agent/RAG 的公共逻辑
3. 在配置了 API Key 时，继续走完整的最小 RAG 闭环
"""

from __future__ import annotations

import os
import re
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_markdown_logger import get_default_llm_logger
from utils.openai_config import resolve_openai_api_key

llm_logger = get_default_llm_logger()


class BaseRetrievalRAGAgent(ABC):
    """检索分析 Agent 的抽象基类。

    基类负责：
    - 构造知识库
    - 构造 Prompt
    - 可选调用 LLM
    - 定义统一的返回结构

    子类只需要关注“如何检索”。
    """

    agent_name = "BaseRetrievalRAGAgent"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://coding.dashscope.aliyuncs.com/v1",
        model: str = "qwen3.5-plus",
        temperature: float = 0.2,
        max_tokens: int = 500,
        top_k: int = 3,
    ):
        self.api_key = resolve_openai_api_key(api_key)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.knowledge_base = self._build_knowledge_base()

    def _build_knowledge_base(self) -> List[Dict[str, Any]]:
        """构造教学用知识库。

        这里额外加入 `question_variants`，方便子类做“问法增强”。
        """
        return [
            {
                "id": "kb1",
                "title": "Token",
                "content": "Token 是大模型处理文本时使用的基本单位。英文里一个 token 大约对应 0.75 个单词，中文通常一个字或词会占用一个或多个 token。",
                "keywords": ["token", "tokens", "分词", "词元"],
                "question_variants": [
                    "模型是按什么单位处理文本的",
                    "为什么提示词长度要看 token",
                ],
            },
            {
                "id": "kb2",
                "title": "Context Window",
                "content": "Context Window 指模型单次请求中最多能处理的上下文长度。上下文越长，模型一次能看到的信息越多，但成本也通常更高。",
                "keywords": ["context", "window", "上下文", "上下文窗口"],
                "question_variants": [
                    "模型一次最多能看到多少内容",
                    "为什么长对话会受长度限制",
                ],
            },
            {
                "id": "kb3",
                "title": "Function Calling",
                "content": "Function Calling 是让模型根据问题决定是否调用外部函数或工具，并把结构化参数交给程序执行的一种能力。",
                "keywords": ["function", "calling", "函数调用", "工具调用", "tool use"],
                "question_variants": [
                    "模型怎么决定调用工具",
                    "让大模型调用本地函数是什么意思",
                ],
            },
            {
                "id": "kb4",
                "title": "Agent Loop",
                "content": "Agent Loop 是一种循环执行机制：模型判断下一步动作，必要时调用工具，再根据工具结果继续判断，直到输出最终答案。",
                "keywords": ["agent loop", "loop", "agent", "循环", "智能体循环"],
                "question_variants": [
                    "智能体为什么要循环决策",
                    "Agent 为什么不是一步就结束",
                ],
            },
            {
                "id": "kb5",
                "title": "RAG",
                "content": "RAG 是 Retrieval-Augmented Generation 的缩写，核心流程是先检索相关资料，再把资料放进 prompt，最后让模型基于资料生成答案。",
                "keywords": ["rag", "retrieval", "generation", "检索增强", "检索增强生成"],
                "question_variants": [
                    "为什么回答前要先查资料",
                    "检索增强生成是什么意思",
                ],
            },
            {
                "id": "kb6",
                "title": "Memory",
                "content": "Memory 用来让 Agent 记住历史信息。短期记忆通常保存在对话上下文里，长期记忆通常保存在数据库或向量库里。",
                "keywords": ["memory", "记忆", "长期记忆", "短期记忆"],
                "question_variants": [
                    "智能体怎么记住之前的内容",
                    "短期记忆和长期记忆有什么区别",
                ],
            },
        ]

    @abstractmethod
    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """子类必须实现自己的检索逻辑。"""

    def build_prompt(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """把检索到的文档拼接到 Prompt。"""
        if retrieved_docs:
            knowledge_text = "\n\n".join(
                [
                    f"[知识{i}] {doc['title']}\n{doc['content']}\n命中原因: {'; '.join(doc['reasons'])}"
                    for i, doc in enumerate(retrieved_docs, start=1)
                ]
            )
        else:
            knowledge_text = "当前没有检索到明显相关的知识，请明确说明知识不足。"

        return (
            "你是一个教学型 AI 助手。\n"
            "请优先依据检索到的资料回答。\n"
            "如果当前知识不足，请直接说明“根据当前知识库无法准确回答”，不要编造。\n\n"
            f"【检索资料】\n{knowledge_text}\n\n"
            f"【用户问题】\n{question}"
        )

    def run_rag(self, question: str) -> Dict[str, Any]:
        """执行一次完整流程：检索、构造 Prompt、可选调用 LLM。"""
        retrieved_docs = self.retrieve(question)
        prompt = self.build_prompt(question, retrieved_docs)
        answer: Optional[str] = None

        # 如果没有配置 API Key，仍然允许做纯检索分析，不强制调用 LLM。
        if self.api_key:
            answer = self._call_api([{"role": "user", "content": prompt}])

        return {
            "agent_name": self.agent_name,
            "question": question,
            "retrieved_docs": retrieved_docs,
            "prompt": prompt,
            "answer": answer,
        }

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """调用兼容 OpenAI 的聊天接口，并记录 Markdown 日志。"""
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
                extra={"agent_name": self.agent_name},
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
                extra={"agent_name": self.agent_name},
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
                extra={"agent_name": self.agent_name},
            )
            raise Exception(f"解析响应失败: {exc}") from exc

    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        """做一个简单分词，用来支持教学型匹配。"""
        parts = re.split(r"[\s,，。？?、:：!！\-\(\)\[\]]+", text.lower())
        return [part for part in parts if part]

    @staticmethod
    def _safe_json(response: Optional[requests.Response]) -> Any:
        if response is None:
            return None
        try:
            return response.json()
        except ValueError:
            return {"raw_text": response.text}


class KeywordRAGAgent(BaseRetrievalRAGAgent):
    """严格关键词检索版本。"""

    agent_name = "KeywordRAGAgent"

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        lowered_question = question.lower()
        question_terms = self._expand_terms(self._extract_terms(question))
        scored_items: List[Dict[str, Any]] = []

        for item in self.knowledge_base:
            score = 0
            reasons: List[str] = []

            title_lower = item["title"].lower()
            content_lower = item["content"].lower()
            keywords_lower = [keyword.lower() for keyword in item["keywords"]]

            if title_lower in lowered_question:
                score += 4
                reasons.append(f"标题命中: {item['title']}")

            for keyword in keywords_lower:
                if keyword in lowered_question:
                    score += 3
                    reasons.append(f"关键词命中: {keyword}")

            for term in question_terms:
                if len(term) >= 2 and term in content_lower:
                    score += 1
                    reasons.append(f"内容词命中: {term}")

            score = self._apply_extra_matching(
                lowered_question=lowered_question,
                item=item,
                reasons=reasons,
                current_score=score,
            )

            if score > 0:
                scored_items.append(
                    {
                        "id": item["id"],
                        "title": item["title"],
                        "content": item["content"],
                        "score": score,
                        "reasons": reasons,
                    }
                )

        scored_items.sort(key=lambda row: row["score"], reverse=True)
        return scored_items[: self.top_k]

    def _expand_terms(self, question_terms: List[str]) -> List[str]:
        """基类版本不做扩展，直接返回原始词。"""
        return question_terms

    def _apply_extra_matching(
        self,
        *,
        lowered_question: str,
        item: Dict[str, Any],
        reasons: List[str],
        current_score: int,
    ) -> int:
        """留给子类扩展额外匹配逻辑。"""
        return current_score


class BoostedKeywordRAGAgent(KeywordRAGAgent):
    """基于继承增强出来的检索版本。

    它在严格关键词检索的基础上，额外增加两类启发式能力：
    1. 同义问法扩展
    2. question_variants 命中
    """

    agent_name = "BoostedKeywordRAGAgent"

    SYNONYM_MAP = {
        "智能体": ["agent"],
        "循环决策": ["loop", "循环"],
        "查资料": ["rag", "检索增强"],
        "检索增强生成": ["rag"],
        "记住": ["memory", "记忆"],
        "工具": ["tool use", "函数调用"],
    }

    def _expand_terms(self, question_terms: List[str]) -> List[str]:
        expanded_terms = list(question_terms)
        for term in question_terms:
            for key, aliases in self.SYNONYM_MAP.items():
                if key in term or term in key:
                    expanded_terms.extend(alias.lower() for alias in aliases)
        # 用 `dict.fromkeys` 去重，同时保留原始顺序，便于调试。
        return list(dict.fromkeys(expanded_terms))

    def _apply_extra_matching(
        self,
        *,
        lowered_question: str,
        item: Dict[str, Any],
        reasons: List[str],
        current_score: int,
    ) -> int:
        score = current_score
        for variant in item.get("question_variants", []):
            variant_lower = variant.lower()
            matched_parts = [
                part
                for part in self._extract_terms(variant_lower)
                if len(part) >= 2 and part in lowered_question
            ]
            if matched_parts:
                score += 2
                reasons.append(f"相近问法命中: {variant}")
                break
        return score


class RetrievalAnalysisRunner:
    """命令行 Runner，负责对比两种检索策略。"""

    def __init__(self, strict_agent: KeywordRAGAgent, boosted_agent: BoostedKeywordRAGAgent):
        self.strict_agent = strict_agent
        self.boosted_agent = boosted_agent

    def run(self) -> None:
        print("=" * 72)
        print("Day9 Retrieval Analysis Demo")
        print("=" * 72)
        print("\n可用命令:")
        print("  list-kb                 - 查看知识库标题")
        print("  compare <问题>          - 对比两种检索策略")
        print("  ask-strict <问题>       - 用严格关键词检索跑一次 RAG")
        print("  ask-boost <问题>        - 用增强检索跑一次 RAG")
        print("  demo-loop               - 测试“智能体为什么要循环决策”")
        print("  demo-rag                - 测试“为什么回答前要先查资料”")
        print("  demo-miss               - 测试知识库未命中问题")
        print("  quit/exit               - 退出程序")
        print()

        while True:
            try:
                user_input = input("[day9-analysis]> ").strip()
                if not user_input:
                    continue

                lower_text = user_input.lower()
                if lower_text in {"quit", "exit"}:
                    print("\n再见!")
                    break

                if lower_text == "list-kb":
                    self._print_knowledge_base()
                    continue

                if lower_text == "demo-loop":
                    self._run_compare("智能体为什么要循环决策？")
                    continue

                if lower_text == "demo-rag":
                    self._run_compare("为什么回答前要先查资料？")
                    continue

                if lower_text == "demo-miss":
                    self._run_compare("什么是 LangGraph？")
                    continue

                if lower_text.startswith("compare "):
                    self._run_compare(user_input[8:].strip())
                    continue

                if lower_text.startswith("ask-strict "):
                    self._run_rag(self.strict_agent, user_input[11:].strip())
                    continue

                if lower_text.startswith("ask-boost "):
                    self._run_rag(self.boosted_agent, user_input[10:].strip())
                    continue

                print("不支持的命令，请输入 compare / ask-strict / ask-boost / demo-* / quit\n")
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as exc:
                print(f"\n错误: {exc}\n")

    def _run_compare(self, question: str) -> None:
        print("\n" + "=" * 72)
        print(f"[Question] {question}")
        print("=" * 72)
        self._print_retrieval_result(self.strict_agent, question)
        self._print_retrieval_result(self.boosted_agent, question)
        print()

    def _run_rag(self, agent: BaseRetrievalRAGAgent, question: str) -> None:
        result = agent.run_rag(question)
        print("\n" + "=" * 72)
        print(f"[Agent] {result['agent_name']}")
        print(f"[Question] {question}")
        print("=" * 72)
        self._print_docs(result["retrieved_docs"])
        print("\n[Prompt Preview]")
        prompt = result["prompt"]
        print(prompt[:700] + ("..." if len(prompt) > 700 else ""))

        if result["answer"] is None:
            print("\n[Answer]")
            print("当前未配置 OPENAI_API_KEY，已完成检索分析和 Prompt 预览。")
        else:
            print("\n[Answer]")
            print(result["answer"])
        print()

    def _print_retrieval_result(self, agent: BaseRetrievalRAGAgent, question: str) -> None:
        print(f"\n[{agent.agent_name}]")
        docs = agent.retrieve(question)
        self._print_docs(docs)

    @staticmethod
    def _print_docs(docs: List[Dict[str, Any]]) -> None:
        if not docs:
            print("  - 未命中任何知识")
            return

        for doc in docs:
            print(f"  - {doc['title']} (score={doc['score']})")
            print(f"    reasons: {'; '.join(doc['reasons'])}")

    def _print_knowledge_base(self) -> None:
        print("\n当前知识库:")
        for item in self.strict_agent.knowledge_base:
            print(f"  - {item['title']}")
        print()


def main() -> None:
    strict_agent = KeywordRAGAgent(
        api_key=None,
        base_url=os.getenv("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"),
        model=os.getenv("OPENAI_MODEL", "qwen3.5-plus"),
    )
    boosted_agent = BoostedKeywordRAGAgent(
        api_key=None,
        base_url=os.getenv("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"),
        model=os.getenv("OPENAI_MODEL", "qwen3.5-plus"),
    )
    runner = RetrievalAnalysisRunner(strict_agent, boosted_agent)
    runner.run()


if __name__ == "__main__":
    main()
