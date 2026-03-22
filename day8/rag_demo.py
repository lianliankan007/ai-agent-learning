#!/usr/bin/env python3
"""
Day8: Minimal RAG teaching demo.

This version uses:
1. a tiny built-in knowledge base
2. simple keyword-based retrieval
3. prompt injection with retrieved context
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_markdown_logger import get_default_llm_logger
from utils.openai_config import resolve_openai_api_key

llm_logger = get_default_llm_logger()


class SimpleRAGDemo:
    """教学型最小 RAG Demo。"""

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
        if not self.api_key:
            raise ValueError(
                "请提供 api_key，或在项目根目录的 .env 中配置 OPENAI_API_KEY"
            )

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.knowledge_base = self._build_knowledge_base()

    def _build_knowledge_base(self) -> List[Dict[str, Any]]:
        """准备一个小型教学知识库。"""
        return [
            {
                "id": "kb1",
                "title": "Token",
                "content": "Token 是大模型处理文本时使用的基本单位。英文里一个 token 大约对应 0.75 个单词，中文通常一个字或词会占用一个或多个 token。",
                "keywords": ["token", "tokens", "tokenizer", "分词", "词元"],
            },
            {
                "id": "kb2",
                "title": "Context Window",
                "content": "Context Window 指模型单次请求中最多能处理的上下文长度。上下文越长，模型一次能看到的信息越多，但成本也通常更高。",
                "keywords": ["context", "window", "上下文", "上下文窗口"],
            },
            {
                "·id": "kb3",
                "title": "Function Calling",
                "content": "Function Calling 是让模型根据问题决定是否调用外部函数或工具，并把结构化参数交给程序执行的一种能力。",
                "keywords": ["function", "calling", "函数调用", "工具调用", "tool use"],
            },
            {
                "id": "kb4",
                "title": "Agent Loop",
                "content": "Agent Loop 是一种循环执行机制：模型判断下一步动作，必要时调用工具，再根据工具结果继续判断，直到输出最终答案。",
                "keywords": ["agent loop", "loop", "agent", "循环", "智能体循环"],
            },
            {
                "id": "kb5",
                "title": "RAG",
                "content": "RAG 是 Retrieval-Augmented Generation 的缩写，核心流程是先检索相关资料，再把资料放进 prompt，最后让模型基于资料生成答案。",
                "keywords": ["rag", "retrieval", "generation", "检索增强", "检索增强生成"],
            },
            {
                "id": "kb6",
                "title": "Memory",
                "content": "Memory 用来让 Agent 记住历史信息。短期记忆通常保存在对话上下文里，长期记忆通常保存在数据库或向量库里。",
                "keywords": ["memory", "记忆", "长期记忆", "短期记忆"],
            },
            {
                "id": "kb7",
                "title": "Planning / Action / Observation",
                "content": "在 Agent 中，Planning 表示判断下一步做什么，Action 表示真正执行动作，Observation 表示执行后获得的新信息。",
                "keywords": ["planning", "action", "observation", "规划", "观察"],
            },
        ]

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """用简单关键词匹配实现最小检索器。"""
        lowered_question = question.lower()
        question_terms = self._extract_terms(question)
        scored_items: List[Dict[str, Any]] = []

        for item in self.knowledge_base:
            score = 0

            title_lower = item["title"].lower()
            content_lower = item["content"].lower()
            keywords_lower = [keyword.lower() for keyword in item["keywords"]]

            if title_lower in lowered_question:
                score += 4

            for keyword in keywords_lower:
                if keyword in lowered_question:
                    score += 3

            for term in question_terms:
                if len(term) >= 2 and term in content_lower:
                    score += 1

            if score > 0:
                scored_items.append(
                    {
                        "id": item["id"],
                        "title": item["title"],
                        "content": item["content"],
                        "score": score,
                    }
                )

        scored_items.sort(key=lambda item: item["score"], reverse=True)
        return scored_items[: self.top_k]

    def build_prompt(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """把检索结果注入 prompt。"""
        if retrieved_docs:
            knowledge_text = "\n\n".join(
                [
                    f"[知识{i}] {doc['title']}\n{doc['content']}"
                    for i, doc in enumerate(retrieved_docs, start=1)
                ]
            )
        else:
            knowledge_text = "当前知识库中没有检索到与该问题明显相关的资料。"

        return (
            "你是一个教学型 AI 助手。\n"
            "请优先基于下面提供的知识回答用户问题。\n"
            "如果知识不足，请明确说明“根据当前知识库无法准确回答”，不要编造。\n\n"
            f"【检索到的知识】\n{knowledge_text}\n\n"
            f"【用户问题】\n{question}"
        )

    def ask(self, question: str) -> str:
        """执行一次完整的最小 RAG 流程。"""
        retrieved_docs = self.retrieve(question)
        prompt = self.build_prompt(question, retrieved_docs)

        print("\n" + "=" * 64)
        print(f"[Question] {question}")
        print("[Retrieve] 命中的知识片段:")
        if not retrieved_docs:
            print("  - 未命中知识库")
        else:
            for doc in retrieved_docs:
                print(f"  - {doc['title']} (score={doc['score']})")

        print("\n[Prompt Preview]")
        print(prompt[:500] + ("..." if len(prompt) > 500 else ""))

        answer = self._call_api(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )
        print(f"\n[Answer] {answer}")
        return answer

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """调用兼容 OpenAI 的聊天接口。"""
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
                extra={"agent_name": "SimpleRAGDemo"},
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
                extra={"agent_name": "SimpleRAGDemo"},
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
                extra={"agent_name": "SimpleRAGDemo"},
            )
            raise Exception(f"解析响应失败: {exc}") from exc

    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        """提取简单关键词，先服务教学理解。"""
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


class SimpleRAGRunner:
    """命令行入口。"""

    def __init__(self, rag_demo: SimpleRAGDemo):
        self.rag_demo = rag_demo

    def run(self) -> None:
        print("=" * 64)
        print("Day8 Minimal RAG Demo")
        print("=" * 64)
        print("\n可用命令:")
        print("  list-kb            - 查看当前知识库标题")
        print("  demo-token         - 测试 Token 问题")
        print("  demo-loop          - 测试 Agent Loop 问题")
        print("  demo-rag           - 测试 RAG 问题")
        print("  demo-miss          - 测试知识库未命中问题")
        print("  quit/exit          - 退出程序")
        print("  <任意文字>         - 直接提问\n")

        while True:
            try:
                user_input = input("[day8-rag]> ").strip()
                if not user_input:
                    continue

                lower_text = user_input.lower()
                if lower_text in {"quit", "exit"}:
                    print("\n再见!")
                    break

                if lower_text == "list-kb":
                    print("\n当前知识库:")
                    for item in self.rag_demo.knowledge_base:
                        print(f"  - {item['title']}")
                    print()
                    continue

                if lower_text == "demo-token":
                    user_input = "什么是 Token？"
                elif lower_text == "demo-loop":
                    user_input = "什么是 Agent Loop？"
                elif lower_text == "demo-rag":
                    user_input = "什么是 RAG？"
                elif lower_text == "demo-miss":
                    user_input = "什么是 LangChain？"

                self.rag_demo.ask(user_input)
                print()
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as exc:
                print(f"\n错误: {exc}\n")


def main() -> None:
    rag_demo = SimpleRAGDemo(
        api_key=None,
        base_url=os.getenv("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"),
        model=os.getenv("OPENAI_MODEL", "qwen3.5-plus"),
    )
    runner = SimpleRAGRunner(rag_demo)
    runner.run()


if __name__ == "__main__":
    main()
