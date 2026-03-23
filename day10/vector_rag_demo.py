#!/usr/bin/env python3
"""
Day10: 最小向量检索 RAG Demo。

这个脚本专门用来教学：
1. 关键词检索和向量检索的差异
2. 一个最小向量检索器是如何工作的
3. 如何用继承复用 RAG 的公共逻辑
"""

from __future__ import annotations

import math
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


class BaseRAGDemo(ABC):
    """RAG Demo 的抽象基类。

    基类负责：
    - 准备知识库
    - 构造 Prompt
    - 调用 LLM
    - 定义统一的 RAG 流程

    子类只负责实现各自的检索方式。
    """

    agent_name = "BaseRAGDemo"

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
        """构造教学知识库。"""
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

    def build_prompt(self, question: str, docs: List[Dict[str, Any]]) -> str:
        """把检索结果注入 Prompt。"""
        if docs:
            knowledge_text = "\n\n".join(
                [
                    f"[知识{i}] {doc['title']}\n{doc['content']}\nscore={doc['score']:.4f}\n原因: {'; '.join(doc['reasons'])}"
                    for i, doc in enumerate(docs, start=1)
                ]
            )
        else:
            knowledge_text = "当前知识库没有检索到明显相关的知识。"

        return (
            "你是一个教学型 AI 助手。\n"
            "请优先依据下面的检索资料回答。\n"
            "如果资料不足，请直接说明“根据当前知识库无法准确回答”，不要编造。\n\n"
            f"【检索资料】\n{knowledge_text}\n\n"
            f"【用户问题】\n{question}"
        )

    def run_rag(self, question: str) -> Dict[str, Any]:
        """执行一次完整的最小 RAG 流程。"""
        docs = self.retrieve(question)
        prompt = self.build_prompt(question, docs)
        answer: Optional[str] = None

        # 允许在没有 API Key 的情况下只做检索分析，不强制调用 LLM。
        if self.api_key:
            answer = self._call_api([{"role": "user", "content": prompt}])

        return {
            "agent_name": self.agent_name,
            "question": question,
            "docs": docs,
            "prompt": prompt,
            "answer": answer,
        }

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
        """做一个轻量分词。"""
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


class KeywordRAGDemo(BaseRAGDemo):
    """纯关键词检索版本。"""

    agent_name = "KeywordRAGDemo"

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        lowered_question = question.lower()
        question_terms = self._extract_terms(question)
        scored_rows: List[Dict[str, Any]] = []

        for item in self.knowledge_base:
            score = 0.0
            reasons: List[str] = []

            if item["title"].lower() in lowered_question:
                score += 4
                reasons.append(f"标题命中: {item['title']}")

            for keyword in item["keywords"]:
                keyword_lower = keyword.lower()
                if keyword_lower in lowered_question:
                    score += 3
                    reasons.append(f"关键词命中: {keyword}")

            for term in question_terms:
                if len(term) >= 2 and term in item["content"].lower():
                    score += 1
                    reasons.append(f"内容词命中: {term}")

            if score > 0:
                scored_rows.append(
                    {
                        "id": item["id"],
                        "title": item["title"],
                        "content": item["content"],
                        "score": score,
                        "reasons": reasons,
                    }
                )

        scored_rows.sort(key=lambda row: row["score"], reverse=True)
        return scored_rows[: self.top_k]


class VectorRAGDemo(BaseRAGDemo):
    """最小向量检索版本。

    这里不接入真实 embedding 模型，而是用一个“教学型向量化方法”：
    - 先做轻量分词
    - 再做同义词归一化
    - 最后构造词频向量并计算余弦相似度

    它不是工业级 embedding，但足够帮助理解向量检索流程。
    """

    agent_name = "VectorRAGDemo"

    SYNONYM_TO_CANONICAL = {
        "智能体": "agent",
        "agent": "agent",
        "循环决策": "loop",
        "循环": "loop",
        "loop": "loop",
        "查资料": "rag",
        "检索增强生成": "rag",
        "检索增强": "rag",
        "rag": "rag",
        "记住": "memory",
        "记忆": "memory",
        "memory": "memory",
        "工具": "tool",
        "工具调用": "tool",
        "函数调用": "tool",
        "tool": "tool",
        "上下文窗口": "context_window",
        "上下文": "context_window",
        "context": "context_window",
        "window": "context_window",
        "token": "token",
        "tokens": "token",
        "词元": "token",
        "分词": "token",
    }

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # 文档向量会在初始化阶段预先计算好，后续查询时直接复用。
        self.vocabulary = self._build_vocabulary()
        self.document_index = self._build_document_index()

    def _build_vocabulary(self) -> List[str]:
        """根据知识库构造词表。"""
        vocabulary: List[str] = []
        for item in self.knowledge_base:
            document_text = self._join_document_text(item)
            for token in self._tokenize_and_normalize(document_text):
                if token not in vocabulary:
                    vocabulary.append(token)
        return vocabulary

    def _build_document_index(self) -> List[Dict[str, Any]]:
        """预计算每条知识的向量。"""
        index_rows: List[Dict[str, Any]] = []
        for item in self.knowledge_base:
            document_text = self._join_document_text(item)
            vector = self._vectorize_text(document_text)
            index_rows.append(
                {
                    "id": item["id"],
                    "title": item["title"],
                    "content": item["content"],
                    "vector": vector,
                }
            )
        return index_rows

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """执行最小向量检索。"""
        query_vector = self._vectorize_text(question)
        query_tokens = self._tokenize_and_normalize(question)
        scored_rows: List[Dict[str, Any]] = []

        for item, index_row in zip(self.knowledge_base, self.document_index):
            similarity = self._cosine_similarity(query_vector, index_row["vector"])
            if similarity <= 0:
                continue

            reasons = [f"余弦相似度: {similarity:.4f}"]

            overlap_tokens = [
                token
                for token in query_tokens
                if token in self._tokenize_and_normalize(self._join_document_text(item))
            ]
            if overlap_tokens:
                reasons.append(f"归一化词命中: {', '.join(sorted(set(overlap_tokens)))}")

            scored_rows.append(
                {
                    "id": item["id"],
                    "title": item["title"],
                    "content": item["content"],
                    "score": similarity,
                    "reasons": reasons,
                }
            )

        scored_rows.sort(key=lambda row: row["score"], reverse=True)
        return scored_rows[: self.top_k]

    def inspect_vector(self, text: str) -> Dict[str, Any]:
        """把一段文本的向量化过程拆开，方便教学观察。"""
        tokens = self._tokenize_and_normalize(text)
        vector = self._vectorize_text(text)
        non_zero_items = [
            {"token": token, "value": value}
            for token, value in zip(self.vocabulary, vector)
            if value > 0
        ]
        return {
            "text": text,
            "tokens": tokens,
            "non_zero_items": non_zero_items,
        }

    def _join_document_text(self, item: Dict[str, Any]) -> str:
        """把标题、正文、关键词、相近问法拼成一条完整文档。"""
        parts = [item["title"], item["content"]]
        parts.extend(item.get("keywords", []))
        parts.extend(item.get("question_variants", []))
        return " ".join(parts)

    def _tokenize_and_normalize(self, text: str) -> List[str]:
        """分词并做同义词归一化。"""
        raw_terms = self._extract_terms(text)
        normalized_terms: List[str] = []

        for term in raw_terms:
            normalized = self.SYNONYM_TO_CANONICAL.get(term, term)
            # 太短的词很容易引入噪音，所以这里直接过滤掉。
            if len(normalized) >= 2:
                normalized_terms.append(normalized)

        return normalized_terms

    def _vectorize_text(self, text: str) -> List[float]:
        """把文本转换成最小词频向量。"""
        tokens = self._tokenize_and_normalize(text)
        token_count: Dict[str, float] = {}

        for token in tokens:
            token_count[token] = token_count.get(token, 0.0) + 1.0

        return [token_count.get(token, 0.0) for token in self.vocabulary]

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> float:
        """计算两个向量的余弦相似度。"""
        if not left or not right:
            return 0.0

        dot_product = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))

        if left_norm == 0 or right_norm == 0:
            return 0.0

        return dot_product / (left_norm * right_norm)


class VectorRAGRunner:
    """命令行入口。"""

    def __init__(self, keyword_demo: KeywordRAGDemo, vector_demo: VectorRAGDemo):
        self.keyword_demo = keyword_demo
        self.vector_demo = vector_demo

    def run(self) -> None:
        print("=" * 72)
        print("Day10 Minimal Vector RAG Demo")
        print("=" * 72)
        print("\n可用命令:")
        print("  list-kb                    - 查看知识库标题")
        print("  compare <问题>             - 对比关键词检索和向量检索")
        print("  ask-keyword <问题>         - 用关键词检索跑一次 RAG")
        print("  ask-vector <问题>          - 用向量检索跑一次 RAG")
        print("  show-vector <问题>         - 查看问题被如何向量化")
        print("  demo-loop                  - 测试“智能体为什么要循环决策”")
        print("  demo-rag                   - 测试“为什么回答前要先查资料”")
        print("  demo-memory                - 测试“智能体怎么记住之前的内容”")
        print("  quit/exit                  - 退出")
        print()

        while True:
            try:
                user_input = input("[day10-vector-rag]> ").strip()
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
                    self._compare("智能体为什么要循环决策？")
                    continue

                if lower_text == "demo-rag":
                    self._compare("为什么回答前要先查资料？")
                    continue

                if lower_text == "demo-memory":
                    self._compare("智能体怎么记住之前的内容？")
                    continue

                if lower_text.startswith("compare "):
                    self._compare(user_input[8:].strip())
                    continue

                if lower_text.startswith("ask-keyword "):
                    self._run_rag(self.keyword_demo, user_input[12:].strip())
                    continue

                if lower_text.startswith("ask-vector "):
                    self._run_rag(self.vector_demo, user_input[11:].strip())
                    continue

                if lower_text.startswith("show-vector "):
                    self._show_vector(user_input[12:].strip())
                    continue

                print("不支持的命令，请输入 compare / ask-keyword / ask-vector / show-vector / quit\n")
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as exc:
                print(f"\n错误: {exc}\n")

    def _compare(self, question: str) -> None:
        print("\n" + "=" * 72)
        print(f"[Question] {question}")
        print("=" * 72)
        self._print_docs("KeywordRAGDemo", self.keyword_demo.retrieve(question))
        self._print_docs("VectorRAGDemo", self.vector_demo.retrieve(question))
        print()

    def _run_rag(self, demo: BaseRAGDemo, question: str) -> None:
        result = demo.run_rag(question)
        print("\n" + "=" * 72)
        print(f"[Agent] {result['agent_name']}")
        print(f"[Question] {question}")
        print("=" * 72)
        self._print_docs(result["agent_name"], result["docs"])
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

    def _show_vector(self, question: str) -> None:
        inspected = self.vector_demo.inspect_vector(question)
        print("\n" + "=" * 72)
        print(f"[Vector Inspect] {question}")
        print("=" * 72)
        print(f"tokens: {inspected['tokens']}")
        print("non_zero_items:")
        for item in inspected["non_zero_items"]:
            print(f"  - {item['token']}: {item['value']}")
        print()

    @staticmethod
    def _print_docs(agent_name: str, docs: List[Dict[str, Any]]) -> None:
        print(f"\n[{agent_name}]")
        if not docs:
            print("  - 未命中任何知识")
            return

        for doc in docs:
            print(f"  - {doc['title']} (score={doc['score']:.4f})")
            print(f"    reasons: {'; '.join(doc['reasons'])}")

    def _print_knowledge_base(self) -> None:
        print("\n当前知识库:")
        for item in self.keyword_demo.knowledge_base:
            print(f"  - {item['title']}")
        print()


def main() -> None:
    keyword_demo = KeywordRAGDemo(
        api_key=None,
        base_url=os.getenv("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"),
        model=os.getenv("OPENAI_MODEL", "qwen3.5-plus"),
    )
    vector_demo = VectorRAGDemo(
        api_key=None,
        base_url=os.getenv("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"),
        model=os.getenv("OPENAI_MODEL", "qwen3.5-plus"),
    )
    runner = VectorRAGRunner(keyword_demo, vector_demo)
    runner.run()


if __name__ == "__main__":
    main()
