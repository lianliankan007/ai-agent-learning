#!/usr/bin/env python3
"""
Day11: 真实向量检索 RAG Demo。

这一天不再停留在“教学型伪向量检索”，而是直接接入：
1. Ollama embedding 模型
2. Qdrant 本地向量库
3. 一个最小但真实可运行的 RAG 检索流程
"""

from __future__ import annotations

import os
import re
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_markdown_logger import get_default_llm_logger
from utils.openai_config import resolve_openai_api_key

llm_logger = get_default_llm_logger()


class BaseRAGDemo(ABC):
    """RAG Demo 抽象基类。

    这里把“知识库、Prompt、RAG 主流程、LLM 调用”放到基类里，
    子类只需要关注“怎么检索”。
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
                "content": "Token 是大模型处理文本时使用的基本单位。提示词长度、输出长度和费用通常都和 token 数量有关。",
                "keywords": ["token", "tokens", "词元", "分词"],
                "question_variants": [
                    "模型按什么单位处理文本",
                    "为什么提示词长度要看 token",
                ],
            },
            {
                "id": "kb2",
                "title": "Context Window",
                "content": "Context Window 指模型一次请求最多能看到的上下文长度。上下文越长，单次能处理的信息越多。",
                "keywords": ["context", "window", "上下文", "上下文窗口"],
                "question_variants": [
                    "模型一次最多能看到多少内容",
                    "为什么长对话会受长度限制",
                ],
            },
            {
                "id": "kb3",
                "title": "Function Calling",
                "content": "Function Calling 是让模型决定是否调用外部工具，并输出结构化参数，再由程序真正执行。",
                "keywords": ["function calling", "tool use", "函数调用", "工具调用"],
                "question_variants": [
                    "模型怎么决定调用工具",
                    "让大模型调本地函数是什么意思",
                ],
            },
            {
                "id": "kb4",
                "title": "Agent Loop",
                "content": "Agent Loop 是模型反复执行“思考、调用工具、读取结果、继续决策”的循环机制，直到产出最终答案。",
                "keywords": ["agent loop", "agent", "loop", "智能体循环", "循环决策"],
                "question_variants": [
                    "智能体为什么要循环决策",
                    "为什么 Agent 不是一步结束",
                ],
            },
            {
                "id": "kb5",
                "title": "RAG",
                "content": "RAG 是先检索资料，再把资料拼进 Prompt，最后让模型基于检索结果生成答案。",
                "keywords": ["rag", "retrieval", "generation", "检索增强", "检索增强生成"],
                "question_variants": [
                    "为什么回答前要先查资料",
                    "检索增强生成是什么意思",
                ],
            },
            {
                "id": "kb6",
                "title": "Memory",
                "content": "Memory 让 Agent 具备记忆能力。短期记忆常放在上下文里，长期记忆通常落到数据库或向量库。",
                "keywords": ["memory", "记忆", "长期记忆", "短期记忆"],
                "question_variants": [
                    "智能体怎么记住之前的内容",
                    "短期记忆和长期记忆有什么区别",
                ],
            },
        ]

    @abstractmethod
    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """子类实现自己的检索方式。"""

    def build_prompt(self, question: str, docs: List[Dict[str, Any]]) -> str:
        """把检索到的知识拼进 Prompt。"""
        if docs:
            knowledge_text = "\n\n".join(
                [
                    f"[知识{i}] {doc['title']}\n{doc['content']}\nscore={doc['score']:.4f}\n原因: {'; '.join(doc['reasons'])}"
                    for i, doc in enumerate(docs, start=1)
                ]
            )
        else:
            knowledge_text = "当前没有检索到相关知识。"

        return (
            "你是一个教学型 AI 助手。\n"
            "回答时优先依据检索资料。\n"
            "如果资料不足，请明确说“根据当前知识库无法准确回答”。\n\n"
            f"【检索资料】\n{knowledge_text}\n\n"
            f"【用户问题】\n{question}"
        )

    def run_rag(self, question: str) -> Dict[str, Any]:
        """执行一次完整的 RAG。"""
        docs = self.retrieve(question)
        prompt = self.build_prompt(question, docs)
        answer: Optional[str] = None

        # 没配大模型 Key 时，仍然允许做检索分析。
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
            raise Exception(f"LLM 调用失败: {exc}") from exc

    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        """做一个轻量分词，便于教学观察。"""
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
    """保留关键词检索，作为 Day11 的对照组。"""

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
                if keyword.lower() in lowered_question:
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


class OllamaEmbedder:
    """负责调用 Ollama 的 embedding 接口。"""

    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed_single(self, text: str) -> List[float]:
        """把一段文本转换成向量。"""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": [text]},
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings", [])
            if not embeddings:
                raise RuntimeError("Ollama 没有返回 embedding 向量")
            return embeddings[0]


class QdrantKnowledgeStore:
    """封装 Day11 需要的 Qdrant 知识库存储能力。"""

    def __init__(self, collection_name: str, host: str = "localhost", port: int = 6333):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        """首次访问时创建 QdrantClient。"""
        if self._client is None:
            self._client = QdrantClient(host=self.host, port=self.port, timeout=30)
        return self._client

    def has_collection(self) -> bool:
        """判断 collection 是否已存在。"""
        collections = self.client.get_collections()
        names = [item.name for item in collections.collections]
        return self.collection_name in names

    def ensure_collection(self, vector_size: int) -> None:
        """确保 collection 存在，不存在时创建。"""
        if self.has_collection():
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def reset_collection(self) -> None:
        """删除已有 collection，便于重新初始化实验数据。"""
        if self.has_collection():
            self.client.delete_collection(collection_name=self.collection_name)

    def upsert_documents(self, rows: List[Dict[str, Any]]) -> None:
        """把文档向量和元数据写入 Qdrant。"""
        if not rows:
            return

        vector_size = len(rows[0]["vector"])
        self.ensure_collection(vector_size)
        points = [
            PointStruct(
                id=row["id"],
                vector=row["vector"],
                payload={
                    "title": row["title"],
                    "content": row["content"],
                    "keywords": row.get("keywords", []),
                    "question_variants": row.get("question_variants", []),
                },
            )
            for row in rows
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
## 重点在这
    def search(self, query_vector: List[float], limit: int = 3) -> List[Any]:
        """执行一次向量检索。"""
        if not self.has_collection():
            return []
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )
        return list(response.points)

    def count_documents(self) -> int:
        """返回当前 collection 中的文档数量。"""
        if not self.has_collection():
            return 0
        return int(self.client.count(collection_name=self.collection_name, exact=True).count)


class QdrantVectorRAGDemo(BaseRAGDemo):
    """真实向量检索版本。

    这一版会：
    1. 先用 Ollama 把知识库转成 embedding
    2. 写入 Qdrant
    3. 查询时把问题也转成 embedding，再去 Qdrant 检索
    """

    agent_name = "QdrantVectorRAGDemo"

    def __init__(
        self,
        embedder: OllamaEmbedder,
        store: QdrantKnowledgeStore,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.embedder = embedder
        self.store = store

    def seed_knowledge_base(self, reset: bool = False) -> None:
        """把内置知识库转成向量后写入 Qdrant。"""
        if reset:
            self.store.reset_collection()

        rows: List[Dict[str, Any]] = []
        for item in self.knowledge_base:
            document_text = self._join_document_text(item)
            vector = self.embedder.embed_single(document_text)
            rows.append(
                {
                    "id": item["id"],
                    "title": item["title"],
                    "content": item["content"],
                    "keywords": item.get("keywords", []),
                    "question_variants": item.get("question_variants", []),
                    "vector": vector,
                }
            )
        self.store.upsert_documents(rows)

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """执行真实向量检索。"""
        query_vector = self.embedder.embed_single(question)
        points = self.store.search(query_vector=query_vector, limit=self.top_k)
        docs: List[Dict[str, Any]] = []

        question_terms = set(self._extract_terms(question))
        for point in points:
            payload = point.payload or {}
            title = str(payload.get("title", ""))
            content = str(payload.get("content", ""))
            keywords = payload.get("keywords", []) or []
            reasons = [f"Qdrant 向量分数: {float(point.score):.4f}"]

            overlap_terms = sorted(
                question_terms
                & {term.lower() for term in keywords if isinstance(term, str)}
            )
            if overlap_terms:
                reasons.append(f"关键词也有重合: {', '.join(overlap_terms)}")

            docs.append(
                {
                    "id": str(point.id),
                    "title": title,
                    "content": content,
                    "score": float(point.score),
                    "reasons": reasons,
                }
            )
        return docs

    @staticmethod
    def _join_document_text(item: Dict[str, Any]) -> str:
        """把标题、正文、关键词、相近问法拼成一个完整文档。"""
        parts = [item["title"], item["content"]]
        parts.extend(item.get("keywords", []))
        parts.extend(item.get("question_variants", []))
        return " ".join(parts)


class RealVectorRAGRunner:
    """Day11 CLI 入口。"""

    def __init__(self, keyword_demo: KeywordRAGDemo, vector_demo: QdrantVectorRAGDemo):
        self.keyword_demo = keyword_demo
        self.vector_demo = vector_demo

    def run(self) -> None:
        print("=" * 72)
        print("Day11 Real Vector RAG Demo")
        print("=" * 72)
        print("\n可用命令:")
        print("  status                      - 查看当前向量库状态")
        print("  init                        - 初始化知识库到 Qdrant")
        print("  init-reset                  - 重建 Qdrant collection")
        print("  list-kb                     - 查看内置知识库标题")
        print("  compare <问题>              - 对比关键词检索和真实向量检索")
        print("  ask-keyword <问题>          - 用关键词检索跑一次 RAG")
        print("  ask-vector <问题>           - 用真实向量检索跑一次 RAG")
        print("  demo-loop                   - 测试 Agent Loop 问题")
        print("  demo-rag                    - 测试 RAG 问题")
        print("  demo-memory                 - 测试 Memory 问题")
        print("  quit/exit                   - 退出")
        print()

        while True:
            try:
                user_input = input("[day11-real-rag]> ").strip()
                if not user_input:
                    continue

                lower_text = user_input.lower()
                if lower_text in {"quit", "exit"}:
                    print("\n再见!")
                    break

                if lower_text == "status":
                    self._print_status()
                    continue

                if lower_text == "init":
                    self.vector_demo.seed_knowledge_base(reset=False)
                    print("\n[OK] 已把知识库写入 Qdrant。\n")
                    continue

                if lower_text == "init-reset":
                    self.vector_demo.seed_knowledge_base(reset=True)
                    print("\n[OK] 已重建 collection 并重新写入知识库。\n")
                    continue

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

                print("不支持的命令，请输入 status / init / compare / ask-keyword / ask-vector / quit\n")
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
        self._print_docs("QdrantVectorRAGDemo", self.vector_demo.retrieve(question))
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

    def _print_status(self) -> None:
        print("\n当前配置:")
        print(f"  - collection: {self.vector_demo.store.collection_name}")
        print(f"  - qdrant: {self.vector_demo.store.host}:{self.vector_demo.store.port}")
        print(f"  - ollama: {self.vector_demo.embedder.base_url}")
        print(f"  - embed_model: {self.vector_demo.embedder.model}")
        print(f"  - collection_exists: {self.vector_demo.store.has_collection()}")
        print(f"  - document_count: {self.vector_demo.store.count_documents()}")
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
    """程序入口。"""
    keyword_demo = KeywordRAGDemo(
        api_key=None,
        base_url=os.getenv("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"),
        model=os.getenv("OPENAI_MODEL", "qwen3.5-plus"),
    )

    embedder = OllamaEmbedder(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"),
    )
    store = QdrantKnowledgeStore(
        collection_name=os.getenv("DAY11_COLLECTION", "day11_real_vector_rag"),
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
    )
    vector_demo = QdrantVectorRAGDemo(
        embedder=embedder,
        store=store,
        api_key=None,
        base_url=os.getenv("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"),
        model=os.getenv("OPENAI_MODEL", "qwen3.5-plus"),
    )

    runner = RealVectorRAGRunner(keyword_demo, vector_demo)
    runner.run()


if __name__ == "__main__":
    main()
