#!/usr/bin/env python3
"""
Day10: Embedding retrieval demo.

Features:
1. Keyword retrieval baseline
2. Embedding retrieval (API mode)
3. Local hash embedding fallback (offline mode)
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm_markdown_logger import get_default_llm_logger
from utils.openai_config import resolve_openai_api_key

llm_logger = get_default_llm_logger()

CONCEPT_GROUPS: Dict[str, List[str]] = {
    "agent_loop": ["agent loop", "智能体循环", "循环决策", "多轮决策", "循环执行"],
    "rag": ["rag", "检索增强", "检索增强生成", "回答前先检索", "先检索再回答"],
    "embedding": ["embedding", "向量化", "向量表示", "语义检索"],
    "token": ["token", "tokens", "词元", "分词", "tokenizer"],
    "function_calling": ["function calling", "函数调用", "工具调用", "tool use"],
    "context_window": ["context window", "上下文窗口", "上下文长度", "窗口长度"],
}


@dataclass
class KnowledgeDoc:
    """知识条目。"""

    doc_id: str
    title: str
    content: str
    keywords: List[str]


def build_docs() -> List[KnowledgeDoc]:
    """构建 Day10 演示知识库。"""

    return [
        KnowledgeDoc(
            doc_id="kb1",
            title="Token",
            content="Token 是大模型处理文本时的基本单位。中文通常按字或词切分为一个或多个 token。",
            keywords=["token", "词元", "分词", "tokenizer"],
        ),
        KnowledgeDoc(
            doc_id="kb2",
            title="Context Window",
            content="Context Window 指单次请求中模型可看到的上下文长度。窗口越大，成本通常越高。",
            keywords=["context window", "上下文窗口", "上下文长度"],
        ),
        KnowledgeDoc(
            doc_id="kb3",
            title="Function Calling",
            content="Function Calling 让模型以结构化参数请求程序执行工具函数。",
            keywords=["function calling", "函数调用", "工具调用", "tool use"],
        ),
        KnowledgeDoc(
            doc_id="kb4",
            title="Agent Loop",
            content="Agent Loop 是智能体在规划、调用工具、观察结果后继续决策的循环机制。",
            keywords=["agent loop", "循环决策", "智能体循环"],
        ),
        KnowledgeDoc(
            doc_id="kb5",
            title="RAG",
            content="RAG 是先检索资料再生成答案的流程，能提升事实性与时效性。",
            keywords=["rag", "检索增强", "检索增强生成", "先检索再回答"],
        ),
        KnowledgeDoc(
            doc_id="kb6",
            title="Embedding",
            content="Embedding 将文本映射为向量，使语义相近文本在向量空间距离更近。",
            keywords=["embedding", "向量化", "向量表示", "语义检索"],
        ),
    ]


class KeywordRetriever:
    """关键词检索器，用于和 Embedding 检索做对比。"""

    def __init__(self, docs: Sequence[KnowledgeDoc], top_k: int = 3):
        """初始化关键词检索器。"""

        self.docs = list(docs)
        self.top_k = top_k

    def search(self, query: str) -> List[Tuple[KnowledgeDoc, float, List[str]]]:
        """按关键词打分返回候选。"""

        query_lower = query.lower()
        terms = self._extract_terms(query)
        results: List[Tuple[KnowledgeDoc, float, List[str]]] = []

        for doc in self.docs:
            score = 0.0
            reasons: List[str] = []

            if doc.title.lower() in query_lower:
                score += 4.0
                reasons.append(f"标题命中: {doc.title}")

            for kw in doc.keywords:
                if kw.lower() in query_lower:
                    score += 3.0
                    reasons.append(f"关键词命中: {kw}")

            content_lower = doc.content.lower()
            overlap = [t for t in terms if len(t) >= 2 and t in content_lower]
            if overlap:
                score += float(len(overlap))
                reasons.append(f"文本片段命中: {', '.join(overlap[:4])}")

            if score > 0:
                results.append((doc, score, reasons))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[: self.top_k]

    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        """提取中英文词项。"""

        parts = re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", text.lower())
        return [p for p in parts if p]


class EmbeddingClient:
    """Embedding 客户端，支持 API 模式与离线 fallback 模式。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        dim: int = 128,
    ):
        """初始化 embedding 客户端。"""

        self.api_key = resolve_openai_api_key(api_key)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        # 没有 key 时自动走本地 hash embedding，保证 demo 可离线运行。
        self.use_local_fallback = not bool(self.api_key)

    def embed(self, text: str) -> List[float]:
        """对单条文本生成向量。"""

        if self.use_local_fallback:
            return self._local_hash_embedding(text)
        return self._embed_by_api(text)

    def _embed_by_api(self, text: str) -> List[float]:
        """调用 embedding API。"""

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": text}

        start = time.perf_counter()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            data = response.json()

            llm_logger.log_exchange(
                provider="openai-compatible",
                model=self.model,
                endpoint=url,
                request_payload=payload,
                request_headers=headers,
                response_payload=data,
                status_code=response.status_code,
                duration_ms=elapsed_ms,
                extra={"agent_name": "Day10EmbeddingDemo", "api_type": "embedding"},
            )

            return data["data"][0]["embedding"]
        except Exception as exc:
            # API 异常时自动降级到离线 embedding，保证流程不断。
            self.use_local_fallback = True
            print(f"[Warn] Embedding API 调用失败，已切换本地模式: {exc}")
            return self._local_hash_embedding(text)

    def _local_hash_embedding(self, text: str) -> List[float]:
        """本地哈希向量化：教学 fallback，不依赖网络。"""

        vec = [0.0] * self.dim
        lowered = text.lower()
        terms = re.findall(r"[a-z0-9]+", lowered)

        # 英文词项特征。
        for term in terms:
            digest = hashlib.md5(term.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vec[idx] += sign

        # 中文双字特征，提升中文问法的稳定性。
        cjk = "".join(ch for ch in lowered if "\u4e00" <= ch <= "\u9fff")
        for i in range(len(cjk) - 1):
            bg = cjk[i : i + 2]
            digest = hashlib.md5(f"bg:{bg}".encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            vec[idx] += 0.8

        # 概念特征，提升同义问法召回。
        for concept, phrases in CONCEPT_GROUPS.items():
            if any(p in lowered for p in phrases):
                digest = hashlib.md5(f"concept:{concept}".encode("utf-8")).hexdigest()
                idx = int(digest[:8], 16) % self.dim
                vec[idx] += 3.0

        return _normalize(vec)


class EmbeddingRetriever:
    """向量检索器：预计算文档向量，查询时做余弦相似度。"""

    def __init__(self, docs: Sequence[KnowledgeDoc], embedding_client: EmbeddingClient, top_k: int = 3):
        """初始化向量检索器并构建索引。"""

        self.docs = list(docs)
        self.embedding_client = embedding_client
        self.top_k = top_k
        self.doc_vectors: Dict[str, List[float]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """预计算并缓存每条文档向量。"""

        for doc in self.docs:
            text = f"{doc.title}\n{doc.content}\n{' '.join(doc.keywords)}"
            self.doc_vectors[doc.doc_id] = self.embedding_client.embed(text)

    def search(self, query: str) -> List[Tuple[KnowledgeDoc, float, List[str]]]:
        """执行向量检索并返回 Top-K。"""

        qvec = self.embedding_client.embed(query)
        results: List[Tuple[KnowledgeDoc, float, List[str]]] = []

        for doc in self.docs:
            dvec = self.doc_vectors[doc.doc_id]
            score = _cosine(qvec, dvec)
            if score <= 0:
                continue

            reasons = [f"余弦相似度: {score:.3f}"]
            if any(k.lower() in query.lower() for k in doc.keywords):
                reasons.append("包含显式关键词（语义+字面同时命中）")
            results.append((doc, score, reasons))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[: self.top_k]


def _normalize(vec: List[float]) -> List[float]:
    """向量归一化到单位长度。"""

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """计算两个向量的余弦相似度。"""

    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def print_results(title: str, rows: List[Tuple[KnowledgeDoc, float, List[str]]]) -> None:
    """打印检索结果。"""

    print(f"\n[{title}]")
    if not rows:
        print("  - 无命中")
        return

    for idx, (doc, score, reasons) in enumerate(rows, start=1):
        print(f"  {idx}. {doc.title} ({doc.doc_id}) | score={score:.3f}")
        for reason in reasons:
            print(f"     - {reason}")


def run_compare(query: str, kw: KeywordRetriever, em: EmbeddingRetriever) -> None:
    """对单个 query 做双检索对比。"""

    print("\n" + "=" * 72)
    print(f"Query: {query}")
    print_results("关键词检索", kw.search(query))
    print_results("Embedding 检索", em.search(query))


def run_pressure_test(kw: KeywordRetriever, em: EmbeddingRetriever) -> None:
    """执行 Day10 3*3 同义问法压力测试。"""

    cases = {
        "Token": ["什么是 token？", "词元是什么意思？", "tokenizer 在做什么？"],
        "Agent Loop": ["什么是 Agent Loop？", "智能体为什么循环决策？", "多轮工具调用是什么机制？"],
        "RAG": ["RAG 是什么？", "为什么回答前要先检索？", "检索增强生成有什么价值？"],
    }

    print("\n" + "=" * 72)
    print("3*3 同义问法压力测试")

    kw_ok = 0
    em_ok = 0
    total = 0
    for topic, queries in cases.items():
        print(f"\n--- Topic: {topic} ---")
        for q in queries:
            total += 1
            kw_top = kw.search(q)[0][0].title if kw.search(q) else "None"
            em_top = em.search(q)[0][0].title if em.search(q) else "None"
            print(f"Q: {q}")
            print(f"  keyword top1  : {kw_top}")
            print(f"  embedding top1: {em_top}")
            if kw_top.lower() == topic.lower():
                kw_ok += 1
            if em_top.lower() == topic.lower():
                em_ok += 1

    print("\n--- Summary ---")
    print(f"Keyword Top1 准确数: {kw_ok}/{total}")
    print(f"Embedding Top1 准确数: {em_ok}/{total}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Day10 embedding retrieval demo")
    parser.add_argument("--query", type=str, default="", help="run one custom query")
    parser.add_argument("--top-k", type=int, default=3, help="top-k results")
    parser.add_argument("--base-url", type=str, default="https://api.openai.com/v1", help="embedding api base url")
    parser.add_argument("--model", type=str, default="text-embedding-3-small", help="embedding model name")
    parser.add_argument("--api-key", type=str, default="", help="explicit api key, optional")
    parser.add_argument("--dim", type=int, default=128, help="local fallback vector dim")
    return parser.parse_args()


def main() -> None:
    """程序入口。"""

    args = parse_args()
    docs = build_docs()

    embedding_client = EmbeddingClient(
        api_key=args.api_key or None,
        base_url=args.base_url,
        model=args.model,
        dim=args.dim,
    )

    if embedding_client.use_local_fallback:
        print("[Info] 未检测到 API Key，当前使用本地哈希向量模式。")
    else:
        print(f"[Info] 使用 Embedding API 模式，model={args.model}")

    kw = KeywordRetriever(docs=docs, top_k=args.top_k)
    em = EmbeddingRetriever(docs=docs, embedding_client=embedding_client, top_k=args.top_k)

    if args.query:
        run_compare(args.query, kw, em)
        return

    sample_queries = [
        "为什么回答前要先检索？",
        "智能体为什么要循环决策？",
        "向量化到底解决了什么问题？",
    ]
    for q in sample_queries:
        run_compare(q, kw, em)

    run_pressure_test(kw, em)


if __name__ == "__main__":
    main()
