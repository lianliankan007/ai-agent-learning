#!/usr/bin/env python3
"""
Day9 Demo: from keyword retrieval to semantic vector-like retrieval.

This demo is intentionally self-contained and offline:
1. KeywordRetriever: baseline from Day8-style literal matching
2. ToyEmbeddingRetriever: semantic-style retrieval with concept mapping + cosine similarity
3. Side-by-side comparison + pressure test

Run:
  python day9/rag_vector_transition_demo.py
  python day9/rag_vector_transition_demo.py --query "智能体为什么要循环决策？"
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class KnowledgeDoc:
    """单条知识文档结构。"""

    doc_id: str
    title: str
    content: str
    keywords: List[str]


def build_knowledge_base() -> List[KnowledgeDoc]:
    """构建用于检索对比的固定知识库。"""

    # 复用 Day8 同主题知识，便于横向比较“检索方式变化”而非“语料变化”。
    return [
        KnowledgeDoc(
            doc_id="kb1",
            title="Token",
            content="Token 是大模型处理文本时使用的基本单位。英文里一个 token 大约对应 0.75 个单词，中文通常一个字或词会占用一个或多个 token。",
            keywords=["token", "tokens", "tokenizer", "分词", "词元"],
        ),
        KnowledgeDoc(
            doc_id="kb2",
            title="Context Window",
            content="Context Window 指模型单次请求中最多能处理的上下文长度。上下文越长，模型一次能看到的信息越多，但成本也通常更高。",
            keywords=["context", "window", "上下文", "上下文窗口"],
        ),
        KnowledgeDoc(
            doc_id="kb3",
            title="Function Calling",
            content="Function Calling 是让模型根据问题决定是否调用外部函数或工具，并把结构化参数交给程序执行的一种能力。",
            keywords=["function", "calling", "函数调用", "工具调用", "tool use"],
        ),
        KnowledgeDoc(
            doc_id="kb4",
            title="Agent Loop",
            content="Agent Loop 是一种循环执行机制：模型判断下一步动作，必要时调用工具，再根据工具结果继续判断，直到输出最终答案。",
            keywords=["agent loop", "loop", "agent", "循环", "智能体循环", "循环决策"],
        ),
        KnowledgeDoc(
            doc_id="kb5",
            title="RAG",
            content="RAG 是 Retrieval-Augmented Generation 的缩写，核心流程是先检索相关资料，再把资料放进 prompt，最后让模型基于资料生成答案。",
            keywords=["rag", "retrieval", "generation", "检索增强", "检索增强生成"],
        ),
        KnowledgeDoc(
            doc_id="kb6",
            title="Memory",
            content="Memory 用来让 Agent 记住历史信息。短期记忆通常保存在对话上下文里，长期记忆通常保存在数据库或向量库里。",
            keywords=["memory", "记忆", "长期记忆", "短期记忆"],
        ),
    ]


class KeywordRetriever:
    """基于关键词和字面重叠的最小检索器。"""

    def __init__(self, docs: List[KnowledgeDoc], top_k: int = 3):
        """初始化关键词检索器。"""

        self.docs = docs
        self.top_k = top_k

    def search(self, query: str) -> List[Dict]:
        """执行一次关键词检索并返回 Top-K 结果。"""

        # 关键词检索核心：字面匹配 + 简单计分。
        query_lower = query.lower()
        terms = self._extract_terms(query)
        hits: List[Dict] = []

        for doc in self.docs:
            score = 0
            reasons: List[str] = []

            if doc.title.lower() in query_lower:
                score += 4
                reasons.append(f"标题命中: {doc.title}")

            for kw in doc.keywords:
                if kw.lower() in query_lower:
                    # 显式关键词命中，给更高分。
                    score += 3
                    reasons.append(f"关键词命中: {kw}")

            content_lower = doc.content.lower()
            overlap_terms = [t for t in terms if len(t) >= 2 and t in content_lower]
            if overlap_terms:
                # 补充“内容片段命中”分，模拟最小全文检索。
                score += len(overlap_terms)
                reasons.append(f"文本片段命中: {', '.join(overlap_terms[:4])}")

            if score > 0:
                hits.append(
                    {
                        "doc": doc,
                        # 统一转 float，便于和语义检索分数并排展示。
                        "score": float(score),
                        "reasons": reasons,
                    }
                )

        # 按分数从高到低排序，再截断为 top_k。
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[: self.top_k]

    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        """提取中英文词项，作为最小化的可比对 term 集合。"""

        # 用非中英文数字字符做分隔，得到粗粒度 term 列表。
        parts = re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", text.lower())
        return [p for p in parts if p]


class ToyEmbeddingRetriever:
    """
    A no-dependency semantic retriever:
    - concept mapping: normalize paraphrases into canonical semantic tokens
    - sparse vector + cosine similarity: simulate embedding-style retrieval
    """

    CONCEPT_GROUPS: Dict[str, List[str]] = {
        "agent_loop": ["agent loop", "智能体循环", "循环决策", "循环执行", "多轮决策"],
        "rag": ["rag", "检索增强", "检索增强生成", "先查资料再回答", "回答前先检索"],
        "token": ["token", "tokens", "词元", "分词", "tokenizer"],
        "memory": ["memory", "记忆", "长期记忆", "短期记忆"],
        "function_calling": ["function calling", "函数调用", "工具调用", "tool use"],
        "context_window": ["context window", "上下文窗口", "上下文长度", "窗口长度"],
        "retrieval": ["retrieval", "检索", "召回", "找资料"],
    }

    def __init__(self, docs: List[KnowledgeDoc], top_k: int = 3):
        """初始化语义检索器并构建离线索引。"""

        self.docs = docs
        self.top_k = top_k
        self.doc_vectors: Dict[str, Dict[str, float]] = {}
        self.idf: Dict[str, float] = {}
        self.doc_norms: Dict[str, float] = {}
        # 初始化时建立离线索引：文档向量 + IDF + 范数。
        self._build_index()

    def search(self, query: str) -> List[Dict]:
        """执行一次语义检索（模拟 embedding + cosine）。"""

        # 查询流程：query 向量化 -> 与每个文档做余弦相似度 -> 取 TopK。
        query_vec, _query_concepts = self._embed_text(query)
        # 查询向量也乘 IDF，降低高频特征对相似度的干扰。
        query_vec = self._apply_idf(query_vec)
        query_norm = self._norm(query_vec)
        if query_norm == 0:
            # 空向量无法计算余弦相似度，直接返回无结果。
            return []

        hits: List[Dict] = []
        for doc in self.docs:
            doc_vec = self.doc_vectors[doc.doc_id]
            sim = self._cosine(query_vec, query_norm, doc_vec, self.doc_norms[doc.doc_id])
            if sim <= 0:
                # 分数 <= 0 代表几乎无关，不作为候选返回。
                continue

            matched_concepts = self._matched_concepts(query, doc)
            reasons = [f"语义相似度: {sim:.3f}"]
            if matched_concepts:
                # 将可解释信息输出给学习者，帮助理解“为什么命中”。
                reasons.append(f"概念对齐: {', '.join(matched_concepts)}")

            hits.append({"doc": doc, "score": sim, "reasons": reasons})

        # 语义分数越高说明向量方向越接近，按降序取前 top_k。
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[: self.top_k]

    def _build_index(self) -> None:
        """构建文档索引：统计 DF/IDF，并缓存每条文档向量。"""

        # 先统计 DF，再计算 IDF，最后得到每条文档的 TF-IDF 风格向量。
        embedded_docs: List[Dict[str, float]] = []
        df_counter: Counter[str] = Counter()

        for doc in self.docs:
            # 将 title/content/keywords 合并，形成该文档的检索表示。
            text = f"{doc.title}。{doc.content}。{' '.join(doc.keywords)}"
            vec, _ = self._embed_text(text)
            embedded_docs.append(vec)
            for token in vec:
                # DF 只统计“该 token 出现在多少文档”。
                df_counter[token] += 1

        n_docs = len(self.docs)
        self.idf = {
            # 平滑 IDF，避免出现除零，并减轻稀有词过度放大问题。
            token: math.log((1 + n_docs) / (1 + df)) + 1.0
            for token, df in df_counter.items()
        }

        for doc, raw_vec in zip(self.docs, embedded_docs):
            # 把原始频次向量映射到带 IDF 权重的空间。
            tfidf_vec = self._apply_idf(raw_vec)
            self.doc_vectors[doc.doc_id] = tfidf_vec
            # 提前缓存范数，查询阶段可复用，减少重复计算。
            self.doc_norms[doc.doc_id] = self._norm(tfidf_vec)

    def _embed_text(self, text: str) -> Tuple[Dict[str, float], List[str]]:
        """将文本编码成稀疏特征向量，并返回命中的概念标签。"""

        # 这是“类 embedding”步骤：把文本转换为稀疏特征向量。
        lowered = text.lower()
        feats: Counter[str] = Counter()
        concepts: List[str] = []

        for concept, phrases in self.CONCEPT_GROUPS.items():
            if any(p in lowered for p in phrases):
                # 概念特征权重更高，提升同义问法的召回能力。
                feats[f"concept:{concept}"] += 3
                concepts.append(concept)

        for word in re.findall(r"[a-z0-9]+", lowered):
            if len(word) >= 2:
                # 英文词项特征。
                feats[f"word:{word}"] += 1

        cjk = "".join(ch for ch in lowered if "\u4e00" <= ch <= "\u9fff")
        for i in range(len(cjk) - 1):
            # 中文双字切分，作为无分词库场景下的简化语义特征。
            bg = cjk[i : i + 2]
            feats[f"bg:{bg}"] += 1

        return dict(feats), concepts

    def _apply_idf(self, vec: Dict[str, float]) -> Dict[str, float]:
        """对输入向量应用 IDF 权重。"""

        # 未出现在 IDF 表的特征使用 1.0，保证查询向量可平滑处理。
        return {k: v * self.idf.get(k, 1.0) for k, v in vec.items()}

    @staticmethod
    def _norm(vec: Dict[str, float]) -> float:
        """计算向量 L2 范数。"""

        # L2 范数用于余弦相似度的分母归一化。
        return math.sqrt(sum(v * v for v in vec.values()))

    @staticmethod
    def _cosine(qv: Dict[str, float], qn: float, dv: Dict[str, float], dn: float) -> float:
        """计算查询向量与文档向量的余弦相似度。"""

        if qn == 0 or dn == 0:
            return 0.0
        # 遍历较小字典，降低稀疏向量点积的计算开销。
        small, large = (qv, dv) if len(qv) < len(dv) else (dv, qv)
        dot = sum(v * large.get(k, 0.0) for k, v in small.items())
        # 余弦值越接近 1，表示方向越一致、语义越接近。
        return dot / (qn * dn)

    def _matched_concepts(self, query: str, doc: KnowledgeDoc) -> List[str]:
        """返回 query 与 doc 共同命中的概念标签，用于可解释输出。"""

        doc_text = f"{doc.title} {doc.content} {' '.join(doc.keywords)}".lower()
        matched: List[str] = []
        query_lower = query.lower()

        for concept, phrases in self.CONCEPT_GROUPS.items():
            query_hit = any(p in query_lower for p in phrases)
            doc_hit = any(p in doc_text for p in phrases)
            if query_hit and doc_hit:
                matched.append(concept)
        return matched


def print_hits(title: str, hits: List[Dict]) -> None:
    """统一打印检索结果，包含分数和命中原因。"""

    print(f"\n[{title}]")
    if not hits:
        print("  - 无命中")
        return

    for idx, item in enumerate(hits, start=1):
        doc = item["doc"]
        # 统一保留 3 位小数，便于比较两种检索器输出。
        print(f"  {idx}. {doc.title} ({doc.doc_id}) | score={item['score']:.3f}")
        for reason in item["reasons"]:
            print(f"     - {reason}")


def run_single_query(query: str, keyword_retriever: KeywordRetriever, embedding_retriever: ToyEmbeddingRetriever) -> None:
    """对单条 query 做两种检索并排展示。"""

    print("\n" + "=" * 72)
    print(f"Query: {query}")

    keyword_hits = keyword_retriever.search(query)
    semantic_hits = embedding_retriever.search(query)

    print_hits("关键词检索", keyword_hits)
    print_hits("向量化语义检索(模拟)", semantic_hits)


def run_pressure_test(keyword_retriever: KeywordRetriever, embedding_retriever: ToyEmbeddingRetriever) -> None:
    """执行同义问法压力测试并统计 Top1 命中数。"""

    print("\n" + "=" * 72)
    print("同义问法压力测试")

    cases = {
        "Token": [
            "什么是 token？",
            "大模型里的词元是啥意思？",
            "分词和tokenizer在做什么？",
        ],
        "Agent Loop": [
            "什么是 Agent Loop？",
            "智能体为什么要循环决策？",
            "为什么 agent 要多轮执行工具？",
        ],
        "RAG": [
            "RAG 是什么？",
            "为什么回答前要先查资料？",
            "检索增强生成到底解决什么问题？",
        ],
    }

    keyword_top1 = 0
    semantic_top1 = 0
    total = 0

    for topic, queries in cases.items():
        print(f"\n--- Topic: {topic} ---")
        for q in queries:
            # 逐题统计样本总数。
            total += 1
            k_hits = keyword_retriever.search(q)
            s_hits = embedding_retriever.search(q)

            k_top = k_hits[0]["doc"].title if k_hits else "None"
            s_top = s_hits[0]["doc"].title if s_hits else "None"
            print(f"Q: {q}")
            print(f"  keyword top1 : {k_top}")
            print(f"  semantic top1: {s_top}")

            # 只看 Top1 是否命中目标主题，便于快速对比稳定性。
            if k_top.lower() == topic.lower():
                keyword_top1 += 1
            if s_top.lower() == topic.lower():
                semantic_top1 += 1

    print("\n--- Summary ---")
    # 这里不追求绝对准确率，而是用于直观看到两类检索的鲁棒性差异。
    print(f"Keyword Top1 准确数: {keyword_top1}/{total}")
    print(f"Semantic Top1 准确数: {semantic_top1}/{total}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Day9 retrieval comparison demo")
    parser.add_argument("--query", type=str, default="", help="run one custom query")
    parser.add_argument("--top-k", type=int, default=3, help="top-k results")
    return parser.parse_args()


def main() -> None:
    """程序入口：支持单 query 模式和默认演示模式。"""

    args = parse_args()
    docs = build_knowledge_base()

    keyword_retriever = KeywordRetriever(docs=docs, top_k=args.top_k)
    embedding_retriever = ToyEmbeddingRetriever(docs=docs, top_k=args.top_k)

    if args.query:
        run_single_query(args.query, keyword_retriever, embedding_retriever)
    else:
        sample_queries = [
            "智能体为什么要循环决策？",
            "为什么回答前要先查资料？",
            "上下文窗口和成本有什么关系？",
        ]
        for q in sample_queries:
            run_single_query(q, keyword_retriever, embedding_retriever)

        run_pressure_test(keyword_retriever, embedding_retriever)


if __name__ == "__main__":
    main()
