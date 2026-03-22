#!/usr/bin/env python3
"""Day11: Hybrid retrieval + lightweight rerank demo."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

CONCEPT_GROUPS: Dict[str, List[str]] = {
    "agent_loop": ["agent loop", "智能体循环", "循环决策", "多轮决策", "循环执行"],
    "rag": ["rag", "检索增强", "检索增强生成", "先检索再回答", "回答前要检索"],
    "embedding": ["embedding", "向量化", "向量表示", "语义检索"],
    "token": ["token", "tokens", "词元", "分词", "tokenizer"],
    "function_calling": ["function calling", "函数调用", "工具调用", "tool use"],
    "context_window": ["context window", "上下文窗口", "上下文长度"],
}


@dataclass
class Doc:
    """Knowledge document."""

    doc_id: str
    title: str
    content: str
    keywords: List[str]


def build_docs() -> List[Doc]:
    """Build demo corpus."""

    return [
        Doc("kb1", "Token", "Token 是模型处理文本的最小单位。", ["token", "词元", "tokenizer"]),
        Doc("kb2", "Context Window", "上下文窗口决定模型单次可见信息长度。", ["context window", "上下文窗口"]),
        Doc("kb3", "Function Calling", "函数调用让模型输出结构化参数调用工具。", ["function calling", "函数调用", "工具调用"]),
        Doc("kb4", "Agent Loop", "Agent Loop 是规划-行动-观察的循环。", ["agent loop", "循环决策", "智能体循环"]),
        Doc("kb5", "RAG", "RAG 是先检索再生成，提高事实性。", ["rag", "检索增强", "检索增强生成"]),
        Doc("kb6", "Embedding", "Embedding 将文本映射为向量，用于语义检索。", ["embedding", "向量化", "语义检索"]),
    ]


def normalize(vec: List[float]) -> List[float]:
    """L2 normalize a vector."""

    n = math.sqrt(sum(v * v for v in vec))
    if n == 0:
        return vec
    return [v / n for v in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity for normalized vectors."""

    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def local_embed(text: str, dim: int = 128) -> List[float]:
    """Local hash embedding fallback."""

    vec = [0.0] * dim
    lowered = text.lower()
    for token in re.findall(r"[a-z0-9\u4e00-\u9fff]+", lowered):
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % dim
        sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign

    # 中文双字特征，提升中文问法鲁棒性。
    cjk = "".join(ch for ch in lowered if "\u4e00" <= ch <= "\u9fff")
    for i in range(len(cjk) - 1):
        bg = cjk[i : i + 2]
        h = hashlib.md5(f"bg:{bg}".encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % dim
        vec[idx] += 0.8

    # 概念特征，提高同义问法的语义召回。
    for concept, phrases in CONCEPT_GROUPS.items():
        if any(p in lowered for p in phrases):
            h = hashlib.md5(f"concept:{concept}".encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % dim
            vec[idx] += 3.0
    return normalize(vec)


def keyword_score(query: str, doc: Doc) -> float:
    """Compute keyword score for one doc."""

    q = query.lower()
    score = 0.0
    if doc.title.lower() in q:
        score += 4.0
    for kw in doc.keywords:
        if kw.lower() in q:
            score += 3.0
    return score


def keyword_recall(query: str, docs: Sequence[Doc], k: int) -> Dict[str, float]:
    """Keyword recall scores."""

    scored = [(d.doc_id, keyword_score(query, d)) for d in docs]
    scored = [(i, s) for i, s in scored if s > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return {i: s for i, s in scored[:k]}


def embedding_recall(query: str, docs: Sequence[Doc], k: int) -> Dict[str, float]:
    """Embedding recall scores."""

    qv = local_embed(query)
    scored: List[Tuple[str, float]] = []
    for d in docs:
        dv = local_embed(f"{d.title} {d.content} {' '.join(d.keywords)}")
        s = cosine(qv, dv)
        if s > 0:
            scored.append((d.doc_id, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return {i: s for i, s in scored[:k]}


def minmax(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize score map."""

    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def hybrid_fuse(kw: Dict[str, float], em: Dict[str, float], alpha: float) -> Dict[str, float]:
    """Fuse keyword and embedding scores."""

    kw_n = minmax(kw)
    em_n = minmax(em)
    all_ids = set(kw_n) | set(em_n)
    fused: Dict[str, float] = {}
    for doc_id in all_ids:
        fused[doc_id] = alpha * em_n.get(doc_id, 0.0) + (1 - alpha) * kw_n.get(doc_id, 0.0)
    return fused


def rerank(query: str, docs_by_id: Dict[str, Doc], fused_scores: Dict[str, float], top_k: int) -> List[Tuple[Doc, float, str]]:
    """Lightweight rerank using phrase overlap as tie-breaker."""

    q_terms = set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", query.lower()))
    rows: List[Tuple[Doc, float, str]] = []
    for doc_id, base in fused_scores.items():
        d = docs_by_id[doc_id]
        d_terms = set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", f"{d.title} {d.content}".lower()))
        overlap = len(q_terms & d_terms)
        final = base + 0.05 * overlap
        reason = f"fuse={base:.3f}, overlap_bonus={0.05 * overlap:.3f}"
        rows.append((d, final, reason))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_k]


def evaluate() -> None:
    """Run 3x3 evaluation for three modes."""

    docs = build_docs()
    docs_by_id = {d.doc_id: d for d in docs}
    cases = {
        "Token": ["什么是 token", "词元是什么意思", "tokenizer 做什么"],
        "Agent Loop": ["什么是 agent loop", "智能体为什么循环决策", "多轮决策机制是什么"],
        "RAG": ["RAG 是什么", "为什么回答前要检索", "检索增强生成是什么意思"],
    }
    modes = {"keyword": 0, "embedding": 0, "hybrid": 0}
    total = 0

    for topic, queries in cases.items():
        for q in queries:
            total += 1
            kw = keyword_recall(q, docs, k=5)
            em = embedding_recall(q, docs, k=5)
            hy = rerank(q, docs_by_id, hybrid_fuse(kw, em, alpha=0.6), top_k=1)

            kw_top = docs_by_id[max(kw, key=kw.get)].title if kw else "None"
            em_top = docs_by_id[max(em, key=em.get)].title if em else "None"
            hy_top = hy[0][0].title if hy else "None"

            if kw_top.lower() == topic.lower():
                modes["keyword"] += 1
            if em_top.lower() == topic.lower():
                modes["embedding"] += 1
            if hy_top.lower() == topic.lower():
                modes["hybrid"] += 1

    print("\n=== 3x3 Evaluation ===")
    print(f"keyword  : {modes['keyword']}/{total}")
    print(f"embedding: {modes['embedding']}/{total}")
    print(f"hybrid   : {modes['hybrid']}/{total}")


def run_query(query: str, alpha: float, candidate_k: int, top_k: int) -> None:
    """Run one query in hybrid mode and print results."""

    docs = build_docs()
    docs_by_id = {d.doc_id: d for d in docs}
    kw = keyword_recall(query, docs, k=candidate_k)
    em = embedding_recall(query, docs, k=candidate_k)
    fused = hybrid_fuse(kw, em, alpha=alpha)
    rows = rerank(query, docs_by_id, fused, top_k=top_k)

    print(f"\nQuery: {query}")
    print(f"alpha={alpha}, candidate_k={candidate_k}, top_k={top_k}")
    print(f"keyword candidates : {kw}")
    print(f"embedding candidates: {em}")
    print("\nFinal (hybrid + rerank):")
    if not rows:
        print("  - 无命中")
        return
    for i, (d, s, reason) in enumerate(rows, start=1):
        print(f"  {i}. {d.title} ({d.doc_id}) score={s:.3f}")
        print(f"     - {reason}")


def parse_args() -> argparse.Namespace:
    """Parse cli args."""

    p = argparse.ArgumentParser(description="Day11 hybrid retrieval demo")
    p.add_argument("--query", type=str, default="", help="single query mode")
    p.add_argument("--alpha", type=float, default=0.6, help="embedding weight in [0,1]")
    p.add_argument("--candidate-k", type=int, default=5, help="recall candidates")
    p.add_argument("--top-k", type=int, default=3, help="final output size")
    p.add_argument("--eval", action="store_true", help="run built-in 3x3 evaluation")
    return p.parse_args()


def main() -> None:
    """Entry point."""

    args = parse_args()
    if args.eval:
        evaluate()
        return
    if args.query:
        run_query(args.query, alpha=args.alpha, candidate_k=args.candidate_k, top_k=args.top_k)
        return
    run_query("为什么回答前要先检索？", alpha=args.alpha, candidate_k=args.candidate_k, top_k=args.top_k)
    evaluate()


if __name__ == "__main__":
    main()
