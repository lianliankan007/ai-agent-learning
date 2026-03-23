#!/usr/bin/env python3
"""
Day13: 检索策略教学 Demo。

这个脚本专门帮助你理解：
1. simple 检索是什么
2. filtered 检索为什么更准
3. hybrid 检索为什么更稳
4. top_k 调整后结果会发生什么变化

这里不接外部向量库，而是用一个教学型的内存检索器，
让你先看清“检索策略”本身的差异。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalResult:
    """统一封装一条检索结果。"""

    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


KNOWLEDGE_BASE = [
    {
        "id": "kb1",
        "content": "RAG 的核心流程是先检索资料，再把资料放进 Prompt，最后让模型基于资料生成答案。",
        "topic": "rag",
        "memory_type": "definition",
        "importance": 0.92,
        "tags": ["rag", "retrieval", "generation"],
    },
    {
        "id": "kb2",
        "content": "chunk 太大时，一个向量会混入多个主题，导致语义不够聚焦，检索结果更容易带噪音。",
        "topic": "chunking",
        "memory_type": "best_practice",
        "importance": 0.88,
        "tags": ["chunk", "chunking", "检索质量"],
    },
    {
        "id": "kb3",
        "content": "chunk 太小时，语义上下文可能不完整，虽然命中了关键词，但不足以支撑稳定回答。",
        "topic": "chunking",
        "memory_type": "best_practice",
        "importance": 0.87,
        "tags": ["chunk", "上下文", "检索质量"],
    },
    {
        "id": "kb4",
        "content": "top_k 太小可能漏掉有用知识，top_k 太大则可能把不相关内容一起塞进 Prompt。",
        "topic": "retrieval",
        "memory_type": "parameter",
        "importance": 0.93,
        "tags": ["top_k", "召回", "噪音"],
    },
    {
        "id": "kb5",
        "content": "过滤检索可以先按 topic、user_id、memory_type 缩小范围，再在局部范围内做相似度搜索。",
        "topic": "retrieval",
        "memory_type": "strategy",
        "importance": 0.91,
        "tags": ["filter", "topic", "memory_type"],
    },
    {
        "id": "kb6",
        "content": "混合检索会综合向量相似度、关键词命中和重要度排序，结果通常比单一信号更稳定。",
        "topic": "retrieval",
        "memory_type": "strategy",
        "importance": 0.95,
        "tags": ["hybrid", "keyword", "importance"],
    },
    {
        "id": "kb7",
        "content": "Memory 系统通常会区分短期记忆和长期记忆，长期记忆常借助向量库做语义检索。",
        "topic": "memory",
        "memory_type": "definition",
        "importance": 0.84,
        "tags": ["memory", "向量库", "长期记忆"],
    },
    {
        "id": "kb8",
        "content": "Function Calling 让模型输出结构化参数，再由程序执行外部工具调用。",
        "topic": "tool_use",
        "memory_type": "definition",
        "importance": 0.72,
        "tags": ["function calling", "tool use", "工具调用"],
    },
]


class TeachingRetriever:
    """教学型内存检索器。"""

    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def simple_search(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """只按轻量语义分数排序，不加过滤。"""
        scored = []
        for row in self.rows:
            score = self._semantic_score(query, row["content"], row.get("tags", []))
            if score > 0:
                scored.append(
                    RetrievalResult(
                        doc_id=row["id"],
                        content=row["content"],
                        score=score,
                        metadata=self._build_metadata(row),
                    )
                )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def filtered_search(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """先按 metadata 过滤，再做相似度排序。"""
        candidates = [row for row in self.rows if self._match_filters(row, filters)]
        scored = []
        for row in candidates:
            score = self._semantic_score(query, row["content"], row.get("tags", []))
            if score > 0:
                scored.append(
                    RetrievalResult(
                        doc_id=row["id"],
                        content=row["content"],
                        score=score,
                        metadata=self._build_metadata(row),
                    )
                )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """综合语义、关键词和重要度做排序。"""
        candidates = [row for row in self.rows if self._match_filters(row, filters)]
        query_tokens = self._tokenize(query)
        rescored: List[RetrievalResult] = []

        for row in candidates:
            semantic = self._semantic_score(query, row["content"], row.get("tags", []))
            if semantic <= 0:
                continue

            content_tokens = self._tokenize(row["content"] + " " + " ".join(row.get("tags", [])))
            keyword = self._keyword_overlap(query_tokens, content_tokens)
            importance = float(row.get("importance", 0.5))
            combined = semantic * 0.7 + keyword * 0.2 + importance * 0.1

            rescored.append(
                RetrievalResult(
                    doc_id=row["id"],
                    content=row["content"],
                    score=combined,
                    metadata=self._build_metadata(row),
                )
            )

        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]

    @staticmethod
    def _build_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "topic": row.get("topic", ""),
            "memory_type": row.get("memory_type", ""),
            "importance": row.get("importance", 0.0),
            "tags": row.get("tags", []),
        }

    @staticmethod
    def _match_filters(row: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        for key, value in filters.items():
            if value in (None, "", []):
                continue
            row_value = row.get(key)
            if isinstance(value, list):
                if isinstance(row_value, list):
                    if not any(item in row_value for item in value):
                        return False
                else:
                    if row_value not in value:
                        return False
            else:
                if row_value != value:
                    return False
        return True

    def _semantic_score(self, query: str, content: str, tags: List[str]) -> float:
        """教学版语义分数。

        这里不追求工业精度，只是用“分词 + 关键词标签”模拟语义接近的效果。
        """
        query_tokens = self._tokenize(query)
        content_tokens = self._tokenize(content + " " + " ".join(tags))
        return self._keyword_overlap(query_tokens, content_tokens)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"[\u4e00-\u9fff]{1,}|[a-zA-Z0-9_]+", text)
            if len(token.strip()) > 1
        }

    @staticmethod
    def _keyword_overlap(query_tokens: set[str], content_tokens: set[str]) -> float:
        if not query_tokens or not content_tokens:
            return 0.0
        return len(query_tokens & content_tokens) / max(len(query_tokens), 1)


class RetrievalStrategyRunner:
    """命令行入口。"""

    def __init__(self, retriever: TeachingRetriever):
        self.retriever = retriever

    def run(self) -> None:
        print("=" * 72)
        print("Day13 Retrieval Strategy Demo")
        print("=" * 72)
        print("\n可用命令:")
        print("  list-kb                              - 查看知识库")
        print("  simple <top_k> <问题>                - 执行 simple 检索")
        print("  filtered <topic> <top_k> <问题>      - 执行 filtered 检索")
        print("  hybrid <topic> <top_k> <问题>        - 执行 hybrid 检索")
        print("  compare <问题>                       - 对比三种检索")
        print("  demo-topk                            - 演示 top_k 差异")
        print("  demo-filter                          - 演示过滤检索差异")
        print("  demo-hybrid                          - 演示混合检索差异")
        print("  quit/exit                            - 退出")
        print()

        while True:
            try:
                user_input = input("[day13-retrieval]> ").strip()
                if not user_input:
                    continue

                lower_text = user_input.lower()
                if lower_text in {"quit", "exit"}:
                    print("\n再见!")
                    break

                if lower_text == "list-kb":
                    self._print_knowledge_base()
                    continue

                if lower_text == "demo-topk":
                    self._demo_topk()
                    continue

                if lower_text == "demo-filter":
                    self._demo_filter()
                    continue

                if lower_text == "demo-hybrid":
                    self._demo_hybrid()
                    continue

                if lower_text.startswith("compare "):
                    self._compare(user_input[8:].strip())
                    continue

                if lower_text.startswith("simple "):
                    # 用默认空白分割，自动兼容多个空格和 tab。
                    parts = user_input.split(maxsplit=2)
                    if len(parts) < 3:
                        print("用法: simple <top_k> <问题>\n")
                        continue
                    self._print_results(
                        "Simple",
                        self.retriever.simple_search(parts[2], top_k=int(parts[1])),
                    )
                    continue

                if lower_text.startswith("filtered "):
                    # 用默认空白分割，自动兼容多个空格和 tab。
                    parts = user_input.split(maxsplit=3)
                    if len(parts) < 4:
                        print("用法: filtered <topic> <top_k> <问题>\n")
                        continue
                    self._print_results(
                        "Filtered",
                        self.retriever.filtered_search(
                            parts[3],
                            top_k=int(parts[2]),
                            filters={"topic": parts[1]},
                        ),
                    )
                    continue

                if lower_text.startswith("hybrid "):
                    # 用默认空白分割，自动兼容多个空格和 tab。
                    parts = user_input.split(maxsplit=3)
                    if len(parts) < 4:
                        print("用法: hybrid <topic> <top_k> <问题>\n")
                        continue
                    self._print_results(
                        "Hybrid",
                        self.retriever.hybrid_search(
                            parts[3],
                            top_k=int(parts[2]),
                            filters={"topic": parts[1]},
                        ),
                    )
                    continue

                print("不支持的命令，请输入 list-kb / compare / simple / filtered / hybrid / quit\n")
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as exc:
                print(f"\n错误: {exc}\n")

    def _compare(self, query: str) -> None:
        print("\n" + "=" * 72)
        print(f"[Question] {query}")
        print("=" * 72)
        self._print_results("Simple", self.retriever.simple_search(query, top_k=3))
        self._print_results(
            "Filtered(topic=retrieval)",
            self.retriever.filtered_search(query, top_k=3, filters={"topic": "retrieval"}),
        )
        self._print_results(
            "Hybrid(topic=retrieval)",
            self.retriever.hybrid_search(query, top_k=3, filters={"topic": "retrieval"}),
        )

    def _demo_topk(self) -> None:
        query = "为什么 top_k 太大会引入噪音？"
        print(f"\n[Demo] {query}")
        for top_k in [1, 3, 5]:
            self._print_results(
                f"Simple top_k={top_k}",
                self.retriever.simple_search(query, top_k=top_k),
            )

    def _demo_filter(self) -> None:
        query = "为什么检索时需要 filter？"
        print(f"\n[Demo] {query}")
        self._print_results("Simple", self.retriever.simple_search(query, top_k=5))
        self._print_results(
            "Filtered(topic=retrieval)",
            self.retriever.filtered_search(query, top_k=5, filters={"topic": "retrieval"}),
        )

    def _demo_hybrid(self) -> None:
        query = "为什么混合检索更稳定？"
        print(f"\n[Demo] {query}")
        self._print_results(
            "Filtered(topic=retrieval)",
            self.retriever.filtered_search(query, top_k=5, filters={"topic": "retrieval"}),
        )
        self._print_results(
            "Hybrid(topic=retrieval)",
            self.retriever.hybrid_search(query, top_k=5, filters={"topic": "retrieval"}),
        )

    @staticmethod
    def _print_results(title: str, results: List[RetrievalResult]) -> None:
        print(f"\n[{title}]")
        if not results:
            print("  - 未命中结果")
            return
        for item in results:
            topic = item.metadata.get("topic", "unknown")
            memory_type = item.metadata.get("memory_type", "unknown")
            importance = item.metadata.get("importance", 0.0)
            print(
                f"  - {item.doc_id} score={item.score:.3f} "
                f"topic={topic} type={memory_type} importance={importance}"
            )
            print(f"    {item.content}")

    def _print_knowledge_base(self) -> None:
        print("\n当前知识库:")
        for row in KNOWLEDGE_BASE:
            print(f"  - {row['id']} topic={row['topic']} type={row['memory_type']}")
            print(f"    {row['content']}")
        print()


def main() -> None:
    retriever = TeachingRetriever(KNOWLEDGE_BASE)
    runner = RetrievalStrategyRunner(retriever)
    runner.run()


if __name__ == "__main__":
    main()
