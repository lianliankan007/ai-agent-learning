from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue


@dataclass
class MemorySearchResult:
    """统一封装一条检索结果。

    这样上层代码就不需要直接依赖 Qdrant 的原始返回结构。
    """

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class QdrantMemoryRetriever:
    """封装基于 Qdrant 的向量检索与混合重排。"""

    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def semantic_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[MemorySearchResult]:
        """执行一次纯向量检索。"""
        query_filter = self._build_filter(filters)
        kwargs: Dict[str, Any] = {
            "collection_name": self.collection_name,
            "query": query_vector,
            "limit": top_k,
        }
        if query_filter is not None:
            kwargs["query_filter"] = query_filter
        if score_threshold is not None:
            kwargs["score_threshold"] = score_threshold

        # 这里直接把 query_vector 发给 Qdrant，让它按向量相似度召回候选结果。
        response = self.client.query_points(**kwargs)
        results: List[MemorySearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            # Qdrant 的 point 结构比较底层，这里统一转换成项目自己的结果对象。
            results.append(
                MemorySearchResult(
                    id=str(point.id),
                    content=str(payload.get("content", "")),
                    metadata={k: v for k, v in payload.items() if k != "content"},
                    score=float(point.score),
                )
            )
        return results

    def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemorySearchResult]:
        """执行混合检索。

        当前实现的思路是：
        1. 先用向量检索扩大召回范围
        2. 再用关键词重合度和 importance 重新排序
        """
        # 先把候选集放大，避免只拿 top_k 时把“关键词更贴切”的结果过早淘汰。
        recalled = self.semantic_search(
            query_vector=query_vector,
            top_k=max(top_k * 3, 10),
            filters=filters,
        )
        query_tokens = self._tokenize(query_text)
        rescored: List[MemorySearchResult] = []

        for item in recalled:
            content_tokens = self._tokenize(item.content)
            keyword_score = self._keyword_overlap(query_tokens, content_tokens)
            importance = float(item.metadata.get("importance", 0.5))
            # 当前权重设计是：语义相似度为主，关键词和重要度为辅。
            combined = item.score * 0.7 + keyword_score * 0.2 + importance * 0.1
            rescored.append(
                MemorySearchResult(
                    id=item.id,
                    content=item.content,
                    metadata=item.metadata,
                    score=combined,
                )
            )

        # 重排后的结果按新分数降序排序，只保留用户需要的前 top_k 条。
        rescored.sort(key=lambda row: row.score, reverse=True)
        return rescored[:top_k]

    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """把简单字典过滤条件转换成 Qdrant 的 Filter 对象。"""
        if not filters:
            return None

        conditions: List[FieldCondition] = []
        for key, value in filters.items():
            # None、空字符串、空列表都不生成过滤条件，避免误过滤。
            if value is None or value == "":
                continue
            # list 用 `MatchAny` 表示“命中任一值即可”，单值则精确匹配。
            if isinstance(value, list):
                if not value:
                    continue
                conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        if not conditions:
            return None
        return Filter(must=conditions)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """做一个非常轻量的分词，用于 hybrid 模式下的关键词重排。"""
        # 中文按连续汉字切分，英文和数字按单词切分；长度为 1 的 token 直接忽略。
        return {
            token.lower()
            for token in re.findall(r"[\u4e00-\u9fff]{1,}|[a-zA-Z0-9_]+", text)
            if len(token.strip()) > 1
        }

    @staticmethod
    def _keyword_overlap(query_tokens: set[str], content_tokens: set[str]) -> float:
        """计算查询词和内容词的重合比例。"""
        if not query_tokens or not content_tokens:
            return 0.0
        intersection = len(query_tokens & content_tokens)
        # 用查询词数量做分母，表示“用户想找的词有多少被命中了”。
        return intersection / max(len(query_tokens), 1)
