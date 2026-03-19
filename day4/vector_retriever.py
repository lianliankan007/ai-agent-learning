from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue


@dataclass
class MemorySearchResult:
    """统一封装检索结果。"""

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

        response = self.client.query_points(**kwargs)
        results: List[MemorySearchResult] = []
        for point in response.points:
            payload = point.payload or {}
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
            combined = item.score * 0.7 + keyword_score * 0.2 + importance * 0.1
            rescored.append(
                MemorySearchResult(
                    id=item.id,
                    content=item.content,
                    metadata=item.metadata,
                    score=combined,
                )
            )

        rescored.sort(key=lambda row: row.score, reverse=True)
        return rescored[:top_k]

    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
        if not filters:
            return None

        conditions: List[FieldCondition] = []
        for key, value in filters.items():
            if value is None or value == "":
                continue
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
        return {
            token.lower()
            for token in re.findall(r"[\u4e00-\u9fff]{1,}|[a-zA-Z0-9_]+", text)
            if len(token.strip()) > 1
        }

    @staticmethod
    def _keyword_overlap(query_tokens: set[str], content_tokens: set[str]) -> float:
        if not query_tokens or not content_tokens:
            return 0.0
        intersection = len(query_tokens & content_tokens)
        return intersection / max(len(query_tokens), 1)
