"""
Simple result reranker.

Re-scores search results by combining the original search score with
additional signals (query-result embedding similarity, freshness).
"""

import math
from typing import Any, Dict, List, Optional

from loguru import logger


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class Reranker:
    """Re-scores and re-orders search results.

    Args:
        score_weight: Weight for original search score (0-1).
        similarity_weight: Weight for query-result embedding similarity (0-1).
        top_k: Maximum results to return after reranking.
    """

    def __init__(
        self,
        score_weight: float = 0.6,
        similarity_weight: float = 0.4,
        top_k: Optional[int] = None,
    ) -> None:
        self._score_weight = score_weight
        self._similarity_weight = similarity_weight
        self._top_k = top_k

    def rerank(
        self,
        results: List[Dict[str, Any]],
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Re-score and re-order results.

        Args:
            results: Search results with optional "score" and "embedding" fields.
            query_embedding: Query vector for computing similarity (optional).

        Returns:
            Re-ranked results with "_rerank_score" field added.
        """
        if not results:
            return []

        for result in results:
            original_score = float(result.get("score", 0) or 0)

            sim_score = 0.0
            if query_embedding and result.get("embedding"):
                sim_score = _cosine_similarity(
                    query_embedding, result["embedding"]
                )

            combined = (
                self._score_weight * original_score
                + self._similarity_weight * sim_score
            )
            result["_rerank_score"] = round(combined, 6)

        results.sort(key=lambda r: r.get("_rerank_score", 0), reverse=True)

        if self._top_k:
            results = results[: self._top_k]

        return results
