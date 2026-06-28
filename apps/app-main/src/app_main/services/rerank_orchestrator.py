"""
Rerank orchestration for the hybrid search path.

Sits between the router and the reranker service: takes the RRF-fused hybrid
results, sends the top-N passage texts to the cross-encoder reranker service,
and reorders the result dicts by the returned scores. On any service failure
(down / timeout / bad response) it falls back to the existing zero-dependency
heuristic :class:`retrieval.reranker.Reranker` so search never 500s on a
reranker outage.

This module owns the wiring so the router stays thin and the fallback path is
unit-testable with the HTTP client mocked.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from retrieval.reranker import Reranker

from app_main.services.reranker_http_client import (
    RerankerHttpClient,
    RerankerServiceError,
)


def _passage_text(result: Dict[str, Any]) -> str:
    """Best available text for a result dict to feed the cross-encoder.

    Hybrid results carry ``content`` (chunk / note / insight body) and
    ``title``; we prefer ``content`` and fall back to ``title`` so every result
    contributes a non-empty passage where possible.
    """
    content = result.get("content")
    if isinstance(content, str) and content.strip():
        return content
    title = result.get("title")
    if isinstance(title, str) and title.strip():
        return title
    return ""


async def rerank_hybrid_results(
    query: str,
    results: List[Dict[str, Any]],
    top_n: int,
    client: Optional[RerankerHttpClient] = None,
    query_embedding: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Rerank the top-N fused results via the reranker service.

    Args:
        query: The search query text.
        results: RRF-fused hybrid result dicts (already ranked).
        top_n: How many leading results to re-score; the tail keeps its order
            beneath the reranked head.
        client: Reranker HTTP client (injectable for tests); defaults to a
            fresh :class:`RerankerHttpClient`.
        query_embedding: Optional query vector, only used by the heuristic
            fallback for embedding similarity.

    Returns:
        The results reordered so the reranked top-N lead (each carrying a
        ``_rerank_score``), followed by the untouched tail. Never raises on a
        reranker outage — it falls back to the heuristic reranker instead.
    """
    if not results:
        return results

    head = results[:top_n]
    tail = results[top_n:]

    passages = [_passage_text(r) for r in head]
    client = client or RerankerHttpClient()

    try:
        scored = await client.rerank(query, passages)
    except RerankerServiceError as e:
        logger.warning(
            f"Reranker service unavailable ({e}); falling back to heuristic "
            f"reranker for {len(head)} results"
        )
        reranked_head = Reranker().rerank(head, query_embedding=query_embedding)
        return reranked_head + tail

    # scored is [(index, score), ...] sorted desc; index refers into ``head``.
    reranked_head: List[Dict[str, Any]] = []
    for index, score in scored:
        if 0 <= index < len(head):
            item = head[index]
            item["_rerank_score"] = float(score)
            reranked_head.append(item)

    # Defensive: if the service returned fewer/odd indices, append any head
    # items it omitted so we never silently drop results.
    seen = {id(r) for r in reranked_head}
    for item in head:
        if id(item) not in seen:
            reranked_head.append(item)

    return reranked_head + tail
