"""
HTTP client for the reranker service.

Thin client that sends ``(query, [passage texts])`` to the cross-encoder
reranker service (port 8105) and returns ``(index, score)`` pairs. Mirrors the
``DoclingHttpClient`` / ``MineruHttpClient`` shape: env-resolved service URL,
plain ``httpx``, no ML libraries in ``app-main``.

The caller owns the richer result objects; this client only deals in passage
text and indices so the heavy ``torch`` / ``sentence-transformers`` stack stays
in the service container.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import httpx
from loguru import logger


DEFAULT_SERVICE_URL = "http://reranker:8105"
RERANK_TIMEOUT_SECONDS = 30.0


class RerankerServiceError(RuntimeError):
    """Raised when the reranker service returns an error or is unreachable."""


class RerankerHttpClient:
    """Calls the reranker service's ``POST /rerank`` over HTTP."""

    def __init__(
        self,
        service_url: Optional[str] = None,
        timeout: float = RERANK_TIMEOUT_SECONDS,
    ) -> None:
        self.service_url = (
            service_url
            or os.environ.get("RERANKER_SERVICE_URL")
            or DEFAULT_SERVICE_URL
        ).rstrip("/")
        self.timeout = timeout

    async def rerank(
        self,
        query: str,
        passages: Sequence[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Re-score ``passages`` against ``query`` via the reranker service.

        Returns a list of ``(index, score)`` tuples sorted by score
        descending, where ``index`` is the position in the input ``passages``
        list. Raises :class:`RerankerServiceError` on any transport or
        service-side failure so the caller can fall back to the heuristic
        reranker.
        """
        if not passages:
            return []

        payload: dict = {"query": query, "passages": list(passages)}
        if top_k is not None:
            payload["top_k"] = top_k

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.service_url}/rerank", json=payload
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            raise RerankerServiceError(
                f"Reranker service request failed: {e}"
            ) from e

        results = data.get("results")
        if results is None:
            raise RerankerServiceError(
                f"Reranker service returned no 'results': {data}"
            )

        return [(int(r["index"]), float(r["score"])) for r in results]
