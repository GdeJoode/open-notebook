"""
Search repositories for text and vector search operations.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query


class SearchRepository:
    """Repository for search operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        self.config = config

    async def text_search(
        self,
        keyword: str,
        results: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Perform a text search across sources and notes.

        Args:
            keyword: Search keyword.
            results: Maximum number of results.
            include_sources: Whether to search sources.
            include_notes: Whether to search notes.

        Returns:
            List of search results.
        """
        if not keyword:
            return []

        try:
            return await execute_query(
                """
                SELECT *
                FROM fn::text_search($keyword, $results, $source, $note)
                """,
                {
                    "keyword": keyword,
                    "results": results,
                    "source": include_sources,
                    "note": include_notes,
                },
                self.config,
            )
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    async def vector_search(
        self,
        embedding: List[float],
        results: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
        minimum_score: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Perform a vector similarity search.

        Args:
            embedding: Query embedding vector.
            results: Maximum number of results.
            include_sources: Whether to search sources.
            include_notes: Whether to search notes.
            minimum_score: Minimum similarity score.

        Returns:
            List of search results with similarity scores.
        """
        try:
            return await execute_query(
                """
                SELECT * FROM fn::vector_search($embed, $results, $source, $note, $minimum_score)
                """,
                {
                    "embed": embedding,
                    "results": results,
                    "source": include_sources,
                    "note": include_notes,
                    "minimum_score": minimum_score,
                },
                self.config,
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    # Reciprocal Rank Fusion damping constant ``k`` in ``w / (k + rank)``.
    # 60 is the value from the original RRF paper (Cormack, Clarke & Buettcher,
    # SIGIR 2009) and the de-facto default in hybrid-search systems; it matches
    # ``shared.retrieval.hybrid_fusion.DEFAULT_RRF_K`` used by Track R's
    # source-level fusion.
    _RRF_K: int = 60

    async def hybrid_search(
        self,
        keyword: str,
        embedding: List[float],
        results: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
        minimum_score: float = 0.2,
        text_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Perform a hybrid text + vector search via Reciprocal Rank Fusion.

        The two signals are on different, incomparable scales — BM25
        ``relevance`` (from ``fn::text_search``) is unbounded, cosine
        ``similarity`` (from ``fn::vector_search``) is ``[0, 1]`` — so a linear
        combination of raw scores would let BM25 silently dominate. We fuse by
        **RRF** instead, which consumes each signal's *rank* (not its raw
        score) and is therefore scale-independent::

            fused(item) = text_weight     · 1/(k + rank_text)
                        + (1-text_weight) · 1/(k + rank_vector)

        Each ``fn::`` function already returns its list best-first, so the
        1-based list position is the rank. An item present in only one list
        still ranks (its missing signal contributes ``0``); an item found by
        both is tagged ``hybrid`` and accrues both terms — biasing corroborated
        results upward.

        Args:
            keyword: Search keyword.
            embedding: Query embedding vector.
            results: Maximum number of results.
            include_sources: Whether to search sources.
            include_notes: Whether to search notes.
            minimum_score: Minimum similarity score for vector search.
            text_weight: RRF weight for the text (BM25) signal (0-1); the
                vector signal gets ``1 - text_weight``.

        Returns:
            Combined, deduplicated, RRF-ranked search results. Each carries
            ``_search_type`` (``text``/``vector``/``hybrid``), ``_combined_score``
            (the fused RRF score), and ``_text_rank``/``_vector_rank``
            provenance (the 1-based rank in each list, or ``None`` if absent).
        """
        vector_weight = 1 - text_weight

        # Each fn:: function returns best-first, so list position == rank.
        text_results = await self.text_search(
            keyword, results * 2, include_sources, include_notes
        )
        vector_results = await self.vector_search(
            embedding, results * 2, include_sources, include_notes, minimum_score
        )

        combined: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(text_results, start=1):
            item_id = item.get("id")
            if not item_id or item_id in combined:
                continue
            item["_search_type"] = "text"
            item["_text_rank"] = rank
            item["_vector_rank"] = None
            item["_combined_score"] = text_weight / (self._RRF_K + rank)
            combined[item_id] = item

        for rank, item in enumerate(vector_results, start=1):
            item_id = item.get("id")
            if not item_id:
                continue
            contribution = vector_weight / (self._RRF_K + rank)
            existing = combined.get(item_id)
            if existing is not None:
                existing["_search_type"] = "hybrid"
                existing["_vector_rank"] = rank
                existing["_combined_score"] += contribution
            else:
                item["_search_type"] = "vector"
                item["_text_rank"] = None
                item["_vector_rank"] = rank
                item["_combined_score"] = contribution
                combined[item_id] = item

        # Sort by fused score desc, with a deterministic id tie-break.
        ranked = sorted(
            combined.values(),
            key=lambda x: (-x.get("_combined_score", 0.0), str(x.get("id"))),
        )
        return ranked[:results]
