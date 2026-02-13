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
        Perform a hybrid text + vector search.

        Args:
            keyword: Search keyword.
            embedding: Query embedding vector.
            results: Maximum number of results.
            include_sources: Whether to search sources.
            include_notes: Whether to search notes.
            minimum_score: Minimum similarity score for vector search.
            text_weight: Weight for text search results (0-1).

        Returns:
            Combined and ranked search results.
        """
        # Get both result sets
        text_results = await self.text_search(
            keyword, results * 2, include_sources, include_notes
        )
        vector_results = await self.vector_search(
            embedding, results * 2, include_sources, include_notes, minimum_score
        )

        # Combine and deduplicate
        seen_ids = set()
        combined = []

        # Add text results with weighted scores
        for item in text_results:
            item_id = item.get("id")
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                item["_search_type"] = "text"
                item["_combined_score"] = item.get("score", 0) * text_weight
                combined.append(item)

        # Add vector results with weighted scores
        vector_weight = 1 - text_weight
        for item in vector_results:
            item_id = item.get("id")
            if item_id:
                if item_id in seen_ids:
                    # Boost score for items found in both searches
                    for existing in combined:
                        if existing.get("id") == item_id:
                            existing["_search_type"] = "hybrid"
                            existing["_combined_score"] += (
                                item.get("score", 0) * vector_weight
                            )
                            break
                else:
                    seen_ids.add(item_id)
                    item["_search_type"] = "vector"
                    item["_combined_score"] = item.get("score", 0) * vector_weight
                    combined.append(item)

        # Sort by combined score and limit results
        combined.sort(key=lambda x: x.get("_combined_score", 0), reverse=True)
        return combined[:results]
