"""
Retrieval service — unified search with automatic query embedding.

Wraps SearchRepository with embedding logic so callers pass text queries
(not pre-computed vectors). Supports text, vector, and hybrid search modes.
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from surrealdb_service.repositories import SearchRepository


class RetrievalService:
    """Unified retrieval: embed query → search → return ranked results.

    Args:
        search_repo: SurrealDB search repository.
        embedding_model: Esperanto EmbeddingModel (optional; required for
            vector and hybrid search). If None, vector/hybrid calls fall
            back to text search.
    """

    def __init__(
        self,
        search_repo: Optional[SearchRepository] = None,
        embedding_model: Any = None,
    ) -> None:
        self._repo = search_repo or SearchRepository()
        self._embedding_model = embedding_model

    async def _embed(self, text: str) -> List[float]:
        """Embed a query string into a vector."""
        if self._embedding_model is None:
            raise ValueError(
                "Embedding model required for vector search. "
                "Configure a default embedding model in Settings → Models."
            )
        vectors = await self._embedding_model.aembed([text])
        return vectors[0]

    async def search(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
        minimum_score: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Unified search dispatcher.

        Args:
            query: Search query text.
            mode: "text", "vector", or "hybrid".
            limit: Maximum results.
            include_sources: Search source embeddings.
            include_notes: Search note embeddings.
            minimum_score: Minimum similarity for vector results.

        Returns:
            Ranked list of result dicts.
        """
        if mode == "text":
            return await self.text_search(
                query, limit, include_sources, include_notes
            )
        elif mode == "vector":
            return await self.vector_search(
                query, limit, include_sources, include_notes, minimum_score
            )
        else:
            return await self.hybrid_search(
                query, limit, include_sources, include_notes, minimum_score
            )

    async def text_search(
        self,
        query: str,
        limit: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
    ) -> List[Dict[str, Any]]:
        """Full-text search (no embedding needed)."""
        return await self._repo.text_search(
            query, limit, include_sources, include_notes
        )

    async def vector_search(
        self,
        query: str,
        limit: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
        minimum_score: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Embed query text, then vector similarity search.

        Falls back to text search if no embedding model is configured.
        """
        try:
            embedding = await self._embed(query)
        except ValueError:
            logger.warning(
                "No embedding model — falling back to text search"
            )
            return await self.text_search(
                query, limit, include_sources, include_notes
            )

        return await self._repo.vector_search(
            embedding, limit, include_sources, include_notes, minimum_score
        )

    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
        minimum_score: float = 0.2,
        text_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Embed query, then combined text + vector search.

        Falls back to text search if no embedding model is configured.
        """
        try:
            embedding = await self._embed(query)
        except ValueError:
            logger.warning(
                "No embedding model — falling back to text search"
            )
            return await self.text_search(
                query, limit, include_sources, include_notes
            )

        return await self._repo.hybrid_search(
            query,
            embedding,
            limit,
            include_sources,
            include_notes,
            minimum_score,
            text_weight,
        )
