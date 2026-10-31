"""
Retrieval service — unified search with automatic query embedding.

Wraps SearchRepository with embedding logic so callers pass text queries
(not pre-computed vectors). Supports text, vector, and hybrid search modes.
Zoeken leest alleen fragmenten (ADR 0004); elke methode vraagt het
classificatiebereik van de aanroeper als verplicht argument (F2-05, ADR 0002 C).
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from shared.models.classification import ClassificationScope
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
        minimum_score: float = 0.2,
        *,
        scope: ClassificationScope,
        history: bool = False,
    ) -> List[Dict[str, Any]]:
        """Unified search dispatcher.

        Args:
            query: Search query text.
            mode: "text", "vector", or "hybrid".
            limit: Maximum results.
            minimum_score: Minimum similarity for vector results.
            scope: Classificatiebereik van de aanroeper (verplicht).
            history: Ook oudere bronversies en runs.

        Returns:
            Ranked list of result dicts.
        """
        if mode == "text":
            return await self.text_search(query, limit, scope=scope, history=history)
        elif mode == "vector":
            return await self.vector_search(
                query, limit, minimum_score, scope=scope, history=history
            )
        else:
            return await self.hybrid_search(
                query, limit, minimum_score, scope=scope, history=history
            )

    async def text_search(
        self,
        query: str,
        limit: int = 10,
        *,
        scope: ClassificationScope,
        history: bool = False,
        hydrate: bool = False,
    ) -> List[Dict[str, Any]]:
        """Full-text search (no embedding needed).

        ``hydrate`` (Track X.2): when ``True``, attach per-hit chunk
        provenance. Off by default so the generic ``/search`` path stays cheap;
        the answer-citation path opts in.
        """
        return await self._repo.text_search(
            query, limit, scope=scope, history=history, hydrate=hydrate
        )

    async def vector_search(
        self,
        query: str,
        limit: int = 10,
        minimum_score: float = 0.2,
        *,
        scope: ClassificationScope,
        history: bool = False,
        hydrate: bool = False,
    ) -> List[Dict[str, Any]]:
        """Embed query text, then vector similarity search.

        Falls back to text search if no embedding model is configured.

        ``hydrate`` (Track X.2): when ``True``, attach per-hit chunk provenance
        (exact matching chunk per source, via the query embedding). Off by
        default; the answer-citation path opts in. On the text-search fallback
        no chunk-level provenance is available regardless.
        """
        try:
            embedding = await self._embed(query)
        except ValueError:
            logger.warning(
                "No embedding model — falling back to text search"
            )
            return await self.text_search(
                query, limit, scope=scope, history=history, hydrate=hydrate
            )

        return await self._repo.vector_search(
            embedding,
            limit,
            minimum_score,
            scope=scope,
            history=history,
            hydrate=hydrate,
        )

    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        minimum_score: float = 0.2,
        text_weight: float = 0.5,
        *,
        scope: ClassificationScope,
        history: bool = False,
        hydrate: bool = False,
    ) -> List[Dict[str, Any]]:
        """Embed query, then combined text + vector search.

        Falls back to text search if no embedding model is configured.

        ``hydrate`` (Track X.2): when ``True``, attach per-hit chunk provenance
        to the fused set. Off by default; the answer-citation path opts in.
        """
        try:
            embedding = await self._embed(query)
        except ValueError:
            logger.warning(
                "No embedding model — falling back to text search"
            )
            return await self.text_search(
                query, limit, scope=scope, history=history, hydrate=hydrate
            )

        return await self._repo.hybrid_search(
            query,
            embedding,
            limit,
            minimum_score,
            text_weight,
            scope=scope,
            history=history,
            hydrate=hydrate,
        )
