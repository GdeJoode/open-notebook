"""
Search service - delegates to the retrieval pipeline.
"""

from typing import Any, Dict, List, Optional


from retrieval.service import RetrievalService
from shared.models.classification import ClassificationScope
from surrealdb_service.repositories import SearchRepository


class SearchService:
    """Service for search business logic.

    Embeds the query through the model gate (F2-08): the embedding model comes
    from the register, and every call goes via ``ModelGate`` to llama-swap.
    Delegates all search operations to the retrieval pipeline's
    RetrievalService.
    """

    def __init__(
        self,
        search_repo: SearchRepository,
        embedding_model: Optional[Any] = None,
    ):
        self.search_repo = search_repo
        self._embedding_model = embedding_model
        self._retrieval: Optional[RetrievalService] = None

    async def _get_retrieval(self) -> RetrievalService:
        """Get or create a RetrievalService with the gate's embedding model."""
        if self._retrieval is not None:
            return self._retrieval

        embedding_model = self._embedding_model
        if embedding_model is None:
            from embeddings.providers import PoortEmbedder

            from app_main.services.model_routing.llm_call import get_aanroeper

            embedding_model = PoortEmbedder(get_aanroeper())

        self._retrieval = RetrievalService(
            search_repo=self.search_repo,
            embedding_model=embedding_model,
        )
        return self._retrieval

    async def text_search(
        self,
        keyword: str,
        results: int = 10,
        *,
        scope: ClassificationScope,
        history: bool = False,
    ) -> List[Dict[str, Any]]:
        """Perform a text search over the source chunks."""
        svc = await self._get_retrieval()
        return await svc.text_search(keyword, results, scope=scope, history=history)

    async def vector_search(
        self,
        keyword: str,
        results: int = 10,
        minimum_score: float = 0.2,
        *,
        scope: ClassificationScope,
        history: bool = False,
    ) -> List[Dict[str, Any]]:
        """Perform a vector search (embeds the query first)."""
        svc = await self._get_retrieval()
        return await svc.vector_search(
            keyword, results, minimum_score, scope=scope, history=history
        )

    async def hybrid_search(
        self,
        keyword: str,
        results: int = 10,
        minimum_score: float = 0.2,
        text_weight: float = 0.5,
        *,
        scope: ClassificationScope,
        history: bool = False,
    ) -> List[Dict[str, Any]]:
        """Perform a hybrid text + vector search."""
        svc = await self._get_retrieval()
        return await svc.hybrid_search(
            keyword,
            results,
            minimum_score,
            text_weight,
            scope=scope,
            history=history,
        )
