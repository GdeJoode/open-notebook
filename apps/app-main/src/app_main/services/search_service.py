"""
Search service - delegates to the retrieval pipeline.
"""

from typing import Any, Dict, List, Optional

from esperanto import EmbeddingModel
from loguru import logger

from llm_manager import ModelManager
from retrieval.service import RetrievalService
from surrealdb_service.repositories import SearchRepository


class SearchService:
    """Service for search business logic.

    Resolves the embedding model from ModelManager, then delegates all
    search operations to the retrieval pipeline's RetrievalService.
    """

    def __init__(
        self,
        search_repo: SearchRepository,
        model_manager: ModelManager,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.search_repo = search_repo
        self.model_manager = model_manager
        self._embedding_model = embedding_model
        self._retrieval: Optional[RetrievalService] = None

    async def _get_retrieval(self) -> RetrievalService:
        """Get or create a RetrievalService with the resolved embedding model."""
        if self._retrieval is not None:
            return self._retrieval

        embedding_model = self._embedding_model
        if embedding_model is None:
            embedding_model = await self._resolve_embedding_model()

        self._retrieval = RetrievalService(
            search_repo=self.search_repo,
            embedding_model=embedding_model,
        )
        return self._retrieval

    async def _resolve_embedding_model(self) -> Optional[EmbeddingModel]:
        """Resolve the default embedding model from DB via ModelManager."""
        defaults = self.model_manager.get_defaults()
        if not defaults or not defaults.default_embedding_model:
            return None

        from surrealdb_service.repositories import ModelRepository

        model_repo = ModelRepository()
        model_record = await model_repo.get(defaults.default_embedding_model)
        if not model_record:
            return None

        model = self.model_manager.get_model_from_config(model_record)
        if not isinstance(model, EmbeddingModel):
            return None

        self._embedding_model = model
        return model

    async def text_search(
        self,
        keyword: str,
        results: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
    ) -> List[Dict[str, Any]]:
        """Perform a text search."""
        svc = await self._get_retrieval()
        return await svc.text_search(
            keyword, results, include_sources, include_notes
        )

    async def vector_search(
        self,
        keyword: str,
        results: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
        minimum_score: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Perform a vector search (embeds the query first)."""
        svc = await self._get_retrieval()
        return await svc.vector_search(
            keyword, results, include_sources, include_notes, minimum_score
        )

    async def hybrid_search(
        self,
        keyword: str,
        results: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
        minimum_score: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Perform a hybrid text + vector search."""
        svc = await self._get_retrieval()
        return await svc.hybrid_search(
            keyword, results, include_sources, include_notes, minimum_score
        )
