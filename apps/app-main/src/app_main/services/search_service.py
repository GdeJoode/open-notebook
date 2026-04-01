"""
Search service - business logic for text and vector search.
"""

from typing import Any, Dict, List, Optional

from esperanto import EmbeddingModel
from loguru import logger

from llm_manager import ModelManager
from surrealdb_service.repositories import SearchRepository


class SearchService:
    """Service for search business logic."""

    def __init__(
        self,
        search_repo: SearchRepository,
        model_manager: ModelManager,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.search_repo = search_repo
        self.model_manager = model_manager
        self._embedding_model = embedding_model

    async def _get_embedding_model(self) -> EmbeddingModel:
        """Resolve the default embedding model."""
        if self._embedding_model is not None:
            return self._embedding_model

        defaults = self.model_manager.get_defaults()
        if not defaults or not defaults.default_embedding_model:
            raise ValueError(
                "Vector search requires an embedding model. "
                "Please configure one in the Models section."
            )

        from surrealdb_service.repositories import ModelRepository

        model_repo = ModelRepository()
        model_record = await model_repo.get(defaults.default_embedding_model)
        if not model_record:
            raise ValueError(
                f"Embedding model '{defaults.default_embedding_model}' not found."
            )

        model = self.model_manager.get_model_from_config(model_record)
        if not isinstance(model, EmbeddingModel):
            raise TypeError(
                f"Model '{defaults.default_embedding_model}' is not an EmbeddingModel."
            )

        self._embedding_model = model
        return model

    async def _embed_query(self, text: str) -> List[float]:
        """Embed a search query into a vector."""
        model = await self._get_embedding_model()
        vectors = await model.aembed([text])
        return vectors[0]

    async def text_search(
        self,
        keyword: str,
        results: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
    ) -> List[Dict[str, Any]]:
        """Perform a text search."""
        return await self.search_repo.text_search(
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
        embedding = await self._embed_query(keyword)
        return await self.search_repo.vector_search(
            embedding, results, include_sources, include_notes, minimum_score
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
        embedding = await self._embed_query(keyword)
        return await self.search_repo.hybrid_search(
            keyword, embedding, results, include_sources, include_notes, minimum_score
        )
