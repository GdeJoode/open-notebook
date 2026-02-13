"""
Search service - business logic for text and vector search.
"""

from typing import Any, Dict, List, Optional

from llm_manager import ModelManager
from surrealdb_service.repositories import SearchRepository


class SearchService:
    """Service for search business logic."""

    def __init__(
        self,
        search_repo: SearchRepository,
        model_manager: ModelManager,
    ):
        self.search_repo = search_repo
        self.model_manager = model_manager

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
        # Get embedding model from defaults
        defaults = self.model_manager.get_defaults()
        if not defaults or not defaults.default_embedding_model:
            raise ValueError(
                "Vector search requires an embedding model. "
                "Please configure one in the Models section."
            )

        # TODO: Embed the keyword using llm_manager when graph is migrated
        # For now, this delegates to the existing vector_search function
        # which handles embedding internally
        raise NotImplementedError(
            "Vector search via service layer requires graph migration. "
            "Use the monolith vector_search for now."
        )

    async def hybrid_search(
        self,
        keyword: str,
        embedding: List[float],
        results: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
        minimum_score: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Perform a hybrid text + vector search."""
        return await self.search_repo.hybrid_search(
            keyword, embedding, results, include_sources, include_notes, minimum_score
        )
