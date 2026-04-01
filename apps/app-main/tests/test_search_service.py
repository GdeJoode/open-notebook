"""Tests for SearchService."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app_main.services.search_service import SearchService
from tests.conftest import make_default_models


class TestSearchServiceTextSearch:

    @pytest.mark.asyncio
    async def test_text_search_delegates(self, search_repo, model_manager):
        search_repo.text_search.return_value = [
            {"id": "source:1", "title": "Result", "score": 0.9},
        ]
        service = SearchService(search_repo, model_manager)

        result = await service.text_search("python", results=5)

        search_repo.text_search.assert_called_once_with("python", 5, True, True)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_text_search_with_filters(self, search_repo, model_manager):
        search_repo.text_search.return_value = []
        service = SearchService(search_repo, model_manager)

        await service.text_search(
            "test", include_sources=False, include_notes=True,
        )

        search_repo.text_search.assert_called_once_with("test", 10, False, True)


class TestSearchServiceVectorSearch:

    @pytest.mark.asyncio
    async def test_vector_search_raises_without_model(self, search_repo, model_manager):
        """When no embedding model is configured, raises ValueError."""
        model_manager.get_defaults.return_value = make_default_models()
        service = SearchService(search_repo, model_manager)

        with pytest.raises(ValueError, match="embedding model"):
            await service.vector_search("test")

    @pytest.mark.asyncio
    async def test_vector_search_embeds_and_delegates(self, search_repo, model_manager):
        """Vector search embeds the query and delegates to the repository."""
        mock_embedding_model = AsyncMock()
        mock_embedding_model.aembed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        search_repo.vector_search.return_value = [
            {"id": "source:1", "score": 0.8},
        ]

        service = SearchService(
            search_repo, model_manager, embedding_model=mock_embedding_model,
        )

        result = await service.vector_search("test query", results=5)

        mock_embedding_model.aembed.assert_called_once_with(["test query"])
        search_repo.vector_search.assert_called_once_with(
            [0.1, 0.2, 0.3], 5, True, True, 0.2,
        )
        assert len(result) == 1


class TestSearchServiceHybridSearch:

    @pytest.mark.asyncio
    async def test_hybrid_search_embeds_and_delegates(self, search_repo, model_manager):
        """Hybrid search embeds the query and delegates to the repository."""
        mock_embedding_model = AsyncMock()
        mock_embedding_model.aembed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        search_repo.hybrid_search.return_value = [{"id": "source:1"}]

        service = SearchService(
            search_repo, model_manager, embedding_model=mock_embedding_model,
        )

        result = await service.hybrid_search("test", results=5)

        mock_embedding_model.aembed.assert_called_once_with(["test"])
        search_repo.hybrid_search.assert_called_once_with(
            "test", [0.1, 0.2, 0.3], 5, True, True, 0.2,
        )
        assert len(result) == 1
