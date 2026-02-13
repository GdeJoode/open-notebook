"""Tests for SearchService."""

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
    async def test_vector_search_not_implemented(self, search_repo, model_manager):
        """Even with a configured model, raises NotImplementedError (not yet migrated)."""
        model_manager.get_defaults.return_value = make_default_models(
            default_embedding_model="model:embed1",
        )
        service = SearchService(search_repo, model_manager)

        with pytest.raises(NotImplementedError):
            await service.vector_search("test query")


class TestSearchServiceHybridSearch:

    @pytest.mark.asyncio
    async def test_hybrid_search_delegates(self, search_repo, model_manager):
        search_repo.hybrid_search.return_value = [{"id": "source:1"}]
        service = SearchService(search_repo, model_manager)
        embedding = [0.1, 0.2, 0.3]

        result = await service.hybrid_search("test", embedding, results=5)

        search_repo.hybrid_search.assert_called_once_with(
            "test", embedding, 5, True, True, 0.2,
        )
        assert len(result) == 1
