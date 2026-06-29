"""Tests for SearchService (delegates to RetrievalService)."""

from unittest.mock import AsyncMock, patch

import pytest

from app_main.services.search_service import SearchService
from tests.conftest import make_default_models


class TestSearchServiceTextSearch:

    @pytest.mark.asyncio
    async def test_text_search_delegates(self, search_repo, model_manager):
        search_repo.text_search = AsyncMock(return_value=[
            {"id": "source:1", "title": "Result", "score": 0.9},
        ])
        service = SearchService(search_repo, model_manager)

        result = await service.text_search("python", results=5)

        search_repo.text_search.assert_called_once_with(
            "python", 5, True, True, hydrate=False
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_text_search_with_filters(self, search_repo, model_manager):
        search_repo.text_search = AsyncMock(return_value=[])
        service = SearchService(search_repo, model_manager)

        await service.text_search(
            "test", include_sources=False, include_notes=True,
        )

        search_repo.text_search.assert_called_once_with(
            "test", 10, False, True, hydrate=False
        )


class TestSearchServiceVectorSearch:

    @pytest.mark.asyncio
    async def test_vector_search_falls_back_without_model(self, search_repo, model_manager):
        """Without embedding model, vector search falls back to text search."""
        model_manager.get_defaults.return_value = make_default_models()
        search_repo.text_search = AsyncMock(return_value=[{"id": "1"}])

        service = SearchService(search_repo, model_manager)
        # Should fall back to text search, not raise
        result = await service.vector_search("test")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_vector_search_embeds_and_delegates(self, search_repo, model_manager):
        """Vector search embeds the query and delegates to the repository."""
        mock_embedding_model = AsyncMock()
        mock_embedding_model.aembed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        search_repo.vector_search = AsyncMock(return_value=[
            {"id": "source:1", "score": 0.8},
        ])

        service = SearchService(
            search_repo, model_manager, embedding_model=mock_embedding_model,
        )

        result = await service.vector_search("test query", results=5)

        mock_embedding_model.aembed.assert_called_once_with(["test query"])
        search_repo.vector_search.assert_called_once_with(
            [0.1, 0.2, 0.3], 5, True, True, 0.2, hydrate=False,
        )
        assert len(result) == 1


class TestSearchServiceHybridSearch:

    @pytest.mark.asyncio
    async def test_hybrid_search_embeds_and_delegates(self, search_repo, model_manager):
        """Hybrid search embeds the query and delegates to the repository."""
        mock_embedding_model = AsyncMock()
        mock_embedding_model.aembed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        search_repo.hybrid_search = AsyncMock(return_value=[{"id": "source:1"}])

        service = SearchService(
            search_repo, model_manager, embedding_model=mock_embedding_model,
        )

        result = await service.hybrid_search("test", results=5)

        mock_embedding_model.aembed.assert_called_once_with(["test"])
        search_repo.hybrid_search.assert_called_once_with(
            "test", [0.1, 0.2, 0.3], 5, True, True, 0.2, 0.5, hydrate=False,
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_hybrid_search_threads_custom_text_weight(
        self, search_repo, model_manager
    ):
        """A non-default text_weight is forwarded to the repository (W.1)."""
        mock_embedding_model = AsyncMock()
        mock_embedding_model.aembed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        search_repo.hybrid_search = AsyncMock(return_value=[{"id": "source:1"}])

        service = SearchService(
            search_repo, model_manager, embedding_model=mock_embedding_model,
        )

        await service.hybrid_search("test", results=5, text_weight=0.7)

        search_repo.hybrid_search.assert_called_once_with(
            "test", [0.1, 0.2, 0.3], 5, True, True, 0.2, 0.7, hydrate=False,
        )

    @pytest.mark.asyncio
    async def test_hybrid_search_fuses_text_and_vector(
        self, model_manager
    ):
        """End-to-end RRF fusion through the real SearchRepository (W.1 AC4).

        Uses a SearchRepository with its text/vector primitives mocked so the
        actual rank-based fusion in ``hybrid_search`` runs: an item present in
        both result sets is tagged ``hybrid`` and outranks single-signal items;
        single-signal items are retained with their origin tag. The raw
        ``relevance``/``similarity`` scales differ (BM25 unbounded vs cosine
        [0,1]) — RRF must ignore magnitude and rank on list position.
        """
        from surrealdb_service.repositories import SearchRepository

        repo = SearchRepository()
        repo.text_search = AsyncMock(return_value=[
            {"id": "source:both", "title": "Both", "relevance": 12.0},
            {"id": "source:text", "title": "TextOnly", "relevance": 9.0},
        ])
        repo.vector_search = AsyncMock(return_value=[
            {"id": "source:both", "title": "Both", "similarity": 0.91},
            {"id": "source:vec", "title": "VecOnly", "similarity": 0.88},
        ])

        mock_embedding_model = AsyncMock()
        mock_embedding_model.aembed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        service = SearchService(
            repo, model_manager, embedding_model=mock_embedding_model,
        )

        results = await service.hybrid_search("test", results=10)

        by_id = {r["id"]: r for r in results}
        # the item found by both signals is fused and tagged hybrid
        assert by_id["source:both"]["_search_type"] == "hybrid"
        # both-signal item accrues two RRF terms -> ranks first
        assert results[0]["id"] == "source:both"
        # fused score is non-degenerate (not 0)
        assert results[0]["_combined_score"] > 0
        # single-signal items retained, tagged by their source
        assert by_id["source:text"]["_search_type"] == "text"
        assert by_id["source:vec"]["_search_type"] == "vector"
        # provenance: per-list ranks carried through
        assert by_id["source:both"]["_text_rank"] == 1
        assert by_id["source:both"]["_vector_rank"] == 1
        assert by_id["source:text"]["_vector_rank"] is None

    @pytest.mark.asyncio
    async def test_hybrid_falls_back_without_model(self, search_repo, model_manager):
        """Without embedding model, hybrid falls back to text search."""
        model_manager.get_defaults.return_value = make_default_models()
        search_repo.text_search = AsyncMock(return_value=[])

        service = SearchService(search_repo, model_manager)
        result = await service.hybrid_search("test")
        assert result == []
