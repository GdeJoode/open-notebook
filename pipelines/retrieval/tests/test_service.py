"""Tests for RetrievalService."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from retrieval.service import RetrievalService
from retrieval.reranker import Reranker


class TestRetrievalServiceTextSearch:
    @pytest.mark.asyncio
    async def test_text_search_delegates(self):
        repo = AsyncMock()
        repo.text_search = AsyncMock(return_value=[{"id": "1", "score": 0.9}])

        svc = RetrievalService(search_repo=repo)
        result = await svc.text_search("python", limit=5)

        repo.text_search.assert_called_once_with("python", 5, True, True)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_text_search_with_filters(self):
        repo = AsyncMock()
        repo.text_search = AsyncMock(return_value=[])

        svc = RetrievalService(search_repo=repo)
        await svc.text_search("test", include_sources=False, include_notes=True)

        repo.text_search.assert_called_once_with("test", 10, False, True)


class TestRetrievalServiceVectorSearch:
    @pytest.mark.asyncio
    async def test_embeds_then_searches(self):
        repo = AsyncMock()
        repo.vector_search = AsyncMock(return_value=[{"id": "1", "score": 0.8}])

        embed_model = AsyncMock()
        embed_model.aembed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        svc = RetrievalService(search_repo=repo, embedding_model=embed_model)
        result = await svc.vector_search("test query", limit=5)

        embed_model.aembed.assert_called_once_with(["test query"])
        repo.vector_search.assert_called_once_with(
            [0.1, 0.2, 0.3], 5, True, True, 0.2
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_falls_back_to_text_without_embedding_model(self):
        repo = AsyncMock()
        repo.text_search = AsyncMock(return_value=[{"id": "1"}])

        svc = RetrievalService(search_repo=repo, embedding_model=None)
        result = await svc.vector_search("test")

        repo.text_search.assert_called_once()
        assert len(result) == 1


class TestRetrievalServiceHybridSearch:
    @pytest.mark.asyncio
    async def test_embeds_then_hybrid(self):
        repo = AsyncMock()
        repo.hybrid_search = AsyncMock(return_value=[{"id": "1"}])

        embed_model = AsyncMock()
        embed_model.aembed = AsyncMock(return_value=[[0.1, 0.2]])

        svc = RetrievalService(search_repo=repo, embedding_model=embed_model)
        result = await svc.hybrid_search("test", limit=5)

        embed_model.aembed.assert_called_once()
        repo.hybrid_search.assert_called_once()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_falls_back_without_model(self):
        repo = AsyncMock()
        repo.text_search = AsyncMock(return_value=[])

        svc = RetrievalService(search_repo=repo)
        await svc.hybrid_search("test")

        repo.text_search.assert_called_once()


class TestRetrievalServiceUnifiedSearch:
    @pytest.mark.asyncio
    async def test_mode_text(self):
        repo = AsyncMock()
        repo.text_search = AsyncMock(return_value=[])

        svc = RetrievalService(search_repo=repo)
        await svc.search("test", mode="text")

        repo.text_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_mode_vector(self):
        repo = AsyncMock()
        repo.vector_search = AsyncMock(return_value=[])

        embed_model = AsyncMock()
        embed_model.aembed = AsyncMock(return_value=[[0.1]])

        svc = RetrievalService(search_repo=repo, embedding_model=embed_model)
        await svc.search("test", mode="vector")

        repo.vector_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_mode_hybrid_default(self):
        repo = AsyncMock()
        repo.hybrid_search = AsyncMock(return_value=[])

        embed_model = AsyncMock()
        embed_model.aembed = AsyncMock(return_value=[[0.1]])

        svc = RetrievalService(search_repo=repo, embedding_model=embed_model)
        await svc.search("test")  # Default mode = hybrid

        repo.hybrid_search.assert_called_once()


class TestReranker:
    def test_reranks_by_score(self):
        reranker = Reranker(score_weight=1.0, similarity_weight=0.0)
        results = [
            {"id": "a", "score": 0.5},
            {"id": "b", "score": 0.9},
            {"id": "c", "score": 0.7},
        ]
        ranked = reranker.rerank(results)
        assert [r["id"] for r in ranked] == ["b", "c", "a"]

    def test_reranks_with_embedding_similarity(self):
        reranker = Reranker(score_weight=0.0, similarity_weight=1.0)
        results = [
            {"id": "a", "score": 0.9, "embedding": [0.0, 1.0]},
            {"id": "b", "score": 0.1, "embedding": [1.0, 0.0]},
        ]
        # Query embedding is [1, 0] — should rank "b" higher
        ranked = reranker.rerank(results, query_embedding=[1.0, 0.0])
        assert ranked[0]["id"] == "b"

    def test_top_k(self):
        reranker = Reranker(top_k=2)
        results = [{"id": str(i), "score": i / 10} for i in range(5)]
        ranked = reranker.rerank(results)
        assert len(ranked) == 2

    def test_empty_results(self):
        reranker = Reranker()
        assert reranker.rerank([]) == []

    def test_no_embedding_still_works(self):
        reranker = Reranker()
        results = [{"id": "a", "score": 0.5}]
        ranked = reranker.rerank(results)
        assert len(ranked) == 1
        assert "_rerank_score" in ranked[0]
