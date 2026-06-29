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

        repo.text_search.assert_called_once_with(
            "python", 5, True, True, hydrate=False
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_text_search_with_filters(self):
        repo = AsyncMock()
        repo.text_search = AsyncMock(return_value=[])

        svc = RetrievalService(search_repo=repo)
        await svc.text_search("test", include_sources=False, include_notes=True)

        repo.text_search.assert_called_once_with(
            "test", 10, False, True, hydrate=False
        )


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
            [0.1, 0.2, 0.3], 5, True, True, 0.2, hydrate=False
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


class TestRetrievalServiceProvenancePassthrough:
    """Track X.1: the repo attaches chunk provenance; the service must surface
    those keys verbatim (it only delegates — no stripping/reshaping).
    """

    @pytest.mark.asyncio
    async def test_vector_search_surfaces_provenance_keys(self):
        hit = {
            "id": "source:s1",
            "similarity": 0.9,
            "chunk_id": "chunk:c1",
            "physical_page": 7,
            "printed_page": 8,
            "section_path": ["Methods"],
            "element_type": "text",
        }
        repo = AsyncMock()
        repo.vector_search = AsyncMock(return_value=[hit])
        embed_model = AsyncMock()
        embed_model.aembed = AsyncMock(return_value=[[0.1, 0.2]])

        svc = RetrievalService(search_repo=repo, embedding_model=embed_model)
        out = await svc.vector_search("q", limit=5)

        assert out[0]["chunk_id"] == "chunk:c1"
        assert out[0]["physical_page"] == 7
        assert out[0]["section_path"] == ["Methods"]
        assert out[0]["element_type"] == "text"

    @pytest.mark.asyncio
    async def test_text_search_surfaces_none_page_for_noteish_hit(self):
        # A note hit has provenance keys present but None (page-less) — must
        # pass through unchanged, never raising.
        hit = {
            "id": "note:n1",
            "relevance": 4.0,
            "chunk_id": None,
            "physical_page": None,
            "section_path": None,
            "element_type": None,
        }
        repo = AsyncMock()
        repo.text_search = AsyncMock(return_value=[hit])

        svc = RetrievalService(search_repo=repo)
        out = await svc.text_search("q")

        assert out[0]["physical_page"] is None
        assert out[0]["relevance"] == 4.0

    @pytest.mark.asyncio
    async def test_hydrate_opt_in_forwarded(self):
        """Track X.2: the answer-citation path opts in with ``hydrate=True``;
        the service must forward that flag to the repo (default stays False so
        the generic ``/search`` path is unaffected — AC5)."""
        repo = AsyncMock()
        repo.text_search = AsyncMock(return_value=[])
        repo.vector_search = AsyncMock(return_value=[])
        embed_model = AsyncMock()
        embed_model.aembed = AsyncMock(return_value=[[0.1, 0.2]])

        svc = RetrievalService(search_repo=repo, embedding_model=embed_model)

        await svc.text_search("q", hydrate=True)
        assert repo.text_search.call_args.kwargs["hydrate"] is True

        await svc.vector_search("q", hydrate=True)
        assert repo.vector_search.call_args.kwargs["hydrate"] is True

        await svc.hybrid_search("q", hydrate=True)
        assert repo.hybrid_search.call_args.kwargs["hydrate"] is True


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


# ===========================================================================
# Edge cases — error paths + boundary behavior
# ===========================================================================


class TestRetrievalServiceEdgeCases:
    @pytest.mark.asyncio
    async def test_search_unknown_mode_defaults_to_hybrid(self):
        repo = AsyncMock()
        repo.hybrid_search = AsyncMock(return_value=[])
        embed_model = AsyncMock()
        embed_model.aembed = AsyncMock(return_value=[[0.1]])

        svc = RetrievalService(search_repo=repo, embedding_model=embed_model)
        await svc.search("anything", mode="bogus-mode")

        repo.hybrid_search.assert_called_once()
        repo.text_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_query_string_still_forwarded(self):
        repo = AsyncMock()
        repo.text_search = AsyncMock(return_value=[])

        svc = RetrievalService(search_repo=repo)
        result = await svc.text_search("")

        repo.text_search.assert_called_once_with(
            "", 10, True, True, hydrate=False
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_repo_results_propagate(self):
        repo = AsyncMock()
        repo.vector_search = AsyncMock(return_value=[])
        embed_model = AsyncMock()
        embed_model.aembed = AsyncMock(return_value=[[0.5, 0.5]])

        svc = RetrievalService(search_repo=repo, embedding_model=embed_model)
        result = await svc.vector_search("query", limit=10)

        assert result == []

    @pytest.mark.asyncio
    async def test_vector_search_passes_explicit_minimum_score(self):
        repo = AsyncMock()
        repo.vector_search = AsyncMock(return_value=[])
        embed_model = AsyncMock()
        embed_model.aembed = AsyncMock(return_value=[[0.1, 0.2]])

        svc = RetrievalService(search_repo=repo, embedding_model=embed_model)
        await svc.vector_search("q", minimum_score=0.8)

        # minimum_score is the 5th positional arg to repo.vector_search
        args, _ = repo.vector_search.call_args
        assert args[4] == 0.8

    @pytest.mark.asyncio
    async def test_hybrid_search_passes_text_weight(self):
        repo = AsyncMock()
        repo.hybrid_search = AsyncMock(return_value=[])
        embed_model = AsyncMock()
        embed_model.aembed = AsyncMock(return_value=[[0.1]])

        svc = RetrievalService(search_repo=repo, embedding_model=embed_model)
        await svc.hybrid_search("q", text_weight=0.7)

        args, _ = repo.hybrid_search.call_args
        # signature: (query, embedding, limit, include_sources,
        #             include_notes, minimum_score, text_weight)
        assert args[6] == 0.7

    @pytest.mark.asyncio
    async def test_embedding_failure_does_not_fall_back(self):
        """If embed_model is set but its call raises (non-ValueError),
        the exception should propagate — only missing-model ValueError
        triggers the text-search fallback.
        """
        repo = AsyncMock()
        embed_model = AsyncMock()
        embed_model.aembed = AsyncMock(side_effect=RuntimeError("network down"))

        svc = RetrievalService(search_repo=repo, embedding_model=embed_model)

        with pytest.raises(RuntimeError, match="network down"):
            await svc.vector_search("query")

        repo.text_search.assert_not_called()
        repo.vector_search.assert_not_called()


class TestRerankerEdgeCases:
    def test_score_none_treated_as_zero(self):
        reranker = Reranker(score_weight=1.0, similarity_weight=0.0)
        results = [
            {"id": "a", "score": None},
            {"id": "b", "score": 0.5},
        ]
        ranked = reranker.rerank(results)
        assert ranked[0]["id"] == "b"
        assert ranked[1]["_rerank_score"] == 0.0  # None coerced to 0

    def test_mixed_embeddings_only_some_get_similarity(self):
        """Results without ``embedding`` get sim_score=0; ones with it
        get a real cosine similarity. Both still rerankable.
        """
        reranker = Reranker(score_weight=0.0, similarity_weight=1.0)
        results = [
            {"id": "with_emb", "score": 0.1, "embedding": [1.0, 0.0]},
            {"id": "no_emb", "score": 0.9},  # no embedding key
        ]
        ranked = reranker.rerank(results, query_embedding=[1.0, 0.0])
        # similarity_weight=1.0, with_emb has sim=1.0, no_emb sim=0
        assert ranked[0]["id"] == "with_emb"
        assert ranked[1]["_rerank_score"] == 0.0

    def test_balanced_weights(self):
        """0.5/0.5 weights should mix score and similarity equally."""
        reranker = Reranker(score_weight=0.5, similarity_weight=0.5)
        results = [
            {"id": "high_score_low_sim", "score": 1.0, "embedding": [0.0, 1.0]},
            {"id": "low_score_high_sim", "score": 0.0, "embedding": [1.0, 0.0]},
        ]
        ranked = reranker.rerank(results, query_embedding=[1.0, 0.0])
        # Both end up with combined score 0.5 — order preserved (stable sort)
        assert ranked[0]["_rerank_score"] == ranked[1]["_rerank_score"] == 0.5

    def test_zero_vector_similarity_safe(self):
        """A zero-norm query embedding should not crash on division."""
        reranker = Reranker(score_weight=0.0, similarity_weight=1.0)
        results = [{"id": "a", "score": 0.5, "embedding": [1.0, 1.0]}]
        ranked = reranker.rerank(results, query_embedding=[0.0, 0.0])
        # Zero-norm → cosine = 0.0 (per _cosine_similarity guard)
        assert ranked[0]["_rerank_score"] == 0.0
