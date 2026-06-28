"""Unit tests for ``SearchRepository.hybrid_search`` RRF fusion (Track W.1).

These exercise the pure rank-based fusion with ``text_search``/``vector_search``
mocked (no DB): the two signals are on incomparable scales (BM25 ``relevance``
unbounded vs cosine ``similarity`` ``[0,1]``), so the fused order must reflect
*ranks* across both lists — never insertion order and never a raw-magnitude
linear combination.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from surrealdb_service.repositories import SearchRepository


def _text(*ids_relevance):
    return [{"id": i, "relevance": r} for i, r in ids_relevance]


def _vector(*ids_similarity):
    return [{"id": i, "similarity": s} for i, s in ids_similarity]


def _repo(text_results, vector_results) -> SearchRepository:
    repo = SearchRepository()
    repo.text_search = AsyncMock(return_value=text_results)
    repo.vector_search = AsyncMock(return_value=vector_results)
    return repo


class TestHybridRRFRanking:
    @pytest.mark.asyncio
    async def test_both_signals_outrank_single_signal(self):
        """An item high in BOTH lists outranks items in only one list."""
        # source:both is rank 2 in text and rank 2 in vector. source:t1 is
        # rank 1 in text only; source:v1 is rank 1 in vector only. With
        # text_weight=0.5 the two single-signal rank-1 items each score
        # 0.5/(60+1); the both item scores 0.5/(60+2)+0.5/(60+2) which is
        # larger -> it must rank first despite being rank 2 in each list.
        repo = _repo(
            _text(("source:t1", 9.9), ("source:both", 8.0)),
            _vector(("source:v1", 0.95), ("source:both", 0.80)),
        )
        out = await repo.hybrid_search("q", [0.1], results=10)

        assert out[0]["id"] == "source:both"
        assert out[0]["_search_type"] == "hybrid"
        # not insertion order: source:t1 was first in the text list
        assert [r["id"] for r in out][0] != "source:t1"

    @pytest.mark.asyncio
    async def test_order_reflects_rank_not_raw_magnitude(self):
        """BM25's larger magnitude must NOT dominate; RRF ranks on position."""
        # source:a is text rank 1 (huge relevance) but absent from vector.
        # source:b is vector rank 1 and text rank 2. With equal weights:
        #   a = 0.5/(61)               ≈ 0.008197
        #   b = 0.5/(62) + 0.5/(61)    ≈ 0.008065 + 0.008197 = 0.016262
        # b wins because it is corroborated, even though a has a far larger
        # raw text score.
        repo = _repo(
            _text(("source:a", 999.0), ("source:b", 1.0)),
            _vector(("source:b", 0.9)),
        )
        out = await repo.hybrid_search("q", [0.1], results=10)

        assert [r["id"] for r in out] == ["source:b", "source:a"]
        assert out[0]["_combined_score"] > out[1]["_combined_score"]

    @pytest.mark.asyncio
    async def test_text_weight_shifts_order(self):
        """Raising text_weight promotes the text-favoured item (knob works)."""
        text = _text(("source:textfav", 5.0))
        vector = _vector(("source:vecfav", 0.9))

        # text_weight high -> text-only item scores higher than vector-only
        out_hi = await _repo(text, vector).hybrid_search(
            "q", [0.1], results=10, text_weight=0.9
        )
        assert out_hi[0]["id"] == "source:textfav"

        # text_weight low -> vector-only item wins
        out_lo = await _repo(text, vector).hybrid_search(
            "q", [0.1], results=10, text_weight=0.1
        )
        assert out_lo[0]["id"] == "source:vecfav"

    @pytest.mark.asyncio
    async def test_combined_score_non_degenerate(self):
        """Fused scores are positive (the pre-W.1 all-zero bug regression)."""
        repo = _repo(
            _text(("source:1", 4.0), ("source:2", 3.0)),
            _vector(("source:1", 0.8), ("source:3", 0.7)),
        )
        out = await repo.hybrid_search("q", [0.1], results=10)
        assert all(r["_combined_score"] > 0 for r in out)
        # strictly descending (sorted by fused score)
        scores = [r["_combined_score"] for r in out]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_dedup_and_provenance_ranks(self):
        """Items are deduped by id; per-list ranks are recorded."""
        repo = _repo(
            _text(("source:x", 5.0), ("source:y", 4.0)),
            _vector(("source:x", 0.9), ("source:z", 0.8)),
        )
        out = await repo.hybrid_search("q", [0.1], results=10)

        by_id = {r["id"]: r for r in out}
        assert len(out) == 3  # x deduped (one entry)
        assert by_id["source:x"]["_search_type"] == "hybrid"
        assert by_id["source:x"]["_text_rank"] == 1
        assert by_id["source:x"]["_vector_rank"] == 1
        assert by_id["source:y"]["_text_rank"] == 2
        assert by_id["source:y"]["_vector_rank"] is None
        assert by_id["source:z"]["_text_rank"] is None
        assert by_id["source:z"]["_vector_rank"] == 2

    @pytest.mark.asyncio
    async def test_limit_applied_after_fusion(self):
        repo = _repo(
            _text(("source:1", 5.0), ("source:2", 4.0), ("source:3", 3.0)),
            _vector(("source:4", 0.9)),
        )
        out = await repo.hybrid_search("q", [0.1], results=2)
        assert len(out) == 2

    @pytest.mark.asyncio
    async def test_empty_results(self):
        repo = _repo([], [])
        out = await repo.hybrid_search("q", [0.1], results=10)
        assert out == []
