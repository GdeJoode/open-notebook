"""Unit tests for the hybrid retrieval service (Track R.3).

Mocks the two underlying signal services (dense R.1 + KG R.2) and asserts the
service: pulls a wider-than-k candidate pool from each, fuses them via RRF
KG-prominently, surfaces per-signal provenance, keeps a one-signal source, and
truncates to k. Also asserts the R.2 carry-in fix — that ``KGRetrievalService``
passes the TRUE corpus source count (``source_repo.count()``) into the pure KG
scorer's IDF — at the seam where the hybrid path depends on it.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from shared.retrieval import BALANCED, KG_HEAVY, FusionWeights

from app_main.services.hybrid_retrieval_service import (
    HybridRetrievalService,
    _pool_size,
)
from app_main.services.kg_retrieval_service import KGRetrievalService


def _dense_svc(rows):
    svc = SimpleNamespace()
    svc.find_related = AsyncMock(return_value=rows)
    return svc


def _kg_svc(rows):
    svc = SimpleNamespace()
    svc.find_related_by_kg = AsyncMock(return_value=rows)
    return svc


def _dense_row(sid, score, title=None):
    return {"id": sid, "title": title, "score": score}


def _kg_row(sid, score, explanation=None, title=None):
    return {
        "id": sid,
        "title": title,
        "score": score,
        "explanation": explanation or [],
    }


def _expl(name, etype="programme", weight=2.0, df=2):
    return [
        {
            "entity_id": "entity:x",
            "name": name,
            "type": etype,
            "document_frequency": df,
            "weight": weight,
            "via_relation": False,
        }
    ]


class TestHybridRetrievalService:
    @pytest.mark.asyncio
    async def test_fuses_both_signals_with_provenance(self):
        dense = _dense_svc(
            [_dense_row("source:a", 0.95), _dense_row("source:b", 0.40)]
        )
        kg = _kg_svc(
            [
                _kg_row("source:b", 4.5, _expl("Regio Deal")),
                _kg_row("source:a", 2.6, _expl("topic", etype="topic")),
            ]
        )
        svc = HybridRetrievalService(dense, kg)

        out = await svc.find_related_hybrid("source:q", k=5, weights=KG_HEAVY)

        ids = [r["id"] for r in out]
        assert set(ids) == {"source:a", "source:b"}
        # KG-heavy: source:b (KG rank1) leads despite dense favouring source:a
        assert ids[0] == "source:b"
        top = out[0]
        assert top["dense"]["present"] and top["kg"]["present"]
        assert top["kg"]["rank"] == 1
        assert top["kg_entities"][0]["name"] == "Regio Deal"
        assert "fused_score" in top

    @pytest.mark.asyncio
    async def test_weight_change_reorders(self):
        """AC3: balanced vs kg-heavy provably reorder the same inputs."""
        dense = [_dense_row("source:a", 0.95), _dense_row("source:b", 0.40)]
        kg = [
            _kg_row("source:b", 4.5, _expl("Regio Deal")),
            _kg_row("source:a", 2.6, _expl("topic", etype="topic")),
        ]
        svc = HybridRetrievalService(_dense_svc(dense), _kg_svc(kg))

        balanced = await svc.find_related_hybrid(
            "source:q", k=5, weights=BALANCED
        )
        svc2 = HybridRetrievalService(_dense_svc(dense), _kg_svc(kg))
        kg_heavy = await svc2.find_related_hybrid(
            "source:q", k=5, weights=KG_HEAVY
        )

        assert [r["id"] for r in balanced][0] == "source:a"
        assert [r["id"] for r in kg_heavy][0] == "source:b"

    @pytest.mark.asyncio
    async def test_one_signal_source_not_dropped(self):
        """AC4: a source present in only one signal still appears."""
        dense = _dense_svc([_dense_row("source:dense_only", 0.9)])
        kg = _kg_svc([_kg_row("source:kg_only", 4.0, _expl("X"))])
        svc = HybridRetrievalService(dense, kg)

        out = await svc.find_related_hybrid("source:q", k=5)
        ids = {r["id"] for r in out}
        assert "source:dense_only" in ids
        assert "source:kg_only" in ids

    @pytest.mark.asyncio
    async def test_preset_resolution_defaults_kg_heavy(self):
        dense = _dense_svc([_dense_row("source:a", 0.9)])
        kg = _kg_svc([_kg_row("source:b", 4.0, [])])
        svc = HybridRetrievalService(dense, kg)
        # preset=None -> kg-heavy default; just assert it runs + ranks both
        out = await svc.find_related_hybrid("source:q", k=5, preset=None)
        assert {r["id"] for r in out} == {"source:a", "source:b"}

    @pytest.mark.asyncio
    async def test_pulls_wider_pool_than_k(self):
        """Both signals are queried over-deep so fusion sees the tail."""
        dense = _dense_svc([])
        kg = _kg_svc([])
        svc = HybridRetrievalService(dense, kg)
        await svc.find_related_hybrid("source:q", k=3)
        pool = _pool_size(3)
        dense.find_related.assert_awaited_once_with("source:q", pool)
        # KG queried with the same pool depth + relation flag
        _, kwargs = kg.find_related_by_kg.call_args
        args = kg.find_related_by_kg.call_args.args
        assert args == ("source:q", pool)
        assert kwargs["expand_relations"] is False

    @pytest.mark.asyncio
    async def test_truncates_to_k(self):
        dense = _dense_svc(
            [_dense_row(f"source:{i}", 0.9 - i * 0.1) for i in range(5)]
        )
        kg = _kg_svc([])
        svc = HybridRetrievalService(dense, kg)
        out = await svc.find_related_hybrid("source:q", k=2)
        assert len(out) == 2

    @pytest.mark.asyncio
    async def test_empty_both_signals_returns_empty(self):
        svc = HybridRetrievalService(_dense_svc([]), _kg_svc([]))
        assert await svc.find_related_hybrid("source:q", k=5) == []


class TestKGTrueSourceCount:
    """AC6 / R.2 carry-in: the KG IDF gets the TRUE corpus source count."""

    @pytest.mark.asyncio
    async def test_passes_count_to_scorer(self, monkeypatch):
        """KGRetrievalService threads source_repo.count() into the scorer."""
        captured = {}

        def _fake_score(query_source_id, entities, **kwargs):
            captured["n_sources"] = kwargs.get("n_sources")
            return []  # empty result is fine; we only assert the IDF input

        monkeypatch.setattr(
            "app_main.services.kg_retrieval_service.score_related_sources",
            _fake_score,
        )

        entity_repo = SimpleNamespace()
        entity_repo.load_active_entity_source_map = AsyncMock(
            return_value=[
                {
                    "id": "entity:1",
                    "canonical_name": "Regio Deal",
                    "entity_type": "programme",
                    "source_documents": ["source:q", "source:b"],
                }
            ]
        )
        source_repo = SimpleNamespace()
        # The corpus has 6 sources even though only 2 carry active entities —
        # the IDF must see 6, not the inferred 2 (the R.2 under-count bug).
        source_repo.count = AsyncMock(return_value=6)

        svc = KGRetrievalService(entity_repo, source_repo)
        await svc.find_related_by_kg("source:q", k=5)

        source_repo.count.assert_awaited_once()
        assert captured["n_sources"] == 6

    @pytest.mark.asyncio
    async def test_explicit_n_sources_skips_count(self, monkeypatch):
        captured = {}

        def _fake_score(query_source_id, entities, **kwargs):
            captured["n_sources"] = kwargs.get("n_sources")
            return []

        monkeypatch.setattr(
            "app_main.services.kg_retrieval_service.score_related_sources",
            _fake_score,
        )
        entity_repo = SimpleNamespace()
        entity_repo.load_active_entity_source_map = AsyncMock(
            return_value=[
                {
                    "id": "entity:1",
                    "canonical_name": "X",
                    "entity_type": "programme",
                    "source_documents": ["source:q", "source:b"],
                }
            ]
        )
        source_repo = SimpleNamespace()
        source_repo.count = AsyncMock(return_value=6)
        svc = KGRetrievalService(entity_repo, source_repo)

        await svc.find_related_by_kg("source:q", k=5, n_sources=9)

        source_repo.count.assert_not_awaited()
        assert captured["n_sources"] == 9

    @pytest.mark.asyncio
    async def test_count_failure_falls_back_to_none(self, monkeypatch):
        captured = {}

        def _fake_score(query_source_id, entities, **kwargs):
            captured["n_sources"] = kwargs.get("n_sources")
            return []

        monkeypatch.setattr(
            "app_main.services.kg_retrieval_service.score_related_sources",
            _fake_score,
        )
        entity_repo = SimpleNamespace()
        entity_repo.load_active_entity_source_map = AsyncMock(
            return_value=[
                {
                    "id": "entity:1",
                    "canonical_name": "X",
                    "entity_type": "programme",
                    "source_documents": ["source:q", "source:b"],
                }
            ]
        )
        source_repo = SimpleNamespace()
        source_repo.count = AsyncMock(side_effect=RuntimeError("db down"))
        svc = KGRetrievalService(entity_repo, source_repo)

        await svc.find_related_by_kg("source:q", k=5)

        # On count failure n_sources is None -> scorer uses observed-sources.
        assert captured["n_sources"] is None
