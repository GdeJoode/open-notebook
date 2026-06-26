"""Unit tests for ``GET /sources/{id}/related-hybrid`` (Track R.3).

Exercise the router contract with the hybrid retrieval service mocked: the
fused provenance shape (dense vs KG contribution + KG entities), preset and
explicit-weight forwarding, k bounds, the empty case, and 404 for a missing
source. The fusion math itself is covered by the pure-fusion unit tests
(``packages/shared``) and the service test.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from shared.retrieval import BALANCED, KG_HEAVY

from app_main.api.routers.sources import router
from app_main.dependencies import (
    get_hybrid_retrieval_service,
    get_source_service,
)


def _make_client(source_svc, hybrid_svc) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_source_service] = lambda: source_svc
    app.dependency_overrides[get_hybrid_retrieval_service] = lambda: hybrid_svc
    return TestClient(app)


def _source_svc(source=SimpleNamespace(id="source:abc")):
    svc = SimpleNamespace()
    svc.get = AsyncMock(return_value=source)
    return svc


def _hybrid_svc(related=None, side_effect=None):
    svc = SimpleNamespace()
    if side_effect is not None:
        svc.find_related_hybrid = AsyncMock(side_effect=side_effect)
    else:
        svc.find_related_hybrid = AsyncMock(return_value=related or [])
    return svc


def _signal(present=True, score=None, rank=None, contribution=0.0):
    return {
        "present": present,
        "score": score,
        "rank": rank,
        "contribution": contribution,
    }


def _result(sid, title, fused, dense, kg, kg_entities=None):
    return {
        "id": sid,
        "title": title,
        "fused_score": fused,
        "dense": dense,
        "kg": kg,
        "kg_entities": kg_entities or [],
    }


def _entity(name, etype="programme", df=2, weight=2.0):
    return {
        "entity_id": "entity:x",
        "name": name,
        "type": etype,
        "document_frequency": df,
        "weight": weight,
        "via_relation": False,
    }


class TestRelatedHybridEndpoint:
    def test_returns_fused_with_per_signal_provenance(self):
        """AC1: fused ranking with dense vs KG contribution + KG entities."""
        ranked = [
            _result(
                "source:b",
                "Convenant B",
                0.061,
                _signal(score=0.82, rank=2, contribution=0.016),
                _signal(score=4.5, rank=1, contribution=0.045),
                [_entity("Regio Deal")],
            ),
            _result(
                "source:c",
                "Convenant C",
                0.030,
                _signal(score=0.91, rank=1, contribution=0.016),
                _signal(present=False, contribution=0.0),
            ),
        ]
        client = _make_client(_source_svc(), _hybrid_svc(related=ranked))

        resp = client.get("/api/sources/source:abc/related-hybrid?k=5")

        assert resp.status_code == 200
        data = resp.json()
        assert [r["id"] for r in data] == ["source:b", "source:c"]
        assert data[0]["fused_score"] >= data[1]["fused_score"]
        # per-signal provenance is surfaced
        assert data[0]["dense"]["rank"] == 2
        assert data[0]["kg"]["rank"] == 1
        assert data[0]["kg"]["contribution"] == 0.045
        # KG driving entities carried through
        assert data[0]["kg_entities"][0]["name"] == "Regio Deal"
        # a one-signal source still appears with the absent signal marked
        assert data[1]["kg"]["present"] is False
        assert data[1]["kg"]["score"] is None

    def test_default_preset_kg_heavy_no_explicit_weights(self):
        hybrid = _hybrid_svc(related=[])
        client = _make_client(_source_svc(), hybrid)
        resp = client.get("/api/sources/source:abc/related-hybrid")
        assert resp.status_code == 200
        # default: no explicit weights -> weights=None, preset=None forwarded
        hybrid.find_related_hybrid.assert_awaited_once_with(
            "source:abc", 5, weights=None, preset=None, expand_relations=False
        )

    def test_named_preset_forwarded(self):
        hybrid = _hybrid_svc(related=[])
        client = _make_client(_source_svc(), hybrid)
        resp = client.get(
            "/api/sources/source:abc/related-hybrid?preset=balanced"
        )
        assert resp.status_code == 200
        hybrid.find_related_hybrid.assert_awaited_once_with(
            "source:abc",
            5,
            weights=None,
            preset="balanced",
            expand_relations=False,
        )

    def test_explicit_weights_build_fusion_weights(self):
        """w_kg/w_dense override the preset and reach the service as weights."""
        hybrid = _hybrid_svc(related=[])
        client = _make_client(_source_svc(), hybrid)
        resp = client.get(
            "/api/sources/source:abc/related-hybrid?w_kg=5&w_dense=1"
        )
        assert resp.status_code == 200
        _, kwargs = hybrid.find_related_hybrid.call_args
        weights = kwargs["weights"]
        assert weights is not None
        assert weights.kg == 5.0
        assert weights.dense == 1.0
        assert weights.kg_prominent

    def test_partial_explicit_weight_fills_from_preset(self):
        """Only w_kg given -> dense falls back to the resolved preset's dense."""
        hybrid = _hybrid_svc(related=[])
        client = _make_client(_source_svc(), hybrid)
        resp = client.get(
            "/api/sources/source:abc/related-hybrid?preset=kg-heavy&w_kg=9"
        )
        assert resp.status_code == 200
        _, kwargs = hybrid.find_related_hybrid.call_args
        weights = kwargs["weights"]
        assert weights.kg == 9.0
        assert weights.dense == KG_HEAVY.dense

    def test_expand_relations_forwarded(self):
        hybrid = _hybrid_svc(related=[])
        client = _make_client(_source_svc(), hybrid)
        resp = client.get(
            "/api/sources/source:abc/related-hybrid?expand_relations=true"
        )
        assert resp.status_code == 200
        _, kwargs = hybrid.find_related_hybrid.call_args
        assert kwargs["expand_relations"] is True

    def test_k_below_one_rejected(self):
        client = _make_client(_source_svc(), _hybrid_svc(related=[]))
        resp = client.get("/api/sources/source:abc/related-hybrid?k=0")
        assert resp.status_code == 422

    def test_k_above_max_rejected(self):
        client = _make_client(_source_svc(), _hybrid_svc(related=[]))
        resp = client.get("/api/sources/source:abc/related-hybrid?k=51")
        assert resp.status_code == 422

    def test_negative_weight_rejected(self):
        client = _make_client(_source_svc(), _hybrid_svc(related=[]))
        resp = client.get(
            "/api/sources/source:abc/related-hybrid?w_kg=-1"
        )
        assert resp.status_code == 422

    def test_source_without_signals_returns_empty(self):
        client = _make_client(
            _source_svc(SimpleNamespace(id="source:lonely")),
            _hybrid_svc(related=[]),
        )
        resp = client.get("/api/sources/source:lonely/related-hybrid")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_missing_source_returns_404(self):
        hybrid = _hybrid_svc(related=[])
        client = _make_client(_source_svc(source=None), hybrid)
        resp = client.get("/api/sources/source:ghost/related-hybrid")
        assert resp.status_code == 404
        hybrid.find_related_hybrid.assert_not_awaited()

    def test_service_error_surfaces_as_500(self):
        hybrid = _hybrid_svc(side_effect=RuntimeError("db down"))
        client = _make_client(_source_svc(), hybrid)
        resp = client.get("/api/sources/source:abc/related-hybrid")
        assert resp.status_code == 500
