"""Unit tests for ``GET /sources/{id}/related-kg`` (Track R.2).

Exercise the router contract with the KG retrieval service mocked: ordering
pass-through, k bounds/validation, the explanation lineage shape, the
no-overlap empty case, and 404 for a missing source. The KG scoring itself is
covered by the pure-scorer unit tests (``packages/shared``) and the live
staging read-only test.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.sources import router
from app_main.dependencies import (
    get_kg_retrieval_service,
    get_source_service,
)


def _make_client(source_svc, kg_svc) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_source_service] = lambda: source_svc
    app.dependency_overrides[get_kg_retrieval_service] = lambda: kg_svc
    return TestClient(app)


def _source_svc(source=SimpleNamespace(id="source:abc")):
    svc = SimpleNamespace()
    svc.get = AsyncMock(return_value=source)
    return svc


def _kg_svc(related=None, side_effect=None):
    svc = SimpleNamespace()
    if side_effect is not None:
        svc.find_related_by_kg = AsyncMock(side_effect=side_effect)
    else:
        svc.find_related_by_kg = AsyncMock(return_value=related or [])
    return svc


def _result(sid, title, score, explanation=None):
    return {
        "id": sid,
        "title": title,
        "score": score,
        "explanation": explanation or [],
    }


def _contrib(eid, name, etype, df, weight, via=False):
    return {
        "entity_id": eid,
        "name": name,
        "type": etype,
        "document_frequency": df,
        "weight": weight,
        "via_relation": via,
    }


class TestRelatedKGEndpoint:
    def test_returns_ranked_with_explanation(self):
        """AC1: ranked sources with the per-pair explanation lineage."""
        ranked = [
            _result(
                "source:b",
                "Convenant B",
                2.4,
                [_contrib("entity:p", "Regio Deal", "programme", 2, 2.4)],
            ),
            _result(
                "source:c",
                "Convenant C",
                0.3,
                [_contrib("entity:t", "gezondheid", "topic", 3, 0.3)],
            ),
        ]
        client = _make_client(_source_svc(), _kg_svc(related=ranked))

        resp = client.get("/api/sources/source:abc/related-kg?k=5")

        assert resp.status_code == 200
        data = resp.json()
        assert [r["id"] for r in data] == ["source:b", "source:c"]
        assert data[0]["score"] >= data[1]["score"]
        # The named-programme pair carries its driving entity in the explanation.
        assert data[0]["explanation"][0]["name"] == "Regio Deal"
        assert data[0]["explanation"][0]["type"] == "programme"
        assert data[0]["explanation"][0]["via_relation"] is False
        # query source never appears
        assert "source:abc" not in [r["id"] for r in data]

    def test_default_k_is_five_and_no_expand(self):
        kg = _kg_svc(related=[])
        client = _make_client(_source_svc(), kg)
        resp = client.get("/api/sources/source:abc/related-kg")
        assert resp.status_code == 200
        kg.find_related_by_kg.assert_awaited_once_with(
            "source:abc", 5, expand_relations=False
        )

    def test_expand_relations_flag_forwarded(self):
        kg = _kg_svc(related=[])
        client = _make_client(_source_svc(), kg)
        resp = client.get(
            "/api/sources/source:abc/related-kg?expand_relations=true"
        )
        assert resp.status_code == 200
        kg.find_related_by_kg.assert_awaited_once_with(
            "source:abc", 5, expand_relations=True
        )

    def test_k_below_one_rejected(self):
        client = _make_client(_source_svc(), _kg_svc(related=[]))
        resp = client.get("/api/sources/source:abc/related-kg?k=0")
        assert resp.status_code == 422

    def test_k_above_max_rejected(self):
        client = _make_client(_source_svc(), _kg_svc(related=[]))
        resp = client.get("/api/sources/source:abc/related-kg?k=51")
        assert resp.status_code == 422

    def test_k_at_upper_bound_accepted(self):
        kg = _kg_svc(related=[])
        client = _make_client(_source_svc(), kg)
        resp = client.get("/api/sources/source:abc/related-kg?k=50")
        assert resp.status_code == 200
        kg.find_related_by_kg.assert_awaited_once_with(
            "source:abc", 50, expand_relations=False
        )

    def test_source_without_overlap_returns_empty(self):
        """AC1: a source that exists but shares no active entities -> []."""
        client = _make_client(
            _source_svc(SimpleNamespace(id="source:lonely")),
            _kg_svc(related=[]),
        )
        resp = client.get("/api/sources/source:lonely/related-kg")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_missing_source_returns_404(self):
        kg = _kg_svc(related=[])
        client = _make_client(_source_svc(source=None), kg)
        resp = client.get("/api/sources/source:ghost/related-kg")
        assert resp.status_code == 404
        kg.find_related_by_kg.assert_not_awaited()

    def test_service_error_surfaces_as_500(self):
        kg = _kg_svc(side_effect=RuntimeError("db down"))
        client = _make_client(_source_svc(), kg)
        resp = client.get("/api/sources/source:abc/related-kg")
        assert resp.status_code == 500
