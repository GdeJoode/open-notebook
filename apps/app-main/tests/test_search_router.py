"""Router tests for ``POST /search`` (Track W.1).

Cover the new ``type="hybrid"`` branch (happy path + empty case + graceful
degradation with no embedding model) and assert the existing ``text`` and
``vector`` branches still call their respective service methods unchanged.
The fusion math itself is covered by the service-level test; here we only
verify routing, parameter forwarding, and response shape.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.search import router
from app_main.dependencies import get_model_service, get_search_service


def _model_svc(embedding_model: str | None = "model:embed") -> SimpleNamespace:
    defaults = SimpleNamespace(default_embedding_model=embedding_model)
    manager = SimpleNamespace(get_defaults=lambda: defaults)
    return SimpleNamespace(model_manager=manager)


def _search_svc(**results) -> SimpleNamespace:
    svc = SimpleNamespace()
    svc.text_search = AsyncMock(return_value=results.get("text", []))
    svc.vector_search = AsyncMock(return_value=results.get("vector", []))
    svc.hybrid_search = AsyncMock(return_value=results.get("hybrid", []))
    return svc


def _make_client(search_svc, model_svc) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_search_service] = lambda: search_svc
    app.dependency_overrides[get_model_service] = lambda: model_svc
    return TestClient(app)


class TestSearchHybrid:
    def test_hybrid_happy_path_returns_fused_results(self):
        """AC1: type='hybrid' reaches hybrid_search and returns its results."""
        fused = [
            {"id": "source:both", "_search_type": "hybrid", "_combined_score": 0.85},
            {"id": "source:text", "_search_type": "text", "_combined_score": 0.30},
        ]
        search_svc = _search_svc(hybrid=fused)
        client = _make_client(search_svc, _model_svc())

        resp = client.post("/api/search", json={"query": "regio deal", "type": "hybrid"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["search_type"] == "hybrid"
        assert data["total_count"] == 2
        assert [r["id"] for r in data["results"]] == ["source:both", "source:text"]
        # text/vector branches were NOT used
        search_svc.text_search.assert_not_called()
        search_svc.vector_search.assert_not_called()

    def test_hybrid_forwards_params(self):
        """text_weight, minimum_score, limit, include flags reach the service."""
        search_svc = _search_svc(hybrid=[])
        client = _make_client(search_svc, _model_svc())

        resp = client.post(
            "/api/search",
            json={
                "query": "q",
                "type": "hybrid",
                "limit": 25,
                "search_sources": False,
                "search_notes": True,
                "minimum_score": 0.3,
                "text_weight": 0.7,
            },
        )

        assert resp.status_code == 200
        search_svc.hybrid_search.assert_awaited_once_with(
            keyword="q",
            results=25,
            include_sources=False,
            include_notes=True,
            minimum_score=0.3,
            text_weight=0.7,
        )

    def test_hybrid_default_text_weight_is_half(self):
        search_svc = _search_svc(hybrid=[])
        client = _make_client(search_svc, _model_svc())

        resp = client.post("/api/search", json={"query": "q", "type": "hybrid"})

        assert resp.status_code == 200
        _, kwargs = search_svc.hybrid_search.call_args
        assert kwargs["text_weight"] == 0.5

    def test_hybrid_empty_results(self):
        """AC3: no-results case returns 200 with an empty list, no 500."""
        search_svc = _search_svc(hybrid=[])
        client = _make_client(search_svc, _model_svc())

        resp = client.post("/api/search", json={"query": "nothing", "type": "hybrid"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []
        assert data["total_count"] == 0
        assert data["search_type"] == "hybrid"

    def test_hybrid_no_embedding_model_does_not_error(self):
        """AC3: with no embedding model configured, hybrid still returns 200.

        Unlike pure vector search, hybrid degrades gracefully downstream
        (falls back to text-only), so the router must NOT short-circuit with
        a 400 the way the vector branch does.
        """
        search_svc = _search_svc(hybrid=[{"id": "source:1"}])
        client = _make_client(search_svc, _model_svc(embedding_model=None))

        resp = client.post("/api/search", json={"query": "q", "type": "hybrid"})

        assert resp.status_code == 200
        search_svc.hybrid_search.assert_awaited_once()

    def test_invalid_type_rejected(self):
        client = _make_client(_search_svc(), _model_svc())
        resp = client.post("/api/search", json={"query": "q", "type": "fuzzy"})
        assert resp.status_code == 422

    def test_text_weight_out_of_range_rejected(self):
        client = _make_client(_search_svc(), _model_svc())
        resp = client.post(
            "/api/search",
            json={"query": "q", "type": "hybrid", "text_weight": 1.5},
        )
        assert resp.status_code == 422


class TestSearchBackwardCompatible:
    """AC2: text and vector branches behave exactly as before W.1."""

    def test_text_branch_unchanged(self):
        search_svc = _search_svc(text=[{"id": "source:t"}])
        client = _make_client(search_svc, _model_svc())

        resp = client.post("/api/search", json={"query": "q", "type": "text"})

        assert resp.status_code == 200
        assert resp.json()["search_type"] == "text"
        search_svc.text_search.assert_awaited_once_with(
            keyword="q",
            results=100,
            include_sources=True,
            include_notes=True,
        )
        search_svc.hybrid_search.assert_not_called()
        search_svc.vector_search.assert_not_called()

    def test_default_type_is_text(self):
        search_svc = _search_svc(text=[])
        client = _make_client(search_svc, _model_svc())

        resp = client.post("/api/search", json={"query": "q"})

        assert resp.status_code == 200
        assert resp.json()["search_type"] == "text"
        search_svc.text_search.assert_awaited_once()

    def test_vector_branch_unchanged(self):
        search_svc = _search_svc(vector=[{"id": "source:v"}])
        client = _make_client(search_svc, _model_svc())

        resp = client.post(
            "/api/search",
            json={"query": "q", "type": "vector", "minimum_score": 0.4},
        )

        assert resp.status_code == 200
        assert resp.json()["search_type"] == "vector"
        search_svc.vector_search.assert_awaited_once_with(
            keyword="q",
            results=100,
            include_sources=True,
            include_notes=True,
            minimum_score=0.4,
        )
        search_svc.hybrid_search.assert_not_called()

    def test_vector_without_embedding_model_still_400(self):
        """The vector branch's 400 guard is unchanged by W.1."""
        search_svc = _search_svc(vector=[])
        client = _make_client(search_svc, _model_svc(embedding_model=None))

        resp = client.post("/api/search", json={"query": "q", "type": "vector"})

        assert resp.status_code == 400
        search_svc.vector_search.assert_not_called()
