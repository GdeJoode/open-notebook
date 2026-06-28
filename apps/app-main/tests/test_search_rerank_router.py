"""Router + orchestration tests for the W.2 reranker wiring.

Covers:
- ``rerank=true`` reorders the top-N fused hybrid results via the (mocked)
  reranker service and attaches ``_rerank_score`` (AC1).
- ``rerank=false`` (default) is byte-identical to W.1 — the reranker is never
  touched (AC2).
- reranker-service-down → graceful fallback to the heuristic ``Reranker``,
  logged, never a 500 (AC3).

The reranker service itself is mocked (no model download); we assert the
wiring/ordering, not model quality.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app_main.services.rerank_orchestrator as orch
from app_main.api.routers.search import router
from app_main.dependencies import get_model_service, get_search_service
from app_main.services.reranker_http_client import (
    RerankerHttpClient,
    RerankerServiceError,
)


# ---------------------------------------------------------------------------
# Fixtures mirroring test_search_router.py
# ---------------------------------------------------------------------------


def _model_svc(embedding_model: str | None = "model:embed") -> SimpleNamespace:
    defaults = SimpleNamespace(default_embedding_model=embedding_model)
    manager = SimpleNamespace(get_defaults=lambda: defaults)
    return SimpleNamespace(model_manager=manager)


def _search_svc(hybrid) -> SimpleNamespace:
    svc = SimpleNamespace()
    svc.text_search = AsyncMock(return_value=[])
    svc.vector_search = AsyncMock(return_value=[])
    svc.hybrid_search = AsyncMock(return_value=hybrid)
    return svc


def _make_client(search_svc, model_svc) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_search_service] = lambda: search_svc
    app.dependency_overrides[get_model_service] = lambda: model_svc
    return TestClient(app)


# Fused hybrid results as returned by the repo (content carries passage text).
def _fused():
    return [
        {"id": "a", "content": "alpha passage", "_combined_score": 0.9},
        {"id": "b", "content": "beta passage", "_combined_score": 0.6},
        {"id": "c", "content": "gamma passage", "_combined_score": 0.3},
    ]


# ---------------------------------------------------------------------------
# AC1 — rerank=true reorders via the (mocked) service
# ---------------------------------------------------------------------------


class TestRerankTrue:
    def test_rerank_true_reorders_and_tags(self, monkeypatch):
        # Service says: passage index 2 (gamma) best, then 0 (alpha), then 1 (beta).
        async def fake_rerank(self, query, passages, top_k=None):
            assert passages == ["alpha passage", "beta passage", "gamma passage"]
            return [(2, 5.0), (0, 3.0), (1, 1.0)]

        monkeypatch.setattr(RerankerHttpClient, "rerank", fake_rerank)

        search_svc = _search_svc(_fused())
        client = _make_client(search_svc, _model_svc())

        resp = client.post(
            "/api/search",
            json={"query": "q", "type": "hybrid", "rerank": True},
        )

        assert resp.status_code == 200
        results = resp.json()["results"]
        assert [r["id"] for r in results] == ["c", "a", "b"]
        assert [r["_rerank_score"] for r in results] == [5.0, 3.0, 1.0]

    def test_rerank_true_respects_top_n_tail_preserved(self, monkeypatch):
        # Only the top-1 is reranked; the tail keeps its fused order beneath.
        async def fake_rerank(self, query, passages, top_k=None):
            assert passages == ["alpha passage"]
            return [(0, 9.9)]

        monkeypatch.setattr(RerankerHttpClient, "rerank", fake_rerank)

        search_svc = _search_svc(_fused())
        client = _make_client(search_svc, _model_svc())

        resp = client.post(
            "/api/search",
            json={
                "query": "q",
                "type": "hybrid",
                "rerank": True,
                "rerank_top_n": 1,
            },
        )

        assert resp.status_code == 200
        results = resp.json()["results"]
        # head (a, reranked) then tail (b, c) untouched
        assert [r["id"] for r in results] == ["a", "b", "c"]
        assert results[0]["_rerank_score"] == 9.9
        assert "_rerank_score" not in results[1]


# ---------------------------------------------------------------------------
# AC2 — rerank=false is byte-identical to W.1 (no reranker call)
# ---------------------------------------------------------------------------


class TestRerankFalseByteIdentical:
    def test_default_no_rerank_does_not_touch_reranker(self, monkeypatch):
        called = {"n": 0}

        async def boom(self, query, passages, top_k=None):
            called["n"] += 1
            raise AssertionError("reranker must not be called when rerank=false")

        monkeypatch.setattr(RerankerHttpClient, "rerank", boom)

        fused = _fused()
        # Capture the exact bytes W.1 would return for this same fused output.
        baseline_svc = _search_svc(copy.deepcopy(fused))
        baseline_client = _make_client(baseline_svc, _model_svc())
        baseline = baseline_client.post(
            "/api/search", json={"query": "q", "type": "hybrid"}
        )

        # Now the same request WITHOUT rerank (explicit false).
        svc = _search_svc(copy.deepcopy(fused))
        client = _make_client(svc, _model_svc())
        resp = client.post(
            "/api/search",
            json={"query": "q", "type": "hybrid", "rerank": False},
        )

        assert resp.status_code == 200
        assert called["n"] == 0
        # Byte-identical to the no-rerank baseline.
        assert resp.content == baseline.content
        # And no _rerank_score leaked in.
        for r in resp.json()["results"]:
            assert "_rerank_score" not in r


# ---------------------------------------------------------------------------
# AC3 — service down → graceful heuristic fallback, never a 500
# ---------------------------------------------------------------------------


class TestRerankServiceDownFallback:
    def test_service_error_falls_back_to_heuristic(self, monkeypatch):
        async def down(self, query, passages, top_k=None):
            raise RerankerServiceError("connection refused")

        monkeypatch.setattr(RerankerHttpClient, "rerank", down)

        # Heuristic reranker orders by original "score"; give the middle item
        # the highest score so we can prove the fallback reordered.
        fused = [
            {"id": "a", "content": "alpha", "score": 0.1, "_combined_score": 0.9},
            {"id": "b", "content": "beta", "score": 0.9, "_combined_score": 0.6},
            {"id": "c", "content": "gamma", "score": 0.5, "_combined_score": 0.3},
        ]
        search_svc = _search_svc(fused)
        client = _make_client(search_svc, _model_svc())

        resp = client.post(
            "/api/search",
            json={"query": "q", "type": "hybrid", "rerank": True},
        )

        # Never a 500 on reranker outage.
        assert resp.status_code == 200
        results = resp.json()["results"]
        # Heuristic reordered by score desc: b (0.9), c (0.5), a (0.1).
        assert [r["id"] for r in results] == ["b", "c", "a"]
        # Heuristic attaches its own _rerank_score.
        assert all("_rerank_score" in r for r in results)


# ---------------------------------------------------------------------------
# Orchestrator-level fallback unit test (client injected, no router)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_fallback_on_service_error():
    class DownClient:
        async def rerank(self, query, passages, top_k=None):
            raise RerankerServiceError("boom")

    results = [
        {"id": "x", "content": "x", "score": 0.2},
        {"id": "y", "content": "y", "score": 0.8},
    ]
    out = await orch.rerank_hybrid_results(
        query="q", results=results, top_n=10, client=DownClient()
    )
    # Heuristic ordered by score desc.
    assert [r["id"] for r in out] == ["y", "x"]
    assert all("_rerank_score" in r for r in out)


@pytest.mark.asyncio
async def test_orchestrator_empty_passthrough():
    out = await orch.rerank_hybrid_results(query="q", results=[], top_n=10)
    assert out == []
