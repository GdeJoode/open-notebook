"""Endpoint tests for POST /sources/{id}/judge-contradictions (Track Z.3).

Drives the route with overridden DI so no DB/LLM/embedding model is needed (the
judge service is mocked). Proves:

* happy path: the endpoint drives the judge and returns the precision-first
  summary (judged / contradicts / reinforces / neutral / below_threshold /
  edges_written);
* query params (k, min_confidence) are forwarded to the service;
* a missing source → 404 (before the judge is touched);
* route-layer input validation (Y.2 discipline): a malformed/injection source id
  → 422, a wrong-table id → 422, and out-of-range k / min_confidence → 422 —
  rejected at the boundary, NEVER a 500 that fell through to the service.

The LLM is mocked (the judge service is an AsyncMock here), satisfying the Z.3
acceptance that the endpoint test mocks the model.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.sources_processing import router
from app_main.dependencies import (
    get_contradiction_judge_service,
    get_source_service,
)
from app_main.services.contradiction_judge_service import JudgeRunResult


def _make_app(*, source_service=None, judge_service=None) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/sources")
    if source_service is not None:
        app.dependency_overrides[get_source_service] = lambda: source_service
    if judge_service is not None:
        # The real factory is async; the override returns the (mock) instance
        # directly — FastAPI handles a plain return for an async dependency.
        app.dependency_overrides[get_contradiction_judge_service] = (
            lambda: judge_service
        )
    return TestClient(app)


def _existing_source():
    s = MagicMock()
    s.id = "source:q"
    return s


def test_judge_happy_path_returns_summary():
    src_svc = AsyncMock()
    src_svc.get.return_value = _existing_source()

    judge_svc = AsyncMock()
    judge_svc.judge_source.return_value = JudgeRunResult(
        source_id="source:q",
        status="judged",
        judged=3,
        contradicts=1,
        reinforces=1,
        neutral=1,
        below_threshold=0,
        edges_written=2,
        candidates_considered=3,
        min_confidence=0.7,
        k=5,
        verdict_pairs=[
            {
                "from": "source:q",
                "to": "source:a",
                "verdict": "contradicts",
                "confidence": 0.9,
                "edge_written": True,
            }
        ],
    )

    client = _make_app(source_service=src_svc, judge_service=judge_svc)
    resp = client.post("/api/sources/source:q/judge-contradictions")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "judged"
    assert data["judged"] == 3
    assert data["contradicts"] == 1
    assert data["reinforces"] == 1
    assert data["neutral"] == 1
    assert data["edges_written"] == 2
    assert data["verdict_pairs"][0]["verdict"] == "contradicts"
    # Params forwarded with the route defaults.
    judge_svc.judge_source.assert_awaited_once()
    _, kwargs = judge_svc.judge_source.await_args
    assert kwargs["k"] == 5 and kwargs["min_confidence"] == 0.7


def test_judge_forwards_query_params():
    src_svc = AsyncMock()
    src_svc.get.return_value = _existing_source()
    judge_svc = AsyncMock()
    judge_svc.judge_source.return_value = JudgeRunResult(
        source_id="source:q", status="judged", min_confidence=0.9, k=3
    )

    client = _make_app(source_service=src_svc, judge_service=judge_svc)
    resp = client.post(
        "/api/sources/source:q/judge-contradictions?k=3&min_confidence=0.9"
    )

    assert resp.status_code == 200
    _, kwargs = judge_svc.judge_source.await_args
    assert kwargs["k"] == 3 and kwargs["min_confidence"] == 0.9


def test_judge_below_threshold_no_edges_is_200():
    """A run that judged pairs but wrote no edge (all neutral/below) is a clean 200."""
    src_svc = AsyncMock()
    src_svc.get.return_value = _existing_source()
    judge_svc = AsyncMock()
    judge_svc.judge_source.return_value = JudgeRunResult(
        source_id="source:q",
        status="judged",
        judged=2,
        neutral=1,
        below_threshold=1,
        edges_written=0,
        candidates_considered=2,
        min_confidence=0.7,
        k=5,
    )

    client = _make_app(source_service=src_svc, judge_service=judge_svc)
    resp = client.post("/api/sources/source:q/judge-contradictions")

    assert resp.status_code == 200
    data = resp.json()
    assert data["edges_written"] == 0
    assert data["below_threshold"] == 1


def test_judge_missing_source_returns_404():
    src_svc = AsyncMock()
    src_svc.get.return_value = None
    judge_svc = AsyncMock()

    client = _make_app(source_service=src_svc, judge_service=judge_svc)
    resp = client.post("/api/sources/source:ghost/judge-contradictions")

    assert resp.status_code == 404
    judge_svc.judge_source.assert_not_awaited()


# ---------------------------------------------------------------------------
# Route-layer input validation (defense-in-depth): bad id / out-of-range params
# are 422 at the boundary, NOT a 500 that fell through to the service.
# ---------------------------------------------------------------------------


def test_injection_source_id_rejected_422_before_service():
    src_svc = AsyncMock()
    judge_svc = AsyncMock()
    client = _make_app(source_service=src_svc, judge_service=judge_svc)

    # A SurrealQL-injection payload in the id. The router strict-validates the
    # id BEFORE touching the service/repo. (URL-encoded so it reaches the route.)
    resp = client.post(
        "/api/sources/source:x%3B%20REMOVE%20TABLE%20source%3B%20--"
        "/judge-contradictions"
    )
    assert resp.status_code == 422
    # Never reached the service/DB.
    judge_svc.judge_source.assert_not_awaited()
    src_svc.get.assert_not_awaited()


def test_wrong_table_id_rejected_422():
    src_svc = AsyncMock()
    judge_svc = AsyncMock()
    client = _make_app(source_service=src_svc, judge_service=judge_svc)

    resp = client.post("/api/sources/note:abc/judge-contradictions")
    assert resp.status_code == 422
    judge_svc.judge_source.assert_not_awaited()


def test_out_of_range_k_rejected_422():
    src_svc = AsyncMock()
    judge_svc = AsyncMock()
    client = _make_app(source_service=src_svc, judge_service=judge_svc)

    # k above the route ceiling (MAX_K=50) and k below 1.
    assert (
        client.post(
            "/api/sources/source:q/judge-contradictions?k=9999"
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/sources/source:q/judge-contradictions?k=0"
        ).status_code
        == 422
    )
    judge_svc.judge_source.assert_not_awaited()


def test_out_of_range_min_confidence_rejected_422():
    src_svc = AsyncMock()
    judge_svc = AsyncMock()
    client = _make_app(source_service=src_svc, judge_service=judge_svc)

    assert (
        client.post(
            "/api/sources/source:q/judge-contradictions?min_confidence=2.0"
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/sources/source:q/judge-contradictions?min_confidence=-1.0"
        ).status_code
        == 422
    )
    judge_svc.judge_source.assert_not_awaited()
