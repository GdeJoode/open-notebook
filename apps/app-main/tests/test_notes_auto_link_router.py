"""Endpoint tests for POST /notes/{id}/auto-link (Track Y.2).

Drives the route with overridden DI so no DB/embedding model is needed. Proves:

* happy path: the endpoint drives the service and returns the summary;
* the no-embedding case returns a clean 200 with status=needs_embedding (the
  route embeds-first via the service; here the service reports it couldn't) —
  never a 500;
* a missing note → 404;
* route-layer input validation: a malformed/injection note id → 422, and
  out-of-range k / min_similarity → 422 (rejected at the boundary, not a 500).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.notes import router
from app_main.dependencies import (
    get_note_auto_link_service,
    get_note_service,
)
from app_main.services.note_auto_link_service import AutoLinkResult


def _make_app(*, note_service=None, auto_link_service=None) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    if note_service is not None:
        app.dependency_overrides[get_note_service] = lambda: note_service
    if auto_link_service is not None:
        # The real factory is async; the override may return the (mock) instance
        # directly — FastAPI awaits a coroutine OR uses a plain return.
        app.dependency_overrides[get_note_auto_link_service] = (
            lambda: auto_link_service
        )
    return TestClient(app)


def _existing_note():
    n = MagicMock()
    n.id = "note:q"
    return n


def test_auto_link_happy_path_returns_summary():
    note_svc = AsyncMock()
    note_svc.get.return_value = _existing_note()

    auto_svc = AsyncMock()
    auto_svc.auto_link.return_value = AutoLinkResult(
        note_id="note:q",
        status="linked",
        created=2,
        below_threshold=1,
        candidates_considered=3,
        embedded=False,
        min_similarity=0.75,
        k=5,
        linked_note_ids=["note:a", "note:b"],
    )

    client = _make_app(note_service=note_svc, auto_link_service=auto_svc)
    resp = client.post("/api/notes/note:q/auto-link")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "linked"
    assert data["created"] == 2
    assert data["below_threshold"] == 1
    assert data["linked_note_ids"] == ["note:a", "note:b"]
    # Params forwarded with the route defaults.
    auto_svc.auto_link.assert_awaited_once()
    _, kwargs = auto_svc.auto_link.await_args
    assert kwargs["k"] == 5 and kwargs["min_similarity"] == 0.75


def test_auto_link_forwards_query_params():
    note_svc = AsyncMock()
    note_svc.get.return_value = _existing_note()
    auto_svc = AsyncMock()
    auto_svc.auto_link.return_value = AutoLinkResult(
        note_id="note:q", status="linked", min_similarity=0.9, k=3
    )

    client = _make_app(note_service=note_svc, auto_link_service=auto_svc)
    resp = client.post("/api/notes/note:q/auto-link?k=3&min_similarity=0.9")

    assert resp.status_code == 200
    _, kwargs = auto_svc.auto_link.await_args
    assert kwargs["k"] == 3 and kwargs["min_similarity"] == 0.9


def test_auto_link_needs_embedding_is_200_not_500():
    note_svc = AsyncMock()
    note_svc.get.return_value = _existing_note()
    auto_svc = AsyncMock()
    auto_svc.auto_link.return_value = AutoLinkResult(
        note_id="note:q",
        status="needs_embedding",
        embedded=False,
        min_similarity=0.75,
        k=5,
    )

    client = _make_app(note_service=note_svc, auto_link_service=auto_svc)
    resp = client.post("/api/notes/note:q/auto-link")

    assert resp.status_code == 200
    assert resp.json()["status"] == "needs_embedding"


def test_auto_link_missing_note_returns_404():
    note_svc = AsyncMock()
    note_svc.get.return_value = None
    auto_svc = AsyncMock()

    client = _make_app(note_service=note_svc, auto_link_service=auto_svc)
    resp = client.post("/api/notes/note:ghost/auto-link")

    assert resp.status_code == 404
    auto_svc.auto_link.assert_not_awaited()


# ---------------------------------------------------------------------------
# Route-layer input validation (defense-in-depth): bad id / out-of-range params
# are 422 at the boundary, NOT a 500 that fell through to the service.
# ---------------------------------------------------------------------------


def test_injection_note_id_rejected_422_before_service():
    note_svc = AsyncMock()
    auto_svc = AsyncMock()
    client = _make_app(note_service=note_svc, auto_link_service=auto_svc)

    # A SurrealQL-injection payload in the id. The router strict-validates the
    # id BEFORE touching the service/repo. (URL-encoded so it reaches the route.)
    resp = client.post(
        "/api/notes/note:x%3B%20REMOVE%20TABLE%20note%3B%20--/auto-link"
    )
    assert resp.status_code == 422
    # Never reached the service/DB.
    auto_svc.auto_link.assert_not_awaited()
    note_svc.get.assert_not_awaited()


def test_wrong_table_id_rejected_422():
    note_svc = AsyncMock()
    auto_svc = AsyncMock()
    client = _make_app(note_service=note_svc, auto_link_service=auto_svc)

    resp = client.post("/api/notes/source:abc/auto-link")
    assert resp.status_code == 422
    auto_svc.auto_link.assert_not_awaited()


def test_out_of_range_k_rejected_422():
    note_svc = AsyncMock()
    auto_svc = AsyncMock()
    client = _make_app(note_service=note_svc, auto_link_service=auto_svc)

    # k above the route ceiling (MAX_K=50) and k below 1.
    assert client.post("/api/notes/note:q/auto-link?k=9999").status_code == 422
    assert client.post("/api/notes/note:q/auto-link?k=0").status_code == 422
    auto_svc.auto_link.assert_not_awaited()


def test_out_of_range_min_similarity_rejected_422():
    note_svc = AsyncMock()
    auto_svc = AsyncMock()
    client = _make_app(note_service=note_svc, auto_link_service=auto_svc)

    assert (
        client.post("/api/notes/note:q/auto-link?min_similarity=2.0").status_code
        == 422
    )
    assert (
        client.post(
            "/api/notes/note:q/auto-link?min_similarity=-5.0"
        ).status_code
        == 422
    )
    auto_svc.auto_link.assert_not_awaited()
