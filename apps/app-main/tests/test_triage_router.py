"""Q.4 triage router: queue GET + operator decision POST + backbone warnings.

The router is a thin HTTP shell over the Q.4 services; these tests override the
DI providers with mocks and assert the HTTP contract:

* ``GET /api/triage/queue`` returns the queue read-model (AC4 fields);
* ``POST /api/triage/queue/{id}/decision`` pins the entity (status +
  ``manual_override``) and closes the row — the override-wins write that a later
  pipeline run does not overwrite (AC7);
* a bad status is rejected (422); a missing row is 404;
* ``GET /api/triage/backbone-warnings`` returns weak-edge warnings.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from app_main.api.routers.triage import router
from app_main.dependencies import get_entity_repo, get_triage_queue_repo
from app_main.services.triage.triage_queue import QueueItem
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _queue_item(**over) -> QueueItem:
    base = dict(
        id="triage_queue:1",
        entity="entity:a",
        name="Org A",
        type="Organisatie",
        structural_degree=0,
        doc_count=2,
        reason="unsure — operator decides",
        decision="open",
        batch_id="b1",
    )
    base.update(over)
    return QueueItem(**base)


def _make_app(*, queue=None, entity_repo=None):
    app = FastAPI()
    app.include_router(router, prefix="/api")
    if queue is not None:
        app.dependency_overrides[get_triage_queue_repo] = lambda: queue
    if entity_repo is not None:
        app.dependency_overrides[get_entity_repo] = lambda: entity_repo
    return TestClient(app)


def test_get_queue_returns_open_rows():
    queue = AsyncMock()
    queue.list_open = AsyncMock(return_value=[_queue_item()])
    client = _make_app(queue=queue)

    resp = client.get("/api/triage/queue")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    item = data["items"][0]
    # AC4 read-model fields.
    assert item["name"] == "Org A"
    assert item["type"] == "Organisatie"
    assert item["structural_degree"] == 0
    assert item["doc_count"] == 2
    assert item["reason"] == "unsure — operator decides"
    queue.list_open.assert_awaited_once()


def test_get_queue_all_includes_decided():
    queue = AsyncMock()
    queue.list_all = AsyncMock(return_value=[_queue_item(decision="active")])
    client = _make_app(queue=queue)

    resp = client.get("/api/triage/queue?all=true")
    assert resp.status_code == 200
    queue.list_all.assert_awaited_once()


def test_decision_pins_entity_and_closes_row():
    """AC7: a decision sets status + manual_override and closes the queue row."""
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=_queue_item())
    queue.set_decision = AsyncMock(
        return_value=_queue_item(decision="active")
    )
    repo = AsyncMock()
    repo.set_manual_override = AsyncMock(return_value=True)
    client = _make_app(queue=queue, entity_repo=repo)

    resp = client.post(
        "/api/triage/queue/triage_queue:1/decision",
        json={"status": "active"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "active"
    assert data["manual_override"] is True
    # The entity was pinned by hand (override-wins write).
    repo.set_manual_override.assert_awaited_once()
    _, kwargs = repo.set_manual_override.call_args
    assert kwargs["status"] == "active"
    assert kwargs["manual_override"] is True
    # The queue row was closed.
    queue.set_decision.assert_awaited_once()


def test_decision_rejects_bad_status():
    queue = AsyncMock()
    repo = AsyncMock()
    client = _make_app(queue=queue, entity_repo=repo)

    resp = client.post(
        "/api/triage/queue/triage_queue:1/decision",
        json={"status": "archived"},
    )
    assert resp.status_code == 422


def test_decision_missing_row_is_404():
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=None)
    repo = AsyncMock()
    client = _make_app(queue=queue, entity_repo=repo)

    resp = client.post(
        "/api/triage/queue/triage_queue:nope/decision",
        json={"status": "active"},
    )
    assert resp.status_code == 404


def test_backbone_warnings_empty_without_ids():
    repo = AsyncMock()
    client = _make_app(entity_repo=repo)

    resp = client.get("/api/triage/backbone-warnings")
    assert resp.status_code == 200
    assert resp.json() == {"count": 0, "warnings": []}
