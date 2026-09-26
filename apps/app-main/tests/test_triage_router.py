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


def test_entity_override_pins_and_closes_open_queue_rows():
    """Q.5 detail-panel toggle: pinning an entity sets override + closes its rows."""
    queue = AsyncMock()
    queue.list_open = AsyncMock(
        return_value=[_queue_item(id="triage_queue:9", entity="entity:a")]
    )
    queue.set_decision = AsyncMock(
        return_value=_queue_item(id="triage_queue:9", decision="active")
    )
    repo = AsyncMock()
    repo.set_manual_override = AsyncMock(return_value=True)
    client = _make_app(queue=queue, entity_repo=repo)

    resp = client.post(
        "/api/triage/entities/entity:a/override",
        json={"manual_override": True, "status": "active"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["entity"] == "entity:a"
    assert data["manual_override"] is True
    assert data["status"] == "active"
    assert data["closed_queue_rows"] == 1
    _, kwargs = repo.set_manual_override.call_args
    assert kwargs["status"] == "active"
    assert kwargs["manual_override"] is True
    queue.set_decision.assert_awaited_once()


def test_entity_override_release_does_not_touch_queue():
    """Releasing a pin (manual_override=false) leaves the queue alone."""
    queue = AsyncMock()
    repo = AsyncMock()
    repo.set_manual_override = AsyncMock(return_value=True)
    client = _make_app(queue=queue, entity_repo=repo)

    resp = client.post(
        "/api/triage/entities/entity:a/override",
        json={"manual_override": False},
    )
    assert resp.status_code == 200
    assert resp.json()["closed_queue_rows"] == 0
    queue.set_decision.assert_not_called()
    _, kwargs = repo.set_manual_override.call_args
    assert kwargs["manual_override"] is False


def test_entity_override_rejects_bad_status():
    repo = AsyncMock()
    client = _make_app(entity_repo=repo)

    resp = client.post(
        "/api/triage/entities/entity:a/override",
        json={"manual_override": True, "status": "archived"},
    )
    assert resp.status_code == 422


def test_entity_override_missing_entity_is_404():
    repo = AsyncMock()
    repo.set_manual_override = AsyncMock(return_value=False)
    client = _make_app(entity_repo=repo)

    resp = client.post(
        "/api/triage/entities/entity:nope/override",
        json={"manual_override": True},
    )
    assert resp.status_code == 404


# --- migration 83: TOOI rows are decided on their own door -------------------

_TOOI_CANDS = [
    {"uri": "https://identifier.overheid.nl/tooi/id/gemeente/gm0014", "label": "Groningen",
     "current_names": ["Groningen", "gemeente Groningen"]},
    {"uri": "https://identifier.overheid.nl/tooi/id/provincie/pv20", "label": "Groningen",
     "current_names": ["Groningen", "GR", "provincie Groningen"]},
]


def _tooi_item(**over) -> QueueItem:
    return _queue_item(kind="tooi", reason="tooi: ambiguous", candidates=_TOOI_CANDS, **over)


def test_a_status_decision_on_a_tooi_row_is_refused_and_pins_nothing():
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=_tooi_item())
    entity_repo = AsyncMock()
    resp = _make_app(queue=queue, entity_repo=entity_repo).post(
        "/api/triage/queue/triage_queue:1/decision", json={"status": "active"})
    assert resp.status_code == 409
    entity_repo.set_manual_override.assert_not_awaited()
    queue.set_decision.assert_not_awaited()


def test_linking_writes_the_chosen_uri_and_its_current_names_and_closes():
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=_tooi_item())
    entity_repo = AsyncMock()
    entity_repo.update_external_ids = AsyncMock(return_value=True)
    resp = _make_app(queue=queue, entity_repo=entity_repo).post(
        "/api/triage/queue/triage_queue:1/tooi-decision",
        json={"action": "link", "uri": _TOOI_CANDS[1]["uri"]})
    assert resp.status_code == 200
    entity_repo.update_external_ids.assert_awaited_once_with(
        "entity:a", [_TOOI_CANDS[1]["uri"]], ["Groningen", "GR", "provincie Groningen"])
    queue.set_decision.assert_awaited_once_with("triage_queue:1", "linked")
    entity_repo.set_manual_override.assert_not_awaited()  # status untouched


def test_linking_a_uri_that_is_not_a_candidate_is_refused():
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=_tooi_item())
    entity_repo = AsyncMock()
    resp = _make_app(queue=queue, entity_repo=entity_repo).post(
        "/api/triage/queue/triage_queue:1/tooi-decision",
        json={"action": "link", "uri": "https://example.org/other"})
    assert resp.status_code == 422
    entity_repo.update_external_ids.assert_not_awaited()


def test_a_refused_write_keeps_the_row_open():
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=_tooi_item())
    entity_repo = AsyncMock()
    entity_repo.update_external_ids = AsyncMock(return_value=False)
    resp = _make_app(queue=queue, entity_repo=entity_repo).post(
        "/api/triage/queue/triage_queue:1/tooi-decision",
        json={"action": "link", "uri": _TOOI_CANDS[0]["uri"]})
    assert resp.status_code == 502
    queue.set_decision.assert_not_awaited()


def test_dismiss_closes_without_writing_anything_else():
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=_tooi_item())
    entity_repo = AsyncMock()
    resp = _make_app(queue=queue, entity_repo=entity_repo).post(
        "/api/triage/queue/triage_queue:1/tooi-decision", json={"action": "dismiss"})
    assert resp.status_code == 200
    queue.set_decision.assert_awaited_once_with("triage_queue:1", "dismissed")
    entity_repo.update_external_ids.assert_not_awaited()
    entity_repo.set_manual_override.assert_not_awaited()


def test_the_tooi_door_refuses_a_status_row():
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=_queue_item())
    resp = _make_app(queue=queue, entity_repo=AsyncMock()).post(
        "/api/triage/queue/triage_queue:1/tooi-decision", json={"action": "dismiss"})
    assert resp.status_code == 409


def test_pinning_an_entity_closes_its_status_rows_only():
    queue = AsyncMock()
    queue.list_open = AsyncMock(return_value=[
        _queue_item(id="triage_queue:s"), _tooi_item(id="triage_queue:t")])
    entity_repo = AsyncMock()
    entity_repo.set_manual_override = AsyncMock(return_value=True)
    resp = _make_app(queue=queue, entity_repo=entity_repo).post(
        "/api/triage/entities/entity:a/override",
        json={"manual_override": True, "status": "active"})
    assert resp.status_code == 200
    assert resp.json()["closed_queue_rows"] == 1
    queue.set_decision.assert_awaited_once_with("triage_queue:s", "active")


def test_the_queue_read_model_carries_kind_and_candidates():
    queue = AsyncMock()
    queue.list_open = AsyncMock(return_value=[_tooi_item()])
    data = _make_app(queue=queue).get("/api/triage/queue").json()
    assert data["items"][0]["kind"] == "tooi"
    assert [c["uri"] for c in data["items"][0]["candidates"]] == [c["uri"] for c in _TOOI_CANDS]
