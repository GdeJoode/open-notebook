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

import pytest
from app_main.api.routers.triage import router
from app_main.dependencies import get_entity_repo, get_triage_queue_repo
from app_main.services.triage import queue_sources
from app_main.services.triage.triage_queue import QueueItem
from fastapi import FastAPI
from fastapi.testclient import TestClient
from surrealdb_service.repositories.base import RevConflictError


@pytest.fixture(autouse=True)
def _no_source_lookup(monkeypatch):
    """GET /queue now resolves each row's documents (Q-B1); these router tests
    mock the queue, so the lookup is mocked too — the test that pins it overrides
    this with its own return value."""
    async def none(entity_ids, config=None):
        return {}

    monkeypatch.setattr(queue_sources, "sources_for_entities", none)
    monkeypatch.setattr(queue_sources, "entity_types", none)


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
    client = _make_app(queue=queue, entity_repo=repo)

    resp = client.post(
        "/api/triage/queue/triage_queue:1/decision",
        json={"status": "active", "rev": 3},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "active"
    assert data["manual_override"] is True
    # F2-06: row closed against its rev, entity pinned in the same transaction.
    queue.set_decision.assert_awaited_once_with(
        "triage_queue:1", "active", rev=3, door="Testpersoon",
        entiteit=("entity:a", {"status": "active", "manual_override": True}),
    )


def test_decision_without_rev_is_422():
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=_queue_item())
    resp = _make_app(queue=queue, entity_repo=AsyncMock()).post(
        "/api/triage/queue/triage_queue:1/decision", json={"status": "active"})
    assert resp.status_code == 422
    queue.set_decision.assert_not_awaited()


def test_decision_with_a_stale_rev_is_409():
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=_queue_item())
    queue.set_decision = AsyncMock(side_effect=RevConflictError(["triage_queue:1"]))
    resp = _make_app(queue=queue, entity_repo=AsyncMock()).post(
        "/api/triage/queue/triage_queue:1/decision", json={"status": "active", "rev": 0})
    assert resp.status_code == 409


def test_decision_rejects_bad_status():
    queue = AsyncMock()
    repo = AsyncMock()
    client = _make_app(queue=queue, entity_repo=repo)

    resp = client.post(
        "/api/triage/queue/triage_queue:1/decision",
        json={"status": "archived", "rev": 0},
    )
    assert resp.status_code == 422


def test_decision_missing_row_is_404():
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=None)
    repo = AsyncMock()
    client = _make_app(queue=queue, entity_repo=repo)

    resp = client.post(
        "/api/triage/queue/triage_queue:nope/decision",
        json={"status": "active", "rev": 0},
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
        json={"manual_override": True, "status": "active", "rev": 2},
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
    assert (kwargs["rev"], kwargs["door"]) == (2, "Testpersoon")
    queue.set_decision.assert_awaited_once()


def test_entity_override_with_a_stale_rev_is_409_and_closes_nothing():
    queue = AsyncMock()
    repo = AsyncMock()
    repo.set_manual_override = AsyncMock(side_effect=RevConflictError(["entity:a"]))
    resp = _make_app(queue=queue, entity_repo=repo).post(
        "/api/triage/entities/entity:a/override",
        json={"manual_override": True, "status": "active", "rev": 0})
    assert resp.status_code == 409
    queue.set_decision.assert_not_called()


def test_entity_override_release_does_not_touch_queue():
    """Releasing a pin (manual_override=false) leaves the queue alone."""
    queue = AsyncMock()
    repo = AsyncMock()
    repo.set_manual_override = AsyncMock(return_value=True)
    client = _make_app(queue=queue, entity_repo=repo)

    resp = client.post(
        "/api/triage/entities/entity:a/override",
        json={"manual_override": False, "rev": 0},
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
        json={"manual_override": True, "status": "archived", "rev": 0},
    )
    assert resp.status_code == 422


def test_entity_override_missing_entity_is_404():
    repo = AsyncMock()
    repo.set_manual_override = AsyncMock(return_value=False)
    client = _make_app(entity_repo=repo)

    resp = client.post(
        "/api/triage/entities/entity:nope/override",
        json={"manual_override": True, "rev": 0},
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
        "/api/triage/queue/triage_queue:1/decision", json={"status": "active", "rev": 0})
    assert resp.status_code == 409
    entity_repo.set_manual_override.assert_not_awaited()
    queue.set_decision.assert_not_awaited()


def test_linking_writes_the_chosen_uri_and_its_current_names_and_closes():
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=_tooi_item())
    entity_repo = AsyncMock()
    koppeling = {"external_ids": [_TOOI_CANDS[1]["uri"]], "aliases": ["Groningen"]}
    entity_repo.external_ids_wijziging = AsyncMock(return_value=koppeling)
    resp = _make_app(queue=queue, entity_repo=entity_repo).post(
        "/api/triage/queue/triage_queue:1/tooi-decision",
        json={"action": "link", "uri": _TOOI_CANDS[1]["uri"], "rev": 0})
    assert resp.status_code == 200
    entity_repo.external_ids_wijziging.assert_awaited_once_with(
        "entity:a", [_TOOI_CANDS[1]["uri"]], ["Groningen", "GR", "provincie Groningen"])
    # F2-06: link and closed row in one transaction.
    queue.set_decision.assert_awaited_once_with(
        "triage_queue:1", "linked", rev=0, door="Testpersoon", entiteit=("entity:a", koppeling))
    entity_repo.update_external_ids.assert_not_awaited()
    entity_repo.set_manual_override.assert_not_awaited()  # status untouched


def test_linking_a_uri_that_is_not_a_candidate_is_refused():
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=_tooi_item())
    entity_repo = AsyncMock()
    resp = _make_app(queue=queue, entity_repo=entity_repo).post(
        "/api/triage/queue/triage_queue:1/tooi-decision",
        json={"action": "link", "uri": "https://example.org/other", "rev": 0})
    assert resp.status_code == 422
    entity_repo.update_external_ids.assert_not_awaited()


def test_a_missing_entity_keeps_the_row_open():
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=_tooi_item())
    entity_repo = AsyncMock()
    entity_repo.external_ids_wijziging = AsyncMock(return_value=None)
    resp = _make_app(queue=queue, entity_repo=entity_repo).post(
        "/api/triage/queue/triage_queue:1/tooi-decision",
        json={"action": "link", "uri": _TOOI_CANDS[0]["uri"], "rev": 0})
    assert resp.status_code == 404
    queue.set_decision.assert_not_awaited()


def test_dismiss_closes_without_writing_anything_else():
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=_tooi_item())
    entity_repo = AsyncMock()
    resp = _make_app(queue=queue, entity_repo=entity_repo).post(
        "/api/triage/queue/triage_queue:1/tooi-decision", json={"action": "dismiss", "rev": 0})
    assert resp.status_code == 200
    queue.set_decision.assert_awaited_once_with(
        "triage_queue:1", "dismissed", rev=0, door="Testpersoon")
    entity_repo.update_external_ids.assert_not_awaited()
    entity_repo.set_manual_override.assert_not_awaited()


def test_the_tooi_door_refuses_a_status_row():
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=_queue_item())
    resp = _make_app(queue=queue, entity_repo=AsyncMock()).post(
        "/api/triage/queue/triage_queue:1/tooi-decision", json={"action": "dismiss", "rev": 0})
    assert resp.status_code == 409


def test_a_tooi_link_with_a_stale_rev_writes_nothing():
    queue = AsyncMock()
    queue.get = AsyncMock(return_value=_tooi_item(rev=4))
    entity_repo = AsyncMock()
    resp = _make_app(queue=queue, entity_repo=entity_repo).post(
        "/api/triage/queue/triage_queue:1/tooi-decision",
        json={"action": "link", "uri": _TOOI_CANDS[0]["uri"], "rev": 3})
    assert resp.status_code == 409
    entity_repo.update_external_ids.assert_not_awaited()
    queue.set_decision.assert_not_awaited()


def test_pinning_an_entity_closes_its_status_rows_only():
    queue = AsyncMock()
    queue.list_open = AsyncMock(return_value=[
        _queue_item(id="triage_queue:s"), _tooi_item(id="triage_queue:t")])
    entity_repo = AsyncMock()
    entity_repo.set_manual_override = AsyncMock(return_value=True)
    resp = _make_app(queue=queue, entity_repo=entity_repo).post(
        "/api/triage/entities/entity:a/override",
        json={"manual_override": True, "status": "active", "rev": 0})
    assert resp.status_code == 200
    assert resp.json()["closed_queue_rows"] == 1
    queue.set_decision.assert_awaited_once_with(
        "triage_queue:s", "active", rev=None, door="Testpersoon")


def test_the_queue_read_model_carries_kind_and_candidates():
    queue = AsyncMock()
    queue.list_open = AsyncMock(return_value=[_tooi_item()])
    data = _make_app(queue=queue).get("/api/triage/queue").json()
    assert data["items"][0]["kind"] == "tooi"
    assert [c["uri"] for c in data["items"][0]["candidates"]] == [c["uri"] for c in _TOOI_CANDS]


# --- Q-B1: the documents behind *Docs* -----------------------------------------


def test_each_row_carries_its_documents(monkeypatch):
    seen = {}

    async def lookup(entity_ids, config=None):
        seen["ids"] = list(entity_ids)
        return {"entity:a": [{"id": "source:1", "title": "49b9320d0e1e__Convenant X"}]}

    monkeypatch.setattr(queue_sources, "sources_for_entities", lookup)
    queue = AsyncMock()
    queue.list_open = AsyncMock(return_value=[
        _queue_item(), _queue_item(id="triage_queue:2", entity="entity:b")])
    items = _make_app(queue=queue).get("/api/triage/queue").json()["items"]
    assert seen["ids"] == ["entity:a", "entity:b"]  # one batched call for the queue
    assert items[0]["sources"] == [{"id": "source:1", "title": "49b9320d0e1e__Convenant X"}]
    assert items[1]["sources"] == []  # an entity with no documents


def test_a_failing_document_lookup_does_not_fail_the_queue(monkeypatch):
    async def boom(entity_ids, config=None):
        raise RuntimeError("db down")

    monkeypatch.setattr(queue_sources, "sources_for_entities", boom)
    queue = AsyncMock()
    queue.list_open = AsyncMock(return_value=[_queue_item()])
    resp = _make_app(queue=queue).get("/api/triage/queue")
    assert resp.status_code == 200
    assert resp.json()["items"][0]["sources"] == []


# --- WB.3: the canonical type next to the stored label -------------------------


def test_each_row_carries_its_canonical_type(monkeypatch):
    async def types(entity_ids, config=None):
        return {"entity:a": "organization"}

    monkeypatch.setattr(queue_sources, "entity_types", types)
    queue = AsyncMock()
    queue.list_open = AsyncMock(return_value=[
        _queue_item(), _queue_item(id="triage_queue:2", entity="entity:gone")])
    items = _make_app(queue=queue).get("/api/triage/queue").json()["items"]
    assert [(i["entity_type"], i["type"]) for i in items] == [
        ("organization", "Organisatie"), ("", "Organisatie")]


def test_a_failing_type_lookup_does_not_fail_the_queue(monkeypatch):
    async def boom(entity_ids, config=None):
        raise RuntimeError("db down")

    monkeypatch.setattr(queue_sources, "entity_types", boom)
    queue = AsyncMock()
    queue.list_open = AsyncMock(return_value=[_queue_item()])
    resp = _make_app(queue=queue).get("/api/triage/queue")
    assert resp.status_code == 200
    assert resp.json()["items"][0]["entity_type"] == ""
