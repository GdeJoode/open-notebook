"""`POST /vocabulary/reconcile` — the D2 door, through FastAPI.

The service and reconciler have their own tests; these pin what only the route
decides: it refuses a register that is not loaded in full (and no longer loads
one itself), it refuses types outside the owner's scope, and a dry run hands the
reconciler no repository and the service no queue.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from app_main.api.routers import vocabulary
from app_main.dependencies import (
    get_entity_repo,
    get_reference_entity_repo,
    get_triage_queue_repo,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient


class _Register:
    def __init__(self, rows: int, keyed: int) -> None:
        self._status = {"rows": rows, "keyed": keyed}

    async def key_status(self, source: str) -> Dict[str, int]:
        return dict(self._status)

    async def lookup_by_key(self, key, source=None) -> List[Dict[str, Any]]:
        return []


def _client(register, monkeypatch, seen: Dict[str, Any]) -> TestClient:
    async def fake_service(reconciler, *, types, dry_run, queue=None, **_):
        from app_main.services.entity_resolution.vocabulary_reconcile_service import (
            ReconcileReport,
        )

        seen.update(types=list(types), dry_run=dry_run, queue=queue,
                    repo=reconciler._repo)
        return ReconcileReport(dry_run=dry_run)

    monkeypatch.setattr(vocabulary, "reconcile_active_entities", fake_service)
    app = FastAPI()
    app.include_router(vocabulary.router)
    app.dependency_overrides[get_reference_entity_repo] = lambda: register
    app.dependency_overrides[get_entity_repo] = lambda: "ENTITY_REPO"
    app.dependency_overrides[get_triage_queue_repo] = lambda: "QUEUE"
    return TestClient(app)


@pytest.mark.parametrize("rows, keyed", [
    (0, 0),        # never loaded
    (10, 10),      # only the bundled seed: every municipality would be "no match"
    (1438, 700),   # partly written before migration 81
])
def test_a_register_not_loaded_in_full_is_refused(monkeypatch, rows, keyed):
    seen: Dict[str, Any] = {}
    r = _client(_Register(rows, keyed), monkeypatch, seen).post(
        "/vocabulary/reconcile", json={})
    assert r.status_code == 409
    assert "/vocabulary/refresh" in r.json()["detail"]
    assert seen == {}  # nothing ran


def test_a_type_outside_the_scope_is_refused(monkeypatch):
    r = _client(_Register(1438, 1438), monkeypatch, {}).post(
        "/vocabulary/reconcile", json={"entity_types": ["topic"]})
    assert r.status_code == 422


def test_a_dry_run_gets_no_repository_and_no_queue(monkeypatch):
    seen: Dict[str, Any] = {}
    r = _client(_Register(1438, 1438), monkeypatch, seen).post(
        "/vocabulary/reconcile", json={})
    assert r.status_code == 200
    assert r.json()["dry_run"] is True
    assert r.json()["tooi_register"] == {"rows": 1438, "keyed": 1438}
    assert seen["dry_run"] is True and seen["repo"] is None and seen["queue"] is None


def test_a_real_run_gets_both(monkeypatch):
    seen: Dict[str, Any] = {}
    r = _client(_Register(1438, 1438), monkeypatch, seen).post(
        "/vocabulary/reconcile", json={"dry_run": False,
                                       "entity_types": ["administrative_area"]})
    assert r.status_code == 200
    assert seen == {"types": ["administrative_area"], "dry_run": False,
                    "queue": "QUEUE", "repo": "ENTITY_REPO"}
