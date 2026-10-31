"""`POST /entity-resolution/apply` — what the server now refuses, and what it logs.

PC.5b. The route merged whatever clusters it was sent, and no test covered it.
Its docstring said "only auto_merge candidates", enforced by the frontend alone,
which also sends review candidates a curator approved. These tests put the route
in each state it now refuses, and check that a refused request merges NOTHING.

The entity table is an in-memory stand-in for the one read the validation makes
(`SELECT id, canonical_name, entity_type, status FROM $ids`); the audit write has
its own test against a real SurrealDB (`test_merge_audit.py`).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from app_main.api.routers import entity_resolution
from app_main.services.entity_resolution import merge_audit
from app_main.services.entity_resolution.recanonicalization_service import MergeResult
from fastapi import FastAPI
from fastapi.testclient import TestClient

ENTITIES: Dict[str, Dict[str, Any]] = {
    "entity:ez": {"canonical_name": "Economische Zaken", "entity_type": "government_organization", "status": "active"},
    "entity:ez2": {"canonical_name": "economische zaken", "entity_type": "government_organization", "status": "active"},
    "entity:old": {"canonical_name": "EZ", "entity_type": "government_organization", "status": "merged"},
    "entity:gone": {"canonical_name": "X", "entity_type": "government_organization", "status": "archived"},
    "entity:topic": {"canonical_name": "Economische Zaken", "entity_type": "topic", "status": "active"},
}


class _Recanon:
    def __init__(self) -> None:
        self.applied: List[Any] = []

    async def apply_merge(self, cluster):
        self.applied.append(cluster)
        losers = [i for i in cluster.loser_ids if ENTITIES[i]["status"] == "active"]
        return MergeResult(new_canonical=cluster.new_canonical, entity_type=cluster.entity_type,
                           winner_id=cluster.winner_id, merged_loser_ids=losers,
                           relations_repointed=0, aliases_created=0, skipped=not losers)


@pytest.fixture
def env(monkeypatch):
    recanon = _Recanon()
    logged: List[Dict[str, Any]] = []

    async def _query(sql, params=None, config=None):
        assert "FROM $ids" in sql
        return [{"id": str(i), **ENTITIES[str(i)]} for i in params["ids"] if str(i) in ENTITIES]

    async def _record(**kw):
        logged.append(kw)
        return len(list(kw["merged_loser_ids"]))

    monkeypatch.setattr(entity_resolution, "execute_query", _query)
    monkeypatch.setattr(merge_audit, "record_merges", _record)
    app = FastAPI()
    app.include_router(entity_resolution.router, prefix="/api")
    app.dependency_overrides[entity_resolution.get_recanonicalization_service] = lambda: recanon
    return TestClient(app), recanon, logged


def _cluster(winner="entity:ez", losers=("entity:ez2",), **over):
    body = {"new_canonical": "Economische Zaken", "entity_type": "government_organization",
            "winner_id": winner, "loser_ids": list(losers), "origin": "curator_approved",
            "method": "fold_equal", "score": 1.0}
    body.update(over)
    return body


def _post(client, *clusters):
    return client.post("/api/entity-resolution/apply", json={"clusters": list(clusters)})


def test_a_valid_curator_merge_is_applied_and_logged(env):
    client, recanon, logged = env
    r = _post(client, _cluster())
    assert r.status_code == 200 and r.json()["applied"] == 1
    [entry] = logged
    assert entry["origin"] == "curator_approved" and entry["method"] == "fold_equal"
    assert entry["author"] == "Testpersoon"
    assert list(entry["merged_loser_ids"]) == ["entity:ez2"]
    assert entry["names"]["entity:ez"] == "Economische Zaken"


def test_the_origin_is_required(env):
    client, recanon, _ = env
    body = _cluster()
    del body["origin"]
    assert _post(client, body).status_code == 422
    assert recanon.applied == []


@pytest.mark.parametrize("cluster, fragment", [
    (_cluster(losers=("entity:nope",)), "not found"),
    (_cluster(winner="entity:old", losers=("entity:ez",)), "the winner must be active"),
    (_cluster(losers=("entity:gone",)), "a loser must be active or merged"),
    (_cluster(losers=("entity:topic",)), "set cross_type"),
    (_cluster(losers=("entity:topic",), cross_type=True, origin="auto_merge"),
     "only a curator may merge across types"),
    (_cluster(losers=("entity:ez", "entity:ez2")), "appears twice"),
])
def test_each_refused_state_merges_nothing(env, cluster, fragment):
    client, recanon, logged = env
    r = _post(client, cluster)
    assert r.status_code == 422
    assert fragment in r.json()["detail"]
    assert recanon.applied == [] and logged == []


def test_a_malformed_id_is_the_callers_error(env):
    client, recanon, _ = env
    r = _post(client, _cluster(losers=("not a record id",)))
    assert r.status_code == 422 and "malformed" in r.json()["detail"]
    assert recanon.applied == []


def test_one_bad_cluster_blocks_the_whole_request(env):
    client, recanon, logged = env
    r = _post(client, _cluster(), _cluster(winner="entity:topic", losers=("entity:gone",),
                                           entity_type="topic"))
    assert r.status_code == 422
    assert recanon.applied == []  # the valid first cluster was NOT applied


def test_an_id_in_two_clusters_is_refused(env):
    client, recanon, _ = env
    r = _post(client, _cluster(), _cluster(winner="entity:ez2", losers=("entity:topic",),
                                           cross_type=True))
    assert r.status_code == 422 and recanon.applied == []


def test_a_curator_may_merge_across_types_when_it_says_so(env):
    client, recanon, logged = env
    r = _post(client, _cluster(losers=("entity:topic",), cross_type=True,
                               method="fold_equal_cross_type"))
    assert r.status_code == 200 and r.json()["applied"] == 1
    assert logged[0]["types"]["entity:topic"] == "topic"


def test_a_repeat_apply_is_skipped_not_refused_and_not_logged(env):
    """Idempotency: a loser already merged is allowed, the service skips it."""
    client, recanon, logged = env
    r = _post(client, _cluster(losers=("entity:old",)))
    assert r.status_code == 200 and r.json()["skipped"] == 1
    assert logged == []
