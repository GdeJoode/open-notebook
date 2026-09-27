"""PC.5b: `record_merges` against a real SurrealDB, read back by the existing reader.

`resolution_log` is SCHEMAFULL (migration 36). A row with a wrong field type is
refused, and `record_merges` swallows that by design (best-effort), so only a
real write proves the rows land — and only `ResolutionLogService`, the reader the
Resolution Log tab uses, proves they are the rows it expects.
"""

from __future__ import annotations

import uuid

import pytest
from app_main.services.entity_resolution.merge_audit import record_merges
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]


async def test_each_merged_pair_is_one_readable_row(live_surrealdb: SurrealDBConfig):
    tag = uuid.uuid4().hex[:6]
    n = await record_merges(
        winner_id="entity:w", merged_loser_ids=["entity:l1", "entity:l2"],
        names={"entity:w": f"Economische Zaken {tag}", "entity:l1": "economische zaken",
               "entity:l2": "Economische zaken"},
        types={"entity:w": "government_organization", "entity:l1": "government_organization"},
        origin="curator_approved", method="fold_equal", score=0.97,
        evidence="type_suffix: economische zaken", config=live_surrealdb,
    )
    assert n == 2
    rows = await execute_query(
        "SELECT * FROM resolution_log WHERE entity_a_text = $a ORDER BY entity_b_text;",
        {"a": f"Economische Zaken {tag}"}, live_surrealdb,
    )
    assert [r["entity_b_text"] for r in rows] == ["Economische zaken", "economische zaken"]
    r = rows[0]
    assert r["match"] is True and r["status"] == "accepted"
    assert r["match_method"] == "curator_approved:fold_equal"
    assert r["confidence"] == pytest.approx(0.97)
    assert r["match_reasoning"] == "type_suffix: economische zaken"
    assert {x["entity_b_label"] for x in rows} == {"government_organization", "UNKNOWN"}
    # Names are not unique; the ids say which records were merged (migration 85).
    assert {x["winner_id"] for x in rows} == {"entity:w"}
    assert {x["loser_id"] for x in rows} == {"entity:l1", "entity:l2"}


async def test_a_plan_cluster_has_no_invented_confidence(live_surrealdb: SurrealDBConfig):
    tag = uuid.uuid4().hex[:6]
    await record_merges(
        winner_id="entity:wp", merged_loser_ids=["entity:lp"],
        names={"entity:wp": f"P{tag}"}, types={},
        origin="curator_approved", method="recanonicalization_plan", config=live_surrealdb,
    )
    [row] = await execute_query(
        "SELECT confidence FROM resolution_log WHERE entity_a_text = $a;",
        {"a": f"P{tag}"}, live_surrealdb,
    )
    assert row["confidence"] == 0.0


async def test_the_existing_reader_lists_them(live_surrealdb: SurrealDBConfig, monkeypatch):
    """The Resolution Log tab's service, filtering on the method as the tab does."""
    from app_main.services import resolution_log_service as mod

    real = mod.execute_query

    async def _bound(q, params=None, config=None):
        return await real(q, params, live_surrealdb)

    monkeypatch.setattr(mod, "execute_query", _bound)
    await record_merges(
        winner_id="entity:w2", merged_loser_ids=["entity:l3"],
        names={"entity:w2": "W", "entity:l3": "L"}, types={},
        origin="auto_merge", method="fold_equal", config=live_surrealdb,
    )
    items, total = await mod.ResolutionLogService().list_candidates(
        match_method="auto_merge:fold_equal")
    assert total >= 1
    assert any(e["entity_b_text"] == "L" for e in items)


async def test_the_whole_reader_works_on_real_rows(live_surrealdb: SurrealDBConfig, monkeypatch):
    """Every method of the Resolution Log service, on rows that exist.

    It read `execute_query`'s result as one list per statement and passed ids as
    plain strings; with no writer and 0 rows that was never exercised."""
    from app_main.services import resolution_log_service as mod

    real = mod.execute_query

    async def _bound(q, params=None, config=None):
        return await real(q, params, live_surrealdb)

    monkeypatch.setattr(mod, "execute_query", _bound)
    tag = uuid.uuid4().hex[:6]
    await record_merges(
        winner_id="entity:w3", merged_loser_ids=["entity:l4"],
        names={"entity:w3": f"W{tag}", "entity:l4": "L4"}, types={},
        origin="curator_approved", method=f"m{tag}", config=live_surrealdb,
    )
    svc = mod.ResolutionLogService()
    items, total = await svc.list_candidates(match_method=f"curator_approved:m{tag}")
    assert total == 1 and items[0]["entity_a_text"] == f"W{tag}"
    rid = str(items[0]["id"])
    assert (await svc.get_candidate(rid))["entity_b_text"] == "L4"
    # A merge record cannot be re-decided: rejecting it would not undo the merge.
    with pytest.raises(mod.MergeRecordImmutable):
        await svc.update_status(rid, "rejected")
    assert await svc.bulk_update_status([rid], "rejected") == 0
    assert (await svc.get_candidate(rid))["status"] == "accepted"
    # A matcher row (no winner_id) still can.
    [m] = await execute_query(
        "CREATE resolution_log CONTENT {entity_a_text: $a, entity_b_text: 'B', "
        "match_method: $m, status: 'pending'};",
        {"a": f"M{tag}", "m": f"matcher{tag}"}, live_surrealdb,
    )
    mid = str(m["id"])
    assert (await svc.update_status(mid, "rejected"))["status"] == "rejected"
    assert await svc.bulk_update_status([mid, rid], "accepted") == 1
    assert f"curator_approved:m{tag}" in await svc.get_methods()
    stats = await svc.get_stats()
    assert stats["total"] >= 1 and stats["accepted"] >= 1


async def test_the_status_route_refuses_a_merge_record():
    """PATCH answers 409 with the reason; bulk counts only the rows it changed."""
    from app_main.api.routers import resolution_log as route
    from app_main.services.resolution_log_service import MergeRecordImmutable
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    class _Svc:
        async def update_status(self, entry_id, status):
            raise MergeRecordImmutable(entry_id)

    app = FastAPI()
    app.include_router(route.router, prefix="/api")
    app.dependency_overrides[route.get_resolution_log_service] = lambda: _Svc()
    r = TestClient(app).patch(
        "/api/resolution-log/resolution_log:x/status", params={"status": "rejected"})
    assert r.status_code == 409 and "applied merge" in r.json()["detail"]
