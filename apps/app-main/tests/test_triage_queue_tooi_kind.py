"""Migration 83: one queue, one row per (entity, kind) — against a real SurrealDB.

D2 review measured three ways a TOOI candidate sharing a status row went wrong:
the next triage run replaced its reason, the only close action pinned status,
and a dismissal was re-created on the next run. Each test here is one of those,
in the state it describes, on the real repository.
"""

from __future__ import annotations

import uuid

import pytest
from app_main.services.triage.triage_queue import (
    DECISION_OPEN,
    KIND_STATUS,
    KIND_TOOI,
    TriageQueueRepository,
)
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query, execute_transaction

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]

_CANDS = [
    {"uri": "https://identifier.overheid.nl/tooi/id/gemeente/gm0014", "label": "Groningen",
     "valid_until": "", "confidence": 0.99, "matched_on": "", "current_names": ["Groningen"]},
    {"uri": "https://identifier.overheid.nl/tooi/id/provincie/pv20", "label": "Groningen",
     "valid_until": "", "confidence": 0.99, "matched_on": "", "current_names": ["Groningen"]},
]


async def _entity(config: SurrealDBConfig) -> str:
    name = f"Groningen-{uuid.uuid4().hex[:8]}"
    rows = await execute_query(
        "CREATE entity SET canonical_name=$n, name_key=string::lowercase($n), name=$n, "
        "entity_type='administrative_area', embedding=[], confidence=0.9, "
        "source_documents=[], status='active' RETURN VALUE id;",
        {"n": name}, config=config,
    )
    return str(rows[0])


async def _rows(config, eid):
    return await execute_query(
        "SELECT * FROM triage_queue WHERE entity = type::thing($e) ORDER BY created_at;",
        {"e": eid}, config=config,
    )


async def test_a_status_flag_does_not_touch_the_tooi_row(live_surrealdb: SurrealDBConfig):
    q = TriageQueueRepository(config=live_surrealdb)
    eid = await _entity(live_surrealdb)
    await q.upsert_tooi(eid, name="Groningen", type="administrative_area",
                        reason="tooi: ambiguous", candidates=_CANDS, batch_id="r1")
    # The Q.4 pipeline's write, with its default replace-the-reason behaviour.
    await q.upsert(eid, name="Groningen", type="administrative_area", structural_degree=0,
                   doc_count=1, reason="isolated core actor", batch_id="t1")
    rows = await _rows(live_surrealdb, eid)
    assert sorted((r["kind"], r["reason"]) for r in rows) == [
        (KIND_STATUS, "isolated core actor"), (KIND_TOOI, "tooi: ambiguous")]
    # And a second status flag refreshes the status row only.
    await q.upsert(eid, name="Groningen", type="administrative_area", structural_degree=0,
                   doc_count=1, reason="unsure — operator decides", batch_id="t2")
    rows = await _rows(live_surrealdb, eid)
    assert {r["kind"]: r["reason"] for r in rows}[KIND_TOOI] == "tooi: ambiguous"
    assert len(rows) == 2


async def test_a_decided_tooi_row_sticks_for_the_same_candidates(
    live_surrealdb: SurrealDBConfig,
):
    q = TriageQueueRepository(config=live_surrealdb)
    eid = await _entity(live_surrealdb)
    rid = await q.upsert_tooi(eid, name="G", type="administrative_area", reason="tooi: x",
                              candidates=_CANDS, batch_id="r1")
    await q.set_decision(rid, "dismissed")
    # The next reconcile run, same question: nothing is queued.
    assert await q.upsert_tooi(eid, name="G", type="administrative_area", reason="tooi: x",
                               candidates=list(reversed(_CANDS)), batch_id="r2") is None
    # A different question (the authority's set changed): queued again.
    new = await q.upsert_tooi(eid, name="G", type="administrative_area", reason="tooi: y",
                              candidates=_CANDS[:1], batch_id="r3")
    assert new is not None and new != rid
    open_rows = [r for r in await _rows(live_surrealdb, eid) if r["decision"] == DECISION_OPEN]
    assert [r["candidate_key"] for r in open_rows] == [_CANDS[0]["uri"]]


async def test_rows_written_before_83_become_status_rows(live_surrealdb: SurrealDBConfig):
    """THE pre-83 state: the row is written while `kind` does not exist."""
    from surrealdb_service.migrations import AsyncMigration
    from surrealdb_service.testing import fixtures as fx

    def sql(v, down=False):
        return AsyncMigration.from_file(
            fx._MIGRATIONS_DIR / f"{v}{'_down' if down else ''}.surrealql").sql

    eid = await _entity(live_surrealdb)
    await execute_transaction(sql(83, down=True), config=live_surrealdb)
    await execute_query(
        "CREATE triage_queue SET entity = type::thing($e), name = 'old', reason = 'r', "
        "batch_id = 'b';", {"e": eid}, config=live_surrealdb)
    await execute_transaction(sql(83), config=live_surrealdb)
    [row] = await _rows(live_surrealdb, eid)
    assert row["kind"] == KIND_STATUS
    # ...and the status path still finds and refreshes it (Q.4 unchanged).
    q = TriageQueueRepository(config=live_surrealdb)
    await q.upsert(eid, name="old", type="", structural_degree=0, doc_count=0,
                   reason="refreshed", batch_id="b2")
    rows = await _rows(live_surrealdb, eid)
    assert [(r["kind"], r["reason"]) for r in rows] == [(KIND_STATUS, "refreshed")]
