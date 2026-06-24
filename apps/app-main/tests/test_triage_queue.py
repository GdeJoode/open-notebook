"""Q.4 triage review-queue: append + dedup-by-entity (``triage_queue`` table).

The queue's load-bearing property is idempotency: re-running the same batch must
NOT grow it. ``TriageQueueRepository.upsert`` enforces at-most-one OPEN row per
entity — a second upsert UPDATEs the existing row (signals/reason/batch refresh)
rather than appending a duplicate. These tests run against a real SurrealDB
testcontainer (so the migration-60 schema + the SQL dedup lookup are exercised,
not mocked) and cover:

* a fresh entity → one CREATE'd open row;
* re-upserting the SAME entity → still ONE row, fields refreshed (no duplicate);
* a DIFFERENT entity → a second distinct row;
* a decided row no longer dedups — a later upsert opens a NEW row;
* ``set_decision`` moves a row off 'open' and drops it from ``list_open``.
"""

from __future__ import annotations

import uuid

import pytest
from app_main.services.triage import triage_queue as tq_module
from app_main.services.triage.triage_queue import (
    DECISION_OPEN,
    TriageQueueRepository,
)
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query

pytestmark = pytest.mark.asyncio


def _u(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _bind_queue(
    monkeypatch: pytest.MonkeyPatch, config: SurrealDBConfig
) -> TriageQueueRepository:
    """Wire the queue repo at the live container.

    The repo passes ``self.config`` through to ``execute_query``; the
    testcontainer config is supplied directly via the constructor.
    """
    return TriageQueueRepository(config=config)


async def _seed_entity(config: SurrealDBConfig, name: str) -> str:
    rows = await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, entity_type='organization', "
        "primary_type='Organisatie', embedding=[], confidence=0.9, "
        "source_documents=[], status='active' RETURN VALUE id;",
        {"n": name},
        config=config,
    )
    return str(rows[0])


@pytest.mark.requires_docker
async def test_fresh_entity_creates_one_open_row(
    live_surrealdb: SurrealDBConfig,
) -> None:
    q = _bind_queue(None, live_surrealdb)  # type: ignore[arg-type]
    eid = await _seed_entity(live_surrealdb, _u("Org"))

    qid = await q.upsert(
        eid,
        name="Org A",
        type="Organisatie",
        structural_degree=0,
        doc_count=1,
        reason="unsure — operator decides",
        batch_id="b1",
    )
    assert qid

    open_items = [i for i in await q.list_open() if i.entity == eid]
    assert len(open_items) == 1
    item = open_items[0]
    assert item.decision == DECISION_OPEN
    assert item.reason == "unsure — operator decides"
    assert item.doc_count == 1


@pytest.mark.requires_docker
async def test_reupsert_same_entity_dedups_and_refreshes(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The idempotency contract: re-running a batch updates, never duplicates."""
    q = _bind_queue(None, live_surrealdb)  # type: ignore[arg-type]
    eid = await _seed_entity(live_surrealdb, _u("Org"))

    first = await q.upsert(
        eid, name="Org", type="Organisatie", structural_degree=0,
        doc_count=1, reason="unsure — operator decides", batch_id="b1",
    )
    # Re-run with refreshed signals (degree grew, new batch).
    second = await q.upsert(
        eid, name="Org", type="Organisatie", structural_degree=4,
        doc_count=3, reason="candidate active", batch_id="b2",
    )
    assert first == second, "the open row id must be stable (no new row)"

    open_items = [i for i in await q.list_open() if i.entity == eid]
    assert len(open_items) == 1, "exactly one open row per entity"
    item = open_items[0]
    assert item.structural_degree == 4
    assert item.doc_count == 3
    assert item.reason == "candidate active"
    assert item.batch_id == "b2"


@pytest.mark.requires_docker
async def test_distinct_entities_get_distinct_rows(
    live_surrealdb: SurrealDBConfig,
) -> None:
    q = _bind_queue(None, live_surrealdb)  # type: ignore[arg-type]
    e1 = await _seed_entity(live_surrealdb, _u("Org"))
    e2 = await _seed_entity(live_surrealdb, _u("Org"))

    id1 = await q.upsert(
        e1, name="A", type="Organisatie", structural_degree=0,
        doc_count=1, reason="r", batch_id="b",
    )
    id2 = await q.upsert(
        e2, name="B", type="Organisatie", structural_degree=0,
        doc_count=1, reason="r", batch_id="b",
    )
    assert id1 != id2
    entities = {i.entity for i in await q.list_open()}
    assert {e1, e2} <= entities


@pytest.mark.requires_docker
async def test_decided_row_does_not_dedup_new_open_row(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Once decided, a later upsert opens a NEW row (only OPEN rows dedup)."""
    q = _bind_queue(None, live_surrealdb)  # type: ignore[arg-type]
    eid = await _seed_entity(live_surrealdb, _u("Org"))

    first = await q.upsert(
        eid, name="A", type="Organisatie", structural_degree=0,
        doc_count=1, reason="r", batch_id="b1",
    )
    decided = await q.set_decision(first, "active")
    assert decided is not None and decided.decision == "active"
    # The decided row is no longer open.
    assert first not in {i.id for i in await q.list_open()}

    second = await q.upsert(
        eid, name="A", type="Organisatie", structural_degree=1,
        doc_count=2, reason="r2", batch_id="b2",
    )
    assert second != first, "a decided row must not be re-opened"
    open_for_entity = [i for i in await q.list_open() if i.entity == eid]
    assert len(open_for_entity) == 1
    assert open_for_entity[0].id == second


@pytest.mark.requires_docker
async def test_set_decision_missing_row_returns_none(
    live_surrealdb: SurrealDBConfig,
) -> None:
    q = _bind_queue(None, live_surrealdb)  # type: ignore[arg-type]
    assert await q.set_decision("triage_queue:does_not_exist", "active") is None


@pytest.mark.requires_docker
async def test_match_conflict_does_not_clobber_status_flag_signals(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Q.4 review minor: a signal-less match-conflict upsert must NOT overwrite a
    prior status-flag row's real degree/doc_count/reason for the same entity.

    The status-flag surfacing (``_triage_one``) records a row with real signals;
    the later match-conflict surfacing (``_surface_match_conflicts``) calls
    upsert with ``structural_degree=0, doc_count=0, merge_reason=True``. The
    fixed upsert preserves the non-zero signals (zero-as-unknown) and APPENDS the
    match-conflict reason rather than replacing it.
    """
    q = _bind_queue(None, live_surrealdb)  # type: ignore[arg-type]
    eid = await _seed_entity(live_surrealdb, _u("Org"))

    # 1) Status-flag surfacing: a real degree/doc_count + a status reason.
    first = await q.upsert(
        eid, name="Org A", type="Organisatie", structural_degree=4,
        doc_count=3, reason="candidate active", batch_id="b1",
    )
    # 2) Match-conflict surfacing: signal-less, merge-reason.
    second = await q.upsert(
        eid, name="Org A", type="Organisatie", structural_degree=0,
        doc_count=0, reason="match conflict: 'A' ~ 'A.'", batch_id="b1",
        merge_reason=True,
    )
    assert first == second, "still one open row per entity"

    open_items = [i for i in await q.list_open() if i.entity == eid]
    assert len(open_items) == 1
    item = open_items[0]
    # Real signals survive the signal-less match-conflict upsert.
    assert item.structural_degree == 4
    assert item.doc_count == 3
    # Both reasons present, status reason not lost.
    assert "candidate active" in item.reason
    assert "match conflict" in item.reason

    # 3) Idempotency: re-running the SAME match-conflict appends nothing.
    await q.upsert(
        eid, name="Org A", type="Organisatie", structural_degree=0,
        doc_count=0, reason="match conflict: 'A' ~ 'A.'", batch_id="b1",
        merge_reason=True,
    )
    again = [i for i in await q.list_open() if i.entity == eid][0]
    assert again.reason == item.reason, "merge_reason must be idempotent"
    assert again.structural_degree == 4
    assert again.doc_count == 3


async def test_merge_reason_appends_and_is_idempotent() -> None:
    """Fast unit test of the pure reason-merge helper (no DB)."""
    merge = tq_module._merge_reason
    # Append a distinct clause.
    assert merge("candidate active", "match conflict") == (
        "candidate active; match conflict"
    )
    # Already-present clause is not re-appended (idempotent).
    assert merge("candidate active; match conflict", "match conflict") == (
        "candidate active; match conflict"
    )
    # Empty existing -> just the incoming.
    assert merge("", "match conflict") == "match conflict"
    # Empty incoming -> existing untouched.
    assert merge("candidate active", "") == "candidate active"
