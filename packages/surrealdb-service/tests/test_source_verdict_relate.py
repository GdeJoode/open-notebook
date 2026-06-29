"""Container tests for `SourceRepository.relate_verdict` (Track Z.1).

`relate_verdict` is the persistence seam for Track Z (contradiction detection):
it idempotently RELATEs a ``source -> source_verdict -> source`` edge carrying an
LLM judge's verdict (reinforces/contradicts/neutral), confidence, reasoning, and
judge model. It mirrors `NoteRepository.relate_note` (Y.1), INCLUDING the
load-bearing injection fix: the RAW id strings are strict-validated against
`_RECORD_ID_RE` via `_validate_record_id` BEFORE interpolation, because RELATE
graph syntax cannot bind a `$param` in the in/out position and `RecordID.parse`
round-trips an injection payload verbatim (so it is NOT a validator).

These tests prove the Z.1 helper acceptance against a real SurrealDB container:

* idempotent — re-relating the same pair yields exactly ONE edge with the latest
  verdict/confidence (clear-before-relate; RELATE is not idempotent);
* refuses a self-edge;
* fields round-trip (verdict/confidence/reasoning/judge_model);
* rejects a wrong-table (non-source) id;
* INJECTION-SAFE — a `;`/`REMOVE TABLE`-bearing id in BOTH the from and to
  positions is REFUSED and neither the `source` table nor the `source_verdict`
  edge row count changes (the Y.1 failure class — must not regress).

Run:

    uv run --project packages/surrealdb-service \\
        pytest packages/surrealdb-service/tests/test_source_verdict_relate.py -v
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.source import SourceRepository


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


async def _make_source(config: SurrealDBConfig) -> str:
    rows = await execute_query(
        "CREATE source SET title = $t RETURN AFTER;",
        {"t": _unique("source")},
        config=config,
    )
    return rows[0]["id"]


async def _edge_rows(
    config: SurrealDBConfig, from_id: str, to_id: str
) -> list[dict]:
    return await execute_query(
        "SELECT in, out, verdict, confidence, reasoning, judge_model, created_at "
        "FROM source_verdict WHERE in = $f AND out = $t",
        {"f": ensure_record_id(from_id), "t": ensure_record_id(to_id)},
        config=config,
    )


# ---------------------------------------------------------------------------
# AC2: relate_verdict is idempotent (re-relate same pair → one row, latest verdict).
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relate_verdict_idempotent_single_edge(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = SourceRepository(config=live_surrealdb)
    a = await _make_source(live_surrealdb)
    b = await _make_source(live_surrealdb)

    assert await repo.relate_verdict(
        a, b, verdict="neutral", confidence=0.4
    ) is True
    # Re-judge the same pair with a different, higher-confidence verdict.
    assert await repo.relate_verdict(
        a, b, verdict="contradicts", confidence=0.92
    ) is True

    rows = await _edge_rows(live_surrealdb, a, b)
    assert len(rows) == 1, (
        f"relating the same pair twice must yield exactly ONE edge, got {len(rows)}"
    )
    # The surviving row carries the LATEST verdict/confidence (clear-before-relate).
    assert rows[0]["verdict"] == "contradicts"
    assert abs(float(rows[0]["confidence"]) - 0.92) < 1e-9


# ---------------------------------------------------------------------------
# AC2: refuses a self-edge.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relate_verdict_refuses_self_edge(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = SourceRepository(config=live_surrealdb)
    a = await _make_source(live_surrealdb)
    assert await repo.relate_verdict(
        a, a, verdict="contradicts", confidence=1.0
    ) is False
    rows = await _edge_rows(live_surrealdb, a, a)
    assert rows == [], "a self-edge must never be written"


# ---------------------------------------------------------------------------
# AC2: fields round-trip.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relate_verdict_fields_round_trip(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = SourceRepository(config=live_surrealdb)
    a = await _make_source(live_surrealdb)
    b = await _make_source(live_surrealdb)

    assert await repo.relate_verdict(
        a,
        b,
        verdict="reinforces",
        confidence=0.81,
        reasoning="both cite the same 2021 trial",
        judge_model="qwen2.5-72b",
    ) is True
    rows = await _edge_rows(live_surrealdb, a, b)
    assert len(rows) == 1
    edge = rows[0]
    assert str(edge["in"]) == a and str(edge["out"]) == b
    assert edge["verdict"] == "reinforces"
    assert abs(float(edge["confidence"]) - 0.81) < 1e-9
    assert edge["reasoning"] == "both cite the same 2021 trial"
    assert edge["judge_model"] == "qwen2.5-72b"
    assert edge["created_at"] is not None, "created_at default must populate"


# ---------------------------------------------------------------------------
# AC2: rejects a wrong-table (non-source) id.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relate_verdict_rejects_non_source_id(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = SourceRepository(config=live_surrealdb)
    a = await _make_source(live_surrealdb)
    # A note id is a valid record id but the wrong table for source_verdict.
    note = await execute_query(
        "CREATE note SET title = $t, content = $t, embedding = [] RETURN AFTER;",
        {"t": _unique("note")},
        config=live_surrealdb,
    )
    assert await repo.relate_verdict(
        a, str(note[0]["id"]), verdict="contradicts", confidence=0.9
    ) is False


# ---------------------------------------------------------------------------
# AC3 (SECURITY): a SurrealQL-injection id is REFUSED in BOTH the from and to
# positions — the `source` table is NOT dropped and no rogue `source_verdict`
# edge is written. `RecordID.parse` splits on the first colon and round-trips an
# injection payload verbatim, so a `startswith("source:")` check would pass it;
# the strict `_RECORD_ID_RE` validator must reject it BEFORE interpolation.
# This is the Y.1 failure class — it must not regress.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relate_verdict_refuses_sql_injection_id(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = SourceRepository(config=live_surrealdb)
    a = await _make_source(live_surrealdb)
    b = await _make_source(live_surrealdb)

    # A drop-table payload. If interpolated,
    # `RELATE source:x; REMOVE TABLE source; --->source_verdict->...` executes as
    # multiple statements and wipes the `source` table (the reproduced blocker).
    payload = "source:x; REMOVE TABLE source; --"

    async def _source_count() -> int:
        rows = await execute_query(
            "SELECT count() AS c FROM source GROUP ALL;", config=live_surrealdb
        )
        return rows[0]["c"] if rows else 0

    async def _verdict_count() -> int:
        rows = await execute_query(
            "SELECT count() AS c FROM source_verdict GROUP ALL;",
            config=live_surrealdb,
        )
        return rows[0]["c"] if rows else 0

    source_before = await _source_count()
    verdict_before = await _verdict_count()
    assert source_before >= 2, "setup: at least the two seeded sources must exist"

    # to-position injection
    assert await repo.relate_verdict(
        a, payload, verdict="contradicts", confidence=0.9
    ) is False
    # from-position injection
    assert await repo.relate_verdict(
        payload, b, verdict="contradicts", confidence=0.9
    ) is False
    # both-positions injection
    assert await repo.relate_verdict(
        payload, payload, verdict="contradicts", confidence=0.9
    ) is False

    # The `source` table is intact (not dropped) and no rogue edge was written.
    assert await _source_count() == source_before, (
        "SECURITY FAIL: the `source` table row count changed — injection executed"
    )
    assert await _verdict_count() == verdict_before, (
        "SECURITY FAIL: a rogue `source_verdict` edge was written from an unsafe id"
    )
