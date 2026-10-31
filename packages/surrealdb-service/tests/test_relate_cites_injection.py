"""SECURITY regression: `SourceRepository.relate_cites` SurrealQL injection.

`relate_cites` is the persistence seam for a ``source -> cites -> source`` edge.
Its endpoints must be interpolated into a ``RELATE`` (RELATE graph syntax cannot
bind a ``$param`` in the in/out position — issue #4232), so the RAW id strings
are strict-validated against `_RECORD_ID_RE` via `_validate_record_id` BEFORE any
interpolation. ``str(ensure_record_id(...))`` (= ``RecordID.parse``) splits only
on the first colon and round-trips a payload like
``source:x; REMOVE TABLE source; --`` VERBATIM, so the SDK's parsing is NOT a
validator — the regex is.

The W.3 MCP ``cite`` tool that forwarded agent-supplied ids here was removed in
F2-02 (ADR 0002 D); the repository seam keeps the guard. The fix is the same
class as the Track Y.1 (`relate_note`) and Z.1 (`relate_verdict`) injection fixes.

This test proves, against a real SurrealDB container, that a `;`/`REMOVE TABLE`
payload in BOTH the citing and cited positions is REFUSED and that neither the
`source` table nor the `cites` edge table loses a row:

    uv run --project packages/surrealdb-service \\
        pytest packages/surrealdb-service/tests/test_relate_cites_injection.py -v
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.source import SourceRepository

# A multi-statement drop-table payload. If interpolated,
# ``RELATE source:x; REMOVE TABLE source; --->cites->...`` executes as several
# statements and wipes the `source` table (the reproduced blocker).
_PAYLOAD = "source:x; REMOVE TABLE source; --"


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


async def _make_source(config: SurrealDBConfig) -> str:
    rows = await execute_query(
        "CREATE source SET title = $t RETURN AFTER;",
        {"t": _unique("source")},
        config=config,
    )
    return str(rows[0]["id"])


async def _source_count(config: SurrealDBConfig) -> int:
    rows = await execute_query(
        "SELECT count() AS c FROM source GROUP ALL;", config=config
    )
    return rows[0]["c"] if rows else 0


async def _cites_count(config: SurrealDBConfig) -> int:
    rows = await execute_query(
        "SELECT count() AS c FROM cites GROUP ALL;", config=config
    )
    return rows[0]["c"] if rows else 0


# ---------------------------------------------------------------------------
# Repository seam: relate_cites refuses an injection id in BOTH positions.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relate_cites_refuses_sql_injection_id(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = SourceRepository(config=live_surrealdb)
    a = await _make_source(live_surrealdb)
    b = await _make_source(live_surrealdb)

    source_before = await _source_count(live_surrealdb)
    cites_before = await _cites_count(live_surrealdb)
    assert source_before >= 2, "setup: at least the two seeded sources must exist"

    # cited-position injection
    assert await repo.relate_cites(a, _PAYLOAD, confidence=1.0) is False
    # citing-position injection
    assert await repo.relate_cites(_PAYLOAD, b, confidence=1.0) is False
    # both-positions injection
    assert await repo.relate_cites(_PAYLOAD, _PAYLOAD, confidence=1.0) is False

    assert await _source_count(live_surrealdb) == source_before, (
        "SECURITY FAIL: the `source` table row count changed — injection executed"
    )
    assert await _cites_count(live_surrealdb) == cites_before, (
        "SECURITY FAIL: a rogue `cites` edge was written from an unsafe id"
    )
