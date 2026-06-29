"""SECURITY regression: `SourceRepository.relate_cites` SurrealQL injection.

`relate_cites` is the persistence seam for a ``source -> cites -> source`` edge.
Its endpoints must be interpolated into a ``RELATE`` (RELATE graph syntax cannot
bind a ``$param`` in the in/out position — issue #4232), so the RAW id strings
are strict-validated against `_RECORD_ID_RE` via `_validate_record_id` BEFORE any
interpolation. ``str(ensure_record_id(...))`` (= ``RecordID.parse``) splits only
on the first colon and round-trips a payload like
``source:x; REMOVE TABLE source; --`` VERBATIM, so the SDK's parsing is NOT a
validator — the regex is.

This matters because the W.3 MCP ``cite`` tool forwards *agent-supplied,
free-string* ids straight to `relate_cites`, so an attacker-influenceable id can
reach the interpolated ``RELATE``. The fix is the same class as the Track Y.1
(`relate_note`) and Z.1 (`relate_verdict`) injection fixes.

These tests prove, against a real SurrealDB container, that a `;`/`REMOVE TABLE`
payload in BOTH the citing and cited positions is REFUSED and that neither the
`source` table nor the `cites` edge table loses a row — both at the repository
seam AND through the `cite` MCP tool (the actual attack surface):

    uv run --project packages/surrealdb-service \\
        pytest packages/surrealdb-service/tests/test_relate_cites_injection.py -v
"""

from __future__ import annotations

import json
import uuid

import pytest
import surrealdb_service.config as config_module
import surrealdb_service.connection as connection_module
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.mcp.server import create_server
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


# ---------------------------------------------------------------------------
# MCP `cite` tool (the actual attack surface): an agent-supplied injection id is
# refused with a CLEAN error (not a 500) and the tables survive.
# ---------------------------------------------------------------------------


@pytest.fixture
async def mcp_global_db(live_surrealdb: SurrealDBConfig):
    """Point the GLOBAL config/pool at the container for the no-config tool path.

    The MCP tool fns build repos with the default (global) config, so bind
    ``_config`` to the container and reset the pool. Restore prior state on
    teardown.
    """
    prev_config = config_module._config
    prev_pool = connection_module._pool
    config_module._config = live_surrealdb
    connection_module._pool = None
    try:
        yield live_surrealdb
    finally:
        connection_module._pool = prev_pool
        config_module._config = prev_config


def _tool(server, name: str):
    return server._tool_manager.get_tool(name).fn


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_cite_tool_refuses_sql_injection_id(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    server = create_server()
    cite = _tool(server, "cite")

    a = await _make_source(cfg)
    b = await _make_source(cfg)

    source_before = await _source_count(cfg)
    cites_before = await _cites_count(cfg)
    assert source_before >= 2

    # Each call must return a clean JSON refusal, never raise / 500. The id never
    # passes `_RECORD_ID_RE`, so the chokepoint (`relate_cites`) — or the tool's
    # own `ensure_record_id` guard for the parse-failure case — rejects it.
    for citing_id, cited_id in (
        (a, _PAYLOAD),
        (_PAYLOAD, b),
        (_PAYLOAD, _PAYLOAD),
    ):
        res = json.loads(
            await cite(citing_source_id=citing_id, cited_source_id=cited_id)
        )
        assert res["created"] is False, (
            f"injection id ({citing_id!r}->{cited_id!r}) must not create an edge"
        )

    assert await _source_count(cfg) == source_before, (
        "SECURITY FAIL: the `source` table row count changed via the cite tool"
    )
    assert await _cites_count(cfg) == cites_before, (
        "SECURITY FAIL: a rogue `cites` edge was written via the cite tool"
    )
