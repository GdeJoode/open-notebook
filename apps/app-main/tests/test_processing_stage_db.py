"""Track PL.2: ``source.processing_stage`` against a live SurrealDB.

Migration 71 adds the strict ``processing_stage`` string (DEFAULT "ingested")
on the SCHEMAFULL ``source`` table. These run against a real SurrealDB
testcontainer (so the actual DEFINE FIELD / strict-field semantics and the S.4
backfill are exercised) and cover:

* the field exists and defaults ``"ingested"`` on a freshly-created source
  (S.4-safe on a FRESH container — AC3);
* ``SourceRepository.set_processing_stage`` advances the stage through the
  pipeline values and is idempotent (re-writing the same stage is fine) — AC3;
* a pre-existing source whose ``processing_stage`` is NONE (the legacy/drift
  case the strict field would otherwise lock) is repaired by the migration-71
  backfill, AND remains writable afterwards (the exact S.4 hazard) — AC3;
* migration 71 down/forward roundtrip removes only the brand-new field and
  restores it; both directions idempotent.

Gated behind ``@requires_docker`` like the other roundtrip suites.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.migrations import AsyncMigration
from surrealdb_service.repositories.source import SourceRepository

pytestmark = pytest.mark.asyncio

_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"


def _u(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _first(result):
    if isinstance(result, list):
        return result[0] if result else {}
    return result or {}


async def _seed_source(config: SurrealDBConfig, title: str) -> str:
    """Create a minimal source row, return its id string."""
    rows = await execute_query(
        "CREATE source SET title=$t RETURN VALUE id;",
        {"t": title},
        config=config,
    )
    return str(rows[0])


# ---------------------------------------------------------------------------
# AC3 — field default on a fresh row (S.4-safe on a FRESH container)
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
async def test_processing_stage_defaults_ingested(
    live_surrealdb: SurrealDBConfig,
) -> None:
    sid = await _seed_source(live_surrealdb, _u("Fresh"))
    rows = await execute_query(
        "SELECT processing_stage FROM source WHERE id = type::thing($id);",
        {"id": sid},
        config=live_surrealdb,
    )
    assert rows and rows[0]["processing_stage"] == "ingested"


# ---------------------------------------------------------------------------
# AC3 — set_processing_stage advances + is idempotent
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
async def test_set_processing_stage_advances_and_is_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = SourceRepository(config=live_surrealdb)
    sid = await _seed_source(live_surrealdb, _u("Advancing"))

    async def _stage() -> str:
        rows = await execute_query(
            "SELECT processing_stage FROM source WHERE id = type::thing($id);",
            {"id": sid},
            config=live_surrealdb,
        )
        return rows[0]["processing_stage"]

    # ingested -> embedded -> extracted, persisted at each step.
    assert await _stage() == "ingested"
    assert await repo.set_processing_stage(sid, "embedded") is True
    assert await _stage() == "embedded"
    assert await repo.set_processing_stage(sid, "extracted") is True
    assert await _stage() == "extracted"

    # Idempotent: re-writing the same stage is a clean no-op-equivalent.
    assert await repo.set_processing_stage(sid, "extracted") is True
    assert await _stage() == "extracted"

    # The gate / failure transitions persist too.
    assert await repo.set_processing_stage(sid, "awaiting_schema_review") is True
    assert await _stage() == "awaiting_schema_review"
    assert await repo.set_processing_stage(sid, "failed") is True
    assert await _stage() == "failed"


# ---------------------------------------------------------------------------
# PL.4 — get_processing_stage reads the persisted stage (the advance_source seam)
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
async def test_get_processing_stage_reads_persisted_value(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = SourceRepository(config=live_surrealdb)
    sid = await _seed_source(live_surrealdb, _u("Readback"))

    # A fresh row defaults ingested (migration 71); the reader returns it.
    assert await repo.get_processing_stage(sid) == "ingested"

    # After a write, the reader returns the new value — this is the round-trip
    # PL.4's advance_source relies on (write stage -> read it -> dispatch next).
    assert await repo.set_processing_stage(sid, "extracted") is True
    assert await repo.get_processing_stage(sid) == "extracted"

    # A missing source reads back None (the driver falls back to ingested).
    assert await repo.get_processing_stage("source:does_not_exist_xyz") is None


# ---------------------------------------------------------------------------
# AC3 — the S.4 hazard: a NONE processing_stage is backfilled AND row stays
# writable. We synthesise the drift by REMOVE-ing the field, creating a row
# (so it has NONE for the field), then re-applying migration 71.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
async def test_migration_71_backfills_none_and_row_stays_writable(
    live_surrealdb: SurrealDBConfig,
) -> None:
    up = AsyncMigration.from_file(_MIGRATIONS_DIR / "71.surrealql")
    down = AsyncMigration.from_file(_MIGRATIONS_DIR / "71_down.surrealql")
    try:
        # Drop the field, then create a row that therefore carries NONE for it
        # (the exact legacy/pre-migration drift the S.4 rule guards against).
        await execute_query(down.sql, config=live_surrealdb)
        sid = await _seed_source(live_surrealdb, _u("Drifted"))
        drifted = await execute_query(
            "SELECT processing_stage FROM source WHERE id = type::thing($id);",
            {"id": sid},
            config=live_surrealdb,
        )
        # Field is undefined -> key absent / NONE.
        assert not drifted[0].get("processing_stage")

        # Re-apply 71: DEFINE FIELD + drift-only backfill. Idempotent.
        await execute_query(up.sql, config=live_surrealdb)
        await execute_query(up.sql, config=live_surrealdb)  # second = no-op

        healed = await execute_query(
            "SELECT processing_stage FROM source WHERE id = type::thing($id);",
            {"id": sid},
            config=live_surrealdb,
        )
        assert healed[0]["processing_stage"] == "ingested", "backfill must repair NONE"

        # The S.4 payoff: a SCHEMAFULL row that had NONE in a strict field is
        # silently unwritable until backfilled. After the backfill, an unrelated
        # UPDATE to the SAME row must succeed.
        await execute_query(
            "UPDATE source SET title=$t WHERE id = type::thing($id);",
            {"id": sid, "t": "rewritten ok"},
            config=live_surrealdb,
        )
        repo = SourceRepository(config=live_surrealdb)
        assert await repo.set_processing_stage(sid, "embedded") is True
    finally:
        # Session-scoped fixture: ALWAYS restore 71 so a mid-test failure can't
        # leave the field dropped for the rest of the suite.
        await execute_query(up.sql, config=live_surrealdb)


# ---------------------------------------------------------------------------
# down / forward roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
async def test_migration_71_down_then_forward_roundtrip(
    live_surrealdb: SurrealDBConfig,
) -> None:
    up = AsyncMigration.from_file(_MIGRATIONS_DIR / "71.surrealql")
    down = AsyncMigration.from_file(_MIGRATIONS_DIR / "71_down.surrealql")

    async def _source_fields() -> dict:
        info = _first(
            await execute_query("INFO FOR TABLE source;", config=live_surrealdb)
        )
        return info.get("fields", {})

    try:
        await execute_query(down.sql, config=live_surrealdb)
        await execute_query(down.sql, config=live_surrealdb)  # idempotent
        assert "processing_stage" not in await _source_fields()

        await execute_query(up.sql, config=live_surrealdb)
        await execute_query(up.sql, config=live_surrealdb)  # idempotent
        assert "processing_stage" in await _source_fields()
    finally:
        await execute_query(up.sql, config=live_surrealdb)
