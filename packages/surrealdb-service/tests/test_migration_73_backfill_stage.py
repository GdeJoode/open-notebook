"""Container tests for migration 73 — legacy `source.processing_stage` backfill.

Migration 71 set every pre-existing source to `"ingested"`; migration 73 derives
the TRUE stage from each source's actual outputs (mentions edges => complete,
entities => extracted, aggregate embedding => embedded) WITHOUT re-running the
pipeline, and only for rows still at `"ingested"` (never clobbering a row the
live auto-chain already advanced).

The `live_surrealdb` fixture applies every migration — including 73 — at session
setup, so merely booting it proves 73 PARSES and executes on the fully-migrated
schema. These tests additionally prove the derivation, idempotency, and the
no-clobber guard against a real SurrealDB v2 container by seeding known states
and applying the exact production migration string.

Run:

    uv run --project packages/surrealdb-service \\
        pytest packages/surrealdb-service/tests/test_migration_73_backfill_stage.py -v
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _migration_73_sql() -> str:
    """The exact production migration-73 string (comment-stripped, one query)."""
    from surrealdb_service.migrations import AsyncMigration
    from surrealdb_service.testing import fixtures as fx

    return AsyncMigration.from_file(fx._MIGRATIONS_DIR / "73.surrealql").sql


async def _make_source(
    config: SurrealDBConfig,
    *,
    stage: str = "ingested",
    embedding: list[float] | None = None,
) -> str:
    if embedding is None:
        rows = await execute_query(
            "CREATE source SET title = $t, processing_stage = $s;",
            {"t": _unique("src"), "s": stage},
            config=config,
        )
    else:
        rows = await execute_query(
            "CREATE source SET title = $t, processing_stage = $s, "
            "embedding = $e;",
            {"t": _unique("src"), "s": stage, "e": embedding},
            config=config,
        )
    return str(rows[0]["id"])


async def _make_entity(
    config: SurrealDBConfig,
    *,
    source_id: str | None = None,
    status: str = "active",
) -> str:
    rows = await execute_query(
        "CREATE entity SET canonical_name = $n, entity_type = 'programme', "
        "embedding = [], status = $st, source_documents = $sd;",
        {
            "n": _unique("ent"),
            "st": status,
            "sd": [source_id] if source_id else [],
        },
        config=config,
    )
    return str(rows[0]["id"])


async def _relate_mentions(config: SurrealDBConfig, src: str, ent: str) -> None:
    await execute_query(
        f"RELATE {src}->mentions->{ent} SET weight = 1.0;",
        config=config,
    )


async def _stage(config: SurrealDBConfig, source_id: str) -> str | None:
    rows = await execute_query(
        "SELECT VALUE processing_stage FROM type::thing($id) LIMIT 1;",
        {"id": source_id},
        config=config,
    )
    return rows[0] if rows else None


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_73_recorded(live_surrealdb: SurrealDBConfig) -> None:
    """The runner discovered and applied migration 73 (proves it parses)."""
    rows = await execute_query(
        "SELECT version FROM _sbl_migrations WHERE version = 73;",
        config=live_surrealdb,
    )
    assert rows, "Migration 73 not recorded — runner did not discover/apply it"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_73_derives_and_is_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Each tier derives the right stage; a second apply changes nothing."""
    # Tier 1 — mentions membership => complete.
    s_complete = await _make_source(live_surrealdb)
    ent_c = await _make_entity(live_surrealdb)
    await _relate_mentions(live_surrealdb, s_complete, ent_c)

    # Tier 2 — an active entity references it (no mentions) => extracted.
    s_extracted = await _make_source(live_surrealdb)
    await _make_entity(live_surrealdb, source_id=s_extracted, status="active")

    # Tier 3 — aggregate embedding, nothing else => embedded.
    s_embedded = await _make_source(live_surrealdb, embedding=[0.1, 0.2, 0.3])

    # Tier 4 — parsed only => stays ingested.
    s_ingested = await _make_source(live_surrealdb)

    # An entity that only references the source via an ARCHIVED row must NOT
    # count (mirrors the read handler's active/reference filter).
    s_archived_only = await _make_source(live_surrealdb)
    await _make_entity(
        live_surrealdb, source_id=s_archived_only, status="archived"
    )

    sql = _migration_73_sql()
    await execute_query(sql, config=live_surrealdb)

    assert await _stage(live_surrealdb, s_complete) == "complete"
    assert await _stage(live_surrealdb, s_extracted) == "extracted"
    assert await _stage(live_surrealdb, s_embedded) == "embedded"
    assert await _stage(live_surrealdb, s_ingested) == "ingested"
    assert await _stage(live_surrealdb, s_archived_only) == "ingested"

    # Idempotent: re-applying derives nothing new.
    await execute_query(sql, config=live_surrealdb)
    assert await _stage(live_surrealdb, s_complete) == "complete"
    assert await _stage(live_surrealdb, s_extracted) == "extracted"
    assert await _stage(live_surrealdb, s_embedded) == "embedded"
    assert await _stage(live_surrealdb, s_ingested) == "ingested"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_73_never_clobbers_advanced_rows(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A row already advanced past `ingested` is left untouched.

    Seed a source at `graphed` that ALSO has mentions edges (which would derive
    `complete` if it were `ingested`). The migration must leave it `graphed`
    because it only ever rewrites rows currently at `ingested`.
    """
    s = await _make_source(live_surrealdb, stage="graphed")
    ent = await _make_entity(live_surrealdb)
    await _relate_mentions(live_surrealdb, s, ent)

    await execute_query(_migration_73_sql(), config=live_surrealdb)

    assert await _stage(live_surrealdb, s) == "graphed"
