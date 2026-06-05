"""Canary roundtrip tests for SCHEMAFULL migrations (Phase B.0).

These tests boot a real SurrealDB container via testcontainers, apply every
migration in ``migrations/``, and then INSERT + SELECT against each
SCHEMAFULL table that lives at the top of the schema. They catch field-name
drift, missing migrations, and SCHEMAFULL/SCHEMALESS regressions at
author-time — exactly the gap Track A's RETRO flagged (lesson #1).

To run locally:

    cd packages/surrealdb-service
    uv run pytest -m requires_docker tests/test_migrations_roundtrip.py

CI: the workflow at ``.github/workflows/db-integration.yml`` runs this file
on every PR. Tests are gated behind ``@pytest.mark.requires_docker`` so they
skip cleanly on machines without Docker.
"""

from __future__ import annotations

import uuid

import pytest

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query


def _unique(prefix: str) -> str:
    """Generate a collision-free identifier for test rows."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Smoke: container booted, migrations applied, version > 0
# --------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migrations_applied(live_surrealdb: SurrealDBConfig) -> None:
    """All migrations applied → ``_sbl_migrations`` has rows up to 43."""
    rows = await execute_query(
        "SELECT * FROM _sbl_migrations ORDER BY version;",
        config=live_surrealdb,
    )
    assert rows, "Expected at least one applied migration"
    versions = [r["version"] for r in rows]
    # We don't pin the exact set (migrations get added over time), but the
    # baseline 1-43 must all be present.
    assert max(versions) >= 43, (
        f"Highest applied migration is {max(versions)}; expected >= 43. "
        f"Has a new migration broken the runner?"
    )


# --------------------------------------------------------------------------
# Per-table roundtrip canaries (SCHEMAFULL tables from migration 39)
# --------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_entity_roundtrip(live_surrealdb: SurrealDBConfig) -> None:
    """Insert an entity using migration-39 field names, read it back.

    This is the *correct* shape — the field-name-drifted variant lives in
    ``test_entity_persistence_drift_xfail`` below.
    """
    canonical = _unique("entity-roundtrip")
    inserted = await execute_query(
        """
        CREATE entity SET
            canonical_name = $name,
            entity_type = $etype,
            description = $desc,
            confidence = $confidence;
        """,
        {
            "name": canonical,
            "etype": "Person",
            "desc": "Round-trip canary",
            "confidence": 0.91,
        },
        config=live_surrealdb,
    )
    assert inserted, "CREATE entity returned no rows"

    rows = await execute_query(
        "SELECT * FROM entity WHERE canonical_name = $name;",
        {"name": canonical},
        config=live_surrealdb,
    )
    assert len(rows) == 1, f"Expected exactly 1 row, got {len(rows)}"
    row = rows[0]
    assert row["canonical_name"] == canonical
    assert row["entity_type"] == "Person"
    assert row["description"] == "Round-trip canary"
    # Default-typed fields should be populated by SurrealDB
    assert row["status"] == "active"
    assert row["extraction_method"] == "llm"
    assert row["confidence"] == pytest.approx(0.91)


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_entity_alias_roundtrip(live_surrealdb: SurrealDBConfig) -> None:
    """Insert an alias linked to an entity, read it back."""
    canonical = _unique("alias-target")
    parent = await execute_query(
        """
        CREATE entity SET
            canonical_name = $name,
            entity_type = "Org";
        """,
        {"name": canonical},
        config=live_surrealdb,
    )
    parent_id = parent[0]["id"]

    alias_text = _unique("alias-text")
    await execute_query(
        f"""
        CREATE entity_alias SET
            alias_text = $alias,
            canonical_entity = {parent_id};
        """,
        {"alias": alias_text},
        config=live_surrealdb,
    )

    rows = await execute_query(
        "SELECT * FROM entity_alias WHERE alias_text = $alias;",
        {"alias": alias_text},
        config=live_surrealdb,
    )
    assert len(rows) == 1
    assert rows[0]["alias_text"] == alias_text
    assert rows[0]["canonical_entity"] == parent_id


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relation_roundtrip(live_surrealdb: SurrealDBConfig) -> None:
    """RELATE two entities, verify the typed edge persists with metadata."""
    a_name = _unique("rel-src")
    b_name = _unique("rel-dst")

    a = await execute_query(
        "CREATE entity SET canonical_name = $n, entity_type = 'Person';",
        {"n": a_name},
        config=live_surrealdb,
    )
    b = await execute_query(
        "CREATE entity SET canonical_name = $n, entity_type = 'Org';",
        {"n": b_name},
        config=live_surrealdb,
    )
    a_id = a[0]["id"]
    b_id = b[0]["id"]

    await execute_query(
        f"""
        RELATE {a_id}->relation->{b_id} SET
            relation_type = "WORKS_AT",
            confidence = 0.83;
        """,
        config=live_surrealdb,
    )

    rows = await execute_query(
        f"""
        SELECT id, relation_type, confidence, status
        FROM relation
        WHERE in = {a_id} AND out = {b_id};
        """,
        config=live_surrealdb,
    )
    assert len(rows) == 1, f"Expected 1 RELATE row, got {len(rows)}"
    assert rows[0]["relation_type"] == "WORKS_AT"
    assert rows[0]["confidence"] == pytest.approx(0.83)
    assert rows[0]["status"] == "active"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_source_roundtrip(live_surrealdb: SurrealDBConfig) -> None:
    """Source roundtrip including the migration-43 metadata bag."""
    title = _unique("source-title")
    created = await execute_query(
        """
        CREATE source SET
            title = $title,
            full_text = "lorem ipsum",
            metadata = { parser_engine: "mineru", confidence: 0.7 };
        """,
        {"title": title},
        config=live_surrealdb,
    )
    assert created
    rows = await execute_query(
        "SELECT title, full_text, metadata FROM source WHERE title = $title;",
        {"title": title},
        config=live_surrealdb,
    )
    assert len(rows) == 1
    assert rows[0]["title"] == title
    assert rows[0]["metadata"]["parser_engine"] == "mineru"
    assert rows[0]["metadata"]["confidence"] == pytest.approx(0.7)


# --------------------------------------------------------------------------
# XFAIL: entity-persistence drift — to be fixed in Phase B.1a
# --------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Pre-existing drift between SCHEMAFULL `entity` table (migration 39) "
        "and apps/app-main/src/app_main/services/entity_persistence_service.py "
        "lines 132-156, which writes `name`, `weight`, `source_ids` (legacy "
        "SCHEMALESS shape) instead of the canonical `canonical_name`, "
        "`source_documents`, and *no* weight field. Track B.1a will fix this "
        "by routing writes through EntityRepository.upsert_entity(). When that "
        "phase lands, flip this xfail to a passing assertion (or delete the "
        "test) and update docs/tracks/B-kg-quality/status.md."
    ),
)
async def test_entity_persistence_drift_xfail(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Mirror the legacy persistence-service write shape.

    Expected to fail until B.1a aligns field names. Once B.1a lands, this test
    should be deleted (or rewritten to assert the canonical shape via the
    repository).

    Pointer for the fixer: ``entity_persistence_service.persist_filtered_result``
    in ``apps/app-main/src/app_main/services/entity_persistence_service.py``,
    lines 132-156.
    """
    legacy_name = _unique("drift")
    # SCHEMAFULL `entity` declares `canonical_name`, NOT `name`. SurrealDB will
    # reject the CREATE with a schema error → xfail.
    await execute_query(
        """
        CREATE entity SET
            name = $name,
            entity_type = "Person",
            weight = 1,
            confidence = 0.9,
            source_ids = ["source:abc"],
            properties = {};
        """,
        {"name": legacy_name},
        config=live_surrealdb,
    )
    # If we ever get here without an error, B.1a fixed the bug — promote
    # to a real assertion at that point.
    rows = await execute_query(
        "SELECT * FROM entity WHERE name = $name;",
        {"name": legacy_name},
        config=live_surrealdb,
    )
    assert rows  # would fail today even if the CREATE silently no-ops
