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
    """Every ``.surrealql`` file in ``migrations/`` is recorded as applied.

    Stronger than ``max() >= 43``: builds the expected set from disk by
    scanning the migrations directory, so this test catches "silently skipped
    middle migration" — the failure mode the self-review flagged for the
    runner's ``already exists`` short-circuit. The historical sequence has
    gaps (1..10 + 26..43), so a contiguous-range check would be wrong;
    deriving from the filesystem is the only correct source of truth.
    """
    from surrealdb_service.testing import fixtures as fx

    expected_versions: set[int] = set()
    for p in fx._MIGRATIONS_DIR.iterdir():
        if not (p.is_file() and p.suffix == ".surrealql"):
            continue
        stem = p.stem
        if stem.endswith("_down"):
            continue
        try:
            expected_versions.add(int(stem))
        except ValueError:
            continue

    rows = await execute_query(
        "SELECT * FROM _sbl_migrations ORDER BY version;",
        config=live_surrealdb,
    )
    assert rows, "Expected at least one applied migration"
    applied = {r["version"] for r in rows}
    missing = expected_versions - applied
    assert not missing, (
        f"Migrations on disk but not recorded as applied: {sorted(missing)}. "
        f"The runner may have silently skipped them, or _sbl_migrations is "
        f"out of date."
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
    # ``embedding`` is FLEXIBLE TYPE array with no DEFAULT on the SCHEMAFULL
    # ``entity`` table (migration 39), so callers MUST provide a value — even
    # an empty list. This mirrors what production callers do.
    inserted = await execute_query(
        """
        CREATE entity SET
            canonical_name = $name,
            entity_type = $etype,
            description = $desc,
            confidence = $confidence,
            embedding = [];
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
            entity_type = "Org",
            embedding = [];
        """,
        {"name": canonical},
        config=live_surrealdb,
    )
    parent_id = parent[0]["id"]

    alias_text = _unique("alias-text")
    # Use ``type::thing()`` to parameterize the record-ID rather than f-string
    # interpolating an unsanitized identifier into the query body.
    await execute_query(
        """
        CREATE entity_alias SET
            alias_text = $alias,
            canonical_entity = type::thing($parent_id);
        """,
        {"alias": alias_text, "parent_id": parent_id},
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
        "CREATE entity SET canonical_name = $n, entity_type = 'Person', embedding = [];",
        {"n": a_name},
        config=live_surrealdb,
    )
    b = await execute_query(
        "CREATE entity SET canonical_name = $n, entity_type = 'Org', embedding = [];",
        {"n": b_name},
        config=live_surrealdb,
    )
    a_id = a[0]["id"]
    b_id = b[0]["id"]

    # SurrealQL's RELATE arrow syntax does not accept function-call
    # expressions like ``type::thing($a)`` in the source/target positions —
    # the parser requires bare record-ID literals (see SurrealDB issue #4232).
    # f-string interpolation is safe here because ``a_id`` / ``b_id`` come
    # directly from ``parse_record_ids`` (no user input) and the format
    # ``entity:<rand>`` is colon-and-alnum only. The SELECT below uses
    # parameterized ``type::thing()`` which IS supported in projection
    # position.
    await execute_query(
        f"""
        RELATE {a_id}->relation->{b_id} SET
            relation_type = "WORKS_AT",
            confidence = 0.83;
        """,
        config=live_surrealdb,
    )

    rows = await execute_query(
        """
        SELECT id, relation_type, confidence, status
        FROM relation
        WHERE in = type::thing($a) AND out = type::thing($b);
        """,
        {"a": a_id, "b": b_id},
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
# Persistence-service field alignment (Phase B.1a — flipped from xfail).
# --------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_entity_persistence_field_alignment(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Guard against the legacy ``name`` / ``weight`` / ``source_ids`` drift.

    Phase B.1a routed entity writes through ``EntityRepository.upsert_entity``
    using the migration-39 canonical fields (``canonical_name`` /
    ``source_documents`` / explicit ``embedding=[]``). This test asserts two
    things at the DB boundary:

    1. The canonical write-path succeeds against SCHEMAFULL (no rejection).
    2. The legacy field shape — ``name`` / ``weight`` / ``source_ids`` — IS
       still rejected. If it ever stops being rejected, the schema has
       silently gone SCHEMALESS and we've lost the canary.

    This file lives in ``surrealdb-service`` (no ``app_main`` dependency), so
    we exercise ``EntityRepository.upsert_entity`` directly with the same
    Entity the persistence service now builds. The service-level wiring is
    covered by the unit suite in ``apps/app-main/tests``.

    Previously this lived as ``test_entity_persistence_drift_xfail`` — the
    xfail marker has been removed because B.1a is the phase that fixes it.
    """
    from shared.models import Entity
    from surrealdb_service.repositories.entity import EntityRepository

    repo = EntityRepository(config=live_surrealdb)

    # --- 1. canonical write succeeds -----------------------------------
    canonical = _unique("b1a-flip")
    ent = Entity(
        canonical_name=canonical,
        entity_type="Person",
        confidence=0.9,
        source_documents=["source:b1a-test"],
        properties={"affiliation": "ACME"},
        embedding=[],
    )
    record_id = await repo.upsert_entity(ent)
    assert record_id, "Canonical upsert returned no record id"

    rows = await execute_query(
        "SELECT * FROM entity WHERE canonical_name = $name;",
        {"name": canonical},
        config=live_surrealdb,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["canonical_name"] == canonical
    assert row["entity_type"] == "Person"
    assert row["confidence"] == pytest.approx(0.9)
    assert row["source_documents"] == ["source:b1a-test"]
    assert row["properties"]["affiliation"] == "ACME"
    assert row["status"] == "active"

    # --- 2. legacy shape still rejected --------------------------------
    legacy_name = _unique("legacy-drift")
    with pytest.raises(Exception):  # noqa: B017 — SurrealDB schema error
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
