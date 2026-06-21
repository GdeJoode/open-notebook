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


# --------------------------------------------------------------------------
# Migration 52 — layered privacy mode (Track J.3)
# --------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_source_private_default_and_override(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """``source.private`` defaults to False (migration 52) and accepts True.

    A source created without ``private`` reads back ``False`` (the SCHEMAFULL
    DEFAULT), and one created with ``private = true`` persists ``True`` — the
    per-document sticky override the J.3 privacy resolver reads.
    """
    # Default path: omit ``private`` -> DEFAULT false.
    default_title = _unique("src-priv-default")
    await execute_query(
        "CREATE source SET title = $title, full_text = 'x';",
        {"title": default_title},
        config=live_surrealdb,
    )
    rows = await execute_query(
        "SELECT title, private FROM source WHERE title = $title;",
        {"title": default_title},
        config=live_surrealdb,
    )
    assert len(rows) == 1
    assert rows[0]["private"] is False

    # Override path: private = true persists.
    private_title = _unique("src-priv-true")
    await execute_query(
        "CREATE source SET title = $title, full_text = 'x', private = true;",
        {"title": private_title},
        config=live_surrealdb,
    )
    rows = await execute_query(
        "SELECT title, private FROM source WHERE title = $title;",
        {"title": private_title},
        config=live_surrealdb,
    )
    assert len(rows) == 1
    assert rows[0]["private"] is True


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_notebook_privacy_mode_optional(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """``notebook.privacy_mode`` is option<string>: NONE by default, settable."""
    inherit_name = _unique("nb-inherit")
    await execute_query(
        "CREATE notebook SET name = $name;",
        {"name": inherit_name},
        config=live_surrealdb,
    )
    rows = await execute_query(
        "SELECT name, privacy_mode FROM notebook WHERE name = $name;",
        {"name": inherit_name},
        config=live_surrealdb,
    )
    assert len(rows) == 1
    # option<string> with no value reads back as None (inherit global).
    assert rows[0].get("privacy_mode") is None

    private_name = _unique("nb-private")
    await execute_query(
        "CREATE notebook SET name = $name, privacy_mode = 'private';",
        {"name": private_name},
        config=live_surrealdb,
    )
    rows = await execute_query(
        "SELECT name, privacy_mode FROM notebook WHERE name = $name;",
        {"name": private_name},
        config=live_surrealdb,
    )
    assert len(rows) == 1
    assert rows[0]["privacy_mode"] == "private"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_settings_default_privacy_mode_seeded_cloud(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The settings singleton's ``default_privacy_mode`` defaults to 'cloud'.

    Migration 52 declares the field with DEFAULT 'cloud' and backfills the
    singleton, so a write to the singleton reads it back as 'cloud' (Track J's
    "cloud by default" — a privacy-first operator flips this one setting).
    """
    # The migration's backfill UPDATE only touches an EXISTING singleton; on a
    # fresh container the record may be absent, so UPSERT it first. The DEFINE
    # FIELD ... DEFAULT 'cloud' applies on the write because migration 52 defined
    # the field on the open_notebook table.
    await execute_query(
        "UPSERT open_notebook:content_settings SET parser_engine = 'docling';",
        config=live_surrealdb,
    )
    rows = await execute_query(
        "SELECT default_privacy_mode FROM open_notebook:content_settings;",
        config=live_surrealdb,
    )
    assert rows, "settings singleton missing"
    assert rows[0]["default_privacy_mode"] == "cloud"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_52_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Re-applying migration 52 is a no-op (all DEFINEs use IF NOT EXISTS).

    Replays the forward migration body against the already-migrated DB and
    asserts it does not raise and leaves the fields intact.
    """
    from surrealdb_service.testing import fixtures as fx

    migration_sql = (fx._MIGRATIONS_DIR / "52.surrealql").read_text()
    # Should not raise on replay.
    await execute_query(migration_sql, config=live_surrealdb)

    # Fields still usable post-replay.
    title = _unique("src-replay")
    await execute_query(
        "CREATE source SET title = $title, full_text = 'x';",
        {"title": title},
        config=live_surrealdb,
    )
    rows = await execute_query(
        "SELECT private FROM source WHERE title = $title;",
        {"title": title},
        config=live_surrealdb,
    )
    assert rows[0]["private"] is False


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_default_models_update_without_privacy_field(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A sibling ``open_notebook`` record updates without default_privacy_mode.

    Regression for the migration 52 bug fixed by migration 53: migration 52
    declared ``default_privacy_mode`` as a REQUIRED ``string`` on the whole
    ``open_notebook`` table (intending only the content_settings singleton), so
    the sibling ``open_notebook:default_models`` record — which never carries the
    field — failed every UPDATE with "Found NONE for field default_privacy_mode,
    expected a string". Migration 53 relaxes it to ``option<string>``. This test
    creates a default_models record WITHOUT the field and updates it; pre-53 this
    raised, post-53 it succeeds.
    """
    await execute_query(
        "UPSERT open_notebook:default_models SET default_chat_model = 'model:x';",
        config=live_surrealdb,
    )
    # Clear the field so this record carries NONE — the exact state the
    # required-string definition rejected on the next write.
    await execute_query(
        "UPDATE open_notebook:default_models UNSET default_privacy_mode;",
        config=live_surrealdb,
    )
    # The update that the bug rejected ("Found NONE for field
    # default_privacy_mode, expected a string") — must not raise now.
    await execute_query(
        "UPDATE open_notebook:default_models SET default_chat_model = 'model:y';",
        config=live_surrealdb,
    )
    rows = await execute_query(
        "SELECT default_chat_model, default_privacy_mode "
        "FROM open_notebook:default_models;",
        config=live_surrealdb,
    )
    assert rows[0]["default_chat_model"] == "model:y"
    # option<string> allows this sibling record to carry NONE (the global
    # default lives on content_settings, not here).
    assert rows[0].get("default_privacy_mode") is None
