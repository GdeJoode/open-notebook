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


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_54_aliases_roundtrip_and_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Migration 54 (K.3): ``aliases`` reads back ``[]``; replay is a no-op.

    The K.3 retroactive merge denormalizes loser surface forms onto the
    winner's ``aliases`` array (migration 54). A fresh entity must read back
    ``aliases=[]`` (safe default), and re-applying the migration body — all
    ``IF NOT EXISTS`` — must not raise and must leave the field usable. Asserts
    the B.8 caution holds: ``status`` defaults to 'active' and the additive
    field does not disturb the canonical write-path.
    """
    from surrealdb_service.testing import fixtures as fx

    name = _unique("alias-rt")
    await execute_query(
        "CREATE entity SET canonical_name = $n, name = $n, "
        "entity_type = 'organization', embedding = [];",
        {"n": name},
        config=live_surrealdb,
    )
    rows = await execute_query(
        "SELECT aliases, status FROM entity WHERE canonical_name = $n;",
        {"n": name},
        config=live_surrealdb,
    )
    assert rows[0]["aliases"] == []
    assert rows[0]["status"] == "active"

    # Replay the forward migration — IF NOT EXISTS makes this a no-op.
    migration_sql = (fx._MIGRATIONS_DIR / "54.surrealql").read_text()
    await execute_query(migration_sql, config=live_surrealdb)

    # The field still writes after replay.
    await execute_query(
        "UPDATE entity SET aliases = ['x', 'y'] WHERE canonical_name = $n;",
        {"n": name},
        config=live_surrealdb,
    )
    rows2 = await execute_query(
        "SELECT aliases FROM entity WHERE canonical_name = $n;",
        {"n": name},
        config=live_surrealdb,
    )
    assert sorted(rows2[0]["aliases"]) == ["x", "y"]


# --------------------------------------------------------------------------
# Migration 57 — re-key reference_entity uniqueness on external_id (K-D2 rev2)
# --------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_57_distinct_same_named_orgs_coexist(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Migration 57: two DISTINCT codes sharing a canonical_name both persist.

    The blocker the K-D2 review caught: under migration 41's
    ``(canonical_name, source_vocabulary)`` UNIQUE index, the two "Bergen"
    gemeenten (gm0373 / gm0893) collapsed into one row at ingest. Migration 57
    re-keys uniqueness on ``(external_id, source_vocabulary)`` so both survive;
    a lookup by the shared name returns BOTH (the ambiguity the reconciler needs).
    """
    from surrealdb_service.repositories.reference_entity import (
        ReferenceEntityRepository,
    )

    repo = ReferenceEntityRepository(config=live_surrealdb)
    name = _unique("Bergen")
    src = _unique("tooi").replace("-", "")

    def _row(code: str, soort_suffix: str) -> dict:
        return {
            "canonical_name": name,
            "entity_type": "organization",
            "source_vocabulary": src,
            "external_uri": f"https://identifier.overheid.nl/tooi/id/gemeente/{code}",
            "external_id": code,
            "aliases": [f"Bergen ({soort_suffix})"],
            "properties": {"soort": "gemeente"},
        }

    code_a = _unique("gm").replace("-", "")
    code_b = _unique("gm").replace("-", "")
    loaded = await repo.bulk_load([_row(code_a, "NH"), _row(code_b, "L")])
    assert loaded == 2  # both persisted (count not collapsed)

    rows = await repo.lookup_by_name(name, source=src)
    assert len(rows) == 2  # distinct orgs coexist under the shared name
    assert {r["external_id"] for r in rows} == {code_a, code_b}


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_57_idempotent_upsert_on_external_id(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Re-loading the same external_id (different name) UPDATEs the one row.

    The upsert key is now ``external_id``: a second load of the same code with a
    refreshed display name updates in place (no duplicate), and a re-apply of
    migration 57's body (OVERWRITE) does not raise.
    """
    from surrealdb_service.repositories.reference_entity import (
        ReferenceEntityRepository,
    )
    from surrealdb_service.testing import fixtures as fx

    repo = ReferenceEntityRepository(config=live_surrealdb)
    src = _unique("tooi").replace("-", "")
    code = _unique("mnre").replace("-", "")

    base = {
        "canonical_name": _unique("Original Name"),
        "entity_type": "organization",
        "source_vocabulary": src,
        "external_uri": f"https://identifier.overheid.nl/tooi/id/ministerie/{code}",
        "external_id": code,
        "aliases": ["ABC"],
        "properties": {},
    }
    await repo.bulk_load([base])
    renamed = dict(base, canonical_name=_unique("Renamed"))
    await repo.bulk_load([renamed])

    rows = await execute_query(
        "SELECT canonical_name FROM reference_entity "
        "WHERE external_id = $eid AND source_vocabulary = $src;",
        {"eid": code, "src": src},
        config=live_surrealdb,
    )
    assert len(rows) == 1  # one row, updated in place (keyed on external_id)
    assert rows[0]["canonical_name"] == renamed["canonical_name"]

    # Re-applying the forward migration body (OVERWRITE) must not raise.
    migration_sql = (fx._MIGRATIONS_DIR / "57.surrealql").read_text()
    await execute_query(migration_sql, config=live_surrealdb)


# --------------------------------------------------------------------------
# Migration 63 — source-level aggregate embedding (Track R.0)
# --------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_63_source_embedding_field(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """``source.embedding`` is option<array<float>>: NONE by default, settable.

    Migration 63 declares the source-level aggregate vector. A source created
    without it reads back NONE (the honest "not yet computed" state), and one
    written with a float array persists it — the source<->source cosine input
    for R.1. The dimension is not schema-pinned (any length accepted).
    """
    # Default path: omit ``embedding`` -> NONE.
    default_title = _unique("src-embed-default")
    await execute_query(
        "CREATE source SET title = $title, full_text = 'x';",
        {"title": default_title},
        config=live_surrealdb,
    )
    rows = await execute_query(
        "SELECT title, embedding FROM source WHERE title = $title;",
        {"title": default_title},
        config=live_surrealdb,
    )
    assert len(rows) == 1
    assert rows[0].get("embedding") is None

    # Set path: a float vector persists with its dimension intact.
    vec_title = _unique("src-embed-set")
    await execute_query(
        "CREATE source SET title = $title, full_text = 'x';",
        {"title": vec_title},
        config=live_surrealdb,
    )
    vector = [float(i) / 10.0 for i in range(1024)]
    await execute_query(
        "UPDATE source SET embedding = $embedding WHERE title = $title;",
        {"title": vec_title, "embedding": vector},
        config=live_surrealdb,
    )
    rows = await execute_query(
        "SELECT embedding FROM source WHERE title = $title;",
        {"title": vec_title},
        config=live_surrealdb,
    )
    assert len(rows) == 1
    assert rows[0]["embedding"] is not None
    assert len(rows[0]["embedding"]) == 1024
    assert rows[0]["embedding"][0] == pytest.approx(0.0)


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_63_idempotent(live_surrealdb: SurrealDBConfig) -> None:
    """Re-applying migration 63 is a no-op (IF NOT EXISTS) and leaves it usable."""
    from surrealdb_service.testing import fixtures as fx

    migration_sql = (fx._MIGRATIONS_DIR / "63.surrealql").read_text()
    await execute_query(migration_sql, config=live_surrealdb)

    title = _unique("src-embed-replay")
    await execute_query(
        "CREATE source SET title = $title, full_text = 'x';",
        {"title": title},
        config=live_surrealdb,
    )
    await execute_query(
        "UPDATE source SET embedding = $embedding WHERE title = $title;",
        {"title": title, "embedding": [0.1, 0.2, 0.3]},
        config=live_surrealdb,
    )
    rows = await execute_query(
        "SELECT embedding FROM source WHERE title = $title;",
        {"title": title},
        config=live_surrealdb,
    )
    assert rows[0]["embedding"] == pytest.approx([0.1, 0.2, 0.3])

    # Clearing to NONE is allowed (option<...>) — the "no chunk vectors" path.
    await execute_query(
        "UPDATE source SET embedding = NONE WHERE title = $title;",
        {"title": title},
        config=live_surrealdb,
    )
    rows = await execute_query(
        "SELECT embedding FROM source WHERE title = $title;",
        {"title": title},
        config=live_surrealdb,
    )
    assert rows[0].get("embedding") is None


# --------------------------------------------------------------------------
# Migration 64 — backfill NONE strict-typed source fields (Track R.0e)
# --------------------------------------------------------------------------
#
# Legacy staging rows predate migration 52 (`private TYPE bool DEFAULT false`),
# so they carry `private = NONE`. A SCHEMAFULL UPDATE re-validates the whole
# record, so that one NONE blocks ANY write to the row — including the R.0
# source-aggregate embedding UPDATE. Migration 64 coalesces the drifted strict
# fields (`private`, plus `topics` for data hygiene) back to their defaults,
# exactly like migration 61 did for `entity`.


async def _force_legacy_source_drift(
    config: SurrealDBConfig, title: str
) -> str:
    """Reproduce a pre-migration-52 row: `private` set to NONE on a SCHEMAFULL row.

    A normal CREATE applies the `DEFAULT false`, and a plain `UPDATE ... UNSET
    private` is itself rejected by the strict revalidation — so neither path
    reaches the legacy state. The real staging rows reached `private = NONE`
    because they were created BEFORE migration 52 declared the field. We recreate
    exactly that history: drop the field definition, UNSET to NONE (now allowed),
    then re-apply migration 52's DEFINE (which, like SurrealDB DEFAULTs, does NOT
    rewrite existing rows). The result is a SCHEMAFULL `source` row carrying
    `private = NONE` — the precise drift migration 64 repairs. Returns the id.
    """
    created = await execute_query(
        "CREATE source SET title = $t, full_text = 'x';",
        {"t": title},
        config=config,
    )
    rid = created[0]["id"] if isinstance(created, list) else created["id"]
    rid = str(rid)

    # Temporarily remove the strict definition so NONE becomes assignable...
    await execute_query("REMOVE FIELD private ON source;", config=config)
    await execute_query("REMOVE FIELD topics ON source;", config=config)
    await execute_query(f"UPDATE {rid} UNSET private, topics;", config=config)
    # ...then restore the exact migration-1 / migration-52 definitions. A bare
    # DEFINE FIELD does not backfill existing rows, so {rid} keeps private=NONE.
    await execute_query(
        "DEFINE FIELD private ON TABLE source TYPE bool DEFAULT false;",
        config=config,
    )
    await execute_query(
        "DEFINE FIELD topics ON TABLE source TYPE option<array<string>>;",
        config=config,
    )

    rows = await execute_query(
        f"SELECT private, topics FROM {rid};",
        config=config,
    )
    assert rows[0].get("private") is None, "setup: private should be NONE pre-64"
    assert rows[0].get("topics") is None, "setup: topics should be NONE pre-64"
    return rid


async def _apply_migration_64(config: SurrealDBConfig) -> None:
    from surrealdb_service.testing import fixtures as fx

    migration_sql = (fx._MIGRATIONS_DIR / "64.surrealql").read_text()
    await execute_query(migration_sql, config=config)


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_64_backfills_drifted_strict_fields(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Criterion 1: NONE strict fields hold their defaults after migration 64.

    A legacy source with `private = NONE` (and `topics = NONE`) reads back
    `private = false` / `topics = []` once the backfill runs.
    """
    title = _unique("src-64-backfill")
    rid = await _force_legacy_source_drift(live_surrealdb, title)

    await _apply_migration_64(live_surrealdb)

    rows = await execute_query(
        f"SELECT private, topics FROM {rid};",
        config=live_surrealdb,
    )
    assert rows[0]["private"] is False
    assert rows[0]["topics"] == []


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_64_unblocks_aggregate_embedding_update(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Criterion 2 (load-bearing): the embedding UPDATE fails pre-64, passes post-64.

    Reproduces the exact staging blocker: a source whose `private` is NONE
    rejects the R.0 aggregate-embedding write with "Found NONE for field
    private". Migration 64 coalesces `private` to false, after which the same
    UPDATE succeeds.
    """
    title = _unique("src-64-embed-unblock")
    rid = await _force_legacy_source_drift(live_surrealdb, title)

    vector = [float(i) / 10.0 for i in range(8)]

    # --- PRE-64: the aggregate-embedding UPDATE is rejected ---------------
    with pytest.raises(Exception) as excinfo:  # noqa: B017 — schema error
        await execute_query(
            f"UPDATE {rid} SET embedding = $embedding;",
            {"embedding": vector},
            config=live_surrealdb,
        )
    # The error names the drifted strict field that blocks the whole-record
    # revalidation — the precise staging symptom.
    assert "private" in str(excinfo.value)

    # --- migration 64 repairs the drift ----------------------------------
    await _apply_migration_64(live_surrealdb)

    # --- POST-64: the SAME UPDATE now succeeds ---------------------------
    await execute_query(
        f"UPDATE {rid} SET embedding = $embedding;",
        {"embedding": vector},
        config=live_surrealdb,
    )
    rows = await execute_query(
        f"SELECT private, embedding FROM {rid};",
        config=live_surrealdb,
    )
    assert rows[0]["private"] is False
    assert rows[0]["embedding"] is not None
    assert len(rows[0]["embedding"]) == 8


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_64_idempotent(live_surrealdb: SurrealDBConfig) -> None:
    """Criterion 3: applying 64 twice is a no-op; a clean row is unchanged.

    Re-running the backfill against an already-repaired DB must not raise, and a
    source that was never drifted (CREATE applied the DEFAULT) keeps its values.
    """
    # A clean row — DEFAULT already applied private=false; topics omitted (NONE,
    # which is valid for option<array<string>>).
    clean_title = _unique("src-64-clean")
    await execute_query(
        "CREATE source SET title = $t, full_text = 'x', private = true, "
        "topics = ['ai', 'db'];",
        {"t": clean_title},
        config=live_surrealdb,
    )

    # First application.
    await _apply_migration_64(live_surrealdb)
    # Second application — must not raise.
    await _apply_migration_64(live_surrealdb)

    rows = await execute_query(
        "SELECT private, topics FROM source WHERE title = $t;",
        {"t": clean_title},
        config=live_surrealdb,
    )
    # `?? default` is a no-op on a non-NONE value: private stays true, topics
    # keeps its real list — the backfill never clobbers real data.
    assert rows[0]["private"] is True
    assert sorted(rows[0]["topics"]) == ["ai", "db"]

    # And the row is still writable after the double-apply.
    await execute_query(
        "UPDATE source SET embedding = [0.1, 0.2] WHERE title = $t;",
        {"t": clean_title},
        config=live_surrealdb,
    )
    rows = await execute_query(
        "SELECT embedding FROM source WHERE title = $t;",
        {"t": clean_title},
        config=live_surrealdb,
    )
    assert rows[0]["embedding"] == pytest.approx([0.1, 0.2])
