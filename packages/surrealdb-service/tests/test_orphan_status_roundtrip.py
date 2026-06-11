"""Roundtrip tests for migration 48 -- orphan-prune lifecycle fields (B.5b).

Boots the testcontainers SurrealDB harness (``live_surrealdb`` fixture from
``surrealdb_service.testing``), applies every migration up to and including
#48, and exercises the four lifecycle fields end-to-end via the
:class:`EntityRepository` writes.

Acceptance criteria mapped to test cases:

* Migration 48 applied + recorded -- ``test_migration_48_recorded``.
* Migration 48 idempotent -- ``test_migration_48_is_idempotent``.
* New fields roundtrip cleanly -- ``test_orphan_lifecycle_fields_roundtrip``.
* ``update_orphan_status`` flips the status field --
  ``test_update_orphan_status_writes_status``.
* ``update_orphan_status(increment_attempts=True)`` bumps the counter --
  ``test_update_orphan_status_increments_attempts``.
* Lifecycle transitions stamp the expected timestamps --
  ``test_update_orphan_status_stamps_timestamps``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytest

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.entity import EntityRepository


def _locate_migration(version: int) -> Path:
    """Walk up from this test file until we find migrations/{version}.surrealql."""
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "migrations" / f"{version}.surrealql"
        if candidate.is_file():
            return candidate
    raise AssertionError(f"Could not locate migrations/{version}.surrealql")


# ---------------------------------------------------------------------------
# Smoke: migration 48 applied + idempotent
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_48_recorded(live_surrealdb: SurrealDBConfig) -> None:
    """Migration 48 is recorded in ``_sbl_migrations``.

    If the harness didn't apply it, every later test in this module
    would skip silently because the orphan-status fields wouldn't
    exist on the ``entity`` table.
    """
    rows = await execute_query(
        "SELECT version FROM _sbl_migrations WHERE version = 48;",
        config=live_surrealdb,
    )
    assert rows, "Migration 48 not recorded -- harness did not apply it"
    assert rows[0]["version"] == 48


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_48_is_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Re-executing migration 48 DDL is a no-op.

    Mirrors the migration-45/47 roundtrip pattern: every DEFINE uses
    ``IF NOT EXISTS`` so replaying the statements against a populated
    DB must succeed without errors and must not produce a duplicate
    migration record.
    """
    migration_path = _locate_migration(48)
    body = migration_path.read_text(encoding="utf-8")
    cleaned = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith("--")
    )
    await execute_query(cleaned, config=live_surrealdb)

    rows = await execute_query(
        "SELECT count() AS n FROM _sbl_migrations "
        "WHERE version = 48 GROUP ALL;",
        config=live_surrealdb,
    )
    assert rows and rows[0]["n"] == 1


# ---------------------------------------------------------------------------
# Roundtrip helpers
# ---------------------------------------------------------------------------


async def _create_test_entity(
    config: SurrealDBConfig,
    canonical_name: str,
    *,
    entity_type: str = "PERSON",
    source_documents: list[str] | None = None,
) -> str:
    """Create one entity row directly and return its record id.

    Bypasses ``upsert_entity`` (which does a SELECT-then-CREATE merge)
    because the roundtrip tests just need a sentinel row to mutate.
    """
    payload: Dict[str, Any] = {
        "canonical_name": canonical_name,
        "entity_type": entity_type,
        "source_documents": list(source_documents or ["source:roundtrip"]),
        "extraction_method": "test",
        "confidence": 1.0,
        "provenance_chain": [],
        "properties": {},
        "embedding": [],
        "status": "active",
        "type_tags": [],
        "primary_type": entity_type,
    }
    result = await execute_query(
        """
        CREATE entity SET
            canonical_name = $canonical_name,
            entity_type = $entity_type,
            description = NONE,
            source_documents = $source_documents,
            extraction_method = $extraction_method,
            confidence = $confidence,
            provenance_chain = $provenance_chain,
            properties = $properties,
            embedding = $embedding,
            status = $status,
            type_tags = $type_tags,
            primary_type = $primary_type;
        """,
        payload,
        config=config,
    )
    assert result, "CREATE entity returned no rows"
    return str(result[0]["id"])


async def _read_orphan_fields(
    config: SurrealDBConfig, entity_id: str
) -> Dict[str, Any]:
    """Read the lifecycle fields back via a record-link SELECT.

    SurrealDB binds ``$id`` as a *plain string*; comparing a RecordID
    column against a string with ``=`` returns no rows. Coerce the
    stringified id via ``type::thing($id)`` so the comparison happens
    in RecordID space -- mirrors the working pattern in
    ``test_notebook_schema_repo_roundtrip`` and the repository's own
    ``update_orphan_status`` query.
    """
    rows = await execute_query(
        "SELECT id, orphan_status, reconnect_attempts, "
        "first_orphaned_at, last_reconnect_attempt_at "
        "FROM entity WHERE id = type::thing($id) LIMIT 1",
        {"id": entity_id},
        config=config,
    )
    assert rows, f"entity {entity_id} not found after roundtrip"
    return rows[0]


# ---------------------------------------------------------------------------
# Schema-level roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_orphan_lifecycle_fields_roundtrip(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The four new fields are queryable + carry the documented defaults."""
    eid = await _create_test_entity(
        live_surrealdb, "RoundtripFields"
    )
    row = await _read_orphan_fields(live_surrealdb, eid)
    # ``option<string>`` fields default to NONE; the int DEFAULT 0.
    assert row["orphan_status"] is None
    assert row["reconnect_attempts"] == 0
    assert row["first_orphaned_at"] is None
    assert row["last_reconnect_attempt_at"] is None


# ---------------------------------------------------------------------------
# update_orphan_status end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_update_orphan_status_writes_status(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The status field accepts the documented values and round-trips."""
    eid = await _create_test_entity(live_surrealdb, "StatusWrite")
    repo = EntityRepository(config=live_surrealdb)

    ok = await repo.update_orphan_status(eid, "pending_reconnect")
    assert ok

    row = await _read_orphan_fields(live_surrealdb, eid)
    assert row["orphan_status"] == "pending_reconnect"

    # Subsequent transition to archived.
    ok = await repo.update_orphan_status(eid, "archived")
    assert ok
    row = await _read_orphan_fields(live_surrealdb, eid)
    assert row["orphan_status"] == "archived"

    # And clearing back to NONE.
    ok = await repo.update_orphan_status(eid, None)
    assert ok
    row = await _read_orphan_fields(live_surrealdb, eid)
    assert row["orphan_status"] is None


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_update_orphan_status_increments_attempts(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """``increment_attempts=True`` bumps the counter monotonically."""
    eid = await _create_test_entity(live_surrealdb, "AttemptsBump")
    repo = EntityRepository(config=live_surrealdb)

    # Three consecutive attempts -> reconnect_attempts == 3.
    for expected in (1, 2, 3):
        await repo.update_orphan_status(
            eid, "pending_reconnect", increment_attempts=True
        )
        row = await _read_orphan_fields(live_surrealdb, eid)
        assert row["reconnect_attempts"] == expected


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_update_orphan_status_stamps_timestamps(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """``set_first_orphaned_at`` / ``set_last_reconnect_attempt_at`` stamp the rows."""
    eid = await _create_test_entity(live_surrealdb, "TimestampStamp")
    repo = EntityRepository(config=live_surrealdb)

    await repo.update_orphan_status(
        eid,
        "pending_reconnect",
        set_first_orphaned_at=True,
        set_last_reconnect_attempt_at=True,
    )
    row = await _read_orphan_fields(live_surrealdb, eid)
    assert isinstance(row["first_orphaned_at"], datetime)
    assert isinstance(row["last_reconnect_attempt_at"], datetime)
    first_at = row["first_orphaned_at"]

    # Second call must NOT overwrite first_orphaned_at -- it is the
    # "FIRST time orphaned" marker and the IF-NONE guard preserves it.
    await repo.update_orphan_status(
        eid,
        "pending_reconnect",
        set_first_orphaned_at=True,
        set_last_reconnect_attempt_at=True,
    )
    row = await _read_orphan_fields(live_surrealdb, eid)
    assert row["first_orphaned_at"] == first_at


# ---------------------------------------------------------------------------
# list_orphans_with_status end-to-end (B.5b attempt 2: B2 fix)
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_list_orphans_with_status_traverses_reference_edge(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The dashboard query finds entities via ``reference`` edges.

    Reproduces B.5b attempt-1 blocker B2: the previous query targeted
    ``source.notebook`` which is not a real column (source -> notebook
    is the ``reference`` RELATE edge, see migration 1 line 54). The
    fixed query walks ``reference`` and stringifies the source ids so
    they match the strings stored in ``entity.source_documents``.

    Setup:
      1. Create a notebook + a source + a ``reference`` edge.
      2. Create an entity whose ``source_documents`` carries that
         source's stringified id.
      3. Mark the entity ``pending_reconnect`` via the repository.
      4. Assert the dashboard query surfaces the entity for that status
         and does NOT surface it for the ``archived`` status.
    """
    # 1. Notebook + source + RELATE edge.
    nb_rows = await execute_query(
        "CREATE notebook SET name = 'orphan-dash-test';",
        config=live_surrealdb,
    )
    assert nb_rows
    notebook_id = str(nb_rows[0]["id"])

    src_rows = await execute_query(
        "CREATE source SET title = 'orphan-dash-src';",
        config=live_surrealdb,
    )
    assert src_rows
    source_id = str(src_rows[0]["id"])

    # SurrealDB rejects ``type::thing()`` inside the RELATE source/target
    # positions; bind RecordIDs directly. Mirrors the production pattern
    # in ``NotebookService.add_source``.
    from surrealdb_service.connection import ensure_record_id

    await execute_query(
        "RELATE $src->reference->$nb;",
        {
            "src": ensure_record_id(source_id),
            "nb": ensure_record_id(notebook_id),
        },
        config=live_surrealdb,
    )

    # 2. Entity carrying the source id in source_documents.
    eid = await _create_test_entity(
        live_surrealdb,
        "OrphanForDashboard",
        source_documents=[source_id],
    )

    repo = EntityRepository(config=live_surrealdb)

    # 3. Flip to pending_reconnect via the repo + stamp first_orphaned_at
    # so the dashboard's ORDER BY clause has a value to sort on (defensive
    # -- without a non-NONE timestamp some SurrealDB versions return zero
    # rows on the ORDER BY clause).
    ok = await repo.update_orphan_status(
        eid,
        "pending_reconnect",
        increment_attempts=True,
        set_first_orphaned_at=True,
        set_last_reconnect_attempt_at=True,
    )
    assert ok

    # 4a. The dashboard surfaces the entity under pending_reconnect.
    pending = await repo.list_orphans_with_status(
        notebook_id, "pending_reconnect"
    )
    pending_ids = {str(row.get("id")) for row in pending}
    assert eid in pending_ids, (
        f"Expected {eid!r} in pending list; got {pending_ids!r}"
    )

    # 4b. The dashboard does NOT surface it under archived.
    archived = await repo.list_orphans_with_status(notebook_id, "archived")
    archived_ids = {str(row.get("id")) for row in archived}
    assert eid not in archived_ids

    # 4c. status=None returns any-non-NONE -- entity should appear.
    any_status = await repo.list_orphans_with_status(notebook_id, None)
    any_ids = {str(row.get("id")) for row in any_status}
    assert eid in any_ids


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_list_orphans_with_status_empty_when_no_sources(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A notebook with no ``reference`` edges returns an empty list.

    Guards against a regression where the query accidentally surfaces
    orphan entities from a different notebook because the source-id
    filter is no-op.
    """
    nb_rows = await execute_query(
        "CREATE notebook SET name = 'empty-orphan-dash';",
        config=live_surrealdb,
    )
    assert nb_rows
    notebook_id = str(nb_rows[0]["id"])

    repo = EntityRepository(config=live_surrealdb)
    pending = await repo.list_orphans_with_status(
        notebook_id, "pending_reconnect"
    )
    assert pending == []
