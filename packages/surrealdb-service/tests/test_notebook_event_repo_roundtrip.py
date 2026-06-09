"""Roundtrip tests for the Phase B.3b ``notebook_event`` table + repository.

Boots the testcontainers SurrealDB harness, applies every migration up to
and including #46 (which lands ``notebook_event`` + ``notebook_schema.excluded_types``)
and exercises the :class:`NotebookEventRepository` end-to-end.

Acceptance criteria mapped to test cases:

* AC#6 (each schema-edit op writes exactly one notebook_event row) is the
  reason this harness exists — the service-layer test in
  ``apps/app-main/tests/test_schema_edit_service.py`` asserts on the count,
  but the underlying CREATE + index + payload roundtrip is exercised here.
* AC migration applies + idempotent — ``test_migration_46_recorded`` +
  ``test_migration_46_is_idempotent``.
* AC notebook_schema.excluded_types accepts an array of strings —
  ``test_notebook_schema_excluded_types_roundtrip``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from shared.models import NotebookEvent, NotebookSchema
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.notebook_event import NotebookEventRepository
from surrealdb_service.repositories.notebook_schema import NotebookSchemaRepository


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


async def _create_notebook(config: SurrealDBConfig) -> str:
    name = _unique("nb")
    rows = await execute_query(
        "CREATE notebook SET name = $name, description = 'b3b roundtrip';",
        {"name": name},
        config=config,
    )
    assert rows, "Expected CREATE notebook to return a row"
    return str(rows[0]["id"])


# ---------------------------------------------------------------------------
# Smoke: migration 46 applied + idempotent
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_46_recorded(live_surrealdb: SurrealDBConfig) -> None:
    """Migration 46 is recorded in ``_sbl_migrations``.

    Without this, every other test in this file would skip silently
    because the table wouldn't exist.
    """
    rows = await execute_query(
        "SELECT version FROM _sbl_migrations WHERE version = 46;",
        config=live_surrealdb,
    )
    assert rows, "Migration 46 not recorded — harness did not apply it"
    assert rows[0]["version"] == 46


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_46_is_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Re-executing migration 46 DDL is a no-op.

    Every DEFINE uses ``IF NOT EXISTS`` so replaying the statements
    against a populated DB must succeed without errors and must not
    produce a duplicate migration record.
    """
    from pathlib import Path

    here = Path(__file__).resolve()
    migration_path = None
    for ancestor in here.parents:
        candidate = ancestor / "migrations" / "46.surrealql"
        if candidate.is_file():
            migration_path = candidate
            break
    assert migration_path is not None, "Could not locate migrations/46.surrealql"

    body = migration_path.read_text(encoding="utf-8")
    cleaned = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith("--")
    )
    await execute_query(cleaned, config=live_surrealdb)

    rows = await execute_query(
        "SELECT count() AS n FROM _sbl_migrations "
        "WHERE version = 46 GROUP ALL;",
        config=live_surrealdb,
    )
    assert rows and rows[0]["n"] == 1


# ---------------------------------------------------------------------------
# NotebookEventRepository round-trip
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_record_persists_event(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """record() lands one row with the right notebook + event_type + payload."""
    notebook_id = await _create_notebook(live_surrealdb)
    repo = NotebookEventRepository(config=live_surrealdb)

    event_id = await repo.record(
        notebook_id,
        "schema_changed",
        {"op": "rename", "old_name": "Researcher", "new_name": "ResearchFellow"},
    )
    assert event_id.startswith("notebook_event:")

    rows = await execute_query(
        "SELECT event_type, payload, notebook, created_at, read_at "
        "FROM notebook_event WHERE notebook = $nb;",
        {"nb": (await _record_id(notebook_id))},
        config=live_surrealdb,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["event_type"] == "schema_changed"
    assert row["payload"]["op"] == "rename"
    assert row["payload"]["old_name"] == "Researcher"
    assert row["read_at"] is None
    assert row["created_at"] is not None


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_list_unread_filters_by_event_type(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """list_unread(event_type=X) returns only events of that type, newest first."""
    notebook_id = await _create_notebook(live_surrealdb)
    repo = NotebookEventRepository(config=live_surrealdb)

    await repo.record(notebook_id, "schema_changed", {"op": "rename"})
    await repo.record(notebook_id, "schema_changed", {"op": "merge"})
    await repo.record(notebook_id, "extension_suggested", {"type_name": "Cohort"})

    schema_events = await repo.list_unread(notebook_id, "schema_changed")
    assert len(schema_events) == 2
    # Newest first — merge inserted after rename.
    assert schema_events[0].payload["op"] == "merge"
    assert schema_events[1].payload["op"] == "rename"

    ext_events = await repo.list_unread(notebook_id, "extension_suggested")
    assert len(ext_events) == 1
    assert ext_events[0].payload["type_name"] == "Cohort"

    all_events = await repo.list_unread(notebook_id)
    assert len(all_events) == 3


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_mark_read_excludes_from_list_unread(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """After mark_read, the event drops out of list_unread but stays in list_by_notebook."""
    notebook_id = await _create_notebook(live_surrealdb)
    repo = NotebookEventRepository(config=live_surrealdb)

    event_id = await repo.record(
        notebook_id, "schema_changed", {"op": "delete"}
    )

    assert len(await repo.list_unread(notebook_id)) == 1
    assert await repo.mark_read(event_id) is True
    assert len(await repo.list_unread(notebook_id)) == 0

    # Still visible via list_by_notebook (history is preserved).
    history = await repo.list_by_notebook(notebook_id)
    assert len(history) == 1
    assert history[0].read_at is not None

    # mark_read is idempotent — second call still returns True.
    assert await repo.mark_read(event_id) is True


# ---------------------------------------------------------------------------
# notebook_schema.excluded_types roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_notebook_schema_excluded_types_roundtrip(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """upsert + get_by_notebook preserves the new excluded_types list.

    Validates that migration 46 added the field correctly and that the
    Pydantic mirror in shared.models.notebook_schema.NotebookSchema
    serialises + deserialises it cleanly.
    """
    notebook_id = await _create_notebook(live_surrealdb)
    repo = NotebookSchemaRepository(config=live_surrealdb)

    await repo.upsert(
        NotebookSchema(
            notebook=notebook_id,
            base_ontology="scholarly",
            excluded_types=["Methodology", "DeprecatedThing"],
        )
    )

    fetched = await repo.get_by_notebook(notebook_id)
    assert fetched is not None
    assert fetched.excluded_types == ["Methodology", "DeprecatedThing"]


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_notebook_schema_excluded_types_default_empty_list(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A schema row written without excluded_types reads back as ``[]``.

    Defends the validator on the Pydantic model that coerces a NONE
    column (legacy rows) to an empty list.
    """
    notebook_id = await _create_notebook(live_surrealdb)
    repo = NotebookSchemaRepository(config=live_surrealdb)
    await repo.upsert(
        NotebookSchema(notebook=notebook_id, base_ontology="scholarly")
    )

    fetched = await repo.get_by_notebook(notebook_id)
    assert fetched is not None
    assert fetched.excluded_types == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _record_id(notebook_id: str):
    """Convert a stringified notebook record id back into a RecordID for SQL.

    The CREATE queries above use ``CREATE notebook SET ...`` which
    returns the id as a stringified form via parse_record_ids. Some
    follow-up assertions need to bind it back into a SurrealQL query as
    a record-id literal — type::thing(stringified) is the canonical way.
    """
    from surrealdb_service.connection import ensure_record_id

    return ensure_record_id(notebook_id)
