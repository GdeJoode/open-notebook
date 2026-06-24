"""Integration tests for migration 59 (Track Q.3) against a live SurrealDB.

Migration 59 adds the ``manual_override`` bool column on ``entity`` and the
``status_change_log`` table. These tests run against a real SurrealDB
testcontainer (so the actual DEFINE FIELD / DEFINE TABLE semantics are
exercised) and cover:

* AC6 — the migration is applied by the session fixture (the container is
  migrated to the latest version, which now includes 59); ``manual_override``
  defaults false on a freshly-created row, and ``entity.status`` accepts
  ``"reference"`` (free string, no ASSERT rejection).
* down/forward roundtrip — applying 59_down removes ONLY the Q.3 objects and
  leaves the pre-existing status field + index intact; re-applying 59 restores
  them. Both directions are idempotent (IF [NOT] EXISTS).
* status_change_log CRUD via the thin repository — append + list_for_entity +
  list_for_batch.
* set_status honours the override guard at the DB layer (no-op when pinned).

Gated behind ``@requires_docker`` like the other roundtrip suites.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from app_main.services.triage.status_change_log import StatusChangeLogRepository
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.migrations import AsyncMigration
from surrealdb_service.repositories.entity import EntityRepository

pytestmark = pytest.mark.asyncio

_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"


def _u(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _first(result):
    """Normalise an execute_query result to a single object.

    ``INFO FOR ...`` statements yield one object; depending on the driver path
    ``execute_query`` may return it bare or wrapped in a one-element list.
    """
    if isinstance(result, list):
        return result[0] if result else {}
    return result or {}


async def _db_tables(config: SurrealDBConfig) -> dict:
    info = _first(await execute_query("INFO FOR DB;", config=config))
    return info.get("tables", {})


async def _table_fields_indexes(config: SurrealDBConfig, table: str):
    info = _first(await execute_query(f"INFO FOR TABLE {table};", config=config))
    return info.get("fields", {}), info.get("indexes", {})


async def _seed_entity(
    config: SurrealDBConfig, name: str, *, status: str = "active"
) -> str:
    """Create a minimal entity, return its id string."""
    rows = await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, entity_type='org', "
        "embedding=[], confidence=0.9, source_documents=[], status=$s "
        "RETURN VALUE id;",
        {"n": name, "s": status},
        config=config,
    )
    return str(rows[0])


# ---------------------------------------------------------------------------
# AC6 — column default + reference status accepted
# ---------------------------------------------------------------------------

@pytest.mark.requires_docker
async def test_manual_override_defaults_false_on_new_row(
    live_surrealdb: SurrealDBConfig,
) -> None:
    eid = await _seed_entity(live_surrealdb, _u("Mover"))
    rows = await execute_query(
        "SELECT manual_override FROM entity WHERE id = type::thing($id);",
        {"id": eid},
        config=live_surrealdb,
    )
    assert rows and rows[0]["manual_override"] is False


@pytest.mark.requires_docker
async def test_entity_status_accepts_reference(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """`entity.status` is a free string (migration 39, no ASSERT)."""
    eid = await _seed_entity(live_surrealdb, _u("Ref"), status="reference")
    rows = await execute_query(
        "SELECT status FROM entity WHERE id = type::thing($id);",
        {"id": eid},
        config=live_surrealdb,
    )
    assert rows and rows[0]["status"] == "reference"


# ---------------------------------------------------------------------------
# down / forward roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.requires_docker
async def test_migration_59_down_then_forward_roundtrip(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Down removes only the Q.3 objects; forward restores them; status stays."""
    up = AsyncMigration.from_file(_MIGRATIONS_DIR / "59.surrealql")
    down = AsyncMigration.from_file(_MIGRATIONS_DIR / "59_down.surrealql")

    try:
        # Down: drop manual_override + status_change_log.
        await execute_query(down.sql, config=live_surrealdb)

        tables = await _db_tables(live_surrealdb)
        assert "status_change_log" not in tables, "down should drop the log table"
        # The pre-existing status field + index must survive the down-migration.
        ent_fields, ent_indexes = await _table_fields_indexes(
            live_surrealdb, "entity"
        )
        assert "status" in ent_fields, (
            "down must NOT drop the migration-39 status field"
        )
        assert "idx_entity_status" in ent_indexes, (
            "down must NOT drop idx_entity_status"
        )
        assert "manual_override" not in ent_fields, "down must drop manual_override"

        # Forward: re-apply 59, restoring both objects. Idempotent.
        await execute_query(up.sql, config=live_surrealdb)
        await execute_query(up.sql, config=live_surrealdb)  # second time = no-op

        tables2 = await _db_tables(live_surrealdb)
        assert "status_change_log" in tables2, (
            "forward should restore the log table"
        )
        ent_fields2, _ = await _table_fields_indexes(live_surrealdb, "entity")
        assert "manual_override" in ent_fields2

        # Down is idempotent too.
        await execute_query(down.sql, config=live_surrealdb)
        await execute_query(down.sql, config=live_surrealdb)
    finally:
        # The fixture is session-scoped; ALWAYS restore 59 so a mid-test failure
        # can't leave the schema dropped for the rest of the suite.
        await execute_query(up.sql, config=live_surrealdb)


# ---------------------------------------------------------------------------
# status_change_log CRUD via the thin repo
# ---------------------------------------------------------------------------

@pytest.mark.requires_docker
async def test_status_change_log_append_and_list(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = StatusChangeLogRepository(config=live_surrealdb)
    eid = await _seed_entity(live_surrealdb, _u("Logged"))
    batch = _u("batch")

    row_id = await repo.append(
        eid, old="reference", new="active", reason="rule X", batch_id=batch
    )
    assert row_id.startswith("status_change_log:")

    by_entity = await repo.list_for_entity(eid)
    assert len(by_entity) == 1
    rec = by_entity[0]
    assert rec["old_status"] == "reference"
    assert rec["new_status"] == "active"
    assert rec["reason"] == "rule X"
    assert rec["batch_id"] == batch
    assert rec.get("changed_at") is not None, (
        f"changed_at missing/empty; row keys={sorted(rec)}"
    )  # schema DEFAULT time::now()
    assert str(rec["entity"]) == eid

    by_batch = await repo.list_for_batch(batch)
    assert len(by_batch) == 1
    assert str(by_batch[0]["entity"]) == eid

    # A second append for the same entity is a distinct row (append-only).
    await repo.append(
        eid, old="active", new="reference", reason="rule Y", batch_id=batch
    )
    assert len(await repo.list_for_entity(eid)) == 2
    assert len(await repo.list_for_batch(batch)) == 2


# ---------------------------------------------------------------------------
# set_status honours the override guard at the DB layer
# ---------------------------------------------------------------------------

@pytest.mark.requires_docker
async def test_set_status_writes_when_not_overridden(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = EntityRepository(config=live_surrealdb)
    eid = await _seed_entity(live_surrealdb, _u("Free"), status="active")

    wrote = await repo.set_status(eid, "reference")
    assert wrote is True

    rows = await execute_query(
        "SELECT status FROM entity WHERE id = type::thing($id);",
        {"id": eid},
        config=live_surrealdb,
    )
    assert rows[0]["status"] == "reference"


@pytest.mark.requires_docker
async def test_set_status_is_noop_when_overridden(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = EntityRepository(config=live_surrealdb)
    eid = await _seed_entity(live_surrealdb, _u("Pinned"), status="reference")
    # Operator pins the entity.
    await execute_query(
        "UPDATE entity SET manual_override = true WHERE id = type::thing($id);",
        {"id": eid},
        config=live_surrealdb,
    )
    pinned = await execute_query(
        "SELECT manual_override FROM entity WHERE id = type::thing($id);",
        {"id": eid},
        config=live_surrealdb,
    )
    assert pinned[0]["manual_override"] is True, "pin must persist before set_status"

    wrote = await repo.set_status(eid, "active", respect_override=True)
    assert wrote is False, "automation must not overwrite a pinned entity"

    rows = await execute_query(
        "SELECT status FROM entity WHERE id = type::thing($id);",
        {"id": eid},
        config=live_surrealdb,
    )
    assert rows[0]["status"] == "reference", "pinned status preserved"

    # The operator path can still force it (respect_override=False).
    forced = await repo.set_status(eid, "active", respect_override=False)
    assert forced is True
    rows2 = await execute_query(
        "SELECT status FROM entity WHERE id = type::thing($id);",
        {"id": eid},
        config=live_surrealdb,
    )
    assert rows2[0]["status"] == "active"
