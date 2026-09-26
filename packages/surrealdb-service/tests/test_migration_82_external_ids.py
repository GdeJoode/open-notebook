"""Migration 82 in THE state it repairs: `external_ids` drifted to `TYPE object`.

Every test container is built from the migrations, so it has the array and can
never show the defect. This file first puts the field in staging's measured shape
(`FLEXIBLE TYPE object DEFAULT {}`), shows the reconciler's write failing there,
then applies 82 and shows the same write succeed.
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query, execute_transaction
from surrealdb_service.repositories.entity import EntityRepository
from surrealdb_service.testing import fixtures as fx

_STAGING_SHAPE = (
    "DEFINE FIELD OVERWRITE external_ids ON entity FLEXIBLE TYPE object DEFAULT {};"
)


def _migration(version: int, *, down: bool = False) -> str:
    from surrealdb_service.migrations import AsyncMigration

    suffix = "_down" if down else ""
    return AsyncMigration.from_file(fx._MIGRATIONS_DIR / f"{version}{suffix}.surrealql").sql


async def _entity(config) -> str:
    name = f"Gemeente Test {uuid.uuid4().hex[:6]}"
    rows = await execute_query(
        "CREATE entity SET canonical_name = $n, name = $n, entity_type = 'administrative_area', "
        "confidence = 0.5, embedding = [], name_key = $k RETURN id;",
        {"n": name, "k": name.lower()},
        config,
    )
    return str(rows[0]["id"])


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_the_reconciler_write_fails_on_the_drifted_shape_and_succeeds_after_82(
    live_surrealdb: SurrealDBConfig,
) -> None:
    await execute_query(_STAGING_SHAPE, None, live_surrealdb)
    eid = await _entity(live_surrealdb)
    rows = await execute_query("SELECT external_ids FROM type::thing($id)", {"id": eid},
                               live_surrealdb)
    assert rows[0]["external_ids"] == {}  # the staging state

    repo = EntityRepository(live_surrealdb)
    uri = "https://identifier.overheid.nl/tooi/id/gemeente/gm0432"
    # The defect: the write is refused, and the repository says so only by False.
    assert await repo.update_external_ids(eid, [uri]) is False

    await execute_transaction(_migration(82), config=live_surrealdb)
    rows = await execute_query("SELECT external_ids FROM type::thing($id)", {"id": eid},
                               live_surrealdb)
    assert rows[0]["external_ids"] == []
    assert await repo.update_external_ids(eid, [uri]) is True
    rows = await execute_query("SELECT external_ids FROM type::thing($id)", {"id": eid},
                               live_surrealdb)
    assert rows[0]["external_ids"] == [uri]


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_82_leaves_existing_arrays_alone(live_surrealdb: SurrealDBConfig) -> None:
    eid = await _entity(live_surrealdb)
    uri = "https://identifier.overheid.nl/tooi/id/ministerie/mnre1130"
    await execute_query("UPDATE type::thing($id) SET external_ids = [$u]",
                        {"id": eid, "u": uri}, live_surrealdb)
    await execute_transaction(_migration(82), config=live_surrealdb)
    rows = await execute_query("SELECT external_ids FROM type::thing($id)", {"id": eid},
                               live_surrealdb)
    assert rows[0]["external_ids"] == [uri]
