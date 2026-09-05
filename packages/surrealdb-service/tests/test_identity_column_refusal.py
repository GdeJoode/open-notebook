"""`upsert_entity` refuses on a database that cannot express identity (PC.3).

Found by running PC.3's code against the config-default database, which sits at
migration 31. Without `name_key` the lookup matches nothing, every upsert falls
through to CREATE, and a re-ingest silently doubles a document's entities. Between
migrations 39 and 79 the old `idx_entity_name_type` turns that into a confusing
index error; below 39 nothing complains at all — and migration 79 removes that
index, which is correct and also removes the accident that was catching it.

The double returns the shape `INFO FOR TABLE` really returns — a dict with a
`fields` mapping keyed by field name — because a double that returns a
convenient shape tests the double.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from surrealdb_service.repositories import entity as entity_repo


class _Config:
    def __init__(self, namespace: str, database: str) -> None:
        self.namespace = namespace
        self.database = database


def _info(fields: List[str]) -> Dict[str, Any]:
    """What SurrealDB returns for `INFO FOR TABLE entity`."""
    return {
        "fields": {
            name: f"DEFINE FIELD {name} ON entity TYPE string PERMISSIONS FULL"
            for name in fields
        }
    }


@pytest.fixture(autouse=True)
def _clear_cache():
    entity_repo._IDENTITY_COLUMN_CHECKED.clear()
    yield
    entity_repo._IDENTITY_COLUMN_CHECKED.clear()


@pytest.mark.asyncio
async def test_refuses_when_name_key_is_not_declared(monkeypatch) -> None:
    async def fake(query, params=None, config=None):
        assert "INFO FOR TABLE entity" in query
        return _info(["canonical_name", "entity_type", "status"])

    monkeypatch.setattr(entity_repo, "execute_query", fake)
    with pytest.raises(entity_repo.IdentityColumnMissing) as exc:
        await entity_repo._assert_identity_column(_Config("open_notebook", "v31db"))

    message = str(exc.value)
    assert "open_notebook/v31db" in message, "the message must name the database"
    assert "79" in message, "the message must name the migration that supplies it"
    assert "backfill_name_key" in message, "and the step migration 79 refuses without"


@pytest.mark.asyncio
async def test_passes_when_name_key_is_declared(monkeypatch) -> None:
    async def fake(query, params=None, config=None):
        return _info(["canonical_name", "entity_type", "status", "name_key"])

    monkeypatch.setattr(entity_repo, "execute_query", fake)
    await entity_repo._assert_identity_column(_Config("ns", "migrated"))


@pytest.mark.asyncio
async def test_checks_once_per_database_not_once_per_write(monkeypatch) -> None:
    """A per-write INFO query would be a real cost on a bulk ingest."""
    calls: List[str] = []

    async def fake(query, params=None, config=None):
        calls.append(query)
        return _info(["name_key"])

    monkeypatch.setattr(entity_repo, "execute_query", fake)
    cfg = _Config("ns", "db1")
    for _ in range(5):
        await entity_repo._assert_identity_column(cfg)
    assert len(calls) == 1, f"checked {len(calls)} times for one database"

    # A DIFFERENT database is a different question and must be asked again.
    await entity_repo._assert_identity_column(_Config("ns", "db2"))
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_a_passing_check_does_not_mask_a_later_failing_one(monkeypatch) -> None:
    """The cache is keyed by database, so one good database cannot vouch for another.

    This is the mutation that matters: cache on a bare boolean and the second
    database inherits the first one's verdict.
    """
    async def fake(query, params=None, config=None):
        return _info(["name_key"]) if config.database == "good" else _info(["id"])

    monkeypatch.setattr(entity_repo, "execute_query", fake)
    await entity_repo._assert_identity_column(_Config("ns", "good"))
    with pytest.raises(entity_repo.IdentityColumnMissing):
        await entity_repo._assert_identity_column(_Config("ns", "bad"))
