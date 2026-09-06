"""`upsert_entity` refuses on a database that cannot express identity (PC.3).

Found by running PC.3's code against the config-default database, which sits at
migration 31. Without `name_key` the lookup matches nothing, every upsert falls
through to CREATE, and a re-ingest silently doubles a document's entities.

THE DOUBLE MIRRORS THE REAL SEAM, WHICH IGNORES ITS CONFIG ARGUMENT. The first
version of these tests branched on `config.database`, so the double honoured a
parameter the production `execute_query` discards: `get_pool(config)` builds the
global pool once and ignores the argument afterwards. That made the tests agree
with a check that was itself unsound — it inspected whichever database the pool
was bound to, named a different one in its message, and cached under a third.
Review demonstrated it live. The check now asks `$session` which database the
CONNECTION is on, and these doubles answer that question rather than reading the
config, because that is what the real one does.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from surrealdb_service.repositories import entity as entity_repo


class _Config:
    """A config object the seam is free to ignore — as the real one does."""

    def __init__(self, namespace: str = "ns", database: str = "asked-for") -> None:
        self.namespace = namespace
        self.database = database


class _Pool:
    """What `get_pool` returns. `ConnectionPool.__init__` sets `self.config =
    config or get_config()`, so the pool knows which database it is on — and it
    IGNORES the argument once built, which is the defect these tests exist for."""

    def __init__(self, connected_to: str) -> None:
        self.config = _Config("ns", connected_to)


def _fake_pool(connected_to: str):
    def get_pool(config: Any = None) -> _Pool:
        # Deliberately ignores `config`, exactly like the production pool.
        return _Pool(connected_to)

    return get_pool


def _fake_execute(*, fields: List[str], calls: List[str] | None = None):
    """A double for `execute_query`. It answers only the schema question now.

    The `$session` round trip is gone: review measured it at ~20 ms per upsert
    because it ran before the cache was consulted, and the pool already holds the
    answer in-process. A double that still answered `$session` would be testing a
    seam that no longer exists.
    """
    async def fake(query: str, params: Any = None, config: Any = None):
        if calls is not None:
            calls.append(query)
        if "INFO FOR TABLE entity" in query:
            return {
                "fields": {
                    name: f"DEFINE FIELD {name} ON entity TYPE string"
                    for name in fields
                }
            }
        raise AssertionError(f"unexpected query: {query}")

    return fake


@pytest.fixture(autouse=True)
def _clear_cache():
    entity_repo._IDENTITY_COLUMN_CHECKED.clear()
    yield
    entity_repo._IDENTITY_COLUMN_CHECKED.clear()


@pytest.mark.asyncio
async def test_refuses_when_name_key_is_not_declared(monkeypatch) -> None:
    monkeypatch.setattr(entity_repo, "get_pool", _fake_pool("v31db"))
    monkeypatch.setattr(
        entity_repo, "execute_query",
        _fake_execute(fields=["canonical_name", "status"]),
    )
    with pytest.raises(entity_repo.IdentityColumnMissing) as exc:
        await entity_repo._assert_identity_column(_Config())

    message = str(exc.value)
    assert "ns/v31db" in message, "the message must name the CONNECTED database"
    assert "asked-for" not in message, (
        "the message named the config's database rather than the connection's — "
        "that is the defect this check was rebuilt to remove"
    )
    assert "79" in message and "backfill_name_key" in message


@pytest.mark.asyncio
async def test_passes_when_name_key_is_declared(monkeypatch) -> None:
    monkeypatch.setattr(entity_repo, "get_pool", _fake_pool("migrated"))
    monkeypatch.setattr(
        entity_repo, "execute_query",
        _fake_execute(fields=["canonical_name", "name_key"]),
    )
    await entity_repo._assert_identity_column(_Config())
    assert entity_repo._IDENTITY_COLUMN_CHECKED == {"ns/migrated"}


@pytest.mark.asyncio
async def test_checks_once_per_connection_not_once_per_write(monkeypatch) -> None:
    """A per-write INFO query would be a real cost on a bulk ingest."""
    calls: List[str] = []
    monkeypatch.setattr(entity_repo, "get_pool", _fake_pool("db1"))
    monkeypatch.setattr(
        entity_repo, "execute_query",
        _fake_execute(fields=["name_key"], calls=calls),
    )
    for _ in range(5):
        await entity_repo._assert_identity_column(_Config())
    # One schema inspection for the first call, and NOTHING for the other four.

    assert len(calls) == 1, (
        f"a cached check made {len(calls)} queries, not 0. Counting only the "
        f"`INFO FOR TABLE` ones was the previous assertion, and it SANCTIONED "
        f"the ~20 ms-per-upsert `$session` round trip review measured instead of "
        f"catching it. Any query on the cached path is the cost: {calls}"
    )


@pytest.mark.asyncio
async def test_a_config_cannot_vouch_for_a_database_it_never_reached(monkeypatch) -> None:
    """THE regression test for what review found.

    Two different configs, one pool. The connection is on a database WITHOUT
    `name_key`; the caller names a database that has one. Keyed on the config,
    the check passed and cached a verdict for a database it never inspected.
    Keyed on the connection, it refuses — because that is where the write goes.
    """
    monkeypatch.setattr(entity_repo, "get_pool", _fake_pool("actually_v31"))
    monkeypatch.setattr(
        entity_repo, "execute_query",
        _fake_execute(fields=["canonical_name"]),
    )
    with pytest.raises(entity_repo.IdentityColumnMissing):
        await entity_repo._assert_identity_column(_Config(database="a_migrated_db"))
    assert not entity_repo._IDENTITY_COLUMN_CHECKED, "a refusal must not be cached"


@pytest.mark.asyncio
async def test_one_connection_does_not_vouch_for_another(monkeypatch) -> None:
    """A good database cannot cover a bad one — the cache is per connection."""
    monkeypatch.setattr(entity_repo, "get_pool", _fake_pool("good"))
    monkeypatch.setattr(
        entity_repo, "execute_query",
        _fake_execute(fields=["name_key"]),
    )
    await entity_repo._assert_identity_column(_Config())

    monkeypatch.setattr(entity_repo, "get_pool", _fake_pool("bad"))
    monkeypatch.setattr(
        entity_repo, "execute_query",
        _fake_execute(fields=["canonical_name"]),
    )
    with pytest.raises(entity_repo.IdentityColumnMissing):
        await entity_repo._assert_identity_column(_Config())


@pytest.mark.asyncio
async def test_upsert_entity_actually_calls_the_refusal(monkeypatch) -> None:
    """Nothing proved the writer used it — deleting the call left 356 tests green.

    The check is only worth anything at the seam it guards, so this exercises
    `upsert_entity` rather than the function in isolation. Review found this by
    deleting the call; it now fails without it.
    """
    from shared.models.entity import Entity

    called: List[str] = []

    async def spy(config: Any = None) -> None:
        called.append("checked")
        raise entity_repo.IdentityColumnMissing("sentinel from the spy")

    monkeypatch.setattr(entity_repo, "_assert_identity_column", spy)

    repo = entity_repo.EntityRepository()
    with pytest.raises(entity_repo.IdentityColumnMissing, match="sentinel"):
        await repo.upsert_entity(
            Entity(canonical_name="Probe", entity_type="organization", confidence=0.9)
        )
    assert called == ["checked"], "upsert_entity did not consult the guard"


@pytest.mark.asyncio
async def test_the_refusal_runs_BEFORE_any_write(monkeypatch) -> None:
    """Pinning that the call happens is not enough — it must happen first.

    Review's minor 3: moving the check below the CREATE would keep the previous
    test green while every write still landed on an unmigrated database.
    """
    from shared.models.entity import Entity

    order: List[str] = []

    async def spy(config: Any = None) -> None:
        order.append("check")

    async def fake_query(query: str, params: Any = None, config: Any = None):
        order.append("query")
        raise RuntimeError("stop here — the order is what is under test")

    monkeypatch.setattr(entity_repo, "_assert_identity_column", spy)
    monkeypatch.setattr(entity_repo, "execute_query", fake_query)

    repo = entity_repo.EntityRepository()
    with pytest.raises(Exception):
        await repo.upsert_entity(
            Entity(canonical_name="Probe", entity_type="organization", confidence=0.9)
        )

    assert order and order[0] == "check", (
        f"the first thing upsert_entity did was {order[:1]}, not the identity "
        f"check — a check that runs after the lookup protects nothing"
    )
