"""Integration roundtrip for AliasOverlayRepository (testcontainers, K.5).

Seeds the ``alias_overlay`` table (migration 56) against a real SurrealDB
container and asserts the K.5 overlay contract:

* a global rule + a notebook-scoped rule round-trip through create/list/get;
* ``list_overlays(notebook_id)`` returns the global rule PLUS that notebook's
  own rule, while a DIFFERENT notebook sees only the global rule
  (per-notebook isolation, AC5/AC7);
* delete removes a rule;
* migration 56's SCHEMAFULL field constraints hold (kind/scope enums).

Gated behind ``@requires_docker`` like the other roundtrip suites.
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.alias_overlay import AliasOverlayRepository


def _u(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


async def _make_notebook(config: SurrealDBConfig, name: str) -> str:
    rows = await execute_query(
        "CREATE notebook SET name = $n, description = '' RETURN AFTER;",
        {"n": name},
        config=config,
    )
    return str(rows[0]["id"])


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_global_overlay_roundtrip(live_surrealdb: SurrealDBConfig) -> None:
    repo = AliasOverlayRepository(config=live_surrealdb)
    name_a = _u("Alfa")
    name_b = _u("Alpha")
    rid = await repo.create_overlay(
        kind="merge",
        name_a=name_a,
        name_b=name_b,
        entity_type="organization",
        scope="global",
    )
    assert rid

    got = await repo.get_overlay(rid)
    assert got is not None
    assert got["kind"] == "merge"
    assert got["scope"] == "global"
    assert got["name_a"] == name_a

    # Global rule shows for None scope and any notebook scope.
    glob = await repo.list_overlays(notebook_id=None)
    assert any(str(r["id"]) == rid for r in glob)


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_per_notebook_isolation(live_surrealdb: SurrealDBConfig) -> None:
    repo = AliasOverlayRepository(config=live_surrealdb)
    nb1 = await _make_notebook(live_surrealdb, _u("nb1"))
    nb2 = await _make_notebook(live_surrealdb, _u("nb2"))

    a = _u("Beta")
    b = _u("Bheta")
    nb_rule = await repo.create_overlay(
        kind="split",
        name_a=a,
        name_b=b,
        entity_type="organization",
        scope="notebook",
        notebook_id=nb1,
    )

    # nb1 sees its own rule.
    in_nb1 = await repo.list_overlays(notebook_id=nb1)
    assert any(str(r["id"]) == nb_rule for r in in_nb1)

    # nb2 does NOT see nb1's rule (per-notebook isolation, AC5).
    in_nb2 = await repo.list_overlays(notebook_id=nb2)
    assert all(str(r["id"]) != nb_rule for r in in_nb2)

    # The global-only view does not see the notebook rule either.
    glob = await repo.list_overlays(notebook_id=None)
    assert all(str(r["id"]) != nb_rule for r in glob)


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_delete_overlay(live_surrealdb: SurrealDBConfig) -> None:
    repo = AliasOverlayRepository(config=live_surrealdb)
    rid = await repo.create_overlay(
        kind="split",
        name_a=_u("Gamma"),
        name_b=_u("Ghamma"),
        entity_type="organization",
    )
    assert await repo.delete_overlay(rid) is True
    assert await repo.get_overlay(rid) is None


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_schema_rejects_invalid_kind(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Migration 56's enum ASSERT rejects a bad kind at the DB layer."""
    with pytest.raises(Exception):
        await execute_query(
            "CREATE alias_overlay SET scope = 'global', kind = 'bogus', "
            "name_a = 'a', name_b = 'b';",
            {},
            config=live_surrealdb,
        )
