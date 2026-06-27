"""Roundtrip test for ``SourceRepository.load_notebook_source_ids`` (U.5).

The U.5 document-layer export scopes the global ``mentions`` / ``cites`` edge
tables to one notebook's sources. The scoping seam resolves notebook → sources
via the ``reference`` RELATE edge (migration 1), mirroring the traversal
``EntityRepository.list_entities_for_notebook`` opens with. This test boots the
testcontainers harness and proves the seam returns exactly the notebook's
sources — and nothing from a sibling notebook.
"""

from __future__ import annotations

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.source import SourceRepository


async def _create_notebook(config: SurrealDBConfig, name: str) -> str:
    rows = await execute_query(
        "CREATE notebook SET name = $name;", {"name": name}, config=config
    )
    assert rows
    return str(rows[0]["id"])


async def _create_source(config: SurrealDBConfig, title: str) -> str:
    rows = await execute_query(
        "CREATE source SET title = $title;", {"title": title}, config=config
    )
    assert rows
    return str(rows[0]["id"])


async def _link(config: SurrealDBConfig, source_id: str, notebook_id: str) -> None:
    await execute_query(
        "RELATE $src->reference->$nb;",
        {"src": ensure_record_id(source_id), "nb": ensure_record_id(notebook_id)},
        config=config,
    )


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_load_notebook_source_ids_scopes_to_notebook(
    live_surrealdb: SurrealDBConfig,
) -> None:
    nb_a = await _create_notebook(live_surrealdb, "nb-a")
    nb_b = await _create_notebook(live_surrealdb, "nb-b")

    s1 = await _create_source(live_surrealdb, "Doc 1")
    s2 = await _create_source(live_surrealdb, "Doc 2")
    s_other = await _create_source(live_surrealdb, "Other")

    await _link(live_surrealdb, s1, nb_a)
    await _link(live_surrealdb, s2, nb_a)
    await _link(live_surrealdb, s_other, nb_b)

    repo = SourceRepository(config=live_surrealdb)

    ids_a = await repo.load_notebook_source_ids(nb_a)
    assert set(ids_a) == {s1, s2}
    assert s_other not in ids_a

    ids_b = await repo.load_notebook_source_ids(nb_b)
    assert set(ids_b) == {s_other}


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_load_notebook_source_ids_empty_notebook(
    live_surrealdb: SurrealDBConfig,
) -> None:
    nb = await _create_notebook(live_surrealdb, "nb-empty")
    repo = SourceRepository(config=live_surrealdb)
    assert await repo.load_notebook_source_ids(nb) == []


@pytest.mark.asyncio
async def test_load_notebook_source_ids_empty_id_returns_empty() -> None:
    """Defensive: empty notebook id short-circuits without a query."""
    repo = SourceRepository()
    assert await repo.load_notebook_source_ids("") == []
