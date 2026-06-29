"""PL.3 integration: the per-notebook ``auto_insights`` toggle (container).

Exercises migration 72 (``notebook.auto_insights TYPE bool DEFAULT true``) and
``NotebookRepository.get_auto_insights`` against a real SurrealDB testcontainer:

* a freshly-created notebook defaults to auto_insights = True (the toggle ON);
* an explicit False reads back False (the opt-out);
* an unknown notebook id reads back True (default ON, conservative).
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.notebook import NotebookRepository

pytestmark = pytest.mark.asyncio


def _u(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@pytest.mark.requires_docker
async def test_new_notebook_defaults_auto_insights_on(
    live_surrealdb: SurrealDBConfig,
) -> None:
    rows = await execute_query(
        "CREATE notebook SET name = $n;", {"n": _u("NB")}, config=live_surrealdb
    )
    nb_id = str(rows[0]["id"])
    repo = NotebookRepository(config=live_surrealdb)

    # Migration 72 default applies to a row created after it -> True.
    assert await repo.get_auto_insights(nb_id) is True


@pytest.mark.requires_docker
async def test_explicit_off_reads_back_false(
    live_surrealdb: SurrealDBConfig,
) -> None:
    rows = await execute_query(
        "CREATE notebook SET name = $n, auto_insights = false;",
        {"n": _u("NB")},
        config=live_surrealdb,
    )
    nb_id = str(rows[0]["id"])
    repo = NotebookRepository(config=live_surrealdb)

    assert await repo.get_auto_insights(nb_id) is False


@pytest.mark.requires_docker
async def test_unknown_notebook_defaults_on(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = NotebookRepository(config=live_surrealdb)
    # Unknown id -> conservative default ON.
    assert await repo.get_auto_insights("notebook:does_not_exist") is True
