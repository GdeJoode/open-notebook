"""PL.3 integration: the per-notebook ``auto_insights`` toggle (container).

Exercises migration 72 (``notebook.auto_insights TYPE bool DEFAULT true``) and
``NotebookRepository.get_auto_insights`` against a real SurrealDB testcontainer:

* a freshly-created notebook defaults to auto_insights = True (the toggle ON);
* an explicit False reads back False (the opt-out);
* an unknown notebook id reads back True (default ON, conservative).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.migrations import AsyncMigration
from surrealdb_service.repositories.notebook import NotebookRepository

pytestmark = pytest.mark.asyncio

_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"


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


# ---------------------------------------------------------------------------
# PL.4 folded minor (a): the S.4 hazard for migration 72, mirroring the
# migration-71 ``test_migration_71_backfills_none_and_row_stays_writable``. A
# pre-existing notebook whose ``auto_insights`` is genuinely NONE (the strict
# field the S.4 backfill must repair) is healed AND remains writable. We
# synthesise the drift by REMOVE-ing the field, creating a row (so it carries
# NONE for the field), then re-applying migration 72.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
async def test_migration_72_backfills_none_and_row_stays_writable(
    live_surrealdb: SurrealDBConfig,
) -> None:
    up = AsyncMigration.from_file(_MIGRATIONS_DIR / "72.surrealql")
    down = AsyncMigration.from_file(_MIGRATIONS_DIR / "72_down.surrealql")
    try:
        # Drop the field, then create a notebook that therefore carries NONE for
        # it (the exact legacy/pre-migration drift the S.4 rule guards against).
        await execute_query(down.sql, config=live_surrealdb)
        rows = await execute_query(
            "CREATE notebook SET name = $n;",
            {"n": _u("Drifted")},
            config=live_surrealdb,
        )
        nb_id = str(rows[0]["id"])
        drifted = await execute_query(
            "SELECT auto_insights FROM notebook WHERE id = type::thing($id);",
            {"id": nb_id},
            config=live_surrealdb,
        )
        # Field undefined -> key absent / NONE.
        assert not drifted[0].get("auto_insights")

        # Re-apply 72: DEFINE FIELD + drift-only backfill. Idempotent.
        await execute_query(up.sql, config=live_surrealdb)
        await execute_query(up.sql, config=live_surrealdb)  # second = no-op

        healed = await execute_query(
            "SELECT auto_insights FROM notebook WHERE id = type::thing($id);",
            {"id": nb_id},
            config=live_surrealdb,
        )
        assert healed[0]["auto_insights"] is True, "backfill must repair NONE"

        # The S.4 payoff: a SCHEMAFULL row that had NONE in a strict field is
        # silently unwritable until backfilled. After the backfill, an unrelated
        # UPDATE to the SAME row must succeed.
        await execute_query(
            "UPDATE notebook SET name = $n WHERE id = type::thing($id);",
            {"id": nb_id, "n": "rewritten ok"},
            config=live_surrealdb,
        )
        # And the toggle is now flippable (the row is genuinely writable).
        await execute_query(
            "UPDATE notebook SET auto_insights = false "
            "WHERE id = type::thing($id);",
            {"id": nb_id},
            config=live_surrealdb,
        )
        repo = NotebookRepository(config=live_surrealdb)
        assert await repo.get_auto_insights(nb_id) is False
    finally:
        # ALWAYS restore 72 so a mid-test failure can't leave the field dropped
        # for the rest of the (session-scoped container) suite.
        await execute_query(up.sql, config=live_surrealdb)
