"""Roundtrip tests for the Phase B.4 ``metrics`` table.

Boots the testcontainers SurrealDB harness (``live_surrealdb`` fixture from
``surrealdb_service.testing``), applies every migration up to and including
#47, and exercises the schema + ``record_metric`` end-to-end.

Acceptance criteria mapped to test cases:

* AC#5 (migration 47 applies + idempotent) — ``test_migration_47_recorded``
  + ``test_migration_47_is_idempotent``.
* AC#3 (record_metric writes the row) — ``test_record_metric_persists_row``.
* AC#3 cont. — payload roundtrip preserves nested dicts.
* Defensive: option<string> fields accept NONE — ``test_record_metric_persists_without_context``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from shared.services.metrics import record_metric
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query


@pytest.fixture
def metrics_pool_bound_to_live(
    live_surrealdb: SurrealDBConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    """Ensure ``record_metric``'s lazy import sees the harness config.

    ``record_metric`` calls ``execute_query`` without a config; that
    branch resolves the default pool from env, which would point at the
    dev DB, not our test container. We monkeypatch the lazy-import
    target so the SQL still runs but is automatically bound to
    ``live_surrealdb``. Hooks in the production code path stay
    unchanged.
    """

    async def _bound(
        query: str,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[SurrealDBConfig] = None,
    ):
        return await execute_query(
            query, params, config=config or live_surrealdb
        )

    # ``record_metric`` does ``from surrealdb_service.connection import
    # execute_query`` lazily inside the function. Patching the module
    # attribute is sufficient — the lazy import resolves to the patched
    # symbol.
    import surrealdb_service.connection as conn_module

    monkeypatch.setattr(conn_module, "execute_query", _bound)
    return live_surrealdb


# ---------------------------------------------------------------------------
# Smoke: migration 47 applied + idempotent
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_47_recorded(live_surrealdb: SurrealDBConfig) -> None:
    """Migration 47 is recorded in ``_sbl_migrations``.

    If the harness didn't apply it, every later test in this module would
    skip silently because ``metrics`` wouldn't exist.
    """
    rows = await execute_query(
        "SELECT version FROM _sbl_migrations WHERE version = 47;",
        config=live_surrealdb,
    )
    assert rows, "Migration 47 not recorded — harness did not apply it"
    assert rows[0]["version"] == 47


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_47_is_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Re-executing migration 47 DDL is a no-op.

    Mirrors the migration-45 roundtrip pattern: every DEFINE uses
    ``IF NOT EXISTS`` so replaying the statements against a populated
    DB must succeed without errors and must not produce a duplicate
    migration record.
    """
    from pathlib import Path

    here = Path(__file__).resolve()
    migration_path = None
    for ancestor in here.parents:
        candidate = ancestor / "migrations" / "47.surrealql"
        if candidate.is_file():
            migration_path = candidate
            break
    assert migration_path is not None, "Could not locate migrations/47.surrealql"

    body = migration_path.read_text(encoding="utf-8")
    cleaned = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith("--")
    )
    await execute_query(cleaned, config=live_surrealdb)

    rows = await execute_query(
        "SELECT count() AS n FROM _sbl_migrations "
        "WHERE version = 47 GROUP ALL;",
        config=live_surrealdb,
    )
    assert rows and rows[0]["n"] == 1


# ---------------------------------------------------------------------------
# record_metric end-to-end
# ---------------------------------------------------------------------------


async def _count_metrics(
    config: SurrealDBConfig, event_type: str
) -> int:
    rows = await execute_query(
        "SELECT count() AS n FROM metrics "
        "WHERE event_type = $event_type GROUP ALL;",
        {"event_type": event_type},
        config=config,
    )
    return int(rows[0]["n"]) if rows else 0


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_record_metric_persists_row(
    metrics_pool_bound_to_live: SurrealDBConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: a record_metric call lands one row with the right payload."""
    live_surrealdb = metrics_pool_bound_to_live
    monkeypatch.delenv("OPEN_NOTEBOOK_DISABLE_METRICS", raising=False)

    before = await _count_metrics(live_surrealdb, "extraction.complete")

    # The shared service calls execute_query without an explicit config;
    # the harness's live_surrealdb has installed itself as the default.
    await record_metric(
        "extraction.complete",
        {
            "entity_count": 7,
            "relation_count": 2,
            "avg_confidence": 0.91,
            "multi_schema": True,
        },
        source="source:fixture-a",
        notebook="notebook:fixture-1",
    )

    after = await _count_metrics(live_surrealdb, "extraction.complete")
    assert after == before + 1, (
        "record_metric did not persist exactly one extraction.complete row"
    )

    # And the row contents survived a roundtrip.
    # NOTE: SurrealDB requires ORDER BY columns to appear in the SELECT
    # projection — including ``created_at`` so the parser can locate it.
    rows = await execute_query(
        "SELECT event_type, payload, source, notebook, created_at FROM metrics "
        "WHERE event_type = 'extraction.complete' "
        "AND source = 'source:fixture-a' "
        "AND notebook = 'notebook:fixture-1' "
        "ORDER BY created_at DESC LIMIT 1;",
        config=live_surrealdb,
    )
    assert rows, "Could not locate the row we just wrote"
    row = rows[0]
    assert row["event_type"] == "extraction.complete"
    assert row["source"] == "source:fixture-a"
    assert row["notebook"] == "notebook:fixture-1"
    payload = row["payload"]
    assert payload["entity_count"] == 7
    assert payload["relation_count"] == 2
    assert payload["avg_confidence"] == pytest.approx(0.91)
    assert payload["multi_schema"] is True


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_record_metric_persists_without_context(
    metrics_pool_bound_to_live: SurrealDBConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional source/notebook → NONE in DB (option<string> schema honours it)."""
    live_surrealdb = metrics_pool_bound_to_live
    monkeypatch.delenv("OPEN_NOTEBOOK_DISABLE_METRICS", raising=False)

    await record_metric(
        "extraction.auto_fallback",
        {"confidence": 0.4, "decision": "fallback", "engine_used": "mineru"},
    )

    rows = await execute_query(
        "SELECT event_type, payload, source, notebook, created_at FROM metrics "
        "WHERE event_type = 'extraction.auto_fallback' "
        "ORDER BY created_at DESC LIMIT 1;",
        config=live_surrealdb,
    )
    assert rows
    row = rows[0]
    assert row["source"] is None
    assert row["notebook"] is None
    assert row["payload"]["decision"] == "fallback"
    assert row["payload"]["engine_used"] == "mineru"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_record_metric_respects_env_disable(
    metrics_pool_bound_to_live: SurrealDBConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPEN_NOTEBOOK_DISABLE_METRICS=1 → no DB row, no error."""
    live_surrealdb = metrics_pool_bound_to_live
    monkeypatch.setenv("OPEN_NOTEBOOK_DISABLE_METRICS", "1")

    before = await _count_metrics(live_surrealdb, "extraction.complete")
    await record_metric(
        "extraction.complete",
        {"entity_count": 99},
        source="source:should-not-write",
    )
    after = await _count_metrics(live_surrealdb, "extraction.complete")
    assert after == before, (
        "OPEN_NOTEBOOK_DISABLE_METRICS=1 did not suppress the write"
    )
