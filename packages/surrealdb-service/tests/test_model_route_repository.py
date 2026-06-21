"""Roundtrip tests for the Track J.1 ``model_route`` table + repository.

Boots the testcontainers SurrealDB harness (``live_surrealdb``), which applies
every migration including #51, and exercises
:class:`ModelRouteRepository` end-to-end.

Acceptance criteria mapped to test cases (J.1 AC#7):

* Migration 51 applies on a fresh DB — ``test_migration_51_recorded``.
* Migration 51 is idempotent on re-run — ``test_migration_51_is_idempotent``.
* ``provider_chain`` ORDER survives a roundtrip — ``test_provider_chain_ordering_roundtrip``.
* Singleton-per-task upsert (no duplicate row) — ``test_upsert_is_singleton_per_task``.
"""

from __future__ import annotations

import pytest

from shared.models.llm import ModelRoute
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.model_route import ModelRouteRepository


# ---------------------------------------------------------------------------
# Smoke: migration 51 applied + idempotent
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_51_recorded(live_surrealdb: SurrealDBConfig) -> None:
    """Migration 51 is recorded in ``_sbl_migrations``."""
    rows = await execute_query(
        "SELECT version FROM _sbl_migrations WHERE version = 51;",
        config=live_surrealdb,
    )
    assert rows, "Migration 51 not recorded — harness did not apply it"
    assert rows[0]["version"] == 51


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_51_is_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Re-executing migration 51 DDL is a no-op (every DEFINE is IF NOT EXISTS)."""
    from pathlib import Path

    here = Path(__file__).resolve()
    migration_path = None
    for ancestor in here.parents:
        candidate = ancestor / "migrations" / "51.surrealql"
        if candidate.is_file():
            migration_path = candidate
            break
    assert migration_path is not None, "Could not locate migrations/51.surrealql"

    body = migration_path.read_text(encoding="utf-8")
    cleaned = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith("--")
    )
    await execute_query(cleaned, config=live_surrealdb)

    rows = await execute_query(
        "SELECT count() AS n FROM _sbl_migrations WHERE version = 51 GROUP ALL;",
        config=live_surrealdb,
    )
    assert rows and rows[0]["n"] == 1


# ---------------------------------------------------------------------------
# ModelRouteRepository round-trip
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_provider_chain_ordering_roundtrip(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """upsert + get_by_task preserves the EXACT provider_chain ordering."""
    repo = ModelRouteRepository(config=live_surrealdb)

    chain = [
        {"provider": "anthropic", "model_id": "model:claude", "enabled": True},
        {"provider": "openai", "model_id": "model:gpt", "enabled": True},
        {"provider": "ollama", "model_id": "model:llama", "enabled": True},
    ]
    private_chain = [
        {"provider": "ollama", "model_id": "model:llama", "enabled": True},
    ]
    await repo.upsert(
        ModelRoute(
            task="entity_extraction",
            provider_chain=chain,
            private_chain=private_chain,
        )
    )

    fetched = await repo.get_by_task("entity_extraction")
    assert fetched is not None
    # Order is load-bearing: the resolver walks the chain head-to-tail.
    assert [e["provider"] for e in fetched.provider_chain] == [
        "anthropic",
        "openai",
        "ollama",
    ]
    assert fetched.provider_chain[0]["model_id"] == "model:claude"
    assert [e["provider"] for e in fetched.private_chain] == ["ollama"]


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_upsert_is_singleton_per_task(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A second upsert for the same task rewrites the row, not duplicates it."""
    repo = ModelRouteRepository(config=live_surrealdb)

    await repo.upsert(
        ModelRoute(
            task="chat",
            provider_chain=[
                {"provider": "anthropic", "model_id": "model:a", "enabled": True}
            ],
        )
    )
    await repo.upsert(
        ModelRoute(
            task="chat",
            provider_chain=[
                {"provider": "openai", "model_id": "model:b", "enabled": True}
            ],
        )
    )

    rows = await execute_query(
        "SELECT count() AS n FROM model_route WHERE task = 'chat' GROUP ALL;",
        config=live_surrealdb,
    )
    assert rows and rows[0]["n"] == 1

    fetched = await repo.get_by_task("chat")
    assert fetched is not None
    assert fetched.provider_chain[0]["provider"] == "openai"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_get_by_task_missing_returns_none(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """An unconfigured task returns None (resolver then uses the default chain)."""
    repo = ModelRouteRepository(config=live_surrealdb)
    assert await repo.get_by_task("summarization") is None


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_list_all_returns_configured_routes(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """list_all surfaces every configured route row."""
    repo = ModelRouteRepository(config=live_surrealdb)
    # Clear any rows left by sibling tests in this session-scoped container.
    await execute_query("DELETE model_route;", config=live_surrealdb)

    await repo.upsert(
        ModelRoute(task="chat", provider_chain=[])
    )
    await repo.upsert(
        ModelRoute(task="summarization", provider_chain=[])
    )

    tasks = {r.task for r in await repo.list_all()}
    assert tasks == {"chat", "summarization"}
