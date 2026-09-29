"""Pooled connections with the real SDK and a real SurrealDB."""

from typing import Any, List

import pytest

from surrealdb_service import connection as conn_mod
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ConnectionPool, execute_query

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]


@pytest.fixture
def pool(live_surrealdb: SurrealDBConfig):
    original = conn_mod._pool
    conn_mod._pool = ConnectionPool(config=live_surrealdb, max_size=4, ping_timeout=5.0)
    yield conn_mod._pool
    conn_mod._pool = original


def _idle(pool: ConnectionPool) -> List[Any]:
    return list(pool._pool._queue)


async def test_an_idle_connection_is_reused(pool):
    """The reuse ping must be a statement SurrealDB accepts: ``SELECT 1`` is a
    parse error on v2, so every checkout dropped every idle connection and
    signed in anew."""
    await execute_query("RETURN 1")
    idle = _idle(pool)
    for i in range(5):
        assert await execute_query(f"RETURN {i}") == i
    assert _idle(pool) == idle
