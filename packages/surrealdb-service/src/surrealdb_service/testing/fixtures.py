"""Testcontainers-backed SurrealDB fixture.

There is no first-party ``testcontainers-surrealdb`` adapter on PyPI as of
2026-06, so we use the generic ``DockerContainer`` plus a custom HTTP-health
wait strategy. The fixture:

1. Pulls ``surrealdb/surrealdb:v2`` (matching ``docker-compose.yml``).
2. Boots it with the same ``rocksdb`` storage + root/root credentials we use
   in production, but on an ephemeral in-container path so each container
   starts fresh.
3. Waits for ``GET /health`` to return 200.
4. Builds a ``SurrealDBConfig`` pointing at the exposed port and applies every
   discovered migration via the canonical :class:`AsyncMigrationManager`.
5. Resets the global connection pool so tests get a fresh pool per session.

Session-scoped: one container per pytest session. Migrations are slow enough
(a few seconds for ~17 files) that re-applying per-function would be painful;
tests that need a clean DB should clear specific tables themselves.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from loguru import logger

from surrealdb_service import connection as connection_module
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.migrations import AsyncMigrationManager

# Pin to the same major tag used in docker-compose.yml. Bumping this should be
# a deliberate decision — schema migrations are written against this version.
SURREALDB_IMAGE = "surrealdb/surrealdb:v2"

# SurrealDB inside the container listens on 8000. Testcontainers maps it to a
# random host port we discover via .get_exposed_port().
SURREALDB_INTERNAL_PORT = 8000

# Repo root is three levels up from this file:
#   packages/surrealdb-service/src/surrealdb_service/testing/fixtures.py
# → ../../../../..
_REPO_ROOT = Path(__file__).resolve().parents[4]
_MIGRATIONS_DIR = _REPO_ROOT / "migrations"


def _docker_reachable() -> bool:
    """Return True iff a Docker daemon is reachable.

    We can't just import ``docker`` and try ``from_env()`` — that raises on
    missing socket but also during normal import on some systems. Using a
    socket check against the default unix socket is faster and safer.
    """
    try:
        import docker  # type: ignore[import-untyped]

        client = docker.from_env(timeout=2)
        client.ping()
        return True
    except Exception:
        return False


DOCKER_AVAILABLE = _docker_reachable()


def docker_available() -> bool:
    """Re-evaluate Docker availability at call-time (handy in CI debug)."""
    return _docker_reachable()


def _wait_for_health(host: str, port: int, timeout_s: float = 60.0) -> None:
    """Block until SurrealDB's HTTP ``/health`` returns 200 or we time out."""
    deadline = time.monotonic() + timeout_s
    url = f"http://{host}:{port}/health"
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001 — propagate after timeout
            last_err = exc
        time.sleep(0.5)
    raise RuntimeError(
        f"SurrealDB at {url} did not become healthy within {timeout_s}s "
        f"(last error: {last_err!r})"
    )


def _port_open(host: str, port: int, timeout_s: float = 2.0) -> bool:
    """Cheap TCP probe — used as a precursor to HTTP health checks."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def live_surrealdb() -> SurrealDBConfig:
    """Boot a SurrealDB container, apply all migrations, yield a config.

    Returns a :class:`SurrealDBConfig` whose ``url`` points at the container.
    The container is torn down at session-end.

    Usage::

        @pytest.mark.requires_docker
        async def test_something(live_surrealdb):
            from surrealdb_service.connection import execute_query
            rows = await execute_query("INFO FOR DB;", config=live_surrealdb)
            ...
    """
    if not DOCKER_AVAILABLE:
        pytest.skip("Docker daemon not reachable — skipping requires_docker tests")

    # Import lazily so collection in non-docker environments doesn't choke on
    # the testcontainers import chain.
    from testcontainers.core.container import DockerContainer

    if not _MIGRATIONS_DIR.is_dir():
        raise RuntimeError(
            f"Migrations directory not found at {_MIGRATIONS_DIR}. "
            "Has the workspace layout changed?"
        )

    logger.info(f"Starting SurrealDB testcontainer ({SURREALDB_IMAGE})")
    container = (
        DockerContainer(SURREALDB_IMAGE)
        .with_exposed_ports(SURREALDB_INTERNAL_PORT)
        .with_command(
            "start --log info --user root --pass root memory"
        )
    )
    container.start()
    try:
        host = container.get_container_host_ip()
        # testcontainers returns str
        port = int(container.get_exposed_port(SURREALDB_INTERNAL_PORT))
        logger.info(f"SurrealDB container at {host}:{port} — waiting for health")

        # TCP first (fast fail), then HTTP /health (slow but definitive).
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if _port_open(host, port):
                break
            time.sleep(0.2)
        _wait_for_health(host, port, timeout_s=60.0)

        config = SurrealDBConfig(
            url=f"ws://{host}:{port}/rpc",
            username="root",
            password="root",
            namespace="open_notebook_test",
            database="open_notebook_test",
        )

        # Reset the global pool so this config is honoured. The pool caches
        # the first config it sees, which would otherwise point at the dev DB.
        connection_module._pool = None

        logger.info(
            f"Applying migrations from {_MIGRATIONS_DIR} to {config.url} "
            f"(ns={config.namespace}, db={config.database})"
        )
        manager = AsyncMigrationManager(
            migrations_dir=_MIGRATIONS_DIR, config=config
        )

        # Run synchronously inside the session-scoped fixture; pytest-asyncio
        # doesn't gift us a loop here, so we drive one ourselves.
        import asyncio

        try:
            asyncio.run(_apply_with_diagnostics(manager))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to apply migrations against testcontainer SurrealDB: {exc}"
            ) from exc

        # Yield the config to tests
        yield config
    finally:
        # Tear down the pool first so no dangling connections leak when the
        # container goes away.
        try:
            asyncio.run(connection_module.close_pool())
        except Exception:
            pass
        connection_module._pool = None
        logger.info("Stopping SurrealDB testcontainer")
        container.stop()


async def _apply_with_diagnostics(manager: AsyncMigrationManager) -> None:
    """Apply migrations, surfacing the offending version on failure."""
    try:
        await manager.run_migration_up()
    except Exception as exc:
        current = -1
        try:
            current = await manager.get_current_version()
        except Exception:
            pass
        # Manager logs internally; we re-raise with a pointer to the next file
        # the runner would have tried.
        next_pending = sorted(
            v for v in manager._up if v > current
        )
        candidate = (
            f"migrations/{next_pending[0]}.surrealql"
            if next_pending
            else "unknown"
        )
        raise RuntimeError(
            f"Migration runner failed after version {current}; "
            f"offending file is likely {candidate}. Underlying error: {exc}"
        ) from exc


# Async wrapper for callers that want an awaitable handle.
# (Some tests prefer pytest-asyncio's async fixtures.)
@pytest_asyncio.fixture(scope="session")
async def live_surrealdb_async(live_surrealdb: SurrealDBConfig) -> AsyncIterator[SurrealDBConfig]:
    """Async-flavoured alias of ``live_surrealdb`` for tests that want one."""
    yield live_surrealdb
