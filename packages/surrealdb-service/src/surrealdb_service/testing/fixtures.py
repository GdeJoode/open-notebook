"""Testcontainers-backed SurrealDB fixture.

There is no first-party ``testcontainers-surrealdb`` adapter on PyPI as of
2026-06, so we use the generic ``DockerContainer`` plus a custom HTTP-health
wait strategy. The fixture:

1. Pulls ``surrealdb/surrealdb:v2`` (matching ``docker-compose.yml``).
2. Boots it with the ``memory`` storage engine and root/root credentials.
   Production uses ``rocksdb`` but no migration we exercise depends on
   rocksdb-specific behaviour; ``memory`` gives a faster boot and a
   guaranteed clean slate per session.
3. Waits for ``GET /health`` to return 200.
4. Builds a ``SurrealDBConfig`` pointing at the exposed port and applies every
   discovered migration (43+ files) via the canonical :class:`AsyncMigrationManager`.
5. Resets the global connection pool so tests get a fresh pool per session,
   bound to the test event loop rather than the migration loop.

Session-scoped: one container per pytest session. Migrations are slow enough
(a few seconds for 43+ files) that re-applying per-function would be painful;
tests that need a clean DB should clear specific tables themselves.
"""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Iterator

import httpx
import pytest
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


def _find_migrations_dir() -> Path:
    """Walk upward from this file until we find a sibling ``migrations/`` dir.

    Magic-number ``parents[N]`` indexing is fragile when files move (the
    attempt-1 bug used ``parents[4]`` and landed in ``packages/``). Walking up
    until we find ``migrations/`` next to a workspace marker (``pyproject.toml``
    declaring ``[tool.uv.workspace]``) is robust to future reshuffles.
    """
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "migrations"
        if candidate.is_dir() and (ancestor / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(
        f"Could not locate the workspace ``migrations/`` directory by walking "
        f"up from {here}. Has the repo layout changed?"
    )


_MIGRATIONS_DIR = _find_migrations_dir()


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


def docker_available() -> bool:
    """Re-evaluate Docker availability at call-time.

    Lazy by design — importing this module should not block on a Docker SDK
    ping for ``pytest -m "not requires_docker"`` runs.
    """
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
def live_surrealdb() -> Iterator[SurrealDBConfig]:
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
    if not docker_available():
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

        # Reset the global pool BEFORE migrations so AsyncMigrationManager
        # builds connections via the test config (not whatever the dev env
        # cached at import time).
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

        # CRITICAL: ``asyncio.run`` closes the loop it created. Any
        # AsyncSurreal connections that the migration runner cached inside the
        # global pool are now bound to a dead loop. Reset the pool so the
        # first test using ``execute_query`` rebuilds connections on its own
        # (pytest-asyncio-owned) loop.
        connection_module._pool = None

        # Yield the config to tests
        yield config
    finally:
        # Tear down the pool first so no dangling connections leak when the
        # container goes away. Pool may be None if tests never touched it.
        try:
            import asyncio

            if connection_module._pool is not None:
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
