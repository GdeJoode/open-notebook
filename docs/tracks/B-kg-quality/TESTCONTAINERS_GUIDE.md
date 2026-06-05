# Testcontainers SurrealDB harness (Phase B.0)

A pytest fixture that boots a real SurrealDB instance, applies every migration
in `migrations/`, and tears down at session-end. Used by every Track B phase
that touches a SCHEMAFULL table — and reusable from any other workspace
member.

## Why it exists

Track A's RETRO (lesson #1) identified that the missing migration #43 would
have been caught at author-time if migrations were exercised against a real
SurrealDB during `pytest`. Mock-only tests pass against the SCHEMALESS path
because nothing validates field-name drift. This harness closes that gap.

## How to use it

### 1. Mark the test

```python
import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_my_schemaful_table(live_surrealdb: SurrealDBConfig) -> None:
    rows = await execute_query(
        "CREATE my_table SET name = 'x';",
        config=live_surrealdb,
    )
    assert rows
```

The `requires_docker` marker:

- skips the test automatically when no Docker daemon is reachable;
- is registered in both the workspace `pyproject.toml` and the
  `packages/surrealdb-service/pyproject.toml`, so other packages can use the
  same name without re-registering.

### 2. Import the fixture in a downstream package

```python
# apps/app-main/tests/conftest.py  (example for B.1a later)
from surrealdb_service.testing import live_surrealdb  # noqa: F401
```

That's it — `pytest` discovers the fixture by name. The fixture is
session-scoped: one container per test session, regardless of how many
modules request it.

## Running locally

```bash
# In the surrealdb-service package
cd packages/surrealdb-service
uv run pytest -m requires_docker

# Or every test (Docker-gated ones skip cleanly when Docker is absent):
uv run pytest
```

Cold-start cost: ~5-10 seconds for the SurrealDB image pull (first run) plus
~2-5 seconds for migration application. Warm cache: ~3 seconds total.

## Running in CI

`.github/workflows/db-integration.yml` runs the `requires_docker` test suite
on every PR that touches:

- `migrations/`
- `packages/surrealdb-service/`
- `packages/shared/src/shared/models/`
- the workflow itself

Docker is available on `ubuntu-latest` GitHub-hosted runners out-of-the-box,
so no extra setup is required. The job runs in parallel with the existing
`e2e.yml` workflow.

## How the fixture works

`packages/surrealdb-service/src/surrealdb_service/testing/fixtures.py`:

1. Pre-flight: pings the Docker daemon. If unreachable → `pytest.skip()`.
2. Boots `surrealdb/surrealdb:v2` (matching `docker-compose.yml`) via the
   generic `testcontainers.core.container.DockerContainer` — there is no
   official `testcontainers-surrealdb` adapter as of 2026-06.
3. Uses `start --user root --pass root memory` so each container is fresh
   (the `memory` storage engine drops on exit; we never see leftover state).
4. Waits for `GET /health` to return 200 (60s timeout, with a TCP-probe
   pre-check to fail fast).
5. Resets `surrealdb_service.connection._pool` so the global pool picks up
   the new config instead of the dev-DB default.
6. Applies all migrations via the canonical `AsyncMigrationManager`. On
   failure, the error message points at the offending migration file.
7. Yields a `SurrealDBConfig` pointing at the container.
8. On session end: closes the pool, stops the container.

## Opting out

There is no environment-variable opt-out today: the harness skips
automatically when Docker is absent (the `_docker_reachable()` probe in
`fixtures.py`). If you want to skip explicitly during local dev (e.g. you have
Docker running but don't want the harness to engage), use:

```bash
uv run pytest -m "not requires_docker"
```

## Adding a new migration

When you add `migrations/N.surrealql`:

1. Add a roundtrip test to `tests/test_migrations_roundtrip.py` that exercises
   any new SCHEMAFULL table or field added by `N.surrealql`.
2. Run `uv run pytest -m requires_docker` locally; the fixture will re-apply
   migrations 1 through N.
3. If the roundtrip test fails, either the migration or the test is wrong —
   the harness pinpoints which.

## Known limitations

- **Session-scoped**: one container per pytest session. Tests that mutate
  global state should clean up after themselves (delete by primary key or by
  `_unique()` prefix — see `test_migrations_roundtrip.py`).
- **No surreal-specific testcontainers adapter**: we use `DockerContainer`
  directly. If/when an official adapter ships (track at
  https://github.com/testcontainers/testcontainers-python), swap it in.
- **WSL2 / Docker Desktop**: works fine; just make sure Docker Desktop's
  integration with your WSL distro is enabled.
