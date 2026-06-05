# Track B — KG quality: rolling status

## Phase B.0 — Testcontainers SurrealDB harness (2026-06-05)

**Branch**: `track/b-kg-foundation`
**State**: code complete, local tests green, ready for review.

### Delivered

- `packages/surrealdb-service/src/surrealdb_service/testing/` — new subpackage
  exposing the `live_surrealdb` pytest fixture. Boots
  `surrealdb/surrealdb:v2` via generic `testcontainers.DockerContainer` (no
  official SurrealDB adapter exists as of 2026-06), waits for `/health`,
  resets the connection-pool singleton, applies all discovered migrations via
  the canonical `AsyncMigrationManager`, and yields a `SurrealDBConfig`. The
  fixture is importable from any workspace member as
  `from surrealdb_service.testing import live_surrealdb`.
- `packages/surrealdb-service/tests/conftest.py` — re-exports the fixture for
  the local test suite (and serves as a template downstream packages can
  copy).
- `packages/surrealdb-service/tests/test_migrations_roundtrip.py` — five
  canary tests:
  - migrations-applied smoke (asserts version ≥ 43);
  - `entity` roundtrip (canonical_name, entity_type, defaults);
  - `entity_alias` roundtrip;
  - `relation` RELATE roundtrip;
  - `source` roundtrip including the migration-43 `metadata` bag;
  - **xfail** for the legacy `entity_persistence_service` write shape
    (`name`/`weight`/`source_ids`) — confirms the bug B.1a will fix and
    documents the exact source location (lines 132-156).
- `.github/workflows/db-integration.yml` — new workflow runs the harness on
  every PR touching `migrations/`, `packages/surrealdb-service/`, or
  `packages/shared/src/shared/models/`. Verifies Docker availability up
  front (Track A's GPU mishap is the cautionary tale).
- `docs/tracks/B-kg-quality/TESTCONTAINERS_GUIDE.md` — usage guide.
- `packages/surrealdb-service/pyproject.toml` — added `testcontainers>=4.0.0`
  to dev deps, registered the `requires_docker` marker.
- Workspace `pyproject.toml` — also registered `requires_docker` so other
  packages can use the marker without re-defining it.

### Decisions taken (all per autopilot defaults)

- **Q-B-1**: legacy persistence drift is *surfaced* via an `xfail(strict=True)`
  test, not fixed here. Strictness means if B.1a accidentally over-fixes, the
  test will turn XPASS and force us to delete/promote it.
- **Storage engine**: `memory` (not rocksdb). Each container is throwaway and
  faster to boot.
- **Session scope**: one container per pytest session. Tests that touch the
  same table use unique IDs (`_unique()` helper) to avoid cross-test
  interference rather than re-applying migrations per test.

### Test results

- `packages/surrealdb-service`: **45 passed, 6 skipped** (the 6 are the
  `requires_docker` tests skipping cleanly because no Docker daemon is
  available in the sandbox where this was authored).
- `apps/app-main`: **367 passed** — no regressions.

### Open items / hand-off notes

- The `requires_docker` tests have not been executed end-to-end yet (no
  Docker in the authoring sandbox). They are designed to skip cleanly when
  Docker is absent and the CI workflow verifies Docker is reachable on the
  runner before running them. **First CI run on the PR is the validation
  gate**.
- B.1a inherits the xfail test in `test_migrations_roundtrip.py` — its
  acceptance criterion #4 should explicitly delete or invert it.
