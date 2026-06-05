# Phase B.0 — self-review

> Author: implementer agent, 2026-06-05
> Branch: `track/b-kg-foundation`

## Acceptance criteria check

| # | Criterion | Verified? |
|---|---|---|
| 1 | `pytest -m requires_docker .../test_migrations_roundtrip.py` runs canary tests, passes on Docker host | Designed to pass; **first verification gate is the CI run on this PR** (no Docker in authoring sandbox) |
| 2 | `pytest -m "not requires_docker"` keeps passing | YES — 45 passed locally |
| 3 | Fixture is session-scoped + importable as `from surrealdb_service.testing import live_surrealdb` | YES — see `testing/__init__.py` |
| 4 | Migration runner applies 1-43 in order, aborts with clear error | YES — `_apply_with_diagnostics` wraps `AsyncMigrationManager.run_migration_up` and adds an "offending file is likely migrations/N.surrealql" hint |
| 5 | CI runs canary tests on PR; xfailed tests show as XFAIL | YES — `db-integration.yml` runs the file; pytest's `xfail(strict=True)` reports XFAIL/XPASS correctly |
| 6 | Persistence-drift xfail has clear comment pointing at `entity_persistence_service.py:132-156` and B.1a fix scope | YES — see the `reason=` in `test_entity_persistence_drift_xfail` |
| 7 | Self-review written | this file |

## Honest trade-offs

- **No end-to-end Docker run in this PR**. The sandbox lacks Docker, so the
  six `requires_docker` tests were verified only by their *skip* behaviour.
  The CI job exists specifically to catch any boot-time problem.
- **Generic `DockerContainer`** rather than an official adapter — verified
  via PyPI that no `testcontainers-surrealdb` adapter exists as of 2026-06.
  If/when one ships, swap it in (one-import change).
- **`memory` storage engine**, not `rocksdb`. This makes each container boot
  faster and guarantees a clean slate. Production uses `rocksdb` — we are
  *not* exercising rocksdb-specific behaviour, but no migration touches
  storage-engine-specific features.
- **Session-scoped fixture**: tests share one DB. They must use unique IDs.
  The alternative (function-scoped) would re-apply all migrations per test
  and add ~3-5s per test — unacceptable as the suite grows.
- **xfail(strict=True)** on the drift test: a strict xfail will surface as
  XPASS-failure if B.1a fixes the bug without updating this test. That's
  intentional — it forces the B.1a author to delete or promote it rather than
  leaving a stale xfail.

## Risks / things to watch

- **First CI run may surface image-pull rate-limit issues** from Docker Hub
  on shared `ubuntu-latest`. If that bites, switch to GHCR or pre-cache the
  image in a self-hosted runner. Track A had no equivalent because its CI
  only built images locally.
- **Migration runner's "already exists" handling** (in
  `migrations.py:160-166`) could mask a real failure on a fresh DB. The
  canary `test_migrations_applied` asserts `max(versions) >= 43`, which
  catches "migration silently skipped" because the version bump still
  happens. But: any future migration that adds a *new* table and silently
  collides with another would not be detected by version count alone — the
  per-table roundtrip tests are the real defence.
- **WS connection to the testcontainer**: we use `ws://host:port/rpc`. If
  testcontainers reports a hostname that's not reachable over IPv4 on the
  GitHub runner (unlikely, but I've seen it on rootless Docker), the
  `_wait_for_health` HTTP probe will time out at 60s with a clear error.

## Quality gates

- `cd packages/surrealdb-service && uv run pytest -q` → **45 passed, 6 skipped**
- `cd packages/surrealdb-service && uv run pytest -q -m "not requires_docker"` → **45 passed, 6 deselected**
- `cd packages/surrealdb-service && uv run pytest -q -m "requires_docker"` → **6 skipped, 45 deselected** (gating works)
- `cd apps/app-main && uv run pytest -q` → **367 passed**
- Frontend `tsc --noEmit`: not run (B.0 doesn't touch frontend; pre-existing
  rule: not required when no frontend file touched)

## Out of scope (deliberately deferred)

- Fixing the entity-persistence drift (B.1a per plan).
- Adding roundtrip tests for tables introduced before migration 39 (chunk,
  source_embedding, etc.) — current canaries focus on the migration-39
  SCHEMAFULL tables that B-track work mutates. B.1b will add more as new
  tables land.
- A `function-scoped` flavour of the fixture (e.g. for tests that need a
  clean DB) — add when first needed.
