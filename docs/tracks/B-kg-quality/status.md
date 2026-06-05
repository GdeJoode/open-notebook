# Track B — KG quality: rolling status

## Phase B.1b — notebook_schema + pass1_results tables + repos (2026-06-05)

**Branch**: `track/b-models-notebook-schema`
**Commits**: `5fc4859` → `e7f0310` → `997ad8f`
**State**: code complete, all quality gates green, ready for review.

### Delivered

- `migrations/45.surrealql` + `migrations/45_down.surrealql` — two new
  SCHEMAFULL tables (`notebook_schema`, `pass1_results`) following the
  migration-43 FLEXIBLE-extension-bag pattern. `UNIQUE` index on
  `notebook_schema.notebook` enforces one row per notebook;
  `idx_pass1_source` covers the hot read path. All `DEFINE` statements
  use `IF NOT EXISTS` so the migration is idempotent.
- `packages/shared/src/shared/models/notebook_schema.py` —
  `NotebookSchema` and `Pass1Result` Pydantic models. Both carry
  bounded confidence/coverage fields, a defensive
  `ensure_metadata_dict` validator on the FLEXIBLE bag, and
  `List[Dict[str, Any]]` for extension-shaped arrays so the dict
  shape can evolve without further migrations.
- `packages/surrealdb-service/src/surrealdb_service/repositories/notebook_schema.py`
  — `NotebookSchemaRepository` (singleton-per-notebook with
  rewrite-on-conflict upsert; plus
  `add_pending_extension` / `accept_pending_extension` /
  `reject_pending_extension`) and `Pass1ResultRepository` (append-only
  + source-scoped / notebook-scoped reads).
- `packages/shared/tests/test_notebook_schema_model.py` — 11 unit
  tests covering construction, full roundtrip, bounds, metadata
  coercion.
- `packages/surrealdb-service/tests/test_notebook_schema_repo_roundtrip.py`
  — 10 `requires_docker` tests covering migration record-keeping +
  idempotence, full roundtrip, UNIQUE-rewrite semantic, direct-CREATE
  blocking, extension lifecycle, and empty-list handling.
- `packages/shared/src/shared/models/__init__.py` + repository
  `__init__.py` — additive exports only. **Coordination note**: B.1a
  (`track/b-models-entity`) touches the same two files to add
  `Entity` / `Relation` and their repos. Both branches are additive
  in distinct sections of `__all__`; merge is expected to be clean
  three-way without semantic conflicts.

### Decisions taken (all per autopilot defaults Q-B-8, Q-B-9)

- **Q-B-9**: migration 45 is reserved for B.1b. (B.1a takes 44.)
- **Q-B-8**: shared `notebook_event` table is NOT introduced here —
  deferred to B.3b as planned.
- **UNIQUE-index handling**: rewrite-on-conflict semantic in the
  repository's `upsert`. Detailed rationale in
  `reviews/phase-B.1b-self-review.md` and inline near `upsert()`.

### Test results

| Suite | Before | After | Note |
|---|---|---|---|
| `packages/shared` | 105 | 116 (+11) | new model tests |
| `packages/surrealdb-service` (not requires_docker) | 52 | 52 | no new non-docker tests; no regressions |
| `packages/surrealdb-service` (requires_docker) | 5 pass, 1 xfail | 15 pass, 1 xfail (+10) | new repo roundtrips |
| `apps/app-main` | 367 | 367 | no regressions |

Final `requires_docker` run summary: `15 passed, 52 deselected, 1 xfailed in 17.63s`.

### Ready for review

PR title: `feat(shared,surrealdb): notebook_schema + pass1_results tables + repos (B.1b)`

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

## Phase B.0 — attempt 2 (2026-06-05)

**State**: revisions addressed, verified end-to-end against a real
SurrealDB container, ready for re-review.

### Fixes vs attempt 1

Reviewer rejected attempt 1 with REVISIONS_NEEDED (review at
`docs/tracks/B-kg-quality/reviews/phase-B.0-attempt-1.md`). Attempt 2
addresses every blocker and major plus several minors. Full per-blocker
table with commit SHAs lives at
`docs/tracks/B-kg-quality/reviews/phase-B.0-self-review.md` → "Attempt 2
fixes".

Highlights:

- **Blocker #1** (migrations-dir off-by-one) → `fixtures.py` now walks
  up from `__file__` looking for a `migrations/` dir sibling to a
  workspace-marker `pyproject.toml`. Robust to file moves.
- **Blocker #2** (no non-Docker safety net) → new
  `tests/test_testing_fixtures.py` (7 tests, no marker) catches
  path-drift, missing migrations files, and dead-code regressions on
  every `pytest -q` run.
- **Major #3** (pool-lifecycle across `asyncio.run`) → pool is now
  reset *after* the migration block, before `yield config`.
- **Major #4** (stale docstring) → rewritten; engine is `memory`, file
  count is "43+".
- **Major #5** (`live_surrealdb_async` dead code) → deleted.
- **Minors #6, #7, #9, #10** → all addressed (see self-review for
  details).

### Verification (attempt 2)

End-to-end run with Docker:

```
cd packages/surrealdb-service && uv run pytest -m requires_docker -v
5 passed, 52 deselected, 1 xfailed in 12.58s
(real 24s including container boot — well under 90s budget)
```

Gating without Docker:

```
cd packages/surrealdb-service && uv run pytest -q -m "not requires_docker"
52 passed, 6 deselected in 0.91s
```

App-main regression check:

```
cd apps/app-main && uv run pytest -q
367 passed in 51.38s
```

### New issue surfaced while running end-to-end

`SCHEMAFULL entity` requires `embedding` to be supplied at CREATE time
(migration 39 declares it as `FLEXIBLE TYPE array` with no DEFAULT).
Tests now pass `embedding = []` to mirror production-correct callers.
**Implication for B.1a**: every `EntityRepository.upsert_entity` write
must include `embedding` — keep this in mind when routing
`entity_persistence_service` through the repository.

### Commit hashes (attempt 2)

- `d2342bb` — `fix(surrealdb-service): robust migrations-dir lookup + pool reset`
- `5de7ed8` — `test(surrealdb-service): non-docker safety net for fixture path drift`
- `37bd30f` — `test(surrealdb-service): roundtrip canaries pass end-to-end against real DB`
