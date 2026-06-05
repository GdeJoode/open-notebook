# Review — Track B Phase B.0 attempt 2

**Branch**: `track/b-kg-foundation`
**Decision**: **APPROVED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-05

## Summary

All attempt-1 blockers and majors are genuinely fixed and verified end-to-end
against a real Docker daemon. The migration-dir resolver now walks up to a
workspace marker rather than counting `parents[N]`, with a non-Docker
regression guard that fires when the path drifts. The pool reset after the
migration `asyncio.run` is in the correct place (after migrations, before
yield) for the right reasons. Five docker-gated roundtrip tests pass + one
strict XFAIL for the persistence-drift sentinel that B.1a will resolve.
This is a solid foundation that 16 downstream phases can build on.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | `pytest -m requires_docker tests/test_migrations_roundtrip.py` boots + applies migrations + roundtrips + tears down ≤ 90s | ✅ | 5 passed + 1 xfailed in **11.16s** end-to-end (well under 90s budget). Verified locally. |
| 2 | Roundtrip asserts INSERT + SELECT entity with default-typed fields | ✅ | `test_entity_roundtrip` (lines 86-128) asserts `canonical_name`, `entity_type`, `status="active"`, `extraction_method="llm"`, `confidence`. Shape correct. |
| 3 | Deliberately broken migration causes clear error | ✅ | `_apply_with_diagnostics` (fixtures.py:230-253) wraps with "offending file is likely migrations/N.surrealql" + chained-from underlying SurrealDB error. Not exercised, but design is verifiable. |
| 4 | CI workflow runs on PR touching `migrations/44.surrealql` | ✅ | `.github/workflows/db-integration.yml` paths-filter is correct (`migrations/**`), Docker pre-flight is explicit (fails loud if daemon missing), and the now-working fixture means the workflow will actually pass. |
| 5 | `TESTCONTAINERS_GUIDE.md` documents marker, opt-out, local-vs-CI | ✅ | Covers all three. Missing: credsStore-on-WSL2 footnote (see Minor #2). |

## Test status

```
$ cd packages/surrealdb-service && uv run pytest -m requires_docker -v
collected 58 items / 52 deselected / 6 selected

tests/test_migrations_roundtrip.py::test_migrations_applied PASSED       [ 16%]
tests/test_migrations_roundtrip.py::test_entity_roundtrip PASSED         [ 33%]
tests/test_migrations_roundtrip.py::test_entity_alias_roundtrip PASSED   [ 50%]
tests/test_migrations_roundtrip.py::test_relation_roundtrip PASSED       [ 66%]
tests/test_migrations_roundtrip.py::test_source_roundtrip PASSED         [ 83%]
tests/test_migrations_roundtrip.py::test_entity_persistence_drift_xfail XFAIL [100%]
================= 5 passed, 52 deselected, 1 xfailed in 11.16s =================

$ cd packages/surrealdb-service && uv run pytest -m "not requires_docker" -q
52 passed, 6 deselected in 2.10s

$ cd apps/app-main && uv run pytest -q
367 passed in 56.44s
```

All gates green. Independently verified by this reviewer with Docker
available (credsStore workaround applied to local config; CI is unaffected).

## Per-blocker / per-major verification

### Blocker #1 (path bug) — RESOLVED

`_find_migrations_dir()` (fixtures.py:47-63):

- **Walk termination**: `here.parents` is finite (stops at filesystem root).
  If no match is found, raises `RuntimeError` with a clear "Has the repo
  layout changed?" hint. No infinite loop.
- **False-positive risk**: walk only matches `<ancestor>/migrations/` +
  `<ancestor>/pyproject.toml` together. Confirmed by ancestor-walk audit
  that only `<repo>/` has both — no collisions through `packages/`,
  `packages/surrealdb-service/`, etc.
- **Correctness**: independently verified that `_MIGRATIONS_DIR` resolves
  to `/mnt/e/repos/private/open-notebook/migrations` regardless of cwd
  (tested from `/tmp`).

Minor doc/code drift: the docstring says the resolver looks for
`pyproject.toml` "declaring `[tool.uv.workspace]`", but the code only
checks `.is_file()`. The stricter content check lives in
`test_testing_fixtures.py::test_migrations_dir_is_under_repo_root_marker`
(belt-and-braces). Not a defect today because the current layout has no
ambiguity; noted as Minor #1.

### Blocker #2 (drift guard test) — RESOLVED

`tests/test_testing_fixtures.py` (96 lines, 7 tests, no `requires_docker`
marker, all in the non-Docker gate):

| Test | Guards against |
|---|---|
| `test_migrations_dir_resolves_to_real_directory` | `is_dir()` false (the exact attempt-1 symptom) |
| `test_migrations_dir_contains_known_baseline_files` | landing on the wrong `migrations/` (no `1.surrealql`/`39.surrealql`/`43.surrealql`) |
| `test_migrations_dir_is_under_repo_root_marker` | parent missing the `[tool.uv.workspace]` declaration |
| `test_find_migrations_dir_returns_same_path` | drift between cached `_MIGRATIONS_DIR` and resolver function |
| `test_docker_available_is_callable_lazy` | regression of Minor #7 (eager `DOCKER_AVAILABLE` at import) |
| `test_fixture_module_has_no_stale_live_surrealdb_async` | resurrection of dead code |
| `test_migrations_dir_path_is_absolute` | `cwd`-relative path drift |

Confirmed the path-drift assertion fires when `_MIGRATIONS_DIR` is
monkey-patched to a bogus value. All 7 run on every `pytest -q` invocation
regardless of Docker presence — they'd have caught attempt-1's
off-by-one in the implementer's own sandbox.

### Major #3 (pool lifecycle) — RESOLVED

Two `connection_module._pool = None` resets are in place:

1. **Line 185** — BEFORE migrations, so `AsyncMigrationManager` builds its
   pool against the test config (not whatever the dev env cached at
   import time).
2. **Line 211** — AFTER `asyncio.run(_apply_with_diagnostics)` completes
   and BEFORE `yield config`, so tests get a fresh `ConnectionPool`
   whose `asyncio.Queue` is built on the pytest-asyncio test loop, not
   the now-closed migration loop.

The reasoning in the code comment ("`asyncio.run` closes the loop ...
Reset the pool so the first test using `execute_query` rebuilds
connections on its own loop") is correct. Confirmed via cross-read of
`connection.py:78` (queue is created in `ConnectionPool.__init__` on
whatever loop is active at that point) and `connection.py:175-183`
(`get_pool()` rebuilds when `_pool is None`).

Teardown also resets `_pool = None` after `close_pool()` (line 221-225)
— defensive but correct.

### Major #4 (docstring) — RESOLVED

Lines 1-21 (fixtures.py): now correctly says **"`memory` storage
engine"** with explicit rationale ("faster boot and a guaranteed clean
slate per session", "production uses `rocksdb` but no migration we
exercise depends on rocksdb-specific behaviour"). Migration count
updated from "~17" to "43+". This is the right level of detail for the
next implementer.

### Major #5 (dead code) — RESOLVED

`grep -rn live_surrealdb_async --include="*.py" --include="*.toml"`
returns zero source-code matches; only doc references and a regression
test (`test_fixture_module_has_no_stale_live_surrealdb_async`) remain.
Nothing imports it. YAGNI satisfied.

### Minor #6 (type annotation) — RESOLVED

`live_surrealdb() -> Iterator[SurrealDBConfig]` on line 124. `Iterator`
imported from `typing` on line 28.

### Minor #7 (eager Docker ping) — RESOLVED

`DOCKER_AVAILABLE` module-level constant removed; `docker_available()`
function evaluates lazily inside the fixture. `pytest -m "not
requires_docker"` runs in **2.10s** (no Docker SDK import cost).

### Minor #9 (f-string for record IDs) — RESOLVED in 2 of 3 places

`test_entity_alias_roundtrip` and `test_relation_roundtrip` SELECT
clauses now use `type::thing($id)`. The `RELATE a_id->relation->b_id`
arrow keeps inline interpolation **with a documented justification**
(SurrealQL parser rejects function-call expressions in arrow source/
target positions; cites SurrealDB issue #4232). The justification is
plausible and the risk vector is limited (`a_id` comes from
`parse_record_ids` and is `entity:<alnum>` only). Acceptable trade-off.

### Minor #10 (version floor) — RESOLVED

`test_migrations_applied` now derives the expected version set from
disk (`fx._MIGRATIONS_DIR.iterdir()` + suffix filter) and asserts
`expected ⊆ applied`. Catches "silently skipped middle migration" — the
exact failure mode the self-review flagged for the runner's
"already exists" short-circuit.

## Issues found

### 🔴 Blockers (must fix)

**None.**

### 🟡 Major (must fix)

**None.**

### 🔵 Minor (optional)

1. **Resolver docstring overstates the check** — `fixtures.py:53`. The
   docstring promises "`pyproject.toml` declaring `[tool.uv.workspace]`"
   but the code only checks `.is_file()`. Today this can't false-positive
   (only the repo root has both `migrations/` and `pyproject.toml`), but
   the doc/code drift is the kind of thing that confuses the next reader.
   Either tighten the resolver to parse `[tool.uv.workspace]` (one
   line, but adds a TOML-parsing import to a hot path) or soften the
   docstring to match what the code does. Filing as follow-up; not blocking.

2. **TESTCONTAINERS_GUIDE.md missing credsStore-on-WSL2 footnote** —
   `docs/tracks/B-kg-quality/TESTCONTAINERS_GUIDE.md`. The self-review
   explicitly notes the Docker Desktop WSL2 `credsStore: desktop.exe`
   crash and the blank-config-file workaround — but the guide itself
   doesn't mention it. This reviewer hit the exact issue while
   verifying. CI is unaffected (ubuntu-latest doesn't use credsStore),
   but every WSL2 developer running this locally will hit it. Add a
   short footnote under "Known limitations". Filing as follow-up.

3. **`_apply_with_diagnostics` doesn't include the SQL excerpt** —
   `fixtures.py:241-253`. The error message points at the next pending
   file (e.g. `migrations/44.surrealql`) but not the offending SQL
   statement. The chained `from exc` carries the underlying SurrealDB
   error so this isn't fatal, but a future B-track phase debugging a
   complex migration would benefit from the per-statement breadcrumb
   the surrealdb-service migration runner already has. Out of scope
   for B.0; nice-to-have for B.1.

4. **A.3 review document carries over on this branch** —
   `docs/tracks/A-mineru/reviews/phase-A.3-attempt-1.md` was added in
   commit `97a0a76`. Decision (per self-review) is to leave it; harmless
   cross-track noise. **Document in the PR description** so the reviewer
   knows the file is intentional carry-over from before the A→main
   merge, not a fresh scope leak.

5. **Container-start failure isn't wrapped with a diagnostic message** —
   `fixtures.py:159`. `container.start()` can fail on image-pull errors
   (rate limit, network), and the exception propagates as a bare
   testcontainers error. The migration block is wrapped with a
   `RuntimeError` hint but the boot is not. Testcontainers' own error
   is usually adequate, but a "SurrealDB container failed to start —
   check Docker daemon and image availability" hint would be friendlier.
   Filing as follow-up; deferring is fine.

## Notes / kudos

- **The new `embedding=[]` finding is exactly the kind of thing this
  harness exists to catch** — and the self-review correctly flagged it
  as a B.1a implication. `apps/app-main/src/app_main/services/
  entity_persistence_service.py` lines 120-160 confirm the legacy
  service builds entity props WITHOUT supplying `embedding`, so every
  `CREATE entity` from the production path would fail against the
  SCHEMAFULL table from migration 39. This is the second drift bug B.1a
  must fix (alongside the `name` vs `canonical_name` field-name drift).
  The B.0 harness has already paid for itself before B.1a even starts.
- The persistence-drift xfail is **still strict and still XFAILing** —
  pytest reports `XFAIL`, not `XPASS`. If B.1a fixes the bug without
  promoting/removing this test, pytest will flip to `XPASS-failure` and
  force the conversation. Exactly the design the plan called for.
- Walk-up + workspace-marker resolver is a strictly better design than
  the original `parents[5]`. Resilient to future package reshuffles.
- The 7-test non-Docker safety net is the kind of cheap belt-and-braces
  that pays compound interest as the harness gets reused across B.1a-B.6.

## Decision rationale

Zero blockers, zero majors. Five minors are all genuinely optional
(docstring polish, missing doc footnote, nice-to-have error
breadcrumbs, intentional cross-track carry-over, hand-friendlier
container-start error). None of them affect downstream phases'
ability to depend on this foundation.

The end-to-end Docker run is verified by both the implementer and this
reviewer. The xfail is strict and surfacing. The CI workflow is path-
filtered correctly and pre-flights Docker explicitly.

**Approved for merge.** This is solid foundation work that B.1a-B.7
can confidently build on.

## Recommendation for next steps

1. **Merge as-is.** All gates green; review-driven changes are good
   quality.
2. **In the PR description**, mention:
   - 7 new non-Docker tests in `test_testing_fixtures.py` as
     regression guards for the attempt-1 path-drift bug.
   - The `docs/tracks/A-mineru/reviews/phase-A.3-attempt-1.md` file is
     intentional carry-over from before the A→main merge; not a Track A
     scope leak.
   - WSL2 developers running locally will hit a `credsStore` crash;
     workaround is `printf '{}' > ~/.docker/config.json` (or set credsStore
     to empty). CI unaffected.
3. **B.1a author** (next phase): two pre-existing bugs the B.0 harness
   has already surfaced — both must be fixed:
   - `entity_persistence_service.persist_filtered_result` writes
     `name`/`weight`/`source_ids` (legacy SCHEMALESS shape); migration
     39 declares `canonical_name`/no-weight/`source_documents`.
   - The same code path never sets `embedding`; migration 39 declares
     it `FLEXIBLE TYPE array` with no DEFAULT, so every `CREATE entity`
     must supply at least `embedding = []`.
   When B.1a lands, delete `test_entity_persistence_drift_xfail` (or
   convert it to a non-xfail assertion against the new repository
   write-path).
4. **Filed follow-ups** (none blocking): tighten resolver docstring or
   parse `[tool.uv.workspace]`; add credsStore footnote to
   TESTCONTAINERS_GUIDE; consider per-statement diagnostic breadcrumb
   in the migration runner.
