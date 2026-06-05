# Review — Track B Phase B.0 attempt 1

**Branch**: `track/b-kg-foundation`
**Decision**: **REVISIONS_NEEDED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-05

## Summary

The phase scope and design are right (testcontainers + canary roundtrip + CI
workflow + xfail for B.1a drift). The implementer's self-review correctly
flagged that the `requires_docker` path is unverified end-to-end, and that
gate was right to flag: the fixture **never runs successfully** in any
environment because the migrations-directory path is off-by-one. CI will
fail loudly on the first PR run for that reason, and several downstream
B-track phases would inherit a broken foundation. Two additional non-trivial
issues (pool-lifecycle around two `asyncio.run()` calls; module docstring
contradicting the actual storage engine) should be resolved before merge.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | `pytest -m requires_docker .../test_migrations_roundtrip.py` boots SurrealDB, applies migrations 1-43, runs roundtrip, tears down ≤ 90s | ❌ | Fixture errors before migrations begin because `_MIGRATIONS_DIR` points at `packages/migrations` (does not exist) — verified locally with a reachable Docker daemon, see Blocker #1. |
| 2 | Roundtrip test asserts INSERT + SELECT entity with default-typed fields | ✅ | `test_entity_roundtrip` (lines 64-102) asserts canonical_name + status default `"active"` + extraction_method default `"llm"` + confidence. Shape is right. |
| 3 | Deliberately broken migration causes clear error | ⚠️ | `_apply_with_diagnostics` (fixtures.py:201-224) wraps with a "likely offending file" hint — design is good. **Not verifiable** because criterion #1 fails first. |
| 4 | CI workflow runs on a PR touching `migrations/44.surrealql` | ⚠️ | Path filter is correct (`migrations/**`). Docker pre-flight is correct. **Will fail** on first run because of Blocker #1. |
| 5 | `TESTCONTAINERS_GUIDE.md` documents marker, opt-out, local-vs-CI | ✅ | Covers marker, gating, local run, CI-trigger paths. The opt-out is documented as "no env variable; skip via `-m 'not requires_docker'`" — reasonable. |

## Test status

```
cd packages/surrealdb-service && uv run pytest -q
...
RuntimeError: Migrations directory not found at /mnt/e/repos/private/open-notebook/packages/migrations.
Has the workspace layout changed?

src/surrealdb_service/testing/fixtures.py:128: RuntimeError
=== short test summary info ===
ERROR tests/test_migrations_roundtrip.py::test_migrations_applied
ERROR tests/test_migrations_roundtrip.py::test_entity_roundtrip
ERROR tests/test_migrations_roundtrip.py::test_entity_alias_roundtrip
ERROR tests/test_migrations_roundtrip.py::test_relation_roundtrip
ERROR tests/test_migrations_roundtrip.py::test_source_roundtrip
45 passed, 1 xfailed, 5 errors in 8.99s
```

(Docker IS reachable in the reviewer's WSL2 environment via the Python SDK,
so the skip path was *not* taken — which is what would happen in CI.)

```
cd apps/app-main && uv run pytest -q
367 passed, 3 warnings in 87.03s
```

## Issues found

### 🔴 Blockers (must fix)

1. **`_MIGRATIONS_DIR` is off-by-one — fixture cannot find migrations** —
   `packages/surrealdb-service/src/surrealdb_service/testing/fixtures.py:48`
   - Issue: `_REPO_ROOT = Path(__file__).resolve().parents[4]` resolves to
     `<repo>/packages/`, not the repo root. `_MIGRATIONS_DIR` is therefore
     `<repo>/packages/migrations`, which does not exist. The comment block
     on lines 45-47 *correctly* shows "../../../../.." (5 `..` segments) but
     `parents[4]` is only four levels up — comment and code disagree, and
     the code is wrong. Reproduction:
     ```
     >>> Path('packages/surrealdb-service/src/surrealdb_service/testing/fixtures.py').resolve().parents[4]
     /mnt/e/repos/private/open-notebook/packages   # WRONG
     >>> ...parents[5]
     /mnt/e/repos/private/open-notebook            # repo root
     ```
   - Impact: every `requires_docker` test errors out (NOT skips) before
     migrations are even attempted, on any environment with a reachable
     Docker daemon — including the CI runner. The acceptance criteria #1,
     #3, and #4 cannot pass in their current form. The self-review's claim
     of "45 passed, 6 skipped" was only true in a no-Docker sandbox; in
     CI it becomes "45 passed, 5 errors, 1 xfailed".
   - Verification: I confirmed locally with `docker.from_env().ping()`
     returning True and the 5 tests erroring (transcript above).

2. **Self-review test summary is misleading** —
   `docs/tracks/B-kg-quality/reviews/phase-B.0-self-review.md:60` and
   `docs/tracks/B-kg-quality/status.md:54`
   - Issue: Self-review reports `requires_docker` tests "skipped 6, gating
     works". That only confirms the *skip* path. The skip path masks
     Blocker #1 because the `_MIGRATIONS_DIR` check (fixtures.py:127-131)
     comes *after* the `pytest.skip` (fixtures.py:120-121). The first
     environment where Docker is reachable surfaces the bug. The status
     line "code complete, local tests green, ready for review" is true
     only because the gate was misinterpreted.
   - Impact: not a code defect per se but the bar for "self-review" is
     that the implementer takes adversarial responsibility for the *first
     real run*. The implementer flagged "first verification gate is CI"
     correctly; what they should have additionally done is at minimum a
     dry-run of `_apply_with_diagnostics` against a real DB — or, since
     no Docker was available, a unit-test that `_MIGRATIONS_DIR.is_dir()`
     resolves to a real directory. (One assertion would have caught
     Blocker #1.)
   - Recommendation: add a tiny non-docker test `def
     test_repo_root_resolves(): assert _MIGRATIONS_DIR.is_dir()` to
     `tests/test_migrations_roundtrip.py` so this class of bug fails fast
     even without Docker.

### 🟡 Major (must fix)

3. **Pool-lifecycle across two `asyncio.run()` invocations is fragile** —
   `packages/surrealdb-service/src/surrealdb_service/testing/fixtures.py:181, 193`
   - Issue: The fixture calls `asyncio.run(_apply_with_diagnostics(manager))`
     (line 181) and then `yield config`. Migrations populate the global
     `connection_module._pool` with a `ConnectionPool` whose pooled
     `AsyncSurreal` connections were created inside that `asyncio.run`'s
     loop — which is closed when `run` returns. The fixture *resets*
     `_pool = None` *before* migrations (line 166) but does NOT reset it
     after, so tests inherit a pool full of connections bound to the dead
     migration-loop. The pool's `_health_check` (connection.py:94-100)
     happens to recover via "dead connection discard + recreate", but only
     because `asyncio.Queue` is loop-agnostic in Python 3.10+. This
     happens to work, but is incidental — any future change in the Queue's
     loop binding semantics (or the pool calling an awaiting method
     before health_check) will break tests with `RuntimeError: loop is
     closed`.
   - Recommendation: reset `connection_module._pool = None` after the
     migration block (after line 185), before `yield config`. One line.
   - Why it matters now: B.1a through B.6 will run many roundtrip tests
     against this fixture. The incidental-recovery path is not
     defensible; a brittle pool that occasionally surfaces "loop closed"
     errors would be hell to debug.

4. **Module docstring contradicts actual storage-engine choice** —
   `packages/surrealdb-service/src/surrealdb_service/testing/fixtures.py:8`
   - Issue: Module docstring claims "Boots it with the same `rocksdb`
     storage + root/root credentials we use in production". Actual
     command on line 138 is `start --log info --user root --pass root
     memory` — i.e. memory storage. Self-review and TESTCONTAINERS_GUIDE
     correctly document `memory`. This single line of stale documentation
     is the kind of thing that bites the next implementer who reads only
     the docstring and assumes parity with prod. Conflict with a clearly
     intentional design choice (memory engine for speed + clean slate).
   - Recommendation: rewrite lines 7-10 to say "memory storage engine for
     fast boot and guaranteed clean state — production uses rocksdb but
     no migration we exercise depends on rocksdb-specific behaviour".
   - Also: line 17 says "a few seconds for ~17 files" but there are 43+
     migrations. Update both numbers.

5. **`live_surrealdb_async` is dead code** —
   `packages/surrealdb-service/src/surrealdb_service/testing/fixtures.py:229-232`
   - Issue: Defined, not exported from `testing/__init__.py`, not used by
     any test in the workspace (`grep -rn live_surrealdb_async` returns
     one hit — the definition itself). The fixture-scope mismatch
     (session-scoped async fixture relying on `asyncio_default_fixture_
     loop_scope = "function"` in `pyproject.toml`) is also questionable.
   - Recommendation: delete. YAGNI; add when the first downstream caller
     actually needs an async-flavoured handle.

### 🔵 Minor (optional)

6. **Type annotation on `live_surrealdb` says `SurrealDBConfig` but the
   function yields** — fixtures.py:106. Should be `Iterator[SurrealDBConfig]`
   or `Generator[SurrealDBConfig, None, None]`. Pytest tolerates this but
   type-checkers don't.

7. **`DOCKER_AVAILABLE` evaluated at module import** — fixtures.py:69.
   Causes a ~2s Docker SDK ping every time the module is imported, even
   for `-m "not requires_docker"` runs. Cheap workaround: lazy-evaluate
   inside the fixture (the function `docker_available()` already exists).

8. **Track A.3 review document landed on this branch** —
   `docs/tracks/A-mineru/reviews/phase-A.3-attempt-1.md` is added in
   commit 97a0a76 ("Track B sprint plan + A.3 review attempt-1 catch-up").
   Scope leak — a Track A artefact in a Track B PR. Not a defect, but next
   time keep PRs clean to a single track to make `git log --oneline
   docs/tracks/<track>` accurate per-track.

9. **F-string interpolation of record IDs in `entity_alias` and `relation`
   tests** — test_migrations_roundtrip.py:124-127, 162-167. Works because
   `parse_record_ids` returns a plain str like `"entity:abc123"`, but
   reads as a SQL-injection-y pattern when the rest of the suite uses
   `$param` placeholders. Prefer:
   `await execute_query("RELATE $a->relation->$b SET ...", {"a": a_id,
   "b": b_id}, ...)` — keeps the style consistent and removes the
   "what if a record-ID parsed by surrealdb-py ever contains a colon
   or quote" foot-gun.

10. **Roundtrip migrations-applied test uses `max(versions) >= 43`** —
    test_migrations_roundtrip.py:51. Correct for the present moment, but
    when B.1a adds migration 44, B.1b adds 45, etc., every later phase
    *should* be bumping the floor. Recommend a helper that asserts every
    integer in `range(1, latest+1)` is in the version set — catches the
    "silently skipped middle migration" case the self-review flagged as
    a known risk (line 44-50). Out of scope for B.0 if you want to keep
    the PR tight, but worth noting for B.1.

## Decision rationale

This is a foundation phase. The fixture **doesn't actually do its job in
any environment with Docker present** (Blocker #1) — the CI run will
visibly fail on this PR, which validates the self-review's "first CI run
is the validation gate" stance but turns the gate into a "definite NO" for
the harness as written. Blocker #1 alone forces REVISIONS_NEEDED; Majors
#3 and #4 compound the risk that the fix is single-line and gets merged
without exercising the pool/storage-engine concerns.

The xfail design (#6 in the self-review) is exactly right — strict=True,
correct file:line citation, accurate field-shape mirroring of the legacy
write path. That part should survive the revision.

## Next steps

Implementer should:

1. Change `parents[4]` to `parents[5]` (or rebuild via a more durable
   strategy, e.g. walking up until `pyproject.toml` with
   `[tool.uv.workspace]` is found). Add a non-docker smoke test asserting
   `_MIGRATIONS_DIR.is_dir()` — catches off-by-one without needing
   Docker (addresses Blocker #1 + #2).
2. Reset `connection_module._pool = None` after the migration block,
   before `yield config` (Major #3).
3. Fix module docstring + migration-count text (Major #4).
4. Delete `live_surrealdb_async` (Major #5).
5. Optionally address minors.
6. Re-run `pytest` in an env with reachable Docker; confirm
   `45 passed, 6 passed, 0 errors` (or whatever the eventual numbers are,
   including the xfail rolling through cleanly).
7. Re-submit for review.
