# Phase B.1b — self-review

> Author: implementer agent, 2026-06-05
> Branch: `track/b-models-notebook-schema`
> Commits: `5fc4859` → `997ad8f`

## Acceptance criteria check

| # | Criterion | Verified? |
|---|---|---|
| 1 | Migration 45 applies on fresh DB AND is idempotent on re-run | YES — covered end-to-end by `test_migration_45_recorded` (records the version row) and `test_migration_45_is_idempotent` (replays every DEFINE statement against a populated DB and asserts `_sbl_migrations` still has exactly one row for #45). Verified with the live container; both pass. |
| 2 | `NotebookSchemaRepository.upsert(...)` round-trips correctly via B.0 fixture | YES — `test_notebook_schema_upsert_roundtrip` covers a fully-populated `NotebookSchema` (accepted + pending extensions, coverage_pct, review toggles, metadata bag) and asserts each field is round-tripped. |
| 3 | `Pass1ResultRepository.record(...)` inserts and `list_by_source(...)` reads it back | YES — `test_pass1_result_record_and_list` records two rows for one source and asserts `list_by_source` returns both, newest-first; `list_by_notebook` is also covered. |
| 4 | UNIQUE index on `notebook_schema.notebook` prevents duplicates | YES, **with rewrite-semantic on the repository's upsert** (see decision below). `test_notebook_schema_unique_index_rewrites` proves the second upsert returns the same record id and the DB still holds exactly one row. `test_notebook_schema_unique_index_blocks_direct_create` proves a raw `CREATE notebook_schema` bypassing the repo *does* raise — the DB-level UNIQUE constraint is intact. |
| 5 | Pre-existing `EntityExtractionService` continues to work — regression guard | YES — pre-existing `test_entity_roundtrip`, `test_entity_alias_roundtrip`, `test_relation_roundtrip`, `test_source_roundtrip`, and `test_entity_persistence_drift_xfail` all behave as before (4 pass, 1 xfail) when migrations now run through #45. No code in `EntityExtractionService` was touched. |

## UNIQUE-index collision decision

`notebook_schema.notebook` is `UNIQUE`. We had three choices for the repository:

1. **Let the DB raise** on the second `CREATE`. Caller would catch and retry as an UPDATE — error-flow business logic, ugly.
2. **Use SurrealDB's `UPSERT`** keyword. SurrealDB v2's `UPSERT` matches on the record id, not on an arbitrary unique field — so it would still need a SELECT-by-notebook to discover the id first.
3. **Repository chooses: SELECT-by-notebook → UPDATE (existing) or CREATE (none)**. Returns the same record id on subsequent upserts, callers never see the unique-violation error.

We chose **option 3 (rewrite-semantic)** because:

- Callers (B.1c) want a "make-this-schema-current" verb, not a primary-key shuffle. Rewrite-semantic matches the conceptual model.
- The UNIQUE index is still active as a safety net for any caller that bypasses the repository (proven by `test_notebook_schema_unique_index_blocks_direct_create`).
- Test `test_notebook_schema_unique_index_rewrites` asserts the second upsert returns the *same* record id (proving UPDATE path, not CREATE) and the DB still has count=1.

Documented in `notebook_schema.py` near `upsert(...)` and in the test docstring.

## Files added / modified

| Path | Change |
|---|---|
| `migrations/45.surrealql` | NEW — two SCHEMAFULL tables with FLEXIBLE extension bags |
| `migrations/45_down.surrealql` | NEW — `REMOVE TABLE IF EXISTS …` for both |
| `packages/shared/src/shared/models/notebook_schema.py` | NEW — `NotebookSchema` + `Pass1Result` |
| `packages/shared/src/shared/models/__init__.py` | MODIFIED — additive (export both) |
| `packages/shared/tests/test_notebook_schema_model.py` | NEW — 11 unit tests |
| `packages/surrealdb-service/src/surrealdb_service/repositories/notebook_schema.py` | NEW — two repositories |
| `packages/surrealdb-service/src/surrealdb_service/repositories/__init__.py` | MODIFIED — additive (export both) |
| `packages/surrealdb-service/tests/test_notebook_schema_repo_roundtrip.py` | NEW — 10 requires_docker tests |

The two `__init__.py` modifications are purely additive; B.1a's branch (`track/b-models-entity`) adds `Entity` / `Relation` exports in the same files. The merge is expected to be a clean additive merge — neither branch removes lines the other added.

## Honest trade-offs / things to watch

- **`UPDATE type::thing($id)` rather than f-string interpolation.** `BaseRepository.update` uses f-string. Both are safe (record ids from `parse_record_ids` are colon+alnum), but parametrised is the better default. Not changing `BaseRepository` here to keep the blast radius small; recommend a follow-up to standardise.
- **`add_pending_extension` requires the schema row to exist** (returns `False` otherwise). Pending extensions are only meaningful once the notebook has chosen a base ontology — failing soft and logging a warning is the right semantic. Tested in `test_notebook_schema_pending_extension_lifecycle`.
- **Coverage bounds are enforced in Pydantic** (`ge=0.0, le=1.0`), not in SurrealDB. A direct DB write *could* insert an out-of-bounds value; we rely on the repository for validation. Consistent with how `Source.metadata` is policed (Pydantic-side).
- **No secondary index on `pass1_results.notebook`.** The plan asked for `idx_pass1_source` only; `list_by_notebook` is currently a table scan. Add when notebook-scoped listing latency becomes a problem (likely never with the expected data volume — pass-1 history per notebook is small).

## Test results

```
packages/shared:
  before:  105 passed
  after:   116 passed (+11)
  command: cd packages/shared && uv run pytest -q

packages/surrealdb-service (non-docker):
  before:  43 passed (pytest-asyncio was missing in initial baseline; after
           uv sync --extra dev, baseline was 52 passed)
  after:   52 passed, 16 deselected
  command: cd packages/surrealdb-service && uv run pytest -m "not requires_docker" -q

packages/surrealdb-service (requires_docker):
  before:  5 passed, 1 xfailed
  after:   15 passed, 1 xfailed (+10 new)
  command: cd packages/surrealdb-service && uv run pytest -m requires_docker -v

apps/app-main:
  baseline: 367 (per plan)
  after:    367 passed in 137.90s — no regressions
  command:  cd apps/app-main && uv run pytest -q
```

## Coordination flags for the reviewer

- B.1a (`track/b-models-entity`) is in flight on its own branch. Both branches touch `packages/shared/src/shared/models/__init__.py` and `packages/surrealdb-service/src/surrealdb_service/repositories/__init__.py`. Our changes there are *purely additive* (new imports + new entries in `__all__`); the merge should be a clean three-way without conflicts beyond textual interleaving.
- No production code outside the two `__init__.py` files was touched. `EntityExtractionService` is unaffected.
- `pyproject.toml` / `uv.lock` were *not* modified.
