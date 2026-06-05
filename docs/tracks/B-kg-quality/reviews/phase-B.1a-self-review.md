# Phase B.1a — Self-review

**Branch**: `track/b-models-entity`
**Commits**: `445c072`, `c0127f7`, `c459fe8`
**Date**: 2026-06-05

## Plan acceptance criteria — status

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Migration 44 applies on fresh DB and is idempotent | PASS | `requires_docker` suite applies all 29 migrations including 44; `IF NOT EXISTS` makes re-run a no-op (mirrors migration-43 pattern) |
| 2 | `Entity(...).model_dump()` round-trips through SurrealDB | PASS | `test_upsert_creates_entity_with_type_tags` + `test_upsert_merges_on_second_call` assert values verbatim after persist + read |
| 3 | `EntityExtractionService.run_extraction()` still ends with `entity` rows | PASS | `app-main` test suite is 368 passing (baseline 367 + 1 new alignment guard); persistence service tests cover the upsert + relation paths |
| 4 | `name`/`canonical_name` drift fixed (Q-B-1 default: fix in this PR) | PASS | Service rewritten to call `EntityRepository.upsert_entity(Entity(...))`; relation block also fixed (`canonical_name` lookup, `source_documents` array write) |
| 5 | `test_entity_persistence_drift_xfail` flips to passing | PASS | Renamed to `test_entity_persistence_field_alignment`, xfail marker removed, now exercises `EntityRepository.upsert_entity` end-to-end + asserts legacy shape IS still rejected |
| 6 | New `test_entity_repository_roundtrip.py` green under B.0 harness | PASS | 3/3 docker-gated tests pass (`test_upsert_creates_entity_with_type_tags`, `test_upsert_merges_on_second_call`, `test_upsert_handles_empty_embedding`) |

## Quality gates

```
cd packages/shared && uv run pytest -q
  116 passed in 2.30s

cd packages/surrealdb-service && uv run pytest -m "not requires_docker" -q
  52 passed, 9 deselected in 2.80s

cd packages/surrealdb-service && uv run pytest -m requires_docker -v
  9 passed, 52 deselected in 6.35s
  Tests:
    - test_migrations_applied
    - test_entity_roundtrip
    - test_entity_alias_roundtrip
    - test_relation_roundtrip
    - test_source_roundtrip
    - test_entity_persistence_field_alignment   (formerly xfail — flipped)
    - test_upsert_creates_entity_with_type_tags (new)
    - test_upsert_merges_on_second_call         (new)
    - test_upsert_handles_empty_embedding       (new)

uv run pytest apps/app-main/tests -q
  368 passed in 57.29s  (baseline 367)
```

## Design notes / decisions

1. **Python-side merge over SurrealQL.** Initial implementation used the
   `LET $existing = ...; IF/UPDATE/CREATE` block with `object::extend`,
   `array::union`, `math::max`. Hit `Parse error: Invalid function/constant
   path` on `object::extend` against SurrealDB v2.x — the function isn't
   available. Refactored to pre-fetch the existing row (one short SELECT),
   merge in Python, then issue a single UPDATE-or-CREATE. Semantics
   identical, contract preserved. The repository docstring documents the
   rationale so future maintainers don't put it back.

2. **`FLEXIBLE TYPE array` required for `type_tags`.** First attempt used
   plain `TYPE array DEFAULT []`. The CREATE succeeded but `type_tags`
   read back as `[]` — SurrealDB SCHEMAFULL silently coerces non-typed
   arrays. Switched to `FLEXIBLE TYPE array DEFAULT []` (mirrors how
   migration 39 declares `source_documents`, `provenance_chain`,
   `embedding`). Documented in status.md as a guideline for future
   migrations.

3. **`embedding` left as required on the model.** Tempting to default it
   to `[]` so callers can omit it, but the SCHEMAFULL column has no DB
   default — making it implicit would mask the real contract. Pydantic
   default of `default_factory=list` is fine for type safety; the
   docstring is explicit that callers MUST pass it. Persistence service
   does this.

4. **Relation drift also fixed.** Wasn't called out separately in the
   plan, but the lookup-by-name in the relation block had the same
   `name`/`canonical_name` drift PLUS wrote `source_id` (scalar) instead
   of `source_documents` (array). Same root cause; fixing it now keeps
   the canary honest for B.1b's relation work.

5. **Existing persistence-service unit tests refactored, not deleted.**
   The 7 tests in `test_entity_persistence_service.py` mocked
   `execute_query` and asserted on the params dict. After the refactor,
   entity writes go through the repository; the tests now patch a
   `MagicMock` repo and assert on the `Entity` model passed to
   `upsert_entity`. Added one new test
   (`test_uses_canonical_schema_field_names`) as a guard against
   re-introducing legacy field names.

## Outstanding warnings / risks

- The `tests/test_migrations.py` suite has 9 pre-existing failures
  ("async def functions are not natively supported") — that's a
  pytest-asyncio plugin/install quirk that exists on `main` and is
  outside the scope of this PR. **Not introduced by B.1a**. The dev-deps
  installation order matters — running `uv sync --extra dev` from the
  package directory leaves dev tools out; running it from the workspace
  root (or `uv sync --all-packages --extra dev`) installs them. Worth a
  small CI doc note in a follow-up phase.

- `EntityRepository.upsert_entity` does NOT touch `embedding` on update
  — by design, since updates today never carry a fresh vector. If B.1b
  introduces an embedding-refresh flow, the merge code will need a
  branch. Documented in the repository docstring.

## Ready for review.
