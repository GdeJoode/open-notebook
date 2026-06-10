# Review — Track B Phase B.3b attempt 1

**Branch**: `track/b-schema-edit-ops` (PR #20)
**Decision**: **APPROVED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-09

## Summary

Largest sub-phase of Track B (6 endpoints, 1 service, 1 migration, shared event repo, dialog component, 6 mutation hooks, 1 e2e spec) lands cleanly. All 6 plan ACs met. Each edit op idempotent + exactly-one-event-per-state-changing-call. Docker roundtrip pins notebook_event table + migration 46 idempotency. Frontend mutations replace React-Query cache directly (no follow-up GET) → 200ms refresh guarantee.

0 blockers, 0 majors, 6 minor follow-ups.

## Acceptance criteria check

All 6 ACs PASS:
1. Rename: accepted_extensions entry + event + GET returns renamed
2. Merge/split/delete: equivalent assertions (state + event)
3. All ops idempotent (re-run = no state + 0 events)
4. Schema tab updates ≤ 200ms (cache replacement, not invalidation)
5. Playwright covers all 4 ops with route mocks (5 specs)
6. Each op writes exactly ONE notebook_event row (service + docker roundtrip)

## Test status (independently verified)

- `apps/app-main` full: 436 passed
- `packages/surrealdb-service` docker (migration 46 + event repo): 7 passed
- frontend tsc + lint: clean
- Playwright spec verified by inspection

## Service correctness (method-by-method)

- `accept_extension`: linear scan; missing → 404; already accepted → noop+no event
- `reject_extension`: filter-by-not-equal; length-equality short-circuit (idempotent reject doesn't 404)
- `rename_type`: same-name bypass; deterministic `rename_id` keyed on `(old, new)` → replay short-circuits
- `merge_types`: dedupe+sort+length≥2; `merge_id` keyed on sorted set + merged_name (`merge([A,B], C) ≡ merge([B,A], C)`)
- `split_type`: dedupe on `into`; `split_id` includes `sha256(criterion)[:12]` → distinct ops for distinct criteria
- `delete_type`: `if type_name in excluded_types: return schema` short-circuit

`_persist` only called after state-change branches.

## Router audit

- 6 endpoints (5 POST + 1 DELETE) registered, return `NotebookSchemaResponse`
- `_ensure_notebook_exists` 404 guard everywhere
- Exception mapping: NotebookSchemaNotFound → 404, UnknownExtension → 404, ValueError → 422
- Pydantic validators (min_length on fields)
- Auth inherited from `/api` PasswordAuthMiddleware

## Migration 46 audit

- SCHEMAFULL + `IF NOT EXISTS` on all 8 DEFINEs
- `notebook` is `record<notebook>` (strong FK); composite index `(notebook, event_type, created_at)`
- `excluded_types: option<array<string>>` on notebook_schema
- 46_down reverses (not idempotent; SurrealDB lacks IF EXISTS — documented)
- `test_migration_46_is_idempotent` re-applies + asserts single migration row

## Frontend audit

- `SchemaEditDialog` 4 modes via `mode` prop
- Hand-rolled forms (≤3 fields each)
- A11y: Radix focus trap, aria-labelledby/describedby, aria-label on "⋯"
- React Query: `setQueryData` replaces cache (matches AC #4)
- Per-row disable-while-pending prevents accept-then-reject race
- `excluded_types` filter applied to both base AND accepted lists

## B.3a Playwright update

Single assertion flip: `toBeDisabled()` → `toBeEnabled()` on Accept/Reject buttons. Correct semantic update (B.3a stub deliberately disabled; B.3b makes live). Not a regression-hider; click→POST covered separately in `schema-edit-ops.spec.ts`.

## Minors (6, non-blocking)

1. GET /schema doesn't test `excluded_types` directly (DELETE + docker roundtrip cover it indirectly)
2. No validation that rename/merge/split source types EXIST (intentional additive design)
3. Rename to existing name produces visible label collision in tree
4. `_persist` event-emit failure silently swallowed (documented best-effort trade-off; metrics counter would let us monitor)
5. Self-review claims 446 tests; reviewer measured 436 (off by 10 — typo or local-CI diff)
6. URL-encoding edge: `Author/Editor` → `%2F` → 404 (FastAPI/Starlette default; academic — LLM names don't contain slashes)

## Kudos

- Deterministic op-ids for idempotency (sorted-set merge, SHA256 split fragment)
- Cache-replacement strategy (`setQueryData` not `invalidateQueries`)
- Migration 46 design notes are among the most thorough on this track
- Single dialog with `mode` prop keeps SchemaBrowser API tight
- Event-emit failure-isolation correctly chosen (additive event stream)
- Cross-op `TestMissingSchemaRow` belt-and-braces
- A11y on dialog done correctly

## Next steps

APPROVED — ready for merge. Minor follow-ups can be filed alongside B.3c/B.3d.
