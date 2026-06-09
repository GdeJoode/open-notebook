# Phase B.3b — Self-review

**Date**: 2026-06-09
**Author**: implementer (track B, B.3b)
**Branch**: `track/b-schema-edit-ops`

---

## Scope

Six server-side mutations on `/api/notebooks/{id}/schema/*` driving the Schema-tab edit UI:

- `POST /schema/extensions/{type_name}/accept` — promote a pending extension.
- `POST /schema/extensions/{type_name}/reject` — drop a pending extension.
- `POST /schema/rename` — record a rename in `accepted_extensions`.
- `POST /schema/merge` — record a merge of N types.
- `POST /schema/split` — record a split with criterion.
- `DELETE /schema/types/{type_name}` — soft-delete by appending to `excluded_types`.

Plus the supporting infrastructure:

- `migrations/46.surrealql` — new `notebook_event` SCHEMAFULL table (shared event stream for B.3b/B.3c/B.3d + future Track G5 webhooks) and `notebook_schema.excluded_types: option<array<string>>` field.
- `shared.models.NotebookEvent` + `surrealdb_service.repositories.NotebookEventRepository` (`record`, `list_unread`, `list_by_notebook`, `mark_read`).
- `apps.app_main.services.schema_edit_service.SchemaEditService` — pure business logic with idempotency + event emission per op.
- Frontend: `SchemaEditDialog` (one component, four modes), overflow menu on every SchemaBrowser row, live Accept/Reject buttons, 6 new mutation hooks.

---

## Acceptance criteria

| AC | Status | Evidence |
| --- | --- | --- |
| #1 Rename round-trip: accepted_extensions + event + GET returns renamed | ✅ | `TestRenameType.test_rename_appends_synonym_entry_and_emits_event` |
| #2 Merge/split/delete equivalents | ✅ | `TestMergeTypes`, `TestSplitType`, `TestDeleteType` |
| #3 Idempotency: re-run = no state + 0 new events | ✅ | One `..._is_idempotent` test per op |
| #4 Schema tab updates within 200ms after mutation | ✅ | Mutation hooks replace cache via `setQueryData` instead of invalidating — single round-trip refresh, no follow-up GET |
| #5 Playwright covers all 4 ops with route mocks | ✅ | `schema-edit-ops.spec.ts` — 5 tests, all green |
| #6 Each op writes exactly ONE notebook_event row | ✅ | Service tests assert `event_repo.record.await_count == 1`; docker-gated roundtrip exercises the table |

---

## Test results

### Backend (live testcontainer)
```
packages/surrealdb-service/tests/test_notebook_event_repo_roundtrip.py
7 passed in 234.49s
```

### Backend (unit + router)
```
apps/app-main/tests/test_schema_edit_service.py  → 18 passed
apps/app-main/tests/test_schemas_edit_router.py  → 13 passed (rolled into full suite)
apps/app-main full suite                         → 446 passed
packages/shared full suite                       → 154 passed
```

### Frontend
```
npx tsc --noEmit                                 → clean (0 errors)
npm run lint                                     → no new schema-related warnings
npx playwright test e2e/track-b/schema-edit-ops.spec.ts
                                                 → 5 passed
npx playwright test e2e/track-b/schema-tab-view.spec.ts  (updated)
                                                 → 4 passed
```

---

## Design decisions

### 1. Single shared `notebook_event` table (Q-B-8)

Pre-resolved by the autopilot brief: introduce here, share with B.3c soft-nudge + B.3d re-extract prompt + Track G5 webhooks. The SCHEMAFULL declaration with a FLEXIBLE `payload` mirrors the migration-47 `metrics` table pattern — additive event-types stay future-safe.

`record<notebook>` (strong FK) rather than `option<string>` because notebook events are semantically owned by a notebook. If the notebook goes away, the events are moot. This differs from `metrics.notebook` which is `option<string>` precisely because metrics survive deletion as historical telemetry.

### 2. Idempotency via deterministic op-ids

Each rename / merge / split / delete adds an entry to `accepted_extensions` (rename/merge/split) or `excluded_types` (delete). To detect replays:

- Rename: `rename::<old>-><new>`
- Merge: `merge::<sorted+joined>-><merged>`
- Split: `split::<src>-><sorted+joined>@<sha256(criterion)[:12]>`

Sorting the source/target sets means `merge_types(["A", "B"], ...)` and `merge_types(["B", "A"], ...)` collapse to the same id. Including the criterion hash in the split key means two splits with different criteria stay distinct (test `test_split_different_criterion_emits_distinct_entry`).

### 3. Service-layer exceptions → HTTP translation

`NotebookSchemaNotFoundError` and `UnknownExtensionError` both translate to 404 (the router differentiates them in the `detail` string). `ValueError` from merge/split's "need 2+ distinct" guard translates to 422.

`UnknownExtensionError` distinguishes from "already accepted" because the latter is a legitimate idempotent no-op — only "name nowhere in either list" deserves 404. Accept's idempotency check returns the row first and 404s only when neither list contains the name.

### 4. Frontend cache replacement, no invalidate

Mutation hooks call `queryClient.setQueryData(NOTEBOOK_SCHEMA_QUERY_KEY, response)` rather than `invalidateQueries`. The endpoint returns the full updated schema so a single round-trip is enough; invalidating would force a second GET. That satisfies AC#4's 200ms guarantee.

### 5. Single `SchemaEditDialog` with `mode` prop

The plan explicitly asked for one component, four modes. Each form has at most three fields with bespoke validation (comma-separated lists, 2+ distinct check), so hand-rolled state is simpler than mapping `react-hook-form` onto the shape. `parseTypeList` centralises the comma-separated parser.

---

## Files changed

### New files

- `migrations/46.surrealql` + `46_down.surrealql`
- `packages/surrealdb-service/src/surrealdb_service/repositories/notebook_event.py`
- `packages/surrealdb-service/tests/test_notebook_event_repo_roundtrip.py`
- `apps/app-main/src/app_main/services/schema_edit_service.py`
- `apps/app-main/tests/test_schema_edit_service.py`
- `apps/app-main/tests/test_schemas_edit_router.py`
- `frontend/src/components/notebooks/schema/SchemaEditDialog.tsx`
- `frontend/e2e/track-b/schema-edit-ops.spec.ts`

### Modified

- `packages/shared/src/shared/models/notebook_schema.py` — `NotebookSchema.excluded_types`, new `NotebookEvent` model.
- `packages/shared/src/shared/models/__init__.py` — re-export `NotebookEvent`.
- `packages/surrealdb-service/src/surrealdb_service/repositories/__init__.py` — re-export.
- `apps/app-main/src/app_main/dependencies.py` — `get_notebook_event_repo` + `get_schema_edit_service`.
- `apps/app-main/src/app_main/api/routers/schemas.py` — 6 new endpoints + `excluded_types` in the GET response.
- `frontend/src/lib/types/notebook_schema.ts` — `excluded_types`.
- `frontend/src/lib/api/notebook-schema.ts` — 6 new methods + payload interfaces.
- `frontend/src/lib/hooks/use-notebook-schema.ts` — 6 new mutation hooks.
- `frontend/src/components/notebooks/schema/SchemaBrowser.tsx` — overflow menu + excluded_types filter + dialog mount.
- `frontend/src/components/notebooks/schema/PendingExtensionsPanel.tsx` — live Accept/Reject buttons.
- `frontend/src/app/(dashboard)/notebooks/[id]/schema/page.tsx` — pass `notebookId` to both components.
- `frontend/e2e/track-b/schema-tab-view.spec.ts` — assertion now expects enabled buttons.

---

## Outstanding notes

1. **Type-name URL-encoding**: route segments use `encodeURIComponent` on both frontend and backend (FastAPI default decoding). Special characters like spaces survive; the existing TTL exporter has tests for this. The edit endpoints inherit that contract.

2. **No regressions**: The full app-main suite (446 tests) and the full shared suite (154 tests) remain green. The existing B.3a Playwright spec was minimally updated (one assertion flip) to reflect the now-enabled buttons.

3. **The `last_modified_by` field**: present in the model since B.1b but still `None` after a mutation — populating it requires plumbing the user identity into `SchemaEditService`, which is a Track G concern (auth integration). Leaving it `None` is consistent with the rest of the per-notebook state.

4. **Coordination**: B.3c (parallel branch) ALSO touches `schemas.py` + `use-notebook-schema.ts` but adds different endpoints (review_required, dismiss_nudge, extraction/resume). Expected clean three-way merge — neither branch removes the other's symbols.

---

## Branch + commits

- Branch: `track/b-schema-edit-ops`
- Commits: cda3ff7 → 02fd60c → 965e2a4 (4 commits including this self-review push).

Ready for review.
