# Phase B.3d — Self-review

**Date**: 2026-06-10
**Author**: implementer (track B, B.3d)
**Branch**: `track/b-reextract-prompt`

---

## Scope

After a B.3b schema edit op emits a `notebook_event{type:"schema_changed"}`, surface a banner inviting the user to re-extract affected sources, and on confirm enqueue background `ENTITY_EXTRACT` jobs.

Delivered surface:

- **`ReextractPromptBanner.tsx`** — polls the unread `schema_changed` stream and renders an amber banner when at least one event exists. Three actions: `[Re-extract all]`, `[Re-extract selected…]` (modal multi-select), `[Later]` (mark-read + dismiss).
- **`reextract_service.py`** (`ReextractService`) — orchestrates job submission for a list of source ids. Uses the existing `CommandService` dispatch path (so the worker sees the same payload as a manual "Run entities" click) and the `JobRepository.get_by_source` slice for per-source dedup.
- **Two new schemas-router endpoints**:
  - `GET /api/notebooks/{id}/schema/reextract_candidates` → `{ notebook_id, source_ids, count }`. V1 returns ALL notebook sources (rationale documented inline).
  - `POST /api/notebooks/{id}/schema/reextract` body `{source_ids: [...]}` → `{ jobs_enqueued, enqueued_source_ids, skipped_source_ids, ... }`.
- **Minimal `notebook_events` HTTP surface** (`GET /events?type=...` + `POST /events/{id}/mark_read`). The B.3c plan originally owned this; B.3c hasn't merged, so the slice the banner needs is included here. The endpoint is intentionally narrow and re-uses the existing `NotebookEventRepository` directly — no new DB shape.
- **Hooks** (`use-notebook-schema.ts`): `useSchemaChangedEvents` (30s poll, `refetchOnWindowFocus: false`), `useReextractCandidates`, `useReextractMutation`, `useMarkEventRead`.

---

## Acceptance criteria

| AC | Status | Evidence |
| --- | --- | --- |
| #1 After rename op, banner shows affected-source count | OK | `schema-reextract.spec.ts` "banner shows count …" + screenshot at `/notebooks/{id}` shows `"3 affected source"` |
| #2 `GET /schema/reextract_candidates` returns source ids JSON | OK | `TestReextractCandidates.test_returns_all_notebook_sources` |
| #3 `[Re-extract all]` triggers one job per source via job-queue | OK | `TestReextractEnqueue.test_enqueues_jobs_for_all_sources_happy_path` + Playwright asserts POST payload `{source_ids: [...]}` |
| #4 `[Re-extract selected]` with subset → only those queued | OK | Playwright "[Re-extract selected…] opens dialog and POSTs only the checked subset" |
| #5 `[Later]` dismisses banner; sources keep "Schema changed" hint | Partial | `[Later]` marks every unread event read and hides banner (verified). The per-source "Schema changed" badge is rendered elsewhere — see Open items |
| #6 Playwright happy path | OK | 4/4 in `schema-reextract.spec.ts` |

---

## Test results

### Backend (unit + router)
```
apps/app-main/tests/test_reextract_service.py  → 13 passed
apps/app-main/tests/test_reextract_router.py   → 14 passed
apps/app-main full suite                        → 473 passed (was 446 before this phase, +27)
```

Test run command:
```
cd apps/app-main && uv run --no-sync python -m pytest tests/ -q --no-header --tb=line
473 passed, 3 warnings in 153.53s
```

### Frontend
```
npx tsc --noEmit                                → clean (0 errors)
npm run lint                                    → no new warnings (pre-existing list unchanged)
```

### E2E (Playwright)
```
e2e/track-b/schema-reextract.spec.ts → 4 passed (5.6s)
All Track-B specs (17 total)         → 17 passed (20.6s)
```

Specs covered:
1. Banner shows count when `schema_changed` events exist; `[Re-extract all]` POSTs every candidate id and dismisses.
2. `[Re-extract selected…]` opens dialog with all candidates pre-checked; un-checking + submit POSTs only the chosen subset.
3. `[Later]` marks every unread event read and hides the banner without posting jobs.
4. Banner is absent when there are no unread `schema_changed` events.

Run command:
```
env PLAYWRIGHT_BASE_URL=http://localhost:18508 npx playwright test e2e/track-b/schema-reextract.spec.ts --reporter=line
```

---

## Design decisions

- **`get_by_source` dedup over a separate "active job" cache**: avoids new state, leans on the existing job table. Inflight = `{queued, processing, retrying}`; completed/failed/cancelled jobs are eligible for re-extraction (the whole point of the banner).
- **Per-source failure ≠ batch abort**: a transient `submit_command_job` error on source N logs and reclassifies that source as skipped; sources N+1..M still get enqueued.
- **V1 candidate list == all notebook sources**: the plan explicitly accepts this; narrowing to "sources containing entities of the modified type" needs an entity → source index that Track G owns. Documented inline in `list_reextract_candidates`.
- **`CommandServiceAdapter` wrapper in DI**: `CommandService` exposes its methods as `@staticmethod`, but the service-level protocol expects an instance. The adapter is 8 lines and keeps the service unit-testable without importing the module-level singleton.
- **Banner returns `null` for the loading state**: the events query polls every 30s. Showing a placeholder on every poll would flicker; users who never touch the schema would never see anything anyway. The `null`-during-load is intentional.

---

## Open items / follow-ups

- **AC#5 "Schema changed" badge on affected sources** — the banner dismissal flow (mark-read) works, but the per-source badge on the workspace SourcesColumn isn't shipped here. It belongs to the SourcesColumn rendering layer; can be added as a small follow-up in B.4 or by extending `SourceCard` to read the same events stream filtered by `op.affected_source_ids`. Flagging for the reviewer to decide whether to require it within B.3d.
- **B.3c overlap on `notebook_events` GET**: the minimal endpoint shipped here (`GET /events`, `POST /events/{id}/mark_read`) is the slice B.3d needs. B.3c (parallel branch) will likely extend the same endpoint with extra event types — review will need to verify the merge produces one cohesive API (no duplicate handler definitions). The endpoint lives in `schemas.py` for now because that's the only consumer in this phase; a `notebook_events.py` router carve-out is a clean follow-up if both phases land.
- **B.3c not merged**: the user-facing plan referred to "the existing notebook_events endpoint (from B.3c)". B.3c didn't ship before this phase; that's why the events endpoint is part of this PR. If B.3c ships a different shape, expect a small reconciliation merge.

---

## Risk + rollout

- **Idempotency**: server-side dedup means a double-click can't stack duplicate jobs. The mutation surface is a single POST; rolling back is a no-op (the events stream is additive).
- **Performance**: the candidates query is bounded by notebook size (typical ≤ 30 sources). The events poll runs every 30s per open tab. No new DB indexes required — the existing `(notebook, event_type, created_at)` index covers `list_unread`.
- **Frontend bundle**: `+ReextractPromptBanner.tsx` adds ~6 KB pre-minify, well under the dashboard route's existing first-load JS budget.

---

## Commit history

- `412a782` — `feat(schemas): re-extract prompt backend + events GET (B.3d)`
- `ba1d923` — `feat(frontend): re-extract prompt banner + hooks (B.3d)`
- `92ce5eb` — `test(e2e/track-b): schema-reextract Playwright spec passes + standalone helpers`

Branch: `track/b-reextract-prompt` (pushed to origin).
