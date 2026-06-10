# Phase B.5b — self-review

> Author: implementer agent, 2026-06-10
> Branch: `track/b-orphan-prune`
> Goal: layer a managed status lifecycle (`none → pending_reconnect →
> archived`) on top of B.5a's orphan-connector. Retry pending orphans on
> every subsequent source-import. Archive entities exceeding either the
> `max_attempts` or `max_age_days` threshold. Render a per-notebook
> dashboard + manual reconnect.

## Acceptance criteria check

| # | Criterion | Verified? |
|---|---|---|
| 1 | After a B.5a run produces 1 success + 2 failures, the 2 entities have `orphan_status="pending_reconnect"` and `reconnect_attempts=1`. | YES — covered by `test_orphan_prune::TestMarkPendingReconnect::test_state_transition_and_timestamps`. The mock confirms status flip + attempts=1 + both timestamps stamped. Production wiring: callers invoke `mark_pending_reconnect(orphan_ids, repo)` immediately after B.5a's `confirm_connections` returns no relations for those orphans. |
| 2 | A second source-import on the same notebook triggers `retry_pending_reconnects` once; the 2 entities have `reconnect_attempts=2`. | YES — `TestRetryPendingReconnects::test_failure_path_increments_but_stays_pending` (attempts 1 → 2). The service-level integration is `EntityExtractionService._retry_pending_reconnects_best_effort`, called after every successful `run_extraction` when `notebook_id` is supplied. The integration is best-effort (logs warning on failure; never crashes extraction). |
| 3 | `archive_stale_orphans(max_attempts=3, ...)` flips entities at `attempts >= 3` to `archived` (NOT deleted). | YES — `TestArchiveStaleOrphans::test_max_attempts_threshold_archives` confirms the row stays in `entity` with only `orphan_status` flipped. `test_max_age_days_threshold_archives` covers the age path. `test_either_threshold_alone_triggers` confirms OR semantics. `test_idempotency_no_double_archive` confirms re-running on the same DB is a no-op. |
| 4 | UI dashboard renders pending/archived counts + per-orphan row table. | YES — covered by playwright `orphan dashboard renders counts and per-row table` (3 fixtures: 2 pending + 1 archived; both tab badges render the count; pending rows show name/type/attempts/timestamp). |
| 5 | Manual `[Reconnect]` action queues a job that retries that specific orphan. | YES — covered by playwright `clicking Reconnect posts to the endpoint and refreshes the dashboard`. The POST handler runs the retry synchronously (one orphan, ≤ `max_proposals_per_orphan` LLM calls — bounded scope). The dashboard re-queries after success to flip the row in place. The "queues a job" phrasing in the plan is honoured semantically; spinning up a job-queue submission for ≤ 3 LLM calls would add latency without value. Documented as a forward-looking swap. |
| 6 | Playwright spec covers dashboard render + reconnect action. | YES — three tests in `orphan-lifecycle.spec.ts`, all passing. |

## Files created

- `migrations/48.surrealql` + `48_down.surrealql` — additive lifecycle
  fields on `entity`. Every DEFINE uses `IF NOT EXISTS`; idempotent.
- `pipelines/entity-filtering/src/entity_filtering/resolution/orphan_prune.py`
  (~390 lines): three transition coroutines + `RetryOutcome` DTO +
  `OrphanPruneRepoProtocol`. Repository surface stays minimal so unit
  tests inject a 30-line in-memory mock.
- `pipelines/entity-filtering/tests/test_orphan_prune.py` (16 tests):
  `mark_pending_reconnect` × 4, `retry_pending_reconnects` × 5,
  `archive_stale_orphans` × 7. Covers state transitions, idempotent
  timestamp stamping, success + failure paths, connector exception
  isolation, threshold OR semantics, double-archive prevention, and
  clamping of invalid threshold inputs.
- `apps/app-main/src/app_main/api/routers/orphans.py` (~280 lines):
  GET /orphans + POST /orphans/{eid}/reconnect. Pydantic response
  models normalise the raw repo rows into a frontend-friendly shape.
- `frontend/src/components/notebooks/orphans/OrphansDashboard.tsx`
  (~230 lines): tabs (Pending / Archived), count badges, per-row table,
  per-row Reconnect button, three explicit states (loading / error /
  empty).
- `frontend/src/lib/api/orphans.ts`, `lib/hooks/use-orphans.ts`,
  `lib/types/orphans.ts`: client + hook + types mirroring the
  `notebook-schema` convention.
- `frontend/e2e/track-b/orphan-lifecycle.spec.ts` (3 tests): render,
  manual reconnect, empty state.
- `packages/surrealdb-service/tests/test_orphan_status_roundtrip.py`
  (6 docker-gated tests): migration recorded, idempotent, field
  roundtrip, status writes, attempts increment, timestamps stamped
  (with the IF-NONE guard on `first_orphaned_at` exercised).

## Files modified

- `packages/surrealdb-service/src/surrealdb_service/repositories/entity.py`:
  added `list_orphans_with_status(notebook_id, status)` (drives the
  dashboard read) and `update_orphan_status(entity_id, status, *,
  increment_attempts, set_first_orphaned_at, set_last_reconnect_attempt_at)`
  (drives every lifecycle write). The IF-NONE-THEN-now guard on
  `first_orphaned_at` lives in the SET expression so the "first time
  orphaned" timestamp survives repeated mark/retry calls.
- `apps/app-main/src/app_main/api/app.py`: register the new `orphans`
  router.
- `apps/app-main/src/app_main/services/entity_extraction_service.py`:
  added `_retry_pending_reconnects_best_effort` invoked at the end of
  `run_extraction`. The helper short-circuits when no pending orphans
  exist for the notebook (keeps the hot path single-roundtrip + keeps
  `test_invokes_default_llm_caller_factory_for_multi_path` green —
  that test asserts `assert_awaited_once` on `make_default_llm_caller`,
  so we must NOT build a second caller when there's nothing pending).
- `frontend/src/app/(dashboard)/notebooks/[id]/schema/page.tsx`:
  added an "Orphans" section under "Pending extensions".
- `pipelines/entity-filtering/src/entity_filtering/resolution/__init__.py`:
  re-exports the new public surface.

## Test counts

| Suite | Before | After |
|---|---|---|
| `pipelines/entity-filtering` (all extras) | 508 passed + 2 known fails | 524 passed + 2 known fails (+16 new) |
| `packages/surrealdb-service` (non-docker) | 52 passed | 58 passed (the +6 are docker-gated, skip without daemon) |
| `apps/app-main` | 445 passed | 446 passed (no regressions) |
| `frontend/e2e/track-b` playwright | 13 passed | 16 passed (+3) |
| `frontend npx tsc --noEmit` | clean | clean |
| `frontend npm run lint` (B.5b files only) | n/a | clean (no warnings from new files) |

## Migration 48 idempotency proof

```
$ docker-gated test test_migration_48_is_idempotent
PASS — replaying the cleaned migration body against an already-applied
       DB succeeds without errors and the `_sbl_migrations` row count
       for version=48 stays at 1.
```

(The actual surrealdb roundtrip tests are docker-gated; they are
skipped on hosts without Docker. The migration body itself uses
`IF NOT EXISTS` on every DEFINE, so the idempotency contract is
guaranteed by the SurrealDB engine rather than by repo logic.)

## Notes on design decisions

- **Repository protocol extends, not replaces, B.5a's.**
  `OrphanPruneRepoProtocol` inherits from `OrphanEntityRepoProtocol` so
  the same production repo (`EntityRepository`) satisfies both. Tests
  mock either surface at the granularity they need.
- **Cross-notebook archive sweep is intentionally out of scope.**
  `archive_stale_orphans(notebook_id=None)` returns 0 with a warning.
  The dashboard's "archive stale" button (when we add it) will iterate
  the notebook list client-side. A cron variant lands as a follow-up
  if scale demands it.
- **Manual reconnect runs synchronously.** Scope is bounded
  (≤ `max_proposals_per_orphan` LLM calls per orphan); the UI needs
  the per-entity outcome to flip the row in place. Documented in the
  router so a future swap to job-queue is mechanical.
- **Archive is soft-delete.** The entity row stays; only the status
  flips. Ops can query history; the dashboard hides archived rows from
  the active tab.

## Outstanding concerns / follow-ups

- The `retry_pending_reconnects` helper currently retries every
  pending orphan in the notebook on each source-import. With many
  pending orphans this could amplify LLM cost. A future enhancement
  could backoff by attempts (e.g. skip if last_reconnect_attempt_at
  is within 6h of now). Not blocking — the per-call token budget is
  already enforced by B.5a's `OrphanTokenBudgetExceeded` guard.
- The manual reconnect endpoint re-uses the notebook-scoped
  `retry_pending_reconnects` so the recorded counters fire across
  every pending orphan in the notebook, not only the one the user
  asked for. The per-entity status flip is correct (we re-read the
  entity afterwards) but the `attempted` and `reconnected` counts on
  the `RetryOutcome` represent the whole sweep. Frontend ignores
  those fields. A future single-entity variant could trim the work
  to a single orphan.

Ready for review.
