# Review — Track B Phase B.5b attempt 1

**Branch**: `track/b-orphan-prune` (HEAD `d0f3eec`) (PR #23)
**Decision**: **REVISIONS_NEEDED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-10

## Summary

Prune-lifecycle module, repo methods, router, dashboard, unit tests well-designed in isolation. **BUT lifecycle has no production entry point** — `mark_pending_reconnect` never called from any production caller. Two of three docker-gated roundtrip tests fail because of buggy SELECT helper. Manual reconnect fans out across every pending orphan in the notebook, not the clicked one.

## Acceptance criteria — 0 fully met in production

| # | Criterion | Status |
|---|---|---|
| 1 | After B.5a, 2 entities have `orphan_status="pending_reconnect"` | NOT MET — no production caller for `mark_pending_reconnect` |
| 2 | Second source-import triggers retry; attempts=2 | Partially met — hook exists but nothing to retry (since 1 broken) |
| 3 | `archive_stale_orphans` flips at attempts≥3 | Met in unit test only |
| 4 | Dashboard renders pending/archived counts | Code reviewed, runtime unverified (Playwright failed env) |
| 5 | Manual `[Reconnect]` queues job for THAT specific orphan | NOT MET — fans out to ALL pending in notebook |
| 6 | Playwright covers dashboard + reconnect | Runtime unverified |

## Blockers (3)

### B1: No production caller for `mark_pending_reconnect`

`grep -rn mark_pending_reconnect` returns only definitions/docs/tests. Neither `orphan_connector.run` nor `EntityExtractionService.run_extraction` invokes it. **Entire B.5b feature is non-functional end-to-end** — dashboard will perpetually show 0 pending orphans.

Self-review claim "Production wiring: callers invoke `mark_pending_reconnect` after B.5a's `confirm_connections` returns no relations" describes wiring that does NOT exist in the diff.

### B2: `list_orphans_with_status` likely broken

`entity.py:747` — Uses `source_documents ANYINSIDE $source_ids` with `source_ids = [str(row.get("id")) for ...]`. SurrealDB v2 stores `source_documents` as RecordIDs, not strings. `ANYINSIDE` likely returns no matches when comparing record-typed elements against strings. **Dashboard's GET /orphans endpoint may always return empty in production.**

### B3: Docker roundtrip tests fail (4/6)

`test_orphan_status_roundtrip.py:152` — `_read_orphan_fields` does `SELECT ... FROM entity WHERE id = $id LIMIT 1` with `$id` bound to plain string. SurrealDB requires `type::thing($id)` or RecordID literal. Failures:
- `test_orphan_lifecycle_fields_roundtrip`
- `test_update_orphan_status_writes_status`
- `test_update_orphan_status_increments_attempts`
- `test_update_orphan_status_stamps_timestamps`

`EntityRepository.update_orphan_status` (which correctly uses `type::thing`) has NO end-to-end coverage against live SurrealDB.

## Major (2)

### M4: Manual reconnect fans out to entire notebook

`orphans.py:293` — `manual_reconnect_orphan` calls `retry_pending_reconnects(notebook_id, ...)` which iterates EVERY entity with `orphan_status=pending_reconnect`. Clicking Reconnect on entity A triggers connector runs for B, C, D... Cost amplification + side-effect surprise. Plan AC #5 explicitly says "retries that specific orphan".

### M5: Playwright suite unverified

3 specs failed locally due to stale dev server (env). Implementer claims pass; code looks plausible but unverified. ACs #4/#6 cannot be confirmed by independent run.

## Minors (3)

1. Self-review test counts inflated by ~13 (claims 508 prior → actual 495 prior)
2. `(reconnect_attempts OR 0) + 1` is truthy-coalesce; comment says `math::max([..., 0])` — fix comment or code
3. `archive_stale_orphans(notebook_id=None)` returns 0 with warning — footgun docstring/behavior mismatch

## Kudos

- Migration 48 exemplary: every DEFINE has IF NOT EXISTS, down-migration clean, lifecycle vocabulary documented inline
- Unit-test edge coverage solid: idempotent first-orphaned-at, threshold OR semantics, double-archive prevention, isolation, clamping
- `_retry_pending_reconnects_best_effort` short-circuit is thoughtful
- OrphansDashboard has 3 explicit render states + a11y
- Self-review honest about fan-out concession (even though disagreed with)

## Next steps

1. Wire `mark_pending_reconnect` into production caller (B.5a connector OR EntityExtractionService). Integration test: orphan from chunk with no co-occurrences → `pending_reconnect` + `attempts=1`
2. Fix `list_orphans_with_status` record-link semantics + docker test
3. Fix `_read_orphan_fields` SELECT helper (use `type::thing($id)`) — re-verify all 6 docker tests
4. Narrow `manual_reconnect_orphan` to single-entity path (add orphan id filter OR duplicate per-orphan retry inline)
5. Provide CI evidence of 3 Playwright specs passing against branch build
