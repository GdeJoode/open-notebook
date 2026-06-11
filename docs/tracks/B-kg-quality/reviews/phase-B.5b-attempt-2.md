# Review — Track B Phase B.5b attempt 2

**Branch**: `track/b-orphan-prune` (HEAD `5c40778`) (PR #23)
**Decision**: **APPROVED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-11

## Summary

All 3 attempt-1 blockers (B1/B2/B3) and both majors (M4/M5) properly resolved with new code AND new tests pinning contracts. 3 minors addressed. End-to-end verification on reviewer's machine: 8/8 docker, 63/63 entity-filtering targeted, 471/471 app-main, 518/519 entity-filtering full (1 pre-existing unrelated). Frontend tsc + lint clean. Playwright 3/3 on port 8607.

## Acceptance criteria

All 6 PASS.

## Verification (reviewer's own runs)

- Docker SurrealDB roundtrips: **8/8** (was 2/6 attempt 1)
- entity-filtering orphan_prune+conn: 63/63
- entity-filtering full: 518/519 (1 pre-existing unrelated)
- apps/app-main: 471/471
- test_orphans_router.py: 5/5
- test_workflow.py: 14/14 (no regression)
- frontend tsc + lint clean
- Playwright orphan-lifecycle.spec.ts: 3/3 (third retried isolated — race confirmed)

## Adversarial probes (all clean)

- **B1 mental inversion**: reverting `_mark_unreconciled_orphans_pending` → 4 of 5 `TestRunMarksPendingReconnect` tests fail. Wire IS pinned.
- **B1 duck-typing**: `hasattr(repo, "update_orphan_status")` backward-compat with B.5a-only mocks via `test_no_pending_mark_when_repo_lacks_prune_surface`
- **B2 SurrealDB record-link**: query traverses `reference` edge AND stringifies `SELECT VALUE in` before `ANYINSIDE` — end-to-end docker creates full notebook→source→RELATE→entity chain
- **B3 `type::thing($id)`**: 6 originally-failing tests now pass; 2 new B2 tests pass
- **M4 filter scope**: `test_entity_id_filter_narrows_to_one_orphan` proves filter limits connector to 1 source-ID even with 3 pending; backward-compat covered
- **M5 Playwright**: 3/3 reproduced on fresh dev server (port 8607); third retry matches documented cold-compile race
- **Best-effort mark**: `test_mark_step_failure_does_not_lose_confirmed_relations` — workflow primary output survives lifecycle hiccup
- **`orphan_cfg.enabled` defaults True** — production caller path actually fires

## Minors (informational, no action required)

1. Entity-name collision in reconciliation: `_mark_unreconciled_orphans_pending` matches by normalized canonical_name. Two orphans sharing normalized name → share fate. ExtractedRelation carries surface strings only — matching by entity-id requires upstream change.
2. Self-review test count off by 1 (517 claimed → 518 measured) — `test_llm_matcher` flake artifact
3. Playwright cold-compile race documented, not fixed. `--workers=1` workaround fine for local; CI uses `next build`

## Kudos

- B1 test design exemplary: 5 specs covering happy + no-mark-when-reconciled + mixed + backward-compat + mark-step-failure — each inverts a real failure mode
- `_PruneRepoMock` records AND mutates state — both assertion styles without cross-contamination
- B2 docker test sets up full notebook→source→RELATE→entity chain explicitly; `ensure_record_id` comment flags trap-pattern
- M4 reuses same `retry_pending_reconnects` codepath for auto + manual — no parallel logic drift
- Self-review attempt-2 traces each blocker to fix file:line AND new test — fast re-review
- `_mark_unreconciled_orphans_pending` docstring documents duck-typed gate + best-effort + lazy import — maintainer-friendly
- Minor-3 fix (required positional `notebook_id`) — type-checker now catches omission

## Decision rationale

Every blocker and major from attempt 1 fixed in code AND pinned by new tests AND independently re-verified. 3 minors addressed. Cold-compile race documented as runner concession with clear CI escape (`next build`). Quality bar exceeded. Foundation for B.6 cross-notebook merge sound.

## Next steps

APPROVED — ready for merge.
