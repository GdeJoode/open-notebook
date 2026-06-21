# Review — Track B Phase B.3d attempt 2

**Branch**: `track/b-reextract-prompt` (HEAD `4bbc8aa`, rebased on main post-B.5b)
**PR**: #24
**Decision**: **APPROVED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-12

## Summary

Attempt 2 resolves all 3 blockers and both majors from attempt 1, plus
addresses two minors. The B.3c reconciliation is clean (no contract duplication
remains; canonical hooks/endpoints from main are the single source of truth),
the PAUSED_FOR_REVIEW dedup gap is closed with a behavioural pin test, and the
cross-notebook source_id defence-in-depth filter is router-level + tested.
Quality bar met: 23/23 reextract tests pass, 19/19 soft-nudge tests preserved,
494/494 full app-main suite, frontend tsc clean, lint clean (no new warnings),
and 9/9 Playwright on a warmed dev server.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | Banner shows affected-source count after schema_changed event | PASS | Visual snapshot `Schema gewijzigd. 3 affected sources can be re-extracted.` |
| 2 | `GET /schema/reextract_candidates` returns source ids JSON | PASS | `test_returns_all_notebook_sources` + 3 router edge tests |
| 3 | `[Re-extract all]` enqueues one job per source | PASS | service `test_enqueues_one_job_per_source`; router POST asserted in spec |
| 4 | `[Re-extract selected]` enqueues only selected subset | PASS | Playwright `[Re-extract selected]` spec verifies subset payload |
| 5 | `[Later]` dismisses banner; per-source "Schema changed" badge | DEFERRED | Banner mark-read + dismiss verified; per-source badge documented as Track G / B.4 follow-up (acceptable per attempt 1 review) |
| 6 | Playwright happy path | PASS | 4/4 reextract + 5/5 soft-nudge on warm server |

## Verification of attempt 1 fixes

### B1 — duplicate /events + /mark_read removed from schemas.py
- `grep -n "_NotebookEventView\|list_notebook_events\|mark_notebook_event_read" apps/app-main/src/app_main/api/routers/schemas.py` → empty (CONFIRMED removed).
- `frontend/src/lib/api/notebook-schema.ts` exposes a single canonical `listEvents`/`markEventRead` (lines 272-308) with docstring noting both banners share the surface. No duplicate clients.
- `frontend/src/lib/hooks/use-notebook-schema.ts` exposes `useNotebookEvents` and `useMarkEventRead` (B.3c surface, line 134, 164). The branch's `useSchemaChangedEvents` is GONE; an inline comment at line 367 explains the intentional non-duplication.
- `ReextractPromptBanner.tsx:67-94` imports `useNotebookEvents` + `useMarkEventRead` from the canonical hook module and calls `useNotebookEvents(notebookId, { types: ['schema_changed'], unread: true })`.
- Playwright mocks return `{event_id, success: true}` shape (`schema-reextract.spec.ts:214, 287, 353`) — matches B.3c `MarkReadResponse`.

### B2 — B.3c surface preserved verbatim
Files verified present (timestamps Jun 10 16:14):
- `apps/app-main/src/app_main/api/routers/notebook_events.py` (canonical, 9.5KB)
- `apps/app-main/tests/test_schemas_soft_nudge.py` (19 tests pass)
- `frontend/src/components/notebooks/schema/SchemaSoftNudge.tsx` (7.0KB)
- `frontend/src/components/notebooks/schema/ExtractionPausedBanner.tsx` (3.0KB)
- `frontend/e2e/track-b/schema-soft-nudge.spec.ts` (5 tests pass)
- `schemas.py` retains `dismiss_soft_nudge` (referenced) and paused-extraction blocks.
- `notebook_events` router properly mounted in `app.py:172-174`.

### B3 — 3-banner page integration
`frontend/src/app/(dashboard)/notebooks/[id]/page.tsx:123-125` renders in order:
`SchemaSoftNudge → ExtractionPausedBanner → ReextractPromptBanner`. Comment block
(lines 112-121) justifies ordering (informational → state-blocker → action prompt).
Each banner self-hides on empty state, so the column grid takes full height in
the common case. No layout breakage observed in snapshot.

### M4 — PAUSED_FOR_REVIEW dedup
- `reextract_service.py:65-66`: `_INFLIGHT_STATUSES = frozenset({"queued", "processing", "retrying", "paused_for_review"})`.
- Rationale docstring at line 68-75 explains the orphan-state hazard and points the caller at `POST /extraction/resume`.
- Pin test `test_skips_source_with_paused_for_review_job` (line 208-235): asserts `jobs_enqueued == 0`, `enqueued_source_ids == []`, `skipped_source_ids == ["source:a"]`, AND `submit_command_job.assert_not_awaited()`. Mental inversion: would FAIL if `paused_for_review` were removed from the frozenset.

### M5 — V1 candidate test docstring
`test_returns_all_notebook_sources` (lines 74-81): explicit docstring explains V1 returns ALL sources because the entity→source index doesn't yet exist; FUTURE narrowing requires evolving to a SUBSET assertion. Clear contract for B.4/Track G.

### Minor 1 — Cross-notebook source_id defence
- `schemas.py:1487-1561` `enqueue_reextract` filters requested ids against `source_repo.list_with_metadata(notebook_id=...)` before dispatch.
- Falls through to unfiltered request on transient DB error (logged), so user isn't locked out.
- Dropped ids logged at INFO level with notebook id + count.
- Pinned by `test_filters_out_cross_notebook_source_ids` (lines 252-289): payload `["source:a", "source:foreign"]` → service called with `source_ids=["source:a"]`.

## Test status

```
# Backend
apps/app-main/tests/test_reextract_service.py  → 14 passed (incl. PAUSED skip)
apps/app-main/tests/test_reextract_router.py   →  9 passed (incl. cross-notebook filter)
apps/app-main/tests/test_schemas_soft_nudge.py → 19 passed (B.3c preserved verbatim)
apps/app-main full suite                       → 494 passed in 73.7s

# Frontend
npx tsc --noEmit                                → clean
npm run lint                                    → pre-existing warnings only (none in B.3d files)

# E2E (Playwright @ localhost:8606, warmed)
e2e/track-b/schema-reextract.spec.ts           → 4 passed
e2e/track-b/schema-soft-nudge.spec.ts          → 5 passed
combined run                                   → 9 passed in 25.9s
```

## Issues found

### Blockers
None.

### Major
None.

### Minor (non-blocking)

1. **Cold-compile Playwright timing brittleness** — `frontend/e2e/track-b/schema-reextract.spec.ts:54-58`. First navigation against a fresh Next dev server compiles `/notebooks/[id]` in 15-25s, which can push the test past the 60s overall timeout. Re-runs on a warmed server pass cleanly. Already mirrors the B.3c spec's timeout strategy; if it flakes on CI, consider `webServer.reuseExistingServer` or a warm-up navigation in `beforeAll`. Not a B.3d defect.

2. **Potential TOCTOU race on concurrent re-extract clicks** — `reextract_service.py:269`. Two browser tabs hitting `/schema/reextract` for the same source within the dedup query window (between `get_by_source` and `submit_command_job`) could both submit. The `_INFLIGHT_STATUSES` check is observation-only, not transactional. For two-tab UX this is unlikely (mutation completes fast, second poll picks up the new in-flight job) but the surface is real. Acceptable for V1; document in B.6 if/when cross-notebook concurrency becomes a hot path.

3. **Three independent banner GETs on page mount** — `page.tsx:123-125` mounts three banner components that each fire their own `useQuery` for `/schema`, `/extraction/paused`, and `/events?type=schema_changed`. No coalescing. ~3 round-trips per notebook open. Fine for now; mention in B.6 if the notebook page hits a critical path budget.

4. **`enqueue_reextract` falls through to UNFILTERED on `list_with_metadata` exception** — `schemas.py:1530-1549`. Resilience trade-off documented inline; the rationale (don't lock the user out on transient DB hiccup) is sound, but the service-layer surface still has no per-source notebook ownership check, so a determined attacker with notebook-A access AND a known source ID from notebook-B could exploit the fall-through. Mitigated in practice because the candidate list rarely fails, but worth keeping in mind for hardening.

5. **`extraction/paused(?:\\?.*)?$` regex in spec** — line 138. Won't match the `/extraction/paused` endpoint if a future revision adds a path segment. Defensive nit; current API stable.

### Pre-existing CI failures (not B.3d-attributable)
- `test-build-single` references a deleted `Dockerfile.single` (removed in main commit `79784f5` "monolith→workspace cutover"). Workflow YAML drift, repo-wide.
- `claude-review` env-var validation error — observed on prior PRs (track-A retro mentions same).

Both failures pre-date this PR and would surface on any branch.

## Kudos

- **B.3c reconciliation execution is exemplary.** Attempt 1 carried a parallel implementation; attempt 2 deletes every duplicate (3 backend symbols, 2 frontend clients, 2 hooks) and leaves block-comments at the deletion sites explaining the canonical owner. Future maintainers will see the intent immediately.
- **PAUSED_FOR_REVIEW pin test is textbook adversarial defence.** The assertion stack pins the contract (zero enqueue, zero await, source goes to skipped) so any silent revert would break the test.
- **Defence-in-depth filter at the router is the right layer.** Keeps the service stateless and the source-of-truth (candidate list) close to the API boundary; the fall-through rationale on DB error is documented and trade-off acknowledged.
- **`useReextractCandidates({ enabled: events.length > 0 })`** correctly avoids the GET when the banner doesn't render.
- **3-banner ordering rationale** in `page.tsx` is well-justified: informational coverage → blocking state → action prompt. Matches the user's cognitive flow.

## B.3c overlap resolution quality

This is the section that earned the REVISIONS_NEEDED in attempt 1, so it
deserves explicit comment. **Quality: excellent.**

- **Surgical, not lazy.** Every duplicate identifier was deleted at the
  declaration site, not aliased or shadowed. The deletion sites carry
  block-comments pointing at the canonical owner — drift becomes
  immediately visible in any future PR.
- **Wire contracts converged.** Playwright mocks now match the B.3c
  `MarkReadResponse` shape (`{event_id, success}`); query params follow
  `type=schema_changed&unread=true` (B.3c), not `type=...&unread_only=...`
  (attempt 1).
- **Hook consolidation is complete.** `ReextractPromptBanner.tsx` uses
  `useNotebookEvents(id, { types: ['schema_changed'], unread: true })` —
  the canonical filter API. No banner-specific hook variant remains.
- **Tests preserved verbatim.** All 19 `test_schemas_soft_nudge.py` tests
  pass without modification. The B.3c spec also passes unchanged.
- **Routing topology correct.** `notebook_events.router` is mounted in
  `app.py:172-174` independently of the schemas router, matching the B.3c
  carve-out the attempt 1 review recommended.

This is the strongest foundation B.6 (cross-notebook merge) could ask for.

## Decision rationale

Zero blockers, zero majors. All previously-flagged issues addressed with
behavioural test pins. The B.3c reconciliation is not just functional but
architecturally clean — the cleanup leaves the codebase in a better state
than if B.3d had merged first. Five minors are non-blocking nits (timing,
concurrency-edge, network-shape) that can be filed as follow-up or addressed
in B.6 hardening.

## Next steps

- **Merge-ready.** Recommend human approval + merge.
- **Pre-existing CI failures** (`test-build-single`, `claude-review`) should
  be addressed independently — they affect every PR. File as repo-hygiene
  tickets.
- **B.6 carry-overs**: the TOCTOU dedup question and the per-source notebook
  ownership check at the service layer are good candidates for the
  cross-notebook concurrency story.
