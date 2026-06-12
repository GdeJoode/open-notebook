# Phase B.7 — Self-review

**Branch**: `track/b-integration-retro`
**Phase**: B.7 (track integration + ARCHITECTURE + RETRO + CHANGELOG)
**Scope**: docs-only (no production code paths touched)

## Attempt 1

Five files: `ARCHITECTURE.md` (Track-B storage layer + service modules),
`CHANGELOG.md` (new, deploy-grade operator notes for migrations 44-48),
`docs/FEATURE_ROADMAP.md` (Track-B banner + phase table), new
`docs/tracks/B-kg-quality/RETRO.md` (7 what-worked / 6 what-hurt / 7
recommendations), `docs/tracks/B-kg-quality/status.md` (final entry).

Acceptance-criteria self-check at submit time:

| AC | Self-claim | Evidence |
|---|---|---|
| 1 | Full corpus E2E | DEFERRED with explicit reasoning in RETRO §Live-test recommendation. Per-phase Playwright specs already exercise production paths against the production API. |
| 2 | Cross-notebook merge sanity | DEFERRED under same envelope as AC1. |
| 3 | ARCHITECTURE reflects new tables + service modules | New §6 "Storage layer additions" lists migrations 44-48, all 4 new tables, the 3 `entity` field additions, all 6 Track-B services. Cross-checked against migration files + actual service paths before commit. |
| 4 | FEATURE_ROADMAP reflects ALL phases complete | All 18 PRs (incl. B.7) marked ✅ in phase table; banner declares completion. |
| 5 | RETRO ≥5 worked / ≥3 hurt / ≥5 recommendations | Actual counts: 7 / 6 / 7. |

Reviewer decision: **REVISIONS_NEEDED**. Major #1 = phase/PR
arithmetic off-by-one (17 enumerated as "17 sub-phases ... 17 PRs"
when the enumeration literally contains 18). Four minors (CHANGELOG
kwarg name, missing review-file links, ARCHITECTURE path comments,
duplicate FEATURE_ROADMAP summary).

## Attempt 2 fixes

Mechanical revisions per reviewer recommendation. No code changes.

### Major #1 — phase/PR count off-by-one

Reframed count statements consistently as **"18 PRs across 17
production sub-phases + 1 integration/retro phase (B.7)"**:

- `RETRO.md` summary header (line 22-34): rewritten to lead with
  "18 PRs across 17 production sub-phases + 1 integration/retro phase
  (B.7)"; reflow keeps the rest of the paragraph intact. Rejection-
  rate denominator clarified to "17 production sub-phases" (47%, not
  the original ambiguous 50%).
- `RETRO.md` "What hurt" CI-noise entry (line 132): "over 17 PRs"
  → "over 18 PRs".
- `RETRO.md` Recommendation #1 (line 138-146): denominator clarified
  to "17 production sub-phases" with paragraph re-flow.
- `RETRO.md` attempt-count table (line 218): added an explanatory
  block note above the table: "Rejection rate calculated on 17
  production sub-phases; B.7 itself is reviewed separately (see
  `reviews/phase-B.7-attempt-1.md`) and not counted in the denominator
  below."
- `RETRO.md` Pattern sentence (line 236): denominator clarified to
  "17 production sub-phases".
- `RETRO.md` Closing (line 283-292): "18 PRs (17 production
  sub-phases + 1 integration/retro phase, B.7)" with rejection-rate
  qualifier "(~47% on the 17 production sub-phases)".
- `FEATURE_ROADMAP.md:128-135`: banner rewritten to "All 18 PRs
  merged (17 production sub-phases — B.0, B.1a–B.1f, B.2a/B.2b,
  B.3a–B.3d, B.4, B.5a/B.5b, B.6 — plus B.7 integration/retro)".

Verification: `grep -c "17 PRs\|17 sub-phases ... merged across 17"
docs/tracks/B-kg-quality/RETRO.md docs/FEATURE_ROADMAP.md` returns
`0` on both files (was `>0` before).

### Minor #1 — CHANGELOG `age_threshold` → `max_age_days`

`CHANGELOG.md:29`: renamed `age_threshold=90 days` to `max_age_days=90`
and added the module-path citation
`pipelines/entity-filtering/.../orphan_prune.py::archive_stale_orphans`
for greppability. Verified against
`pipelines/entity-filtering/src/entity_filtering/resolution/orphan_prune.py:446`
(`max_age_days: int = DEFAULT_MAX_AGE_DAYS`).

### Minor #2 — RETRO attempt-count table review-file links

`RETRO.md` attempt-count table (line 218 onwards): every row now
hyperlinks to the specific review file:

- REVISIONS rows → `reviews/phase-B.N-attempt-1.md` (the rejection
  attempt that documents the issue).
- APPROVED rows → `reviews/phase-B.N-attempt-1.md` (the approval
  attempt).
- B.0 → `phase-B.0-attempt-1.md`; B.1a → `self-review.md` (no separate
  reviewer attempt file exists; the substantive content is in the
  self-review).

This addresses the Track G inversion-test reviewer use case from
review §minor #2 (six-months-later cross-track audience).

### Minor #3 — ARCHITECTURE service module path parentheticals

`ARCHITECTURE.md:143-144`: added explicit `(in <workspace>/)`
parenthetical to the two rows that deviate from the "live in
`apps/app-main/src/app_main/services/` unless noted otherwise"
default:

- `shared.services.metrics.record_metric` → `(in packages/shared/)`
- `entity_filtering.resolution.orphan_connector` → `(in
  pipelines/entity-filtering/)`

For symmetry; the four `apps/`-resident rows are left bare, matching
the "unless noted" caveat.

### Minor #4 — FEATURE_ROADMAP duplicate one-line summary

`FEATURE_ROADMAP.md:128-138`: dropped the redundant **One-line
summary** sentence that repeated "Multi-schema KG extraction, schema-
edit UX, telemetry, orphan-lifecycle, cross-notebook merge" verbatim.
Verified via `grep -n "One-line summary\|multi-schema KG extraction"
docs/FEATURE_ROADMAP.md` (returns nothing); the substantive content
remains in the banner opener at line 128-129.

## Final verification

```
$ git diff --stat main...HEAD
 ARCHITECTURE.md                    |  63 +++++++-
 CHANGELOG.md                       |  30 ++++
 docs/FEATURE_ROADMAP.md            |  33 +++++
 docs/tracks/B-kg-quality/RETRO.md  | 297 +++++++++++++++++++++++++++++++++++++
 docs/tracks/B-kg-quality/status.md |  38 +++++
 5 files changed, 460 insertions(+), 1 deletion(-)
```

Five doc files, no code paths touched (`grep -E '\.(py|ts|tsx|js)$'`
on the diff stat returns nothing).

## Ready for review

All four issue classes (1 major + 4 minors) closed with mechanical
fixes. No new issues encountered during the fix-up; the only judgment
call was the symmetry decision on minor #3 (add path parentheticals to
both deviating rows rather than just `orphan_connector`).
