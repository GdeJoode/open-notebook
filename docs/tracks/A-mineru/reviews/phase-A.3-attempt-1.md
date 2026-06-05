# Review — Track A Phase A.3 attempt 1

**Branch**: `track/a-mineru-integration`
**Decision**: APPROVED (with documented concerns and minor follow-ups)
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-05

## Summary

Phase A.3 ships the threshold-tuning report, integration Playwright spec
(2 sub-tests), four-file documentation set (README, ARCHITECTURE,
troubleshooting, RETRO + status + roadmap), `score_pdf_corpus.py` CLI
helper, and three synthetic test fixtures. Every quality gate was
re-run by the reviewer and passes: 367 backend tests, 105 shared, 10/10
Playwright, clean `tsc --noEmit`, lint clean of new code. The
threshold-tuning decision (keep 0.95) is defensible but borderline —
all five real-corpus PDFs hit fallback because `table_success` is
hard-floored by docling's table-parser failure — see Major #1 for the
detailed adversarial analysis. Several documentation artefacts are
stale (claim the CI workflow file is still pending when it has lived
at `.github/workflows/e2e.yml` since commit `732a1cf`); these are
docs-only inaccuracies and do not affect shipping behaviour.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | `threshold-tuning.md` has ≥3 real-PDF rows with name + score + decision + manual quality call | PASS | 5 real-PDF rows + 3 synthetic baseline rows; per-signal breakdown + dominant_signal column. Math checks out against the implementation. |
| 2 | Default 0.95 confirmed or new default proposed | PASS | Kept 0.95, decision sentence at end is correctly bolded. Skipped commit 4 of the planned sequence per the conditional rule. See Major #1 for adversarial pushback on the decision quality. |
| 3 | `parser-engine-integration.spec.ts` exists + passes via route mocks | PARTIAL PASS | Spec exists, both sub-tests pass under `--workers=1`. Deviates from plan §5 step 4 (upload flow): no POST `/api/sources` mock, no `Upload source` click — the spec instead navigates straight to `/sources/{id}` with a mocked GET. See Major #2. |
| 4 | README has "Parser engines" subsection | PASS | `README.md:324-333` matches the plan §7 copy block verbatim; describes all four engines. |
| 5 | `architecture.md` mentions dispatcher + `Source.metadata` | PASS | `docs/development/architecture.md:139-148` — paragraph + bullet block matches plan §8. Inserted before §5 Background Processing (effectively after §4 AI Processing Layer per pre-resolved decision). |
| 6 | `troubleshooting/parser-engines.md` exists with ≥3 scenarios + index link | PASS | 4 scenarios; `docs/troubleshooting/index.md:25-26` links. Commands are concrete (exact `docker compose` / `uv run` invocations, not "check logs"). |
| 7 | `RETRO.md` has "What worked" / "What hurt" / "Recommendations for tracks B-G" | PASS | All three sections present with concrete bullet evidence (per-phase attempt counts, named files, named bugs). One factual inaccuracy — see Minor #1. |
| 8 | No regressions: pytest + lint + tsc + Playwright all green | PASS | Verified by reviewer (see "Test status" below). |
| 9 | `status.md` final entry titled `## Phase A.3 — IMPLEMENTED` ending with "Track A — COMPLETE" | PASS | Entry at line 1052; closing block at line 1170 reads "Track A — COMPLETE on 2026-06-04". |
| 10 | `FEATURE_ROADMAP.md` Track A marked done with pointers | PASS | `docs/FEATURE_ROADMAP.md:62-69` — ✅ COMPLETE banner with RETRO + tuning-report links. |

## Test status

Reviewer re-ran every gate from scratch:

```
$ uv run --project apps/app-main pytest apps/app-main/tests/ -q
…
367 passed, 3 warnings in 128.47s (0:02:08)

$ uv run --project packages/shared pytest packages/shared/tests/ -q
…
105 passed in 2.67s

$ cd frontend && npx tsc --noEmit
(no output — clean)

$ cd frontend && npm run lint
(only pre-existing warnings in unrelated files;
 no new warnings or errors)

$ cd frontend && npx playwright test e2e/track-a/ --reporter=line --workers=1
10 passed (1.3m)
```

**Note on backend count**: implementer's self-review reports **357
passed**; the reviewer measures **367**. Likely cause: the self-review
was captured before a final rebase or before all new tests were
collected. The delta is in the implementer's favour (more tests
green) and is not a blocker — but it shows the self-review's "Quality
gates run" block was not refreshed at hand-off time.

## Issues found

### 🔴 Blockers (must fix)

None.

### 🟡 Major (consider before merge)

1. **Threshold decision is borderline — 100% fallback rate on a 5-PDF
   corpus deserves a more cautious framing** — `docs/tracks/A-mineru/threshold-tuning.md:42-50, 88-113`
   - Issue: All five real PDFs scored 0.725-0.850, every single one
     hit fallback. The decision argues this is "acting as designed"
     because `table_success=0.00` is informative — but a closer look
     shows the threshold 0.95 is **structurally unreachable** for any
     doc where docling fails > 1/3 of tables: with all other signals
     perfect, max overall = 0.85 + 0.15 * table_score, so any
     `table_score < 0.66` mathematically forbids the doc from crossing
     0.95. The corpus shows table_score = 0.00 universally. Net
     effect: in any deployment with non-trivial PDFs (which is most),
     "auto" mode is functionally indistinguishable from "always
     fallback to MinerU" — at substantial GPU cost (5-8 GB model
     download, ~30s/doc parse latency) and with no user signal that
     the default is mis-calibrated for their corpus.
   - The implementer's three counter-arguments are real but partial:
     (a) sample size 5 is too small to definitively recommend a
     change either, (b) users can lower the slider — but only if they
     know to look, and (c) "auto" is documented as "recommended" in
     `README.md:330`. The asymmetric error is real: a conservative
     bias (more fallback) is "safer" only if MinerU strictly beats
     docling on every fallback case, which is not validated.
   - Recommendation: the report and RETRO already note this as a
     follow-up; that is correct. Before merging this PR, **add one
     sentence to `threshold-tuning.md` Decision section** acknowledging
     that the "keep 0.95" decision is contingent on the small corpus
     and that B-track telemetry should re-evaluate. This is a
     framing improvement, not a code change. Severity is Major
     because the "Track A complete" sign-off should not silently
     embed a borderline calibration call as a settled question.

2. **Integration spec skips the upload-flow leg of the plan**
   — `frontend/e2e/track-a/parser-engine-integration.spec.ts:232-240`
   - Issue: plan-A.3.md §5 step 4 explicitly describes "Mock POST
     `/api/sources` (upload): … returns source.id='src_test_e2e' …",
     and step 6 says "Navigate to /notebooks/{some_id}, click 'Upload
     source', select `synthetic_clean_text.pdf`, submit". The shipped
     spec mocks neither and navigates directly to
     `/sources/${SOURCE_ID}` with a pre-mocked GET. The badge +
     reprocess-override sequence IS covered (which is the highest-value
     part), but the spec falls short of the "full chain" claim in
     plan §10 acceptance criterion #3 and self-review row 3.
   - Mock specificity is otherwise good: the captured POST body for
     reprocess is asserted to `toMatchObject({ parser_engine: 'docling' })`,
     which would break if a refactor broke the override plumbing.
   - Severity is Major because the spec is the *single* "integration"
     spec — the unique value-add over the four A.2 sub-specs is the
     end-to-end sequence, and the upload leg is what makes it
     end-to-end rather than a slightly bigger badge test. The plan
     budgeted ~30 LOC for this; the spec is already 334 LOC, so
     adding the upload step would not blow the budget.
   - Decision rationale for not blocking: the badge → reprocess →
     POST-override sequence is the more failure-prone path and IS
     covered. The skipped upload click is a UI traversal already
     exercised by manual smoke and (transitively) by the A.2 specs.
     This is fixable as a follow-up commit or a separate PR;
     escalating to blocker would punish a docs-and-tuning PR that is
     otherwise complete.

### 🔵 Minor (file as follow-up)

1. **RETRO + status + self-review all describe the CI workflow as
   "still pending"** — `docs/tracks/A-mineru/RETRO.md:47-49,144-154`;
   `docs/tracks/A-mineru/status.md:1145-1149`;
   `docs/tracks/A-mineru/reviews/phase-A.3-self-review.md:88-91`
   - Reality: the file was `git mv`'d to `.github/workflows/e2e.yml` in
     commit `732a1cf` ("ci: install Playwright E2E workflow at canonical
     .github/workflows path (A.0 final)") and further patched in
     `cf1ad8b`. There is no `e2e-workflow.yml.pending` file on this
     branch (verified with `find` and `git log --all`).
   - Impact: the retrospective declares a problem solved without
     reflecting that the solution actually shipped. This is a
     documentation accuracy issue — the implementer is undersell­ing
     the team's own delivery. Doesn't affect runtime behaviour.

2. **`table_success` characterisation in RETRO is slightly inaccurate**
   — `docs/tracks/A-mineru/RETRO.md:60-62`;
   `docs/tracks/A-mineru/threshold-tuning.md:165-167`
   - Both notes say `table_success` "zeros the moment any table has
     zero parsed rows" / "one bad table shouldn't completely zero the
     signal". The implementation
     (`apps/app-main/src/app_main/services/parsing/confidence.py:220-231`)
     actually returns `non_empty / len(tables)`, so one bad table out
     of five yields 0.8, not 0.0. The signal IS sensitive (one missed
     table costs 1/N of the signal) but it does not "zero" on a
     single failure.
   - In the actual A.3 corpus the score did hit 0.0 because docling
     parsed 0/N tables across every doc; the prose just over-states
     the failure mode.

3. **`score_pdf_corpus.py` docling-host bridge is the load-bearing
   path but is only tested through the format helpers**
   — `apps/app-main/scripts/score_pdf_corpus.py:124-227`;
   `apps/app-main/tests/test_score_pdf_corpus.py`
   - `_run_docling_via_host_bridge` is ~100 LOC of staging + httpx
     call + docling response translation. The smoke test in
     `tests/test_score_pdf_corpus.py` only pins the markdown row
     format and the helper functions; it never exercises the bridge.
     A regression there would only be caught when a developer next
     re-tunes. Acceptable for a one-shot tuning script; flagged for
     visibility.

4. **`mockSourceEndpoints` runs inside the second test after
   navigation has already mounted the dashboard chrome** —
   `frontend/e2e/track-a/parser-engine-integration.spec.ts:325-327`
   - In the graceful-degradation test, the source-detail mocks are
     registered after `gotoAndWait('/settings')` and before
     `gotoAndWait('/sources/...')`. The order is functionally correct
     because Playwright's `page.route` applies to the next navigation,
     but the second test relies on this implicit ordering being
     understood. Documenting it would help future maintainers.

5. **Self-review's `## Quality gates run` block lists `357 passed`
   but the reviewer measures `367`** —
   `docs/tracks/A-mineru/reviews/phase-A.3-self-review.md:43-46`
   - Likely a stale capture from before the final commit. Not a
     correctness issue — more tests passing is in the implementer's
     favour. Worth refreshing if the self-review file ever serves
     as a reference for future reviewers.

## Decision rationale

**APPROVED with major issues as advisory.** No blockers. The two
major issues are framings-and-completeness concerns, not correctness
defects:

- Major #1 (threshold decision framing) is a request for the
  decision sentence in `threshold-tuning.md` to acknowledge that the
  "keep 0.95" call is provisional given the 5-PDF corpus. The
  implementer already documents this in the Follow-ups list and in
  RETRO.md — adding one explicit clause to the Decision section
  itself would close the loop. This is a docs nit upgraded to Major
  because the file is the *track-closing calibration call* and its
  current wording reads more confidently than the data warrants.
- Major #2 (integration spec missing the upload leg) is real, but
  the spec covers the higher-risk path (badge + reprocess override)
  and the lower-risk upload modal is exercised by A.2 specs
  transitively. Honestly classified as "should have been included
  per plan, but skipping it doesn't break the gate".

All test artefacts are green. The threshold-tuning math checks out
against the implementation (e.g. Convenant 0.850 = 0.30*1 + 0.20*1 +
0.15*1 + 0.15*0 + 0.10*1 + 0.10*1; Bennett_test 0.7255 = 0.30 + 0.20
+ 0.15*0.23 + 0 + 0.10*0.91 + 0.10). The CLI uses the production
scoring function (no re-implementation). The synthetic fixtures are
appropriately disclaimed as not part of the decision corpus. The
Playwright spec passes on a real `next dev` instance (reviewer
verified, not mocked).

Track A is genuinely closeable. The two majors should be addressed
in a follow-up PR or as an amendment commit on this branch before
merge; if the implementer chooses the amendment route, the changes
are < 20 LOC of documentation. Not gated on the merge happening
today.

## Next steps

**If accepted as APPROVED**: ready for human review and merge. The
two majors and five minors can land as a single docs cleanup commit
either on this branch (preferred) or as the first commit of a future
clean-up PR. Recommended sequence for the cleanup commit:

1. Add one sentence to `threshold-tuning.md` Decision section
   acknowledging 5-PDF corpus contingency.
2. Fix the "still pending" CI-workflow notes in RETRO.md,
   status.md, and self-review.md (or delete the self-review file,
   since this review supersedes it).
3. Correct the "any one bad table → 0.0" wording in RETRO.md and
   threshold-tuning.md follow-ups.
4. (Optional) Append the upload-flow leg to the integration spec
   (Major #2).

**If the implementer prefers to reject this APPROVED and rework**:
that is also fine — the changes above are all docs and would land
inside a single sub-day amendment.
