# Phase A.3 — Self-review

> Implementer self-review prior to handing off to the reviewer agent.
> Date: 2026-06-04. Branch: `track/a-mineru-integration`.

## Plan adherence

Eight-commit sequence from `docs/tracks/A-mineru/plan-A.3.md` §12.
Commit 4 (default-threshold bump) was conditional on tuning concluding
"change default" — tuning kept 0.95 (see decision in
`threshold-tuning.md`) so commit 4 is intentionally skipped. Final
sequence:

| # | Plan | Actual SHA | Notes |
|---|------|------------|-------|
| 1 | fixtures + generator | `f229ea7` | Three PDFs ≤ 17 KB each via stdlib + Pillow. No `reportlab` dep added. |
| 2 | score_pdf_corpus CLI + smoke test | `98af375` | 8 smoke tests passing; CLI driven against docling HTTP service. |
| 3 | threshold-tuning report | `5f8f6d7` | 5 real PDFs + 3 synthetics; decision: keep 0.95. |
| 4 | (conditional bump) | — | Skipped per tuning outcome. |
| 5 | integration spec | `ab2b8d3` | Two test blocks: happy path + graceful degradation. 10/10 Track A passing. |
| 6 | README + ARCHITECTURE.md | `07f16ab` | Insertion points per pre-resolved decisions. |
| 7 | troubleshooting page | `badd747` | 4 scenarios; index pointer added. |
| 8 | RETRO + status + roadmap | this commit | Track A marked COMPLETE. |

## Acceptance criteria check

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `threshold-tuning.md` has ≥ 3 real-PDF rows with name + score + decision + manual quality call | ✅ 5 real-PDF rows + 3 synthetic baseline rows |
| 2 | 0.95 confirmed or new default proposed with code change | ✅ kept 0.95 with documented rationale |
| 3 | `parser-engine-integration.spec.ts` exists + passes via route mocks | ✅ 2 tests, both green |
| 4 | README has "Parser engines" subsection | ✅ under Advanced Features |
| 5 | `architecture.md` mentions dispatcher + `Source.metadata` | ✅ inserted after §4 AI Processing Layer |
| 6 | `troubleshooting/parser-engines.md` exists with ≥ 3 scenarios + index link | ✅ 4 scenarios + linked from index |
| 7 | `RETRO.md` has the three required sections | ✅ "What worked" / "What hurt" / "Recommendations for tracks B-G" |
| 8 | No regressions: pytest + lint + tsc + Playwright all green | ✅ all green |
| 9 | `status.md` has final entry with "Track A — COMPLETE" | ✅ |
| 10 | `FEATURE_ROADMAP.md` Track A marked done | ✅ banner added with RETRO + tuning pointers |

## Quality gates run

```
$ uv run --project apps/app-main pytest apps/app-main/tests/ -q
…
357 passed
```

```
$ uv run --project packages/shared pytest packages/shared/tests/ -q
…
105 passed
```

```
$ cd frontend && npx tsc --noEmit
# (clean)

$ cd frontend && npm run lint
# (no new warnings; pre-existing warnings only)

$ cd frontend && PLAYWRIGHT_BASE_URL=http://localhost:8503 \
    npx playwright test e2e/track-a/ --reporter=line --workers=1
10 passed (1.0m)
```

## What changed since the plan was written

Nothing material — the plan was written within hours of the
implementation and all open questions were pre-resolved in the
prompt. One minor adjustment: the Reprocess modal's parser-engine
dropdown uses bare engine names (`Docling`, `MinerU`) rather than
the `Docling (default)` qualifier used on the global settings
dropdown. The integration spec was updated accordingly.

## Risks / known gaps

1. The host-side docling bridge in `score_pdf_corpus.py` is a
   workaround for the docker bind-mount path mismatch (host
   `docling_input/` mounted as container `/data/input`). Documented
   in RETRO.md as a follow-up.
2. Live SurrealDB round-trip testing for `Source.metadata` is still
   deferred. The integration spec covers the API → UI side; the
   DB → API read-back is fixture-mocked. Belongs to a testcontainers
   PR in Track B.
3. The `table_success` confidence signal zeros on any one bad
   table — documented in `threshold-tuning.md` Follow-ups and
   `RETRO.md` "What hurt".
4. Phase A.0's CI workflow (`docs/tracks/A-mineru/e2e-workflow.yml.pending`)
   is still pending the `git mv` to `.github/workflows/e2e.yml`.
   Orchestrator task; Track A's CI smoke contract does not fire
   until resolved.

## Hand-off

Branch pushed to origin. Reviewer can:

1. Read `docs/tracks/A-mineru/RETRO.md` for the track-level summary.
2. Read `docs/tracks/A-mineru/threshold-tuning.md` for the calibration
   decision.
3. Spot-check the integration spec at
   `frontend/e2e/track-a/parser-engine-integration.spec.ts`.
4. Run the local quality gates above to verify.
