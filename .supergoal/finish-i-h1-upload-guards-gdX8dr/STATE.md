# State: Finish Phase I.H1 — upload guards + per-IP rate limiting

**Status:** COMPLETE (mode B — direct in-session via WSL)
**Current phase:** 3 (done)
**Started:** 2026-06-19
**Last update:** 2026-06-19
**Run root:** .supergoal/finish-i-h1-upload-guards-gdX8dr
**Baseline ref:** 21e0108 (main)    <!-- deliverables diffed against main -->

<!-- ENV NOTE: all code/test/git commands run via `wsl bash -lc` against the
     repo at /mnt/e/repos/private/open-notebook using the root .venv
     (/mnt/e/repos/private/open-notebook/.venv/bin/python). Windows Git Bash
     picks the wrong python (no pytest). Pre-flight: 7/7 passed via WSL. -->



## Phase progress

| # | Phase | Status | Started | Completed | Notes |
|---|-------|--------|---------|-----------|-------|
| 1 | Adversarial review | done | 2026-06-19 | 2026-06-19 | REVISIONS_NEEDED: 2 BLOCKER + 2 Major + 2 Minor |
| 2 | Address revisions | done | 2026-06-19 | 2026-06-19 | all findings resolved; 9/9 tests pass; commits 527e920/fbec50d/ae8bc72 |
| 3 | Verify, push & PR | done | 2026-06-19 | 2026-06-19 | 9/9 tests + 599 collected clean; pushed to origin (210b260); PR URL surfaced |

## Final AC spot-check (5 plan ACs for I.H1)

| AC | Verdict | Evidence |
|---|---|---|
| 1 — 413 oversize file, detail references limit | pass | `test_oversize_file_rejected_with_413` (asserts 413 + "MAX_FILE_SIZE_MB" in detail + no DB call) |
| 2 — 422 oversize pages via pypdfium | pass | `test_oversize_pages_rejected_with_422`; guard now streams pypdfium from spooled file |
| 3 — 429 + Retry-After on burst | pass | `test_burst_from_one_ip_trips_429_with_retry_after` (asserts header) |
| 4 — RateLimitError handler still active | trust-prior | app.py handler unchanged by revisions; reviewer confirmed app.py wiring correct |
| 5 — per-IP keying | pass (keying); documented limitation (multi-worker) | `test_production_limiter_keyed_per_ip` + `test_second_ip_is_unaffected`; in-memory store = per-process, deferred |

## Engineering check status

- Build: — (n/a, no build step this phase)
- Typecheck: — (n/a, python)
- Lint: —
- Tests: —

## Notable events

- 2026-06-19 — Plan locked, 3 phases.

## Failure log

(none yet)
