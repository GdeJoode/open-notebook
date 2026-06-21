SUPERGOAL_PHASE_START
Phase: 3 of 3 — Verify, push & PR (Polish & Harden)
Task: Re-verify against the I.H1 plan, push the branch to origin, surface the PR, update the status ledger.
Type: brownfield
Mandatory commands: cd apps/app-main && python -m pytest tests/test_upload_guards.py tests/test_rate_limiter.py -x, cd apps/app-main && python -m pytest --collect-only -q, ruff check apps/app-main/src/app_main/api apps/app-main/tests/test_upload_guards.py apps/app-main/tests/test_rate_limiter.py, git push -u origin track/i-upload-guards, git ls-remote --heads origin track/i-upload-guards
Acceptance criteria: 6
Evidence required: command outputs, PR compare URL, final diff --stat, per-AC spot-check table
Depends on phases: 1, 2

## Why

Final independent re-verification against the original plan, then ship the branch so the human can merge. This enforces "every aspect verified".

## Work

- Re-run the full mandatory command set: the two test files, a `pytest --collect-only -q` import-health sweep (proves the slowapi wiring + new module don't break collection suite-wide), and ruff.
- Review `git diff main...track/i-upload-guards` for stray debug prints, session TODO/FIXME, or dead imports introduced by this run. Remove any found (commit scoped `(I.H1)`).
- Spot-check the 5 I.H1 plan ACs against the final code: (1) 413 on oversize file with limit in detail; (2) 422 on oversize page count via pypdfium; (3) 429 with `Retry-After` on burst; (4) `RateLimitError` handler still registered in app.py; (5) per-IP keying via get_remote_address. Mark each pass / trust-prior with the evidence used.
- Update the I.H1 row in `docs/tracks/I-docling-studio/status.md` from "in progress" to reflect pushed / ready-for-review (include date 2026-06-19). Commit it scoped `docs(track-i): ... (I.H1)`.
- `git push -u origin track/i-upload-guards`.
- Print the PR compare URL `https://github.com/GdeJoode/open-notebook/compare/main...track/i-upload-guards` and state clearly that opening the PR is a manual one-click step (gh CLI unavailable in this environment).

## Acceptance criteria (all must pass — verify each in transcript)

- The two test files pass AND `pytest --collect-only -q` reports 0 import errors AND ruff is clean.
- `git diff main...track/i-upload-guards` reviewed; no stray debug prints / session TODO-FIXME from this run remain.
- `git ls-remote --heads origin track/i-upload-guards` returns a non-empty ref (branch is on origin).
- The PR compare URL is printed with the explicit note that PR open is manual (no gh).
- The status-ledger row for I.H1 is updated and committed.
- A per-AC spot-check table for the 5 I.H1 plan ACs is printed, each pass / trust-prior with evidence.

## Mandatory commands (run each, surface last ~10 lines + exit code)

- `cd apps/app-main && python -m pytest tests/test_upload_guards.py tests/test_rate_limiter.py -x`
- `cd apps/app-main && python -m pytest --collect-only -q`
- `ruff check apps/app-main/src/app_main/api apps/app-main/tests/test_upload_guards.py apps/app-main/tests/test_rate_limiter.py`
- `git push -u origin track/i-upload-guards`
- `git ls-remote --heads origin track/i-upload-guards`

## Evidence required in transcript

- Each mandatory command's last ~10 lines + exit code.
- The printed PR compare URL.
- Final `git diff --stat main...track/i-upload-guards`.
- Per-AC spot-check table (5 rows).

## Notes

If `git push` fails on auth (no cached credentials), do NOT loop on it — surface the failure, leave the branch committed locally, and print the manual `git push -u origin track/i-upload-guards` + compare URL for the user. That counts as the deliverable boundary given the environment, recorded honestly.
