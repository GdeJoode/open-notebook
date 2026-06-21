SUPERGOAL_PHASE_START
Phase: 2 of 3 — Address revisions
Task: Resolve every BLOCKER/Major review finding on track/i-upload-guards with surgical commits; keep tests green.
Type: brownfield
Mandatory commands: cd apps/app-main && python -m pytest tests/test_upload_guards.py tests/test_rate_limiter.py -x, ruff check apps/app-main/src/app_main/api apps/app-main/tests/test_upload_guards.py apps/app-main/tests/test_rate_limiter.py
Acceptance criteria: 5
Evidence required: per-finding resolution table, test output, ruff output
Depends on phases: 1

## Why

Turn review findings into a clean, mergeable branch without scope creep.

## Work

- Read the findings from `docs/tracks/I-docling-studio/reviews/phase-I.H1-attempt-1.md`.
- For each BLOCKER and Major: implement the smallest correct fix. One commit per finding, conventional message scoped `fix(api): ... (I.H1)`, ending with the trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- For each Minor: fix if cheap, else defer with a one-line reason recorded in the review doc.
- If a finding requires a design decision rather than a surgical fix (e.g. "AC5 needs a Redis-backed shared store to be truly per-IP across workers"), do NOT bluff a fix. Record the trade-off, implement the defensible minimal option (or document the limitation), and only escalate (FAILURE_HANDOFF) if it genuinely blocks merge.
- Append an "Attempt 1 — revisions" section to the review doc: a table mapping each finding → resolution (fix commit sha, or justified wontfix/defer).
- Re-run the two test files and ruff after the fixes.

## Acceptance criteria (all must pass — verify each in transcript)

- Every BLOCKER and Major finding is fixed (with a commit) or carries a written, defensible wontfix justification in the review doc.
- Every Minor finding is addressed or explicitly deferred with a one-line reason.
- `git diff main...track/i-upload-guards --stat` touches only I.H1-scope files (config.py, app.py, rate_limit.py, sources_upload.py, pyproject.toml, uv.lock, the two test files, Track I docs) — no unrelated refactors.
- `test_upload_guards.py` + `test_rate_limiter.py` pass.
- If Phase 1 was APPROVED with zero BLOCKER/Major: this phase records "no revisions required" in the review doc and still re-runs the tests green.

## Mandatory commands (run each, surface last ~10 lines + exit code)

- `cd apps/app-main && python -m pytest tests/test_upload_guards.py tests/test_rate_limiter.py -x`
- `ruff check apps/app-main/src/app_main/api apps/app-main/tests/test_upload_guards.py apps/app-main/tests/test_rate_limiter.py`

## Evidence required in transcript

- A per-finding resolution table (finding → severity → resolution → commit/justification).
- Test pass count and ruff clean status.

## Notes

Surgical discipline (global rule): every changed line must trace to a specific finding. Do not "improve" adjacent code. If you spot something unrelated, note it in the review doc as a follow-up, don't fix it here.
