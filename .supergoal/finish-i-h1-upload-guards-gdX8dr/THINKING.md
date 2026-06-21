# THINKING — Finish Phase I.H1 (upload guards + per-IP rate limiting)

## Goals
- Take the existing branch `track/i-upload-guards` (5 commits, I.H1 implementation + tests + self-review) from "implemented" to "ready for merge".
- Run a genuine adversarial review (the established Track I workflow gate), fix what it finds, and get the branch pushed with a PR ready.

## Constraints
- Branch already exists with committed work; do NOT re-implement from scratch. Only review + revise.
- Surgical edits only (global CLAUDE.md rule): every changed line traceable to a review finding.
- Follow Track I conventions: per-phase review doc under `docs/tracks/I-docling-studio/reviews/`, status-ledger row, conventional commits scoped `(I.H1)`, co-author trailer.
- Do NOT touch `main`. All work on `track/i-upload-guards`.

## Risks (top 3)
1. **Test env split.** Implementer ran pytest in a WSL worktree against a provisioned root venv (slowapi installed there). The Windows conda env (`E:\Python\Anaconda3`) may lack `slowapi`/`pypdfium2`, so `pytest` could fail on import here — a false "red" unrelated to code quality. Mitigation: Stage 6.5 pre-flight runs the test command; if red due to missing deps, `pip install slowapi` into the active env or surface honestly. Likelihood: medium.
2. **`gh` CLI unavailable.** Verified absent. The autonomous run can `git push` but cannot open the PR. Mitigation: push the branch and print the GitHub compare URL (`.../compare/main...track/i-upload-guards`) for the user to click. PR creation is a manual one-click step, documented as such. Likelihood: certain (already known).
3. **Adversarial reviewer finds a BLOCKER requiring real rework** (e.g. the AC5 per-IP/multi-worker honesty gap, or a guard that reads the whole upload into memory defeating the OOM purpose). Mitigation: phase 2 has the 3-strike + fix-spec loop; if a finding needs a design decision beyond surgical fixing, escalate (FAILURE_HANDOFF) rather than bluff. Likelihood: medium.

## Dependencies / ordering
- Phase 2 (revisions) depends entirely on Phase 1's findings.
- Phase 3 (push + PR) depends on Phase 2 leaving tests green and the reviewer's blockers resolved.
- Working tree currently on `main` with unrelated modified/untracked docs; run must `git checkout track/i-upload-guards` first (untracked docs carry over harmlessly).

## Open questions (assumed, correct at gate)
- PR target = `origin` (GdeJoode fork), base `main`. Not `upstream`.
- "Ready for merge" = reviewer has no unaddressed BLOCKER/Major findings + tests green + branch pushed + PR URL surfaced. Actual merge stays the user's call.

## Tools / skills relied on
- `adversarial-reviewer` agent (project agent) for Phase 1.
- git (push); `gh` NOT available → PR via printed compare URL.
- pytest / ruff for verification.
- No Context7/WebSearch needed (no external SDK design work).

## Memory
- No project memory index present. Worth writing at run end: a `project_*` memory noting Track I state + the WSL/Windows test-env split (a real gotcha for future runs).
