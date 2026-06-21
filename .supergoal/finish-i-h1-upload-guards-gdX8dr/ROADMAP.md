# Roadmap: Finish Phase I.H1 — upload guards + per-IP rate limiting

**Task:** Drive `track/i-upload-guards` to ready-for-merge: adversarial review → address revisions → push + PR.
**Type:** brownfield
**Created:** 2026-06-19
**Total phases:** 3

## Context summary

- **Stack:** Python monorepo (FastAPI app at `apps/app-main`), uv-managed; Next.js frontend (not touched this phase).
- **Package manager:** uv (python); pytest with `asyncio_mode = "auto"`, `testpaths = ["tests"]`.
- **Build / test / lint commands:** `python -m pytest` (in `apps/app-main`), `ruff check`.
- **Risky areas:** test env is WSL-provisioned (slowapi/pypdfium2 may be missing in Windows conda env); `gh` CLI unavailable (no autonomous PR open).

## Assumptions

If any are wrong, stop the run and tell us:

- The 5 commits on `track/i-upload-guards` are the implementation under review; we review + revise, never re-implement.
- PR target is `origin` (GdeJoode fork), base `main`. The branch will be pushed; the PR is opened by the user via the printed compare URL (no `gh`).
- "Ready for merge" = no unaddressed BLOCKER/Major review findings + tests green + branch on origin + PR URL surfaced. The actual merge click stays the user's decision.
- The known AC5 limitation (slowapi in-memory store is per-process, not a shared cross-worker IP budget) is acceptable as a documented limitation, not a blocker — unless the reviewer argues otherwise with a concrete exploit.

## Risk top 3

1. **Test env split (WSL vs Windows)** — likelihood: medium, mitigation: pre-flight runs the test command; `pip install slowapi pypdfium2` into the active env if missing, else surface honestly.
2. **`gh` unavailable** — likelihood: certain, mitigation: `git push` + print GitHub compare URL; PR open is a manual one-click step.
3. **Reviewer finds a real BLOCKER needing rework** — likelihood: medium, mitigation: 3-strike + fix-spec loop in phase 2; escalate (FAILURE_HANDOFF) if a design decision is required rather than bluffing a fix.

## Phase map

| # | Phase | Depends on | Deliverable |
|---|-------|------------|-------------|
| 1 | Adversarial review | — | `reviews/phase-I.H1-attempt-1.md` with verdict + enumerated findings |
| 2 | Address revisions | 1 | Each BLOCKER/Major resolved (commit or justified wontfix); tests green |
| 3 | Verify, push & PR | 1, 2 | Branch on origin; PR compare URL printed; status ledger updated |

---

## Phase 1 — Adversarial review

**Why:** The Track I gate. An independent skeptic must try to break the branch before it merges (mirrors the I.A two-round review).

**Deliverables:**
- `docs/tracks/I-docling-studio/reviews/phase-I.H1-attempt-1.md`

**Acceptance criteria:**
- [ ] `track/i-upload-guards` is checked out in the working tree (`git branch --show-current` == `track/i-upload-guards`).
- [ ] The `adversarial-reviewer` agent reviewed the full diff `main...track/i-upload-guards` against the I.H1 plan section + its 5 acceptance criteria.
- [ ] A review doc exists at the deliverable path containing: an explicit verdict (`APPROVED` or `REVISIONS_NEEDED`) and, if the latter, a numbered list of findings each tagged BLOCKER / Major / Minor with file:line and rationale.
- [ ] The AC5 multi-worker/per-IP claim and the upload-read-into-memory behavior are explicitly assessed in the review (these are the two highest-risk spots).

**Mandatory commands:**
- `git checkout track/i-upload-guards`
- `git diff --stat main...track/i-upload-guards`

**Evidence required:**
- The verdict line printed into the transcript.
- The findings list (or "no findings — APPROVED") printed into the transcript.

**Dependencies:** none

---

## Phase 2 — Address revisions

**Why:** Turn review findings into a clean, mergeable branch without scope creep.

**Deliverables:**
- One commit per resolved BLOCKER/Major finding on `track/i-upload-guards` (conventional message scoped `(I.H1)`, co-author trailer).
- An "Attempt 1 — revisions" section appended to `reviews/phase-I.H1-attempt-1.md` mapping each finding → resolution (fix commit sha, or justified wontfix).

**Acceptance criteria:**
- [ ] Every BLOCKER and Major finding from phase 1 is either fixed (with a commit) or has a written, defensible wontfix justification in the review doc.
- [ ] Minor findings are addressed or explicitly deferred with a one-line reason.
- [ ] Edits are surgical: `git diff main...track/i-upload-guards` touches only files within I.H1 scope (`config.py`, `app.py`, `rate_limit.py`, `sources_upload.py`, `pyproject.toml`, `uv.lock`, the two test files, and Track I docs). No unrelated refactors.
- [ ] `test_upload_guards.py` + `test_rate_limiter.py` pass after revisions.
- [ ] If phase 1 verdict was APPROVED with zero BLOCKER/Major, this phase is a documented no-op (record "no revisions required") and still re-runs the tests.

**Mandatory commands:**
- `cd apps/app-main && python -m pytest tests/test_upload_guards.py tests/test_rate_limiter.py -x`
- `ruff check apps/app-main/src/app_main/api apps/app-main/tests/test_upload_guards.py apps/app-main/tests/test_rate_limiter.py`

**Evidence required:**
- Per-finding resolution table printed into the transcript.
- Test output (pass count) and ruff output (clean) printed into the transcript.

**Dependencies:** 1

---

## Phase 3 — Verify, push & PR (Polish & Harden)

**Why:** Final independent re-verification against the plan, then ship the branch so the human can merge. This is the "every aspect verified" gate.

**Deliverables:**
- `track/i-upload-guards` pushed to `origin`.
- GitHub PR compare URL printed for the user.
- `docs/tracks/I-docling-studio/status.md` ledger row for I.H1 updated to reflect "pushed / ready for review".

**Acceptance criteria:**
- [ ] Full mandatory command set re-run clean (the two test files + a `pytest --collect-only` import-health sweep + ruff).
- [ ] No stray debug prints / session TODO-FIXME introduced by this run (`git diff main...track/i-upload-guards` reviewed).
- [ ] Branch present on origin (`git ls-remote --heads origin track/i-upload-guards` non-empty).
- [ ] PR compare URL `https://github.com/GdeJoode/open-notebook/compare/main...track/i-upload-guards` printed (gh unavailable → manual one-click open, stated explicitly).
- [ ] Status ledger row updated and committed.
- [ ] The 5 plan ACs for I.H1 spot-checked against final code (413 size guard, 422 page guard, 429+Retry-After, RateLimitError handler intact, per-IP keying) — each marked pass / trust-prior with evidence.

**Mandatory commands:**
- `cd apps/app-main && python -m pytest tests/test_upload_guards.py tests/test_rate_limiter.py -x`
- `cd apps/app-main && python -m pytest --collect-only -q`
- `ruff check apps/app-main/src/app_main/api apps/app-main/tests/test_upload_guards.py apps/app-main/tests/test_rate_limiter.py`
- `git push -u origin track/i-upload-guards`
- `git ls-remote --heads origin track/i-upload-guards`

**Evidence required:**
- Each mandatory command's last ~10 lines + exit code.
- The printed PR compare URL.
- Final `git diff --stat main...track/i-upload-guards`.
- Per-AC spot-check table.

**Dependencies:** 1, 2
