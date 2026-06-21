SUPERGOAL_PHASE_START
Phase: 1 of 3 — Adversarial review
Task: Independently review the track/i-upload-guards branch (Phase I.H1) and produce a verdict + findings list.
Type: brownfield
Mandatory commands: git checkout track/i-upload-guards, git diff --stat main...track/i-upload-guards
Acceptance criteria: 4
Evidence required: verdict line, findings list
Depends on phases: none

## Why

The Track I merge gate. An independent skeptic must try to break the branch before it merges, mirroring the two-round adversarial review that I.A went through.

## Work

- Check out `track/i-upload-guards` in the working tree (currently on `main`). The untracked Track-I docs in the tree carry over harmlessly.
- Read the phase spec for I.H1: `docs/tracks/I-docling-studio/plan.md` → section "### Phase I.H1 — Upload guards + per-IP rate limiting" (its 5 acceptance criteria) and the implementer's self-review `docs/tracks/I-docling-studio/reviews/phase-I.H1-self-review.md`.
- Invoke the `adversarial-reviewer` agent (via the Agent tool, subagent_type: "adversarial-reviewer") on the diff `main...track/i-upload-guards`. Give it: the 5 plan ACs, the self-review's two flagged gaps (AC5 multi-worker per-IP; upload-rewind not asserted), and an explicit instruction to scrutinize (a) whether the size guard reads the whole upload into memory before rejecting — which would defeat the OOM-prevention purpose — and (b) whether the 429 path actually emits `Retry-After`.
- Capture the reviewer's output verbatim into `docs/tracks/I-docling-studio/reviews/phase-I.H1-attempt-1.md`: verdict (APPROVED or REVISIONS_NEEDED) + numbered findings, each tagged BLOCKER / Major / Minor with file:line and rationale.

## Acceptance criteria (all must pass — verify each in transcript)

- `git branch --show-current` returns `track/i-upload-guards`.
- The adversarial-reviewer agent ran against the full `main...track/i-upload-guards` diff with the I.H1 plan ACs as the rubric.
- `docs/tracks/I-docling-studio/reviews/phase-I.H1-attempt-1.md` exists and contains an explicit verdict plus (if REVISIONS_NEEDED) a numbered, severity-tagged findings list with file:line references.
- The review explicitly assesses the AC5 per-IP/multi-worker claim AND the upload-read-into-memory behavior (the two highest-risk spots).

## Mandatory commands (run each, surface last ~10 lines + exit code)

- `git checkout track/i-upload-guards`
- `git diff --stat main...track/i-upload-guards`

## Evidence required in transcript

- The verdict line (APPROVED / REVISIONS_NEEDED).
- The findings list, or "no findings — APPROVED".

## Notes

Do not fix anything in this phase — review only. Fixes are Phase 2. If the reviewer returns APPROVED with zero BLOCKER/Major, that is a valid outcome; Phase 2 becomes a documented no-op + test re-run.
