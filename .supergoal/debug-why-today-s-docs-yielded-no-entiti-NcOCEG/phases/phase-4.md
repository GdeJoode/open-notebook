SUPERGOAL_PHASE_START
Phase: 4 of 4 — B.8d Polish, Harden + track ledger
Task: Regression-green the suites, consolidate findings (incl. schema drift + deferred atomicity), update track ledger + roadmap, write memory.
Type: brownfield, hardening
Mandatory commands: git diff --name-only b796f7c -- '*.py' | xargs -r uv run ruff check, uv run pytest packages/surrealdb-service/tests apps/app-main/tests packages/shared/tests -q
Acceptance criteria: 9
Evidence required: lint clean, suites green, findings doc, schema-drift decision, deferred-items note, status ledger + roadmap update, memory, reviewer APPROVED, clean git status
Depends on phases: 1, 2, 3

## Why
Lock in the B.8 fixes, capture non-obvious findings durably, and leave the track ledger consistent so this follow-up is a first-class part of Track B's audit trail.

## Work
- Branch `track/b8d-polish`.
- Run full ruff + the relevant test suites across apps + packages + shared; fix any regressions from B.8a-c.
- Write `docs/tracks/B-kg-quality/reviews/phase-B.8-findings.md` consolidating: operational no-entities root cause, model switch, extraction_method provenance fix, ORDER BY fix + deploy, default-filter finding, and the resolution verdict + M4/Q9 recommendation.
- Codify the schema-drift decision: `idx_entity_fulltext` + the `name` field + index exist on the live DB but in NO migration (the documented Q-B-1 drift). Decide: add a migration to codify them OR a documented drop. Pick the safe option and justify.
- Record deferred items explicitly (upsert SELECT-then-UPDATE atomicity from B.1e; full TOOI/Crossref resolution = Track M4) in the findings doc — not silently dropped.
- Update docs/tracks/B-kg-quality/status.md (B.8 complete row) and add a B.8 follow-up note under Track B in docs/FEATURE_ROADMAP.md.
- Remove temp/debug artifacts (container probe scripts, stray files); confirm `git status` shows only intended changes.
- Write a `project_open-notebook-kg.md` memory (live extraction path, qwen model, resolution V1 ceiling, schema-drift caveat) + link in MEMORY.md.
- adversarial-reviewer until APPROVED (max 3).

## Acceptance criteria (all must pass — verify each in transcript)
- Ruff clean on files changed across B.8 (scope to the diff vs baseline b796f7c; the ~224 pre-existing repo-wide import-sort errors are explicitly out of scope).
- `uv run pytest packages/surrealdb-service/tests apps/app-main/tests packages/shared/tests -q` passes (pre-existing/`requires_docker`-skipped explicitly noted).
- `docs/tracks/B-kg-quality/reviews/phase-B.8-findings.md` exists and covers all six findings.
- The schema-drift item has a concrete codified decision (migration path OR documented drop).
- Deferred items (atomicity, M4 resolution) are explicitly recorded.
- status.md has a B.8-complete row AND FEATURE_ROADMAP has a B.8 note.
- A `project_open-notebook-kg.md` memory is written + indexed in MEMORY.md.
- `git status` shows only intended files; no leftover debug artifacts.
- adversarial-reviewer returns APPROVED.

## Mandatory commands (run each, surface last ~10 lines + exit code)
- uv run ruff check apps packages
- uv run pytest packages/surrealdb-service/tests apps/app-main/tests packages/shared/tests -q

## Evidence required in transcript
- Lint + test output.
- The findings doc + section list; the schema-drift decision (file path).
- status.md + FEATURE_ROADMAP updates.
- `git status` clean changeset; reviewer APPROVED.

## Notes
This is the "every aspect is perfect" gate. If B.8c concluded resolution is capped by V1, that is the honest verdict — record it with the M4 recommendation, do not fabricate a pass. Do not delete the user's data or the 12 ingested Convenant sources.
