SUPERGOAL_PHASE_START
Phase: 1 of 4 — B.8a Model + provenance fix (Backend)
Task: Switch extraction to qwen2.5:14b, fix extraction_method provenance, surface swallowed extraction errors.
Type: brownfield, bugfix, backend
Mandatory commands: curl -s --max-time 5 http://localhost:11434/api/tags, git diff --name-only b796f7c -- '*.py' | xargs -r uv run ruff check, uv run pytest packages/shared/tests apps/app-main/tests -k "extract or entity or persist" -q
Acceptance criteria: 8
Evidence required: tag-confirm, config diff, provenance-fix diff, before/after entity_method, tests green, adversarial-reviewer APPROVED, status ledger entry
Depends on phases: none

## Why
The user chose qwen2.5:14b; today EXTRACTION_MODEL=llama3.1:8b. And `extraction_method` is never populated (always defaults "llm"), so we can't even prove which path/model produced an entity — fix that so the live validation in B.8c is trustworthy.

## Work
- Branch `track/b8a-model-provenance`.
- Confirm the exact qwen Ollama tag via `/api/tags` (plan-time saw `qwen2.5:14b-instruct-q5_K_M`); set `EXTRACTION_MODEL` (docker-compose.yml:130) + any hardcoded extraction default to it.
- Fix `extraction_method` propagation: thread the real method/model from the extraction path into the persisted Entity (entity_persistence_service.py ~156-163 builds Entity without setting it; handlers.py ~250 knows the type but it's not threaded). Record the model used (provenance_chain or a model field) so B.8c can prove qwen ran.
- Replace silent `except: return []` / `pass` in the entity write + extraction-result path with logged, surfaced errors (the swallowed-error pattern hid the KG ORDER BY bug too).
- Add/extend unit tests for: extraction_method is set to the real path; an extraction error surfaces instead of returning empty.
- Run adversarial-reviewer on the branch; revise until APPROVED (max 3 attempts; escalate per docs/tracks/README.md on impasse).
- Append a status.md ledger row to docs/tracks/B-kg-quality/status.md (phase B.8a, branch, date, outcome).

## Acceptance criteria (all must pass — verify each in transcript)
- The qwen tag used is confirmed present in `/api/tags` (exact string).
- `EXTRACTION_MODEL` + any hardcoded default now reference the qwen tag (show diff).
- `extraction_method` is populated from the real extraction path (not the model default) — shown by a unit test asserting a non-default value end-to-end.
- At least one previously-swallowed extraction failure now logs/raises (shown by a test).
- `uv run pytest ... -k "extract or entity or persist"` exits 0.
- Ruff is clean on THIS phase's changed Python files (scope to the diff vs baseline b796f7c; do NOT fix the ~224 pre-existing repo-wide import-sort errors — that debt is out of scope).
- adversarial-reviewer returns APPROVED (paste verdict).
- A B.8a row is appended to docs/tracks/B-kg-quality/status.md.

## Mandatory commands (run each, surface last ~10 lines + exit code)
- curl -s --max-time 5 http://localhost:11434/api/tags
- git diff --name-only b796f7c -- '*.py' | xargs -r uv run ruff check   # changed files only; baseline has ~224 pre-existing import-sort errors (xargs -r = no-op if nothing changed)
- uv run pytest packages/shared/tests apps/app-main/tests -k "extract or entity or persist" -q

## Evidence required in transcript
- `/api/tags` proving the qwen tag.
- The two diffs (config + provenance).
- Test output showing extraction_method populated + error surfaced.
- adversarial-reviewer APPROVED verdict + the status.md ledger row.

## Notes
Do NOT rebuild the image here (B.8b). The multi-schema orchestrator is the live path when notebook_id is supplied + extractor_type=="llm". Don't touch resolution logic (assess-only is B.8c). Upsert atomicity (B.1e-deferred) is sequential-safe for this run — document it in B.8d, don't fix it here unless review demands.
