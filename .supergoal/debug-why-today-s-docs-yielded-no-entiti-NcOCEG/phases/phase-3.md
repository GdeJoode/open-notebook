SUPERGOAL_PHASE_START
Phase: 3 of 4 — B.8c Live validation + 11-doc resolution assessment (Integration)
Task: Trigger qwen extraction on the parsed docs + ingest/extract the 11 Convenant PDFs; measure cross-doc resolution; honest ceiling + M4 recommendation (assess-only).
Type: brownfield, integration, investigation, data
Mandatory commands: ls tests/pdfs, uv run pytest packages/surrealdb-service/tests -k "entity or resolution or merge" -q
Acceptance criteria: 9
Evidence required: trigger result on 2 docs, 11-doc ingest+extract, per-doc counts, resolution metric, 3+ shared-entity examples, honest verdict, review doc, reviewer APPROVED, ledger entry
Depends on phases: 1, 2

## Why
The user's core goal: confirm extraction actually produces entities (the operational "no entities" question) AND assess how well the related Regio Deal / Convenant docs resolve to shared canonical entities — knowing the V1 name_normalizer caps this (full TOOI/Crossref = deferred Track M4).

## Work
- Branch `track/b8c-live-validation`.
- Trigger entity extraction for the 2 already-parsed docs via the real seam (`POST /api/sources/{id}/run-entities`, resume if PAUSED_FOR_REVIEW). Confirm entities now exist with qwen + correct extraction_method. This answers the operational "why no entities" definitively (was never triggered / was paused).
- Ingest all 11 `tests/pdfs/Convenant*.pdf` into a fresh notebook; run extraction on each (sequential; checkpoint each doc's entity count to STATE.md Notable events).
- Document the resolution mechanism precisely (name_normalizer V1 + idx_entity_name_type exact match + embedding_dedup@0.90 in filtering) with file:line.
- Measure cross-document resolution: distinct canonical entities vs total mentions; top shared entities and how many of the 11 docs each spans; any exact-duplicate canonical_name+type rows that should have merged.
- Write `docs/tracks/B-kg-quality/reviews/phase-B.8c-resolution-assessment.md`: the metric, 3+ concrete shared-entity examples, an HONEST verdict (met / partially met / capped-by-V1), and a recommendation on whether to pull Track M4/Q9 (TOOI/Crossref + NL normalization) forward. NO resolver code change this run.
- adversarial-reviewer until APPROVED (max 3); append B.8c status.md ledger row.

## Acceptance criteria (all must pass — verify each in transcript)
- Extraction triggered on the 2 parsed docs; entity count goes from 0 to >0 for them (DB-verified) with qwen extraction_method.
- The operational root cause of "no entities" is stated with evidence (never-triggered vs paused-for-review).
- All 11 Convenant PDFs ingested (11 sources confirmed) and extraction run on each (per-doc counts shown; failures surfaced).
- Resolution mechanism documented with file:line (exact-match + normalizer + embedding-dedup).
- A cross-doc resolution metric is computed (distinct canonical vs mentions) + 3+ shared entities spanning ≥2 docs each.
- An honest resolution verdict + M4/Q9 recommendation is written to the review doc.
- `uv run pytest ... -k "entity or resolution or merge"` exits 0.
- adversarial-reviewer returns APPROVED.
- A B.8c row is appended to docs/tracks/B-kg-quality/status.md.

## Mandatory commands (run each, surface last ~10 lines + exit code)
- ls tests/pdfs
- uv run pytest packages/surrealdb-service/tests -k "entity or resolution or merge" -q

## Evidence required in transcript
- The 2-doc trigger result (0 → >0).
- 11-source ingest confirmation + per-doc entity counts.
- Resolution metric + 3+ shared-entity examples.
- The review doc path + verdict + recommendation.
- reviewer APPROVED + ledger row.

## Notes
qwen2.5:14b over 11 PDFs is slow — sequential, checkpoint per doc to STATE so progress survives a restart. Assess-only: report the honest ceiling, do NOT change the resolver. Do not delete the user's data or the ingested sources. If the 2 parsed docs were paused for schema review, resuming is the fix — say so plainly.
