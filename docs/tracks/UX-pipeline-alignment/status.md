# Track UX — Status

## Phase UX.1 — `processing_stage` TS type + stage-polling hook (+ backend list field + backfill migration)

**Branch:** `track/ux1-processing-stage-types-hook` (off `main`) — NOT merged, NOT pushed.
**Commits:** `c47b78c` (Part A), `69c2e90` (Part B), `4ca6079` (Part C).
**State:** Ready for review.

### Part A — Frontend types + hook
- `frontend/src/lib/pipeline/processing-stage.ts` (new): `ProcessingStage` 7-value union mirroring the backend enum, `TERMINAL_STAGES`/`GATED_STAGES` constants, `isTerminalStage`/`isGatedStage` helpers (+ `isPollableStage`).
- `frontend/src/lib/types/api.ts`: import the union; add `processing_stage?: ProcessingStage` to `SourceListResponse` (inherited by `SourceDetailResponse`).
- `frontend/src/lib/hooks/use-sources.ts`: `useSourcePipeline(id, enabled?)` polling GET `/sources/{id}` every 2s while the stage can still advance; `pipelineRefetchInterval` exported for unit test. 404 retry guard mirrors `useSourceStatus`. `useSource`/`useSourceStatus` untouched.
- Tests: `processing-stage.test.ts` (all 7 values + helpers), `use-sources.pipeline.test.ts` (2000 for non-terminal/undefined, false for the 3 stop stages).

### Part B — Backend: processing_stage on the LIST endpoint
- `apps/app-main/src/app_main/api/schemas.py`: `SourceListResponse.processing_stage: str = "ingested"`.
- `packages/surrealdb-service/.../repositories/source.py`: `list_with_metadata` projects `processing_stage` (both branches).
- `apps/app-main/.../routers/sources_crud.py`: list handler populates `processing_stage` from the row with `row.get("processing_stage", "ingested") or "ingested"`.
- Test: `apps/app-main/tests/test_sources_list_processing_stage.py` (schema default/serialization + handler mapping via mocked repo).

### Part C — Backfill migration 73
- `migrations/73.surrealql` (new): drift-only backfill deriving the true stage for rows still at `"ingested"` — mentions edge => complete, active/reference entity => extracted, non-empty aggregate embedding => embedded, else ingested. Idempotent; guards `embedding != NONE` before `array::len`.
- `migrations/73_down.surrealql` (new): documented no-op (data-only forward migration).
- Test: `packages/surrealdb-service/tests/test_migration_73_backfill_stage.py` — container-verified derivation, archived-only exclusion, idempotency, no-clobber.

### Commands
- `frontend/`: `npm run lint` — pass (only pre-existing warnings, none in new files); `npx tsc --noEmit` — 0 errors; `npx vitest run src/lib/pipeline src/lib/hooks/use-sources.pipeline.test.ts` — 36 passed.
- Backend: `pytest test_sources_list_processing_stage.py` 4 passed; `test_schemas_soft_nudge.py` 19 passed (no regression); `test_sources_crud.py` 5 passed; migration 73 container tests 3 passed.
