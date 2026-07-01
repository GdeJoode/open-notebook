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

---

## Phase UX.2 — Canonical `PipelineStatus` component + stage state machine

**Branch:** `track/ux2-pipeline-status-component` (off `main`) — NOT merged, NOT pushed.
**Commits:** `95c6f69` (state machine), `0ac008c` (component-test infra), `7314097` (component + barrel).
**State:** Ready for review.

### Pure state machine — `frontend/src/lib/pipeline/pipeline-stages.ts` (new, no React)
- Reuses `ProcessingStage` from `processing-stage.ts` (UX.1); does not redefine it.
- `SPINE_NODES` = ordered `[Ingest, Embed, Extract, Graph, Complete]` (Embed before Extract); `INSIGHTS_NODE` is the parallel branch off Embed (appended with `parallel: true`, never inline).
- `derivePipelineNodes({ processingStage, jobStatus, counts })` → `PipelineNode[]` with `state ∈ pending|active|done|gated|failed`. Rules as implemented:
  - Spine index from `processing_stage` (`ingested→0 … graphed→3`, `awaiting_schema_review→Extract(2)`): earlier nodes `done`; the current node is `active` ONLY when `jobStatus ∈ {new,queued,running}`, else `done`.
  - `awaiting_schema_review` ⇒ Extract `gated` + `review-schema` action.
  - `complete` ⇒ every spine node incl. Complete `done`.
  - `failed` (stage overwrites position on the backend) ⇒ surfaced at the entry node (Ingest) `failed` + `retry` action, later nodes `pending` (no count-based inference). Pre-overwrite window: `jobStatus==='failed'` on a normal stage localises the failure to the current node.
  - `undefined` stage ⇒ all spine `pending` (no inference even with stray counts).
  - Counts hydrate `done` nodes only; a count of `0` sets the raw `count` but no label and never changes state (graphed + `entity_count=0` ⇒ Extract still `done`).
  - `deepLinkTab`: Ingest/Embed→`chunks`, Extract/Graph→`entities`, Insights→`insights`, Complete→none.
  - Insights: `done` when `insights_count>0`; `active` when past Embed with a running job and no insight yet; else `pending`. Never affects the spine.
- Test: `pipeline-stages.test.ts` — full stage×jobStatus×counts matrix (55 assertions incl. AC1–AC6).

### Component — `frontend/src/components/sources/pipeline/PipelineStatus.tsx` (new)
- Props: `variant`, `processingStage`, `jobStatus`, `counts`, `onNodeClick(deepLinkTab)`, `onNodeAction(action, node)`, `children` (live streaming-log slot), `defaultOpen` (detail).
- `live` = horizontal tracker (spinner on active) + children slot; `card` = 5-segment mini-bar + current-stage label; `detail` = collapsible spine row with recovery actions kept visible while collapsed.
- Per-node icons: pending Circle (muted), active Loader2 spin (blue), done Check (green), gated AlertTriangle (amber) + Review schema, failed XCircle (red) + Retry. Insights rendered as a small parallel node off Embed.
- a11y: each node is a focusable `<button>` with an aria-label describing stage + state (+ count); gated/failed action buttons are standalone (not nested), Tab-reachable, aria-labelled. Uses `cn` + shadcn Button/Collapsible + lucide, matching adjacent components.
- Barrel: `frontend/src/components/sources/pipeline/index.ts` (new) exports `PipelineStatus` + `PipelineStatusProps`.
- Test: `__tests__/PipelineStatus.test.tsx` — all 5 states × 3 variants, count enrichment on done nodes, gated/failed actions present + aria-labelled + firing, node-click deep-link callback, keyboard reachability, undefined ⇒ all-pending.

### Test infra added (deviation, see below)
- `frontend/vitest.config.ts`: include `*.test.tsx`, add `@vitejs/plugin-react` (Vitest 4 uses oxc; app tsconfig is `jsx: preserve`). Dev deps added: `jsdom`, `@testing-library/react`, `@testing-library/dom`, `@testing-library/user-event`, `@vitejs/plugin-react`. `.test.tsx` opts into jsdom via a per-file docblock; `.test.ts` stays in node.

### Commands (from `frontend/`)
- `npm run lint` — pass (only pre-existing warnings in untouched files; none in new files).
- `npx tsc --noEmit` — 0 errors.
- `npx vitest run src/lib/pipeline src/components/sources/pipeline` — 92 passed (3 files).
- Full `npx vitest run` — 199 passed (16 files), no regression.

### Deviations
- Added component-test infrastructure (jsdom + testing-library + vite react plugin) — none existed in `frontend/` before. Justified by the plan's test strategy ("Component (vitest + RTL)") which mandates `.test.tsx` component tests from UX.2 onward. Additive, dev-only.
- `failed` node placement: with no failed-at position on the spine and counts barred from setting state, the failure is surfaced at the entry (Ingest) node. A future signal (`failedStage`) could localise it; documented in-code.
