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

---

## Phase UX.3 — SourceCard 5-segment mini progress bar — DONE (ready for review)

**Branch**: `track/ux3-sourcecard-mini-bar` (off `main`, UX.1+UX.2 merged).
**Commits**: `716c812` (counts adapter) · `1dd0187` (SourceCard rewrite).

### Counts adapter (AC0, BLOCKING) — `frontend/src/lib/pipeline/source-counts.ts` (new)
- `toPipelineCounts(source)` maps `0 ⇒ undefined` for the count-gated stages
  (`embedded_chunks`, `entity_count`, `insights_count`) and derives
  `graph_present` from `relation_count > 0`. Positive counts pass through.
- Why: the list endpoint types those counts as required `number` (default `0`).
  On the `failed` path `derivePipelineNodes` locates the failure at the first
  spine stage with no output signal, treating `undefined` = not-reached and `0`
  = reached-but-empty. Passing a raw list `0` reads as "Ingest reached" and
  mislabels a parse failure as "Ingest done / Embed failed". Collapsing
  `0 ⇒ undefined` at the seam restores "Ingest failed". Documented in-module.
- Test: `source-counts.test.ts` — 0⇒undefined + pass-through of >0 + graph_present,
  plus end-to-end failed-path assertions (incl. a guard test proving raw zeros
  WOULD mislabel Ingest as done).

### SourceCard — `frontend/src/components/sources/SourceCard.tsx` (modified)
- Removed the `STATUS_CONFIG`/`useSourceStatus` badge block, the processing
  message, the bottom failed-retry block, and the job-progress bar. Replaced with
  a single `<PipelineStatus variant="card" />` fed from the card's own
  `processing_stage` + `toPipelineCounts(effectiveSource)`.
- Polling gated: `useSourcePipeline(source.id, enabled)` with
  `enabled = propStage !== undefined ? isPollableStage(propStage) : (command_id || active job || wasProcessing)`.
  Terminal (`complete`/`failed`) and gated (`awaiting_schema_review`) cards ⇒
  `enabled=false` (no repeating `/sources/{id}`); the hook's own
  `refetchInterval` is the second stop-gate. Prefer polled data
  (`pipelineData ?? source`) for the effective stage/counts.
- Completion refresh: a stage-based `useEffect` calls `onRefresh()` once an
  in-flight card settles to a terminal stage (replaces the old
  `useSourceStatus` completion detection).
- Actions: gated node → `onClick(source.id)` (deep-link to detail/review); failed
  node → existing `onRetry(source.id)`. Dropdown Retry/Delete/Remove preserved
  (Retry gated on `stage === 'failed'`). Insights/topic badges preserved for
  completed (and unknown-stage) cards. Added `aria-label="Source actions"` on the
  icon-only menu trigger.
- Test: `frontend/src/components/sources/__tests__/SourceCard.test.tsx` — bar per
  stage (ingested/embedded/extracted/graphed/complete/awaiting_schema_review/
  failed); AC0 parse-failure (Ingest failed) + embedded-then-failed (Extract
  failed); gated Review-schema deep-link + failed Retry; terminal cards
  `enabled=false`, non-terminal `enabled=true`; undefined ⇒ all-pending + title.

### Commands (from `frontend/`)
- `npm run lint` — pass (only pre-existing warnings in untouched files; none in new/changed files).
- `npx tsc --noEmit` — 0 errors.
- `npx vitest run src/components/sources src/lib/pipeline` — 119 passed (5 files).
- Full `npx vitest run` — 226 passed (18 files); no regression (203 baseline + 23 new UX.3 tests).

### Deviations
- Dropped `useSourceStatus` from the card entirely (the plan only mandated
  replacing the badge block). Its two jobs are preserved: the active-node spinner
  now reads the job axis from the source payload's `status` field via
  `PipelineStatus jobStatus`, and completion-refresh is driven by the stage
  transition. Fewer request axes per card.
- Gated/failed cards also stop polling (`enabled=false`), not only terminal ones:
  `isPollableStage` already treats `awaiting_schema_review` as non-pollable (no
  automatic progress until the user acts), matching the UX.1 hook contract.
