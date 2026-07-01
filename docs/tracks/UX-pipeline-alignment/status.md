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

---

## Phase UX.4 — Source-detail spine + Graph signal + drop stale regen text — DONE (ready for review)

**Branch**: `track/ux4-detail-spine-graph` (off `main`, UX.1+UX.2+UX.3 merged) — NOT merged, NOT pushed.
**Commits**: `85f2ea7` (drop stale regenerate guidance) · `a91d28e` (source-detail spine + graph signal).

### SourceDetailContent — `frontend/src/components/source/SourceDetailContent.tsx` (modified)
- Replaced the "Processing Status Bar" — the four independent output badges
  (chunks / embedded / entities / insights, old l.567–608) — with a single
  `<PipelineStatus variant="detail" />` collapsed spine, fed from
  `effectiveStage` + `toPipelineCounts(effectiveSource)` (the shared
  `0 ⇒ undefined` adapter; no raw list zeros). `ParserEngineBadge` + the Zotero
  badge are retained beneath the spine (not pipeline-output badges).
- **Live polling**: `useSourcePipeline(sourceId, pipelineEnabled)` where
  `pipelineEnabled = source != null && isPollableStage(source.processing_stage)`.
  In-flight sources advance the spine live; terminal (`complete`/`failed`) and
  gated (`awaiting_schema_review`) pages pass `enabled=false` (bounded). The
  spine reads `effectiveSource = pipelineData ?? source` so counts/stage are the
  freshest available.
- **Graph node/signal**: resolves from `processing_stage` via the state machine
  (`done` at graphed/complete, `pending` earlier) and is enriched by
  `graph_present` (`relation_count > 0`). No new derivation added — the counts
  adapter + stage make `derivePipelineNodes` resolve the Graph node.
- **Deep-link wiring**: the `Tabs` were made controlled
  (`value={activeTab} onValueChange={setActiveTab}`); `onNodeClick(deepLinkTab)`
  sets the tab (Ingest/Embed→`chunks`, Extract/Graph→`entities`,
  Insights→`insights`; Complete→undefined→no-op). The pipeline `deepLinkTab`
  values already equal the detail tab `value`s, so the map is the identity.
- **Gate + failed actions**: `onNodeAction('review-schema')` opens the Entities
  tab (where entity/schema review lives); `onNodeAction('retry')` calls a new
  `handleRetry` wired to `sourcesApi.retry` then `fetchSource`.
- All 6 tab triggers still mount and the header actions dropdown is unchanged
  (manual runners are reframed in UX.6, not here).
- Test: `frontend/src/components/source/__tests__/SourceDetailContent.spine.test.tsx`
  — spine state per `processing_stage`; old badges gone; expand reveals per-node
  counts; Graph node graphed-vs-non-graphed; Extract→Entities and
  Insights→Insights tab switches; gated Review-schema (→Entities) + failed Retry
  (→`sourcesApi.retry`); 6 tabs + dropdown present; terminal `enabled=false`,
  in-flight `enabled=true`. Heavy children + react-markdown mocked; PipelineStatus
  + Radix Tabs kept real.

### DocumentGraphView — `frontend/src/app/(dashboard)/knowledge-graph/components/DocumentGraphView.tsx` (modified)
- Removed the stale empty-state guidance ("Regenerate the document graph …
  Run `POST /knowledge-graph/document-graph/regenerate`"); replaced with a note
  that the document graph is built automatically as sources reach the `graphed`
  stage. Grep confirms 0 hits for both removed strings under `frontend/src`.
- Guard test: `frontend/src/app/(dashboard)/knowledge-graph/components/__tests__/DocumentGraphView.regen-text.test.ts`.

### Commands (from `frontend/`)
- `npm run lint` — pass (only pre-existing warnings; `Network`/`Play` in
  SourceDetailContent pre-date this branch on `main`; none in new/changed code).
- `npx tsc --noEmit` — 0 errors.
- `npx vitest run src/components/source src/app` — 84 passed (5 files).
- Full `npx vitest run` — 246 passed (20 files); no regression (226 baseline + 20
  new UX.4 tests: 17 spine + 3 regen-guard).

### Deviations
- The failed Retry action uses `sourcesApi.retry` (the source requeue endpoint),
  as no dedicated retry handler existed on the detail page. The plan permitted
  reusing "whatever retry/reprocess handler already exists"; `retry` is the
  closest semantic match (full requeue) and was already in the API layer.

---

## Phase UX.5 — Lean 4-step creation flow + config relocation

**Branch:** `track/ux5-lean-create-flow` (off `main`) — NOT merged, NOT pushed.
**Commits:** `d282b2d` (components), `53db087` (flow), `045ac24` (tests + e2e).
**State:** Ready for review.

### New — `frontend/src/components/sources/steps/AdvancedIngestionSettings.tsx`
- Collapsed Radix `Collapsible` disclosure on the Input screen: parser engine /
  OCR engine / table mode, every field defaulting to `Auto`. Emits a
  `processing_overrides` key only for a field moved off `Auto` (all-Auto ⇒ `{}`,
  backend auto-routes). Keyboard-accessible: Tab to the trigger, Enter/Space
  toggle, `aria-expanded` reflects state.

### New — `frontend/src/components/sources/pipeline/ProcessingLogConsole.tsx`
- The streaming-log console mounted UNDER the live tracker (SSE via
  `useProcessingLogs` while in-flight; falls back to persisted
  `fetchProcessingLogs` once settled). Extracted so the create flow no longer
  depends on `ExtractionTab` for logs.

### `frontend/src/components/sources/pipeline/CreateSourcePipeline.tsx` (rewritten)
- `STEP_LABELS` reduced to 4: `Input → Organize → Processing → Done`. The
  mandatory Config step and the Extract/Postprocess/Classification/Entities/Embed
  manual tabs are gone.
- Deleted: `derivePipelineStatuses`, `manualStatuses`, `handleStartEntities`,
  `handleStartEmbed`, the auto-detect-manual-completion effect, the job-status
  inference block, `handleExtractionContinue`/`handlePostprocessContinue`,
  `classificationReady`, the debug `console.log` polling effect.
- Progress now comes from one live `<PipelineStatus variant="live">` fed by
  `useSourcePipeline(sourceId)` (the `processing_stage` spine); `useSourceStatus`
  drives only the current node's spinner. `ProcessingLogConsole` is passed as its
  children (mounted beneath the tracker).
- Flow control: an effect watches `processingComplete`. `complete` ⇒
  `phase='complete'`, `activeTab=Done`. `awaiting_schema_review` keeps
  `phase='processing'` and the tracker parks the Extract node as `gated` with a
  "Review schema" action (no false Done). `failed` keeps `phase='processing'` and
  the tracker renders the failed node + Retry (wired to `useRetrySource`); we do
  NOT flip to the `error` phase (the footer has no error rendering, and the node
  action is the recovery path). Node actions: `retry → retrySource.mutate`,
  `review-schema → router.push('/sources/{id}')`.
- Multi-file batching preserved. Each entry now carries its OWN
  `processing_stage`: the multi-source poll calls `sourcesApi.get(entry.id)` (was
  `sourcesApi.status`) and maps stage → entry status
  (`complete→completed`, `failed→failed`, else `processing`). The Processing step
  renders one `<PipelineStatus variant="card">` per entry driven by
  `entry.stage`, so each source advances independently.
- `AdvancedIngestionSettings` mounts collapsed under `SourceTypeStep` on the
  Input step; its overrides thread into `processingOverrides` → the create call's
  `processing_overrides` (undefined when empty).
- Removed dead imports from this file only (tab files left in place for UX.6):
  `ProcessingConfigStep`, `ExtractionTab`, `EntitiesTab`, `EmbeddingTab`,
  `PreprocessingTab`, `SummariesTab`.

### `frontend/src/components/sources/pipeline/PipelineFooter.tsx` (modified)
- Added a `lastConfigStep` prop; Submit now shows on the last config step
  (Organize = 2) instead of the hard-coded old `3`.

### `frontend/src/components/source/SourceDetailContent.tsx` (modified)
- Reprocess dialog copy reframed as the PRIMARY parser/OCR config home ("The
  primary place to change parsing / OCR settings …"). Still calls
  `sourcesApi.reprocess` via `PipelineConfigPanel` — no endpoint/behavior change.

### AdvancedIngestionSettings override contract
- Contribution = the set of fields moved off `Auto`
  (`parser_engine` / `docling_ocr_engine` / `docling_table_mode`). Collapsed +
  all-Auto (default) ⇒ `{}` (no overrides). Changed fields persist across
  collapse (no silent config loss). The parent sends `processing_overrides` only
  when the object is non-empty.

### Tests
- `frontend/src/components/sources/steps/__tests__/AdvancedIngestionSettings.test.tsx`
  — 6 tests: collapsed default (`aria-expanded=false`), Enter/Space keyboard
  toggle, `{}` on mount, Auto defaults, single-field override, multi-field
  override omitting untouched fields. (Radix Select driven via pointer-capture +
  scrollIntoView polyfills.)
- `frontend/src/components/sources/pipeline/__tests__/CreateSourcePipeline.test.tsx`
  — 6 tests: 4 steps / no Config-Extract-Embed-Classification; live tracker mounts
  with Embed-before-Extract + log console; `complete → Done`; parks on
  `awaiting_schema_review` (Review schema, no Done, Extract `gated`); no
  Start-Entities/Start-Embed buttons; multi-file batch drives per-source card
  state from each source's polled `processing_stage`.
- `frontend/e2e/track-ux/create-source-pipeline.spec.ts` — route-mocked create →
  live tracker walks `ingested→embedded→extracted→graphed→complete` → Done.

### Commands (from `frontend/`)
- `npm run lint` — pass (0 errors; only pre-existing warnings; none in new/changed files).
- `npx tsc --noEmit` — 0 errors.
- `npx vitest run src/components/sources` — 66 passed (4 files).
- Full `npx vitest run` — 258 passed (22 files); no regression (246 UX.4 baseline + 12 new).
- `npx playwright test e2e/track-ux/create-source-pipeline.spec.ts` — 1 passed
  (10.7s). Ran against a self-served `next dev` on port 8599 (port 8502 was held
  by a Docker container serving the OLD build); pass the same `PLAYWRIGHT_BASE_URL`
  to reproduce. The spec is fully route-mocked (no live backend / DB / worker).

### Deviations
- `PipelineFooter` gained a `lastConfigStep` prop (the plan said "trim the
  stepper to 4 steps"; the stepper itself is data-driven and needed no change, but
  the footer had the old last-config-step hard-coded to `3`).
- `failed` stays in `phase='processing'` (tracker shows the failed node + Retry)
  rather than switching to the `error` phase — the footer has no `error` rendering
  and the recovery action lives on the node, matching AC3.
- The streaming console was extracted into a new `ProcessingLogConsole` component
  (the old console was embedded inside `ExtractionTab`, which the lean flow drops).

---

## Phase UX.5 — Review revisions (Blocker + Major)

**Branch:** `track/ux5-lean-create-flow` — NOT merged, NOT pushed.
**State:** Ready for re-review. Addresses the two review defects; the approved
single-source spine / log streaming / 4-step model / reprocess relocation were
left untouched.

### Blocker — multi-file batch wedged on the schema-review gate
`frontend/src/components/sources/pipeline/CreateSourcePipeline.tsx`:
- New `stageToEntryStatus` maps `awaiting_schema_review → 'gated'` (a settled,
  poll-stopping state) instead of the non-terminal `'processing'`. `isSettledEntry`
  (complete/failed/gated) and `isPollableEntry` (creating/processing) replace the
  old `isTerminalEntry`.
- Poll effect now filters on `isPollableEntry` and clears the interval once no
  entry is pollable — a gated entry no longer polls `GET /sources/{id}` forever.
- The batch advances to Done only on `multiAllComplete` (pure success), mirroring
  the single-source rule. A settled-with-issues batch (`multiAllSettled` but some
  gated/failed) stays on Processing in a non-hanging terminal state with a settled
  banner + per-entry recovery cards; `processingStepStatus` reports `failed`/
  `completed` accordingly (no false green Done).
- Per-entry cards now pass `onNodeAction={handleMultiNodeAction(entry.id, …)}`:
  review-schema ⇒ `router.push('/sources/{id}')`; retry ⇒ `retrySource.mutate(id)`
  and re-arm the entry to `processing` so its poll resumes.

### Major — advanced overrides discarded on back-navigation
- `frontend/src/components/sources/steps/AdvancedIngestionSettings.tsx` is now a
  CONTROLLED presentational component: parent owns `parserEngine`/`ocrEngine`/
  `tableMode` + the open state (exported choice types). Removed the mount-time
  `onOverridesChange({})` reset that wiped selections on remount.
- Parent `CreateSourcePipeline` holds the four pieces of state and derives
  `processing_overrides` via `useMemo` (all-Auto ⇒ `{}`, only changed fields
  emitted). Input→Organize→Back now rehydrates the disclosure (still expanded,
  Docling still selected) and the create call still sends `{parser_engine:'docling'}`.

### Minor — interval churn
- Poll effect keyed on a stable `multiPollKey` (joined entry ids) and reads live
  entries from `multiSourcesRef`; the 3s interval is created once and lives until
  the stop condition (no per-tick teardown). Cadence + stop semantics unchanged.

### Tests
- `CreateSourcePipeline.test.tsx` +3: gated multi-batch (polling stops — no further
  `sourcesApi.get` after settle, settled banner, Review schema → `push('/sources/source:a')`);
  failed multi-batch (Retry → `retryMutate('source:a')`); override persistence across
  Input→Organize→Back reaching the create call with `processing_overrides`.
- `AdvancedIngestionSettings.test.tsx` rewritten for the controlled API (+1 remount
  persistence test); a11y + Auto-defaults + only-changed-fields contract preserved
  via a controlled harness.

### Commands (from `frontend/`)
- `npm run lint` — pass (0 errors; only pre-existing warnings; none in changed files).
- `npx tsc --noEmit` — 0 errors.
- `npx vitest run src/components/sources` — 70 passed (4 files).
- Full `npx vitest run` — 262 passed (22 files); no regression (258 baseline + 4 new).

## Round 3 — Blocker: per-entry Retry never resumed polling after full settle

### Root cause
- The multi-file poll effect was keyed on `multiPollKey` = the joined list of ALL
  entry ids. When every entry settled, the interval self-cleared inside its tick,
  but the id list was unchanged, so the key was unchanged and the effect never
  re-ran. `handleMultiNodeAction(id, 'retry')` re-armed an entry to `processing`
  but — id set unchanged ⇒ key unchanged ⇒ effect did not re-fire — the cleared
  interval stayed dead and the re-armed entry spun forever, never polled again.

### Fix (`CreateSourcePipeline.tsx`)
- Effect key changed from ALL ids to the sorted set of POLLABLE ids.
  - Before: `multiPollKey = multiSources.map(s=>s.id).filter(Boolean).join(',')`,
    dep `[phase, isMulti, multiPollKey]`.
  - After: `pollableKey = multiSources.filter(s => s.id && isPollableEntry(s.status))
    .map(s=>s.id).sort().join(',')`, dep `[phase, isMulti, pollableKey]`.
- Added an empty-set guard: `if (phase !== 'processing' || !isMulti || pollableKey
  === '') return` — no interval runs when nothing is pollable.
- Self-correcting: settle ⇒ key `''` (interval clears); Retry re-arms one entry to
  `processing` ⇒ key `'' → 'source:a'`, effect re-fires, interval recreated and the
  entry is polled again; re-settle ⇒ key back to `''`. Key changes only when the
  pollable SET changes (not on per-tick count updates) so round-2's no-churn
  property holds. Cleanup clears the prior interval before recreating — no double
  interval. Single-source path untouched (React Query invalidation).

### Test (upgraded, replaces the shallow retry test)
- New `resumes polling a re-armed entry after Retry once the batch has fully
  settled`: `{source:a failed, source:b complete}` batch, advance PAST the clearing
  tick (poll a >3s window with no new `sourcesApi.get` ⇒ interval proven dead) and
  freeze the get() count; then click Retry and assert a NEW `sourcesApi.get('source:a')`
  lands beyond the frozen count — polling actually resumed, not merely that
  `retryMutate` fired. Confirmed to FAIL on the old all-ids key (`resumed: false`).

### Commands (from `frontend/`)
- `npm run lint` — pass (0 errors; only pre-existing warnings; none in changed files).
- `npx tsc --noEmit` — 0 errors.
- `npx vitest run src/components/sources` — 70 passed (4 files).
- Full `npx vitest run` — 262 passed (22 files); no regression (retry test upgraded
  in place, count unchanged at 262).

### Commit
- `a8a6111` fix(ux5): key multi-poll on pollable ids so retry resumes polling
