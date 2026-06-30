# Track PL — status

## Phase PL.1 — Fix the orphaned `source.embedding` aggregate ✅ (ready for review)

**Branch**: `track/pl1-source-aggregate-embedding` (off `main`)
**Commit**: `5de3f95` feat(embeddings): write source.embedding aggregate in the live embed step

### What changed
- **Promoted helper** `populate_source_embedding` out of `scripts/backfill_chunk_embeddings.py`
  into a reusable, non-script module: `pipelines/embeddings/src/embeddings/aggregate.py`
  (depends only on `SourceRepository` + the pure `shared.utils.vectors.mean_pool`).
- **Live call site**: `pipelines/embeddings/src/embeddings/service.py` —
  `EmbeddingService.embed_source` now calls `populate_source_embedding` right after
  the per-chunk vectors are written (both the structural-chunk path and the
  text-split fallback). The early no-text/no-chunks return also clears the aggregate
  to NONE (the per-chunk rows were just deleted, so a stale aggregate would dangle).
- **Script** `scripts/backfill_chunk_embeddings.py` now imports + re-exports the
  promoted helper (`__all__ = ["populate_source_embedding"]`); its behaviour and CLI
  are unchanged.

### Idempotency / graceful handling
- Idempotent: `populate_source_embedding` recomputes purely from the source's
  current `source_embedding` rows; `embed_source` deletes + rebuilds those rows on
  every run, so a re-embed yields a fresh, correct aggregate (no dup, no error).
- Graceful: a source with no/empty chunk vectors → `mean_pool` returns `None` →
  `set_aggregate_embedding(None)` writes NONE; no crash. Dim is chunk-vector derived
  (1024 in prod), never hardcoded.

### Test evidence (per AC)
- **AC1** (aggregate present, correct dim, == mean_pool): `@requires_docker`
  `apps/app-main/tests/test_backfill_chunk_embeddings_db.py::test_embed_source_writes_aggregate_equal_to_mean_pool`
  seeds source+chunks → `embed_source` → asserts `src.embedding == mean_pool(chunk_vectors)`
  and dim == chunk dim, and that a second source's `find_related_by_embedding` returns it. PASS.
- **AC2** (idempotent re-embed): `test_embed_source_reembed_recomputes_aggregate` — re-embed
  keeps 3 chunk vectors, aggregate stays == mean_pool, no dup/error. PASS.
- **AC2** (graceful empty): `test_embed_source_no_chunks_no_text_clears_aggregate` (DB) +
  `pipelines/embeddings/tests/test_aggregate.py::test_embed_source_no_chunks_no_text_clears_aggregate`
  + `test_populate_source_embedding_empty_writes_none` — NONE written, no crash. PASS.
- **AC3** (reusable non-script home; script re-exports, unchanged): all 7 of
  `apps/app-main/tests/test_backfill_chunk_embeddings.py` (which import the script's
  `populate_source_embedding`) stay green.
- **AC4** (no regression): `pipelines/embeddings/tests/` 45 passed;
  `test_handle_process_source_autoembed.py` + `test_source_processing_service.py`
  + `test_source_related_db.py` → 62 passed (the 3 `TestBuildIngestionConfig` docling
  failures pre-exist on `main` — verified by checkout — and are unrelated);
  `test_backfill_chunk_embeddings_db.py` 11 passed.

### Live (AC5) — staging backfill
- DB: `SURREAL_DATABASE=staging`.
- **Pre-probe**: 12 sources, all with chunk vectors, only **6** had an aggregate →
  **6 missing** (e.g. `source:79jeo38ux1ekb1sghzxa`).
- **Backfill**: `python scripts/backfill_chunk_embeddings.py --source-embeddings` →
  `populated=12 empty=0 (aggregate dim=1024)`.
- **Post-probe** (read-only): `have_aggregate=12`, `missing_aggregate_but_have_chunkvecs=0`;
  the previously-NULL sample `source:79jeo38ux1ekb1sghzxa` now has a 1024-dim aggregate,
  appears in another source's `find_related_by_embedding`, and its own `find_related`
  returns 5 results.

### New files
- `pipelines/embeddings/src/embeddings/aggregate.py`
- `pipelines/embeddings/tests/test_aggregate.py`

### Modified
- `pipelines/embeddings/src/embeddings/service.py`
- `scripts/backfill_chunk_embeddings.py`
- `apps/app-main/tests/test_backfill_chunk_embeddings_db.py`

---

## Phase PL.2 — Auto-chain EXTRACT after EMBED + `processing_stage` — DONE

**Branch**: `track/pl2-autochain-extract` (off `main` with PL.1 merged).

### What shipped
- **Auto-chain seam**: `_handle_embed_source` (`apps/app-main/.../handlers.py`) now,
  after a successful SOURCE embed, best-effort enqueues `run_entities`
  (`CommandService.submit_command_job("open_notebook","run_entities",{source_id})`) —
  mirroring the `DOCUMENT_PARSE→embed` chain exactly (try/except, log, never fail the
  embed). Sources only; the note path (`_handle_embed_single_item`) still chains
  `NOTE_AUTO_LINK`. This closes the foundational gap (the KG never built automatically).
- **`source.processing_stage`** — migration **71** (strict `string DEFAULT "ingested"`
  on the SCHEMAFULL `source` table + drift-only S.4 backfill, mirroring migration 65
  `source.private`; `71_down` REMOVEs the brand-new field). Model field
  `Source.processing_stage` + `ProcessingStage` enum (`shared/types/enums.py`) +
  `SourceRepository.set_processing_stage` (best-effort).
- **Stage transitions** (handler-level, best-effort via `_set_processing_stage`):
  parse OK → `ingested`; embed OK → `embedded`; extract OK → `extracted`; schema gate →
  `awaiting_schema_review` (then reraise → worker parks `PAUSED_FOR_REVIEW`); hard
  failure (parse/embed/extract) → `failed`.
- **Gate respected**: the auto-enqueued extract uses the existing `run_extraction`
  path, so an unreviewed-schema notebook raises `SchemaReviewPendingError` → handler
  sets `awaiting_schema_review` and reraises → no entities, no crash.

### Per-criterion evidence
- **AC1** (ingest→entities with no manual call): seam proven in
  `apps/app-main/tests/test_handle_embed_source_autoextract.py::test_autoextract_enqueued_after_embed`
  (embed → exactly one `run_entities` enqueue for the source + stage=`embedded`). The
  plan permits the seam assertion in lieu of a full LLM roundtrip.
- **AC2** (gate parks, zero entities): `@requires_docker`
  `test_handle_entity_extract_gate_db.py::test_auto_extract_parks_on_schema_review_gate`
  drives the REAL `handle_entity_extract` against a live container with a
  `review_required` `NotebookSchema` → `SchemaReviewPendingError`,
  `processing_stage == awaiting_schema_review`, 0 `entity` rows. PASS.
- **AC3** (`processing_stage` advances, persisted, idempotent; S.4-safe fresh):
  `@requires_docker` `test_processing_stage_db.py` — default `ingested` on a fresh row,
  `set_processing_stage` advances `ingested→embedded→extracted→awaiting_schema_review→failed`
  (idempotent re-write), migration-71 backfill repairs a NONE row AND keeps it writable
  (the S.4 hazard), down/forward roundtrip. 4 passed.
- **AC4** (best-effort preserved; suites green): triage best-effort unchanged
  (`_run_triage` keeps its own log-and-continue guard); embed→extract chaining
  best-effort proven by
  `test_handle_embed_source_autoextract.py::test_enqueue_failure_does_not_fail_embed`
  and embed-hard-failure→`failed`+no-chain by `test_embed_failure_sets_stage_failed`;
  entity-extract stage matrix in `test_handle_entity_extract_stage.py`. Regression:
  app-main non-docker `-k "extract or triage or embed or entity or source or stage ..."`
  → 438 passed (only the 3 pre-existing `TestBuildIngestionConfig` docling failures,
  unrelated); shared 151 passed; surrealdb-service source 13 passed; embeddings
  aggregate 6 passed.

### PL.1 fold-in (the populated→NONE stale-clear)
- `pipelines/embeddings/tests/test_aggregate.py::test_populate_source_embedding_clears_stale_populated_aggregate`
  — seed a non-null aggregate, strip the chunk vectors, re-run → aggregate overwritten
  to NONE. PASS.

### New files
- `migrations/71.surrealql`, `migrations/71_down.surrealql`
- `apps/app-main/tests/test_handle_embed_source_autoextract.py`
- `apps/app-main/tests/test_handle_entity_extract_stage.py`
- `apps/app-main/tests/test_handle_entity_extract_gate_db.py`
- `apps/app-main/tests/test_processing_stage_db.py`

### Modified
- `apps/app-main/src/app_main/handlers.py`
- `packages/shared/src/shared/models/source.py`
- `packages/shared/src/shared/types/enums.py`
- `packages/surrealdb-service/src/surrealdb_service/repositories/source.py`
- `pipelines/embeddings/tests/test_aggregate.py`

---

## Phase PL.3 — Auto-chain GRAPH (mentions refresh) + INSIGHTS toggle — READY FOR REVIEW

**Branch**: `track/pl3-autochain-graph-insights` (off `main` with PL.1 + PL.2 merged).
**Commits**: `af4bd6f` (primitives) → `0ee4bee` (handler wiring + seam tests) →
`e0610de` (DB tests) → `1b0349a` (notebook API toggle surface).

### What shipped
- **Source-scoped mentions refresh seam** — `MentionsProjectionService.refresh_source(source_id)`:
  runs the FULL corpus projection (so each edge keeps its global R.2 weight × R.6
  IDF / df — the weighting is inherently cross-source), then keeps only edges
  whose `source_id == this source`, and writes only those:
  `EntityRepository.clear_mentions_for_source` (a scoped `DELETE mentions WHERE
  in = $src`) + the same idempotent `relate_mention` RELATE loop the global
  `regenerate` uses. Source-scoped + idempotent; R.6 noise normalization
  preserved (df==1 singletons never become edges).
- **`graphed`/`complete` transitions** — `handle_entity_extract`, after a
  successful extract (`extracted`), calls the new best-effort
  `_refresh_source_mentions(source_id)` helper: refresh THIS source's mentions →
  set `graphed` → set `complete`. A refresh failure logs and leaves the stage at
  `extracted` (best-effort; extraction already persisted). `complete` is set
  right after `graphed` — the KG chain (embed→extract→graph) is the spine;
  INSIGHTS does NOT gate `complete` (simple, no join).
- **Auto-INSIGHTS chain, per-notebook toggle** — the flag lives on
  **`notebook.auto_insights`** (migration **72**, strict `bool DEFAULT true`,
  S.4 drift-only backfill mirroring migration 71; `Notebook.auto_insights` model
  field; `NotebookRepository.get_auto_insights` defaults ON on
  missing/legacy/unknown/error). `_handle_embed_source` chains
  `INSIGHT_EXTRACT`/`run_summaries` PARALLEL to the extract chain via
  `_maybe_chain_insights`: resolve the owning notebook → read the toggle →
  enqueue when ON, skip when OFF; unlinked source → default ON. Best-effort
  (toggle-read error → ON; queue hiccup → logged, embed still returns).
  Produces `source_insight` rows; does NOT touch `processing_stage`. Settable
  via the notebook PUT API (`NotebookUpdate.auto_insights` /
  `NotebookResponse.auto_insights`).
- **Folded PL.2 minor (chunk-count guard)** — both the extract enqueue AND the
  insights enqueue in `_handle_embed_source` are now guarded on
  `embedded_chunks > 0` (the orchestrator's count), mirroring the
  `DOCUMENT_PARSE→embed` `chunk_count > 0` guard. A zero-chunk source spawns no
  no-op `run_entities` / `run_summaries` jobs.

### Per-criterion evidence
- **AC1** (source-scoped mentions appear with NO manual regenerate; endpoint
  returns them): `@requires_docker`
  `test_handle_entity_extract_graph_db.py::test_auto_extract_materializes_source_graph_and_completes`
  drives the REAL `handle_entity_extract` (extraction mocked-but-persisted as a
  real extraction would) → the source's `mentions` edge materializes via the
  handler's own refresh, `get_document_entity_edges(source_id)` returns it, and
  the OTHER source (sharing the concept) gets no edges (scoped). Plus
  `test_mentions_refresh_source_db.py::test_refresh_source_creates_only_this_sources_edges`.
- **AC2** (source-scoped + idempotent; R.6 preserved):
  `test_mentions_refresh_source_db.py::test_refresh_source_is_scoped_and_idempotent`
  (refresh c0 leaves c1's edges byte-identical; re-run cleared==created==2, no
  dup pairs; spoke never an endpoint) +
  `test_refresh_weights_match_global_regenerate` (scoped weights == global
  regenerate weights).
- **AC3** (INSIGHTS on/off, best-effort): seam tests
  `test_handle_embed_source_insights_chain.py` — toggle ON → `run_summaries`
  enqueued; OFF → skipped (extract still fires); unlinked → default ON; zero
  chunks → neither enqueued; insights-enqueue failure best-effort. Toggle DB
  evidence: `test_notebook_auto_insights_db.py` (default ON, explicit off,
  unknown→ON).
- **AC4** (`processing_stage` reaches `graphed`→`complete`; suites green):
  `test_handle_entity_extract_graph_db.py` asserts `complete` end-to-end;
  `test_handle_entity_extract_stage.py::test_success_sets_stage_extracted_then_graphed_complete`
  asserts the `extracted→graphed→complete` order and
  `test_graph_refresh_failure_keeps_stage_extracted` the best-effort fallback.
  Regression: app-main non-docker **1370 passed** (only the 3 pre-existing
  `TestBuildIngestionConfig` docling failures); PL.3 + adjacent DB suite **18
  passed** (incl. the existing global `test_mentions_regenerate_db` 7 +
  `test_processing_stage_db` 4); shared notebook/settings 19; surrealdb-service
  non-docker entity/notebook/mention/source 19.

### New files
- `migrations/72.surrealql`, `migrations/72_down.surrealql`
- `apps/app-main/tests/test_handle_embed_source_insights_chain.py`
- `apps/app-main/tests/test_mentions_refresh_source_db.py`
- `apps/app-main/tests/test_notebook_auto_insights_db.py`
- `apps/app-main/tests/test_handle_entity_extract_graph_db.py`

### Modified
- `apps/app-main/src/app_main/handlers.py`
- `apps/app-main/src/app_main/services/mentions_projection_service.py`
- `apps/app-main/src/app_main/api/schemas.py`
- `apps/app-main/src/app_main/api/routers/notebooks.py`
- `packages/shared/src/shared/models/notebook.py`
- `packages/surrealdb-service/src/surrealdb_service/repositories/notebook.py`
- `packages/surrealdb-service/src/surrealdb_service/repositories/entity.py`
- `apps/app-main/tests/test_handle_embed_source_autoextract.py`
- `apps/app-main/tests/test_handle_entity_extract_stage.py`
- `apps/app-main/tests/test_notebooks_router.py`

---

## Phase PL.4 — One `SourcePipeline` definition + status surfaced — READY FOR REVIEW

**Branch**: `track/pl4-pipeline-definition` (off `main` with PL.1 + PL.2 + PL.3 merged).
**Commits**: `5d72fd6` (consolidation) → `9c26e64` (API exposure + pipeline unit
tests) → `6024aba` (migration-72 minor + get_processing_stage roundtrip).

### What shipped — pure consolidation, NO behavior change vs PL.3

- **One declarative `SOURCE_PIPELINE` + the `advance_source` driver** —
  `apps/app-main/src/app_main/services/source_pipeline.py`. The PL.1–PL.3
  chaining was scattered across the handlers (each handler wrote
  `processing_stage` AND held the ad-hoc next-stage enqueue inline). PL.4 moves
  ALL of it into one place:
  - `SOURCE_PIPELINE` is an ordered list of `PipelineStage` dataclasses, each
    with `name` / `produces` (the `processing_stage` value) / `auto` / `gate` /
    `depends_on` / `enqueue_command` (None = inline) / `parallel`. The spine is
    `INGEST(ingested) → EMBED(embedded) → EXTRACT(extracted, gate=schema_review)
    → GRAPH(graphed→complete, inline)`; INSIGHTS is a `parallel` branch off EMBED
    (`gate=auto_insights`, `produces=None` — it never advances the spine).
  - `advance_source(source_id, *, embedded_chunks=None)` reads
    `source.processing_stage` via the new `SourceRepository.get_processing_stage`,
    finds "where the source is", and dispatches the next allowed stage(s): the
    spine successor (enqueue a job, or run GRAPH inline + advance graphed→complete)
    plus any parallel branch (INSIGHTS, toggle-gated). It honours `auto`,
    `depends_on`, the gates, the `embedded_chunks > 0` guard (EMBED fan-out only),
    and is best-effort + idempotent (every enqueue/inline run is guarded; the
    downstream jobs are themselves idempotent so a double-dispatch is safe).
  - Parked/terminal stages (`awaiting_schema_review`, `failed`, `complete`) do
    NOT auto-advance — they need an explicit resume (a review, a re-run).
- **Handlers became THIN** — `handle_process_source`, `_handle_embed_source`,
  `handle_entity_extract` now do their stage's work, write their produced stage
  (`_set_processing_stage`, re-exported from the pipeline module), then call
  `advance_source`. **No source-pipeline ad-hoc enqueue remains in handlers.py**
  (grep: the only `submit_command_job` left is the Y.3 NOTE auto-link in
  `_handle_embed_single_item` — a different, non-source pipeline). The
  schema-review gate stays IN the extract handler (it must reraise
  `SchemaReviewPendingError` for the worker → `PAUSED_FOR_REVIEW`);
  `advance_source` is only reached on the success path.
- **`MentionsProjectionService.refresh_source` invariant PRESERVED unchanged** —
  GRAPH still runs the FULL corpus projection then filters the write (global
  R.2 weight × R.6 IDF/df); PL.4 only relocated the *call site* (handler →
  `advance_source._run_graph_inline`), not the projection.
- **`processing_stage` on the source read API** — added to `SourceResponse`
  (`api/schemas.py`) and surfaced in `get_source` / `update_source`
  (`api/routers/sources_crud.py`), so the UI can show per-document progress
  (`ingested → … → complete`, or a parked `awaiting_schema_review` / `failed`).

### `complete`-with-zero-edges semantics (folded PL.3 minor b)

`complete` means "the KG chain RAN", not "the KG is non-empty". A source whose
entities share no `df>1` concept with the rest of the corpus produces 0
`mentions` edges (R.6 noise normalization drops `df==1` singletons), yet its
extraction succeeded and its GRAPH "refreshed to empty" — so it correctly
reaches `graphed → complete`. This is a legitimate terminal state; the UI should
read "complete with 0 edges" as a healthy outcome, not a failure. (Documented in
the `source_pipeline.py` module docstring too.)

### Per-criterion evidence

- **AC1** (chain driven by `advance_source`; handlers hold no ad-hoc enqueues;
  stage-transition table tested): grep proof above. Unit tests
  `apps/app-main/tests/test_source_pipeline.py` (13) pin the table — each stage →
  the correct next action (`ingested`→EMBED, `embedded`→EXTRACT+INSIGHTS,
  `extracted`→GRAPH inline→graphed→complete), gates respected (toggle off skips
  INSIGHTS, KG chain still fires), the chunk-count guard (0 chunks → neither
  chained), parked/terminal stages do not advance, best-effort enqueue never
  raises, plus a structural assertion on the definition's shape.
- **AC2** (`processing_stage` returned by the read endpoint): `test_sources_crud.py`
  `TestGetSourceProcessingStage` — `GET /sources/{id}` echoes the source's stage
  (`graphed`) and falls back to `ingested` when unset.
- **AC3** (behavior-identical / no regression): the PL.1–PL.3 seam tests
  (`test_handle_process_source_autoembed`, `test_handle_embed_source_autoextract`,
  `test_handle_embed_source_insights_chain`, `test_handle_entity_extract_stage`)
  still green AFTER routing through `advance_source` (the mocks now supply
  `get_processing_stage` — the only test-side change, the assertions on what gets
  enqueued / which stages are written are UNCHANGED). The end-to-end equality:
  `@requires_docker test_handle_entity_extract_graph_db` drives the REAL
  `handle_entity_extract` against a live container → through the consolidated
  `advance_source` → GRAPH inline refresh → the SAME source-scoped `mentions`
  edge materializes and `processing_stage == complete`, scoped (the other source
  gets no edges) — identical to PL.3. The gate park
  (`test_handle_entity_extract_gate_db`) and the source-scoped refresh
  (`test_mentions_refresh_source_db`) suites also green.
- **AC4** (full app-main suite green + migration-72 S.4 added): app-main
  non-docker **1385 passed** (only the 3 pre-existing `TestBuildIngestionConfig`
  docling failures — baseline, docling not installed). The folded migration-72
  S.4 writability test (`test_notebook_auto_insights_db::
  test_migration_72_backfills_none_and_row_stays_writable`) mirrors the
  migration-71 one (a genuinely-NONE `notebook.auto_insights` row is repaired by
  the drift-only backfill AND stays writable) → green. `mypy` clean on
  `source_pipeline.py` (the 2 reported errors are pre-existing `shared.*`
  missing-stub warnings, unrelated).

### New files
- `apps/app-main/src/app_main/services/source_pipeline.py`
- `apps/app-main/tests/test_source_pipeline.py`

### Modified
- `apps/app-main/src/app_main/handlers.py` (now thin — delegates to `advance_source`)
- `apps/app-main/src/app_main/api/schemas.py` (+`SourceResponse.processing_stage`)
- `apps/app-main/src/app_main/api/routers/sources_crud.py` (surface the stage)
- `packages/surrealdb-service/src/surrealdb_service/repositories/source.py`
  (+`get_processing_stage`)
- `apps/app-main/tests/test_handle_process_source_autoembed.py`
- `apps/app-main/tests/test_handle_embed_source_autoextract.py`
- `apps/app-main/tests/test_handle_embed_source_insights_chain.py`
- `apps/app-main/tests/test_handle_entity_extract_stage.py`
- `apps/app-main/tests/test_handle_entity_extract_graph_db.py` (docstring)
- `apps/app-main/tests/test_processing_stage_db.py` (+get_processing_stage roundtrip)
- `apps/app-main/tests/test_notebook_auto_insights_db.py` (+migration-72 S.4 test)
- `apps/app-main/tests/test_sources_crud.py` (+processing_stage API assertion)
