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
