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
