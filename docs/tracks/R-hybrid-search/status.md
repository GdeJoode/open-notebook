# Track R — status

## Phase R.0 — Embedding foundation — MERGED + APPROVED; live backfill BLOCKED (→ R.0b)
- O.2a-style adversarial review: **APPROVED attempt 1** (0 blockers/majors, 3 minors). Merged to `main` (`119ecd3`).
- **Live backfill BLOCKED**: running the chunk backfill against staging failed on all 6 sources —
  `Ollama API error: the input length exceeds the context length`. `mxbai-embed-large` has a ~512-token
  context; chunk texts exceed it and the embedding service does no truncation. The container test missed it
  (fake model, no context limit). → **R.0b** (embedding context-length fix, branch `track/r0b-embed-context-fix`)
  must land + be reviewed before the live backfill can complete.
- Decisions locked + Purview lessons added (`purview-lessons.md`). Orphan chunks GC'd (10,501→1,448).
- Review minors (follow-up): `embed_source` embeds `is_content=False` noise chunks (→ R.6); dead `embed:True` flag.

---

## Phase R.0 — Embedding foundation (forward-fix + backfill) — Ready for review

**Branch**: `track/r0-embedding-foundation` (off `main`)
**Commits** (`cc530ac..a32fa87`):
- `cc530ac` feat(embeddings): auto-enqueue embed_source after ingest (forward-fix)
- `69b9fff` feat(source): aggregate embedding field + mean-pool populate (migration 63)
- `66386ef` feat(scripts): backfill_chunk_embeddings.py (Track-P pattern)
- `a32fa87` test(embeddings): end-to-end backfill container test

### Forward-fix seam
Enqueue the existing `embed_source` job from the `DOCUMENT_PARSE` handler
(`apps/app-main/src/app_main/handlers.py:handle_process_source`) once
`process_source` returns `chunk_count > 0`. Chosen over inline embedding and
over editing `SourceProcessor` because: (a) it keeps the domain orchestrator
free of a job-queue dependency; (b) every ingest path (async upload, sync
upload, reprocess) funnels through this single handler; (c) `embed_source` is
already idempotent. Enqueue is best-effort — a queue hiccup logs but never
fails the already-persisted extraction.

### Migration 63 (`migrations/63.surrealql`)
```surrealql
DEFINE FIELD IF NOT EXISTS embedding ON TABLE source TYPE option<array<float>>;
```
`option<...>` so NONE = "not yet computed / no chunk vectors" (distinct from
`[]`); legacy rows read back NONE; dimension not schema-pinned (follows the
model). Down: `REMOVE FIELD IF EXISTS embedding ON TABLE source;`.

### Acceptance criteria — all PASS
1. **Fresh ingest auto-embeds** — handler enqueues `embed_source` when chunks
   exist (`test_handle_process_source_autoembed.py`); `embed_source` → real
   `source_embedding` rows proven on a container
   (`test_backfill_chunk_embeddings_db.py::test_backfill_chunks_real_run...`).
2. **Backfill idempotent + model-derived dim** — dry-run reports the missing
   set without writing; real run populates `source_embedding`; re-discovery → 0
   missing; dimension model-derived (logged), never hardcoded.
3. **`source.embedding` aggregate** — migration 63 + `populate_source_embedding`
   mean-pools chunk vectors; no chunk vectors → NONE gracefully.
4. **No hardcoded dim; Track J guardrail green** — no `768`/`1024` literal in
   touched code; `test_embedding_local_guardrail.py` 6/6 pass.
5. **Suites green** — see below.

### Test evidence
- `apps/app-main` (`-k "embedding or backfill or source_processor or source_embedding"`, scoped to `apps/app-main/tests`): **50 passed**
- Track J guardrail (`test_embedding_local_guardrail.py`): **6 passed**
- `pipelines/embeddings` (`pipelines/embeddings/tests`): **28 passed**
- Migration roundtrip (`test_migrations_roundtrip.py`): **16 passed** (incl. 2 new migration-63)
- Backfill container tests (`test_backfill_chunk_embeddings_db.py`): **4 passed**
- mypy on touched files: clean (the one reported error is the pre-existing
  `get_embedding_count` return-Any, not R.0 code).

> The legacy top-level `tests/test_domain.py|test_graphs.py|test_models_api.py|test_utils.py`
> have pre-existing collection import errors unrelated to R.0 — scope pytest to
> the package `tests/` dir to avoid them.

### LIVE backfill (gated checkpoint — for the operator to run)
Run inside the app env (e.g. the app container or `uv run --project apps/app-main`),
against the live staging DB, in order:

```bash
# 1. chunk embeddings — dry-run first, then real, then confirm idempotent
python scripts/backfill_chunk_embeddings.py --dry-run
python scripts/backfill_chunk_embeddings.py
python scripts/backfill_chunk_embeddings.py --dry-run        # expect 0 missing

# 2. source-level aggregate vectors (mean-pool; no model calls)
python scripts/backfill_chunk_embeddings.py --source-embeddings --dry-run
python scripts/backfill_chunk_embeddings.py --source-embeddings
```
Expect the chunk run to log `Embedding dimension = 1024`. Post-run verification
query:
```surrealql
-- chunk embeddings present for the 6 live sources, and aggregate set
SELECT id,
  count(SELECT id FROM source_embedding WHERE source = $parent.id) AS chunk_vecs,
  (embedding != NONE) AS has_aggregate,
  array::len(embedding) AS agg_dim
FROM source;
```
All 6 live sources should show `chunk_vecs > 0`, `has_aggregate = true`,
`agg_dim = 1024`.

The orchestrator can call `populate_all_source_embeddings()` /
`populate_source_embedding(source_id)` from
`scripts/backfill_chunk_embeddings.py` directly for the aggregate step.
