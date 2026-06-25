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

---

## R.0b — chunk-embedding context-length guard (2026-06-25)

**Branch**: `track/r0b-embed-context-fix` (off `main`)
**Commits**: `08d1751` fix, `ba62b7d` tests
**Status**: Ready for review. Full 6/14-source backfill is the human-gated step.

### Bug
Live R.0 backfill failed on every source: `Ollama API error: the input length
exceeds the context length` → `embedded_sources=0 failed=N`. `mxbai-embed-large`
has a ~512-token context; chunk passages exceed it and the embedding service
applied no length guard.

### Root cause (confirmed live)
The esperanto Ollama provider embeds **one text per HTTP call** against the
legacy `/api/embeddings` endpoint, so this is per-text over-length, not a
batch-total. That legacy endpoint **ignores `truncate` and `options.num_ctx`** —
verified directly: `truncate=true`, `truncate=false`, and `num_ctx=8192` all
return the same 500 on a 9000-char input. Only the newer `/api/embed` endpoint
honours `truncate` (default true → 1024-dim returns), but esperanto does not use
it. So threading `truncate=true` through the wrapper would NOT have fixed it.
`num_ctx` does not raise mxbai's 512-token pin either; only truncation works.

### Fix (client-side, model-aware truncation)
`pipelines/embeddings/src/embeddings/{config.py,service.py}`:
- `max_input_chars=2048` (≈512 tok × 4 chars/tok, Latin) cap applied before
  every embed call; tail-truncate (keep head), log each cut.
- `truncate_on_context_error=True`: on a residual context-length rejection
  (dense/CJK text the char cap misses — char count is not a token proxy: CJK
  fails ~600 chars, dense Latin survives ~2500), recover **per-text by halving**
  until accepted. A single over-long text never fails its batch-mates and **no
  chunk is dropped**. Dimension stays model-derived (1024), never hardcoded;
  embeddings stay LOCAL (no `model_routing` import).

### Evidence (per acceptance criterion)
1. **Real-Ollama over-long embed** — `test_service_ollama_context.py` (runs
   against local mxbai, not a fake): a ~9000-char text fails raw (`context
   length`), then embeds to **dim 1024** through the guarded service; dense CJK
   recovers via adaptive shrink to 1024. PASS.
2. **R.0 unit/container tests unaffected** — `pipelines/embeddings` 36 passed;
   `apps/app-main` backfill + autoembed 12 passed.
3. **Truncation observable, no silent drops** — `logger.warning` on every char
   cap + every adaptive shrink; batch order/shape preserved; unit tests assert
   no chunk dropped.
4. **Track J guardrail green** — `test_embedding_local_guardrail.py` 6 passed;
   no cloud routing; dim not hardcoded.
5. **Live smoke** — `--dry-run` reports 14 sources / 1960 chunks; `--limit 1`
   ran twice, embedded 2 sources (140 chunks each), **dim=1024, failed=0**,
   remaining 14→13→12 (idempotent + resumable). Stored vectors confirmed dim
   1024 in staging.

### Note on current staging data
Staging has been re-ingested since the bug report (now 14 sources / smaller
chunks, max chunk **1519 chars**). The 200 longest current chunks embed raw
without the guard, so the per-chunk failure does **not** reproduce on today's
data — but the original ~2000-char chunks did exceed mxbai's context (reproduced
directly with 9000-char + CJK input). The guard is correct defence-in-depth for
whenever over-long input occurs.

### Remaining
Human-gated full backfill: `python scripts/backfill_chunk_embeddings.py`
(12 sources left), then `--source-embeddings` for aggregates.
