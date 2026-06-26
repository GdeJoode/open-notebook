# Track R — status

## Phase R.0 — ✅ LIVE COMPLETE (2026-06-26)
The embedding foundation is fully live on `staging`. All 6 sources: non-empty chunks ==
`source_embedding` count (284/209/135/260/280/279), source-level aggregate vector set, **dim 1024**, local.
Required FIVE fixes, each adversarially reviewed + merged — the original R.0 was green on tests but the
REAL run exposed data/infra realities the synthetic container tests didn't:
- **R.0** (`119ecd3`) — auto-embed on ingest + migration 63 (`source.embedding`) + mean-pool aggregate + backfill.
- **R.0b** (`4f42537`) — chunk-embedding context-length guard (mxbai ~512 tok; esperanto uses Ollama legacy `/api/embeddings` which ignores truncate → client-side cap + per-text halving).
- **R.0c** (`c69d16f`) — backfill detects PARTIALLY-embedded sources (`chunk_count > embedding_count`, was `= 0`).
- **R.0d** (`835d7e6`) — skip empty/whitespace chunks (table/image chunks) on embed AND exclude them from completeness detection (coupled).
- **R.0e** (`977634a`) — migration 64: coalesce drifted strict `source.private` (NONE→false) so source rows are writable (the aggregate UPDATE was blocked).

Live ops run: orphan GC (10,501→1,448 chunks), migrations 63+64 applied to staging (v62→64), full chunk
backfill (1447 non-empty chunks, dim 1024), source aggregates (6/6). Follow-ups (→ R.6 / later): `embed_source`
embeds `is_content=False` noise chunks; `embed_note`/`embed_insight` whole-doc head-truncation; distinct-vector
alignment test; 2 comment nits in `migrations/64.surrealql`. **Systematic strict-field drift → Track S.**

---

## Phase R.0 — Embedding foundation — MERGED + APPROVED; live backfill BLOCKED (→ R.0b) [superseded by the LIVE COMPLETE entry above]
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

## R.0c — Partial-source detection in chunk backfill (2026-06-25)

Branch `track/r0c-partial-detection` (off `main` w/ R.0 + R.0b merged).
Commits: `cc7c2c3` (fix), `d8cca05` (tests).

### Bug (measured live on staging)
`backfill_chunk_embeddings.py --dry-run` reported **0 sources / 0 chunks** while
staging had only **390 of 1448 chunks embedded** — all 6 sources PARTIALLY
embedded (70/284, 80/209, 80/135, 10/260, 100/281, 50/279). Backfill silently
did nothing; the "resumable/idempotent" claim was false for partial sources.

### Root cause
`list_sources_missing_chunk_embeddings()` selected sources with
`count(source_embedding) = 0`. A partial source has `count >= 1`, so the
`AND ... = 0` clause excluded it — any source with ≥1 embedding was treated as
done. Partial states arise because `_embed_chunks` commits per-batch; the R.0b
context-length failure left sources embedded part-way.

### Fix
Detect on counts-differ: `count(chunk) > count(source_embedding)` (fully OR
partially unembedded). `embed_source` deletes+rebuilds a source's embeddings, so
source-level "counts differ" is sufficient — no per-chunk granularity needed.
`count_missing_chunks` now reports only the unembedded chunks of a partial source
(`chunk_count - embedding_count`), not its whole chunk set. Dim stays
model-derived (1024), local, no hardcoding.

### Evidence (per acceptance criterion)
1. **Partial flagged (fails-old/passes-new)** —
   `test_discovery_query_flags_partially_embedded_source`: seeds 3-chunk source
   with 1 embedding, asserts it is flagged. FAILS against reverted `= 0` query
   (`assert partial in missing` -> AssertionError), PASSES against the fix.
   The updated `test_discovery_query_finds_unembedded_sources` also fails-old.
2. **Fully embedded NOT flagged** — same test, `full` (3/3) absent from missing.
3. **Zero embeddings still flagged** — same test, `zero` (0/3) present (no
   regression).
4. **embed_source completes partial** —
   `test_embed_source_completes_partial_then_clean`: stale 1-of-3 row, run
   embed_source (local fake model), assert count==3 and re-detection returns 0.
5. **Existing R.0 tests green / dry-run writes nothing** — full suite
   `test_backfill_chunk_embeddings.py` + `_db.py` = **11 passed**. Dry-run path
   unchanged (`get_embedding_count` left clamped, no model resolved on dry-run).

NOT run: real backfill against staging (human-gated). Expectation once gated run
fires: dry-run reports all 6 partial sources / the 1058 missing chunks.

## R.0d — skip empty/whitespace chunks when embedding (live backfill fix)

Branch: `track/r0d-empty-chunk-skip` (off `main` after R.0c).
Commits: `7e19c57` (embeddings service skip), `7a4b4bd` (detection exclude + tests).

### Bug (measured live on staging)
The full chunk backfill embedded 5 of 6 sources to dim 1024 but source
`dndibxmjveoxk7tfqfsl` FAILED with `Embedding failed after 3 attempts: Text
cannot be empty`. Root cause: that source has exactly 1 chunk with empty `text`
— a `table`-type chunk (`chunk:ik1k140l5lbwol7vt1`, element_type=`table`, order
280, no extracted text). The model rejects empty input, aborting the whole
source. (1 such chunk of 1448 corpus-wide today, but table/image chunks with no
text are a recurring class.)

### Root cause
Two coupled gaps: (1) `_embed_chunks` / `_embed_text_chunks` passed every
chunk's text to the model, including `""`; the model raised `Text cannot be
empty` and `_embed_with_retry` re-raised after retries, killing the source.
(2) the R.0c detection counted ALL chunks, so even if the empty chunk were
skipped, `chunk_count > embedding_count` would flag the source forever and
re-embed it every run (eternal-incomplete).

### Fix (two-sided)
1. **Embed side** (`pipelines/embeddings/src/embeddings/service.py`): new guard
   `_is_empty_text` (None / empty / `str.strip()` whitespace). Both embed paths
   build an `embeddable` list of `(original_order, chunk)` for non-empty chunks
   only, embed just those, and zip vectors back — empties get no model call and
   no `source_embedding` row; batch-mates still embed; original `order`
   preserved.
2. **Detection side** (`scripts/backfill_chunk_embeddings.py`):
   `list_sources_missing_chunk_embeddings` counts only
   `text != NONE AND string::trim(text) != ""` chunks;
   `count_missing_chunks` filters empties in Python via `_is_empty_chunk`
   (shared None/empty/whitespace logic, works for `Chunk` objects and raw
   strings). A source whose every non-empty chunk is embedded reads COMPLETE.

Dim stays model-derived (1024), local, no hardcoding.

### Evidence (per acceptance criterion)
1. **Empty/whitespace skipped, source succeeds (fails-old/passes-new)** —
   `test_service.py::TestEmbedSourceEmptyChunkSkip::test_empty_and_whitespace_chunks_skipped_mixed_batch`:
   normal + empty(table) + whitespace + normal, strict model rejecting empties;
   asserts 2 rows, model only sees the 2 normals, orders [0,3]. FAILS against
   reverted service (`Text cannot be empty`), PASSES with fix (verified by
   `git stash` of the service file -> 4 fail; restored -> pass).
2. **Not eternally flagged** (DB, `@requires_docker`) —
   `test_backfill_chunk_embeddings_db.py::test_empty_chunks_skipped_and_not_eternally_flagged`:
   5 chunks (3 normal + empty table + whitespace), `embed_source` writes 3 rows
   and does not raise; detection does NOT flag despite chunk_count(5) >
   embedding_count(3). FAILS against reverted detection query (source stays
   flagged), PASSES with fix (verified by `git stash` of the script -> fail;
   restored -> pass).
3. **Genuinely partial still flagged (no R.0c regression)** (DB) —
   `test_partial_nonempty_source_still_flagged_with_empties`: 3 non-empty + 1
   empty, 1 embedded -> flagged, `count_missing_chunks == 2`.
4. **count_missing_chunks counts only non-empty unembedded** —
   DB-free `test_count_missing_chunks_excludes_empty` (3 of 5, then 0) + the AC3
   DB assertion above.
5. **No regressions / dry-run writes nothing / Track J green** —
   embeddings suite **40 passed**; backfill DB-free+DB **15 passed**; Track J
   `test_embedding_local_guardrail.py` **6 passed**. Dry-run path unchanged.

NOT run: real backfill against staging (human-gated). Expectation once gated:
source `dndibxmjveoxk7tfqfsl` embeds its non-empty chunks to dim 1024, skips the
1 empty table chunk, and does NOT reappear in the missing set.

## R.0e — backfill NONE strict `source` fields (schema-drift fix) (2026-06-26)

Branch `track/r0e-source-field-backfill` (off `main`). Ready for review.

### Blocker
The R.0 source-aggregate write (`set_aggregate_embedding`, an
`UPDATE source:x SET embedding=...`) failed for every staging source with
`Found NONE for field 'private', with record 'source:...', but expected a bool`.
Legacy rows predate migration 52's `private TYPE bool DEFAULT false`, so they
carry `private = NONE`. A SCHEMAFULL UPDATE re-validates the WHOLE record, so
that one NONE blocks ANY write to the row — the aggregate embedding AND the
app's own source updates. Identical drift class migration 61 fixed for `entity`.

### Drifted-field analysis (empirical, full 1->63 schema)
Booted a fresh container, ran `INFO FOR TABLE source` + a per-field
NONE-rejection probe (UNSET each field, attempt a write). Result:

| field | type | rejects NONE? |
|-------|------|---------------|
| `private` | `bool DEFAULT false` | **YES — the only one** |
| `topics` | `option<array<string>>` | no (NONE valid; coalesced to `[]` for hygiene) |
| `embedding` | `option<array<float>>` | no — NONE intended (mig 63); NOT touched |
| `asset`, `command`, `full_text`, `metadata`, `title`, `zotero_*` | `option<...>` | no |
| `created`, `updated` | computed `VALUE` clause | no (never NONE) |

So `private` is the sole strict, non-`option<>`, no-VALUE field — the load-bearing
fix. Task statement's claim that `topics` would block next is incorrect for the
migrated schema (it's `option<array<string>>`); coalesced anyway as data hygiene.

### Migration 64 body
```surrealql
UPDATE source SET
    private = private ?? false,
    topics  = topics  ?? [];
```
Idempotent (no-op on clean rows), no schema change. `64_down.surrealql` = documented
no-op (mirror 61_down; data backfill has no faithful inverse).

### Evidence (per acceptance criterion, all `@requires_docker`, all green)
1. **Defaults restored** — `test_migration_64_backfills_drifted_strict_fields`:
   forged legacy row (`private=NONE`, `topics=NONE`) reads back `private=false`,
   `topics=[]` after 64.
2. **UPDATE fails pre-64 / passes post-64 (load-bearing)** —
   `test_migration_64_unblocks_aggregate_embedding_update`: the
   `UPDATE source:x SET embedding=[...]` raises "Found NONE for field private"
   pre-64 (`pytest.raises` + asserts "private" in message), succeeds on the SAME
   row post-64.
3. **Idempotent** — `test_migration_64_idempotent`: applying 64 twice doesn't
   raise; a clean row with `private=true` / `topics=['ai','db']` is unchanged
   (`?? default` no-ops on real values) and stays writable.
4. **Clean 1->64 chain on fresh container** — existing `test_migrations_applied`
   derives the expected version set from disk; 64 is applied + recorded.
5. **64_down present + sane** — comment-only no-op (0 non-comment lines, same
   shape as 61_down).

Full migrations roundtrip suite: **19 passed** (no regressions).

Drift reproduced faithfully via `REMOVE FIELD -> UNSET -> re-DEFINE` (recreates
the pre-mig-52 history; a bare DEFINE doesn't backfill existing rows). A plain
`UPDATE ... UNSET private` is itself rejected by the strict revalidation, so it
cannot reach the legacy state — hence the remove/redefine dance.

NOT run: the live aggregate / live migration against staging (human-gated).
Systematic staging strict-field drift: `entity` fixed by mig 61, `source` by
mig 64; likely other tables carry the same drift.
