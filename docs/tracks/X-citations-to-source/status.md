# Track X — Status

## Phase X.1 — Thread chunk provenance through retrieval (Backend)

**Branch**: `track/x1-retrieval-provenance` (off `main` @ `5420666`, all of Track W merged)
**State**: complete — ready for review.

### Hit→chunk provenance mapping (the investigation)

The `fn::text_search`/`fn::vector_search` SurrealDB functions (`migrations/4.surrealql`)
**collapse** a source's many matching `source_embedding` rows to a *single source-level
hit*: they `SELECT source.id AS id` and `GROUP BY id` with `math::max(similarity|relevance)`.
So the returned hit `id`/`parent_id` is the **`source` id** — the specific matching chunk
(and its page) is lost *inside the function*.

The provenance lives on the `chunk` table (`physical_page`/`printed_page`/`section_path`/
`element_type`), reachable from an embedding via the **`source_embedding.chunk` link**
(`record<chunk>`, added in `migrations/27.surrealql`). Verified on `staging` (read-only):
`source_embedding` rows are **100%** chunk-linked (1447/1447); chunks carry real pages.

### Hydration approach (repo-layer batch SELECT — `fn::` untouched)

`SearchRepository.hydrate_provenance(hits, embedding)` runs **one** batched follow-up
`SELECT` over `source_embedding` joined to `chunk`, keyed on the hit source ids:

- **vector / hybrid** (embedding present): re-score each hit-source's embeddings by
  `vector::similarity::cosine` and take the top chunk per source. Because
  `fn::vector_search` collapses with `math::max(cosine)`, this top chunk is **exactly**
  the row that produced the source's winning score → the attached `physical_page` is the
  page of the *actual chunk the hit came from*. Verified equal to the `fn::` collapsed max
  on staging to **1e-9**.
- **text-only** (no embedding): BM25 `search::score` is **not** reproducible outside its
  originating query context (measured 8.32 vs 5.64 on a re-run on staging), so we do **not**
  fabricate a chunk/page. We attach source-level structural provenance
  (`section_path`/`element_type` of the source's first chunk) and leave `physical_page`
  `None` rather than assert an unverifiable page.
- **notes / non-source hits / lookup failure**: every provenance key → `None`; never raises
  (the hydration is wrapped so a DB failure degrades silently and leaves existing keys intact).

`hybrid_search` calls its text/vector legs with `hydrate=False` and hydrates the **fused**
set once, with the embedding (so the page reflects the vector match, not the text leg).

### Surfaced shape (additive)

Each hit from `RetrievalService.{vector,hybrid,text}_search` (and `SearchRepository`) now
carries, alongside its existing fields (`id`/`parent_id`/`title`/`relevance|similarity`/
fusion fields):

```
chunk_id, physical_page, printed_page, section_path, element_type, source
```

All keys are always present (stable shape); value-or-`None`. Existing callers that ignore
them are unaffected — `RetrievalService` and `app-main/search_service` delegate without the
new `hydrate` kwarg, so signatures are unchanged.

### `fn::` functions: UNTOUCHED

No migration added; `migrations/1.surrealql`/`4.surrealql` `fn::text_search`/`fn::vector_search`
are not modified. Hydration is pure Python + a read-only follow-up `SELECT`.

### Test evidence (per acceptance criterion)

- **AC1** (provenance keys, correct mapping, None-handling): `test_search_provenance.py`
  (17 tests, DB-mocked) — real chunk page/section on source hits, stable key set, notes/miss
  → `None`, no-overwrite-with-None. **Exact mapping** proven on staging:
  `test_search_provenance_staging.py::test_hydrated_page_matches_the_chunk_the_hit_came_from`
  (cosine top-1 == `fn::` collapsed `math::max`, 1e-9).
- **AC2** (backward-compat): existing suites green — `test_search_hybrid_fusion.py` (7),
  `test_repositories.py` (11), retrieval `test_service.py` (26, incl. 2 new passthrough),
  app-main `test_search_service.py`+`test_search_router.py` (19). Full non-docker
  surrealdb-service suite: **92 passed, 2 skipped** (the staging probe self-skips without the
  env var), no regressions. Fusion tests pass even with the DB unreachable (graceful degrade).
- **AC3** (unit shape + staging probe): `test_search_provenance.py` (shape) +
  `test_search_provenance_staging.py` (real `physical_page`/`section_path` on a known
  multi-page source `source:dndibxmjveoxk7tfqfsl`, 15 pages, "Convenant Regio Deal
  Midden-Limburg"). Run: `SURREAL_DATABASE=staging uv run --project packages/surrealdb-service
  pytest packages/surrealdb-service/tests/test_search_provenance_staging.py` → **2 passed**.

### Files

- `packages/surrealdb-service/src/surrealdb_service/repositories/search.py` — `hydrate_provenance`,
  `_best_chunk_per_source`, `_hit_source_id`, `hydrate` flag on the three search methods.
- `packages/surrealdb-service/tests/test_search_provenance.py` — unit (mocked).
- `packages/surrealdb-service/tests/test_search_provenance_staging.py` — read-only staging probe.
- `pipelines/retrieval/tests/test_service.py` — passthrough tests.

### Commits (on `track/x1-retrieval-provenance`)

- `a2a8c24` feat(search): hydrate per-hit chunk provenance in SearchRepository (X.1)
- `666d6a1` test(search): provenance hydration unit tests + read-only staging probe (X.1)
- `d4c1f17` test(retrieval): assert RetrievalService surfaces provenance keys verbatim (X.1)

### Note for X.2 / memory

- The hit id is a **source id**, not a chunk/embedding id; X.2 should consume the new
  `chunk_id`/`physical_page`/`section_path` keys (already on each hit), not re-derive them.
- Per-hit precision differs by mode: vector/hybrid give an *exact* chunk page; text-only gives
  source-level only (no page). The `ask` graph defaults to vector search, so the precise path
  is the primary one. The X.3 faithfulness guard should membership-check the surfaced
  `chunk_id` (present only for vector/hybrid hits).
