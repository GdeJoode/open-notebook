# Track W — status

## Phase W.1 — Expose hybrid search via the router (Feature 5a) — READY FOR REVIEW

**Branch**: `track/w1-hybrid-endpoint` (off `main`)
**Commits**:
- `982a135` feat(search): expose hybrid (BM25+vector) search via /search router
- `043ce43` test(search): router + service tests for hybrid /search
- `1b42c96` fix(search): rank hybrid results by RRF, not broken raw-score sum

### Scope (confirmed minimal)
The hybrid fusion chain already existed end-to-end
(`SearchService.hybrid_search` → `RetrievalService.hybrid_search` →
`SearchRepository.hybrid_search`). The only gaps were: the router branch, the
request field, and `text_weight` thread-through. No fusion logic, no `fn::`
SurrealDB functions, and no embedding model were modified.

### Files touched
- `apps/app-main/src/app_main/api/schemas.py` — `SearchRequest.type` Literal
  gains `"hybrid"`; new `text_weight: float = 0.5` (ge=0, le=1).
- `apps/app-main/src/app_main/services/search_service.py` —
  `hybrid_search` now forwards `text_weight` (was silently dropped, always
  defaulting to 0.5 downstream).
- `apps/app-main/src/app_main/api/routers/search.py` — new `type == "hybrid"`
  branch calling `search_svc.hybrid_search(...)`; degrades gracefully with no
  embedding model (text-only fallback in retrieval layer) rather than the 400
  the vector branch raises.
- `apps/app-main/tests/test_search_router.py` — new router tests.
- `apps/app-main/tests/test_search_service.py` — `text_weight` thread-through
  + real-repo RRF fusion test.
- `packages/surrealdb-service/src/surrealdb_service/repositories/search.py` —
  **RRF fusion** (replaces the broken raw-`score` linear sum; see below).
- `packages/surrealdb-service/tests/test_search_hybrid_fusion.py` — new
  rank-based fusion unit tests (7).

### Request schema change
`type` now accepts `"text" | "vector" | "hybrid"` (default `"text"`,
unchanged). New `text_weight` (0..1, default 0.5) used only by hybrid; the
vector weight is `1 - text_weight`. Existing `minimum_score`,
`search_sources`/`search_notes`, `limit` are reused for hybrid.

### Acceptance criteria
1. **Hybrid reachable / fused** — `POST /search {type:"hybrid"}` returns
   BM25+vector fused, deduped, tagged, **RRF-ranked** results. Router test
   `test_hybrid_happy_path_returns_fused_results`; real-`SearchRepository`
   ranking proofs in `packages/surrealdb-service/tests/test_search_hybrid_fusion.py`
   (`test_both_signals_outrank_single_signal`,
   `test_order_reflects_rank_not_raw_magnitude`,
   `test_combined_score_non_degenerate`) + service-level
   `test_hybrid_search_fuses_text_and_vector`.
2. **text/vector byte-identical** — `TestSearchBackwardCompatible`
   (`test_text_branch_unchanged`, `test_default_type_is_text`,
   `test_vector_branch_unchanged`, `test_vector_without_embedding_model_still_400`).
   Full `apps/app-main/tests`: 1229 passed (3 pre-existing `docling`-missing
   failures unrelated to W.1, confirmed identical on `main`).
3. **Local embedding, graceful empty/no-model** — embedding resolved via DB
   default `mxbai-embed-large` (1024-dim, no hardcoded dim);
   `test_hybrid_empty_results`, `test_hybrid_no_embedding_model_does_not_error`.
4. **Tests** — router (happy + empty + no-model) and service (RRF fusion +
   text_weight) added; 7 repo-level RRF tests; retrieval pipeline suite green
   (24 passed); surrealdb-service non-docker suite green (69 passed).
5. **Staging probe** (read-only, `SURREAL_DATABASE=staging`) — non-degenerate
   RRF scores and sensible order:
   - `"Regio Deal"` → 4 `hybrid` hits (both signals), scores strictly
     descending `0.016393 > 0.016129 > 0.015873 > 0.015625` (was all `0.0000`
     before the fix).
   - `"regiodeal samenwerking gemeente"` → 4 `vector`-only hits,
     `0.008197 > 0.008065 > 0.007937 > 0.007812`.
   Embedding resolved to local `mxbai-embed-large` (verified 1024-dim).

### Resolved during W.1: RRF ranking (commit `1b42c96`)
The repo-layer fusion read `item.get("score", 0)`, but `fn::text_search`
returns `relevance` and `fn::vector_search` returns `similarity` — neither is
`score`. So `_combined_score` collapsed to 0 and "ranked" output degenerated to
insertion order, **failing AC1**. Renaming + linearly combining would let the
unbounded BM25 signal dominate the bounded cosine signal, so the fix uses
**Reciprocal Rank Fusion** (scale-independent; consumes rank not magnitude),
matching Track R's validated `shared.retrieval.hybrid_fusion` (k=60). The
formula is reimplemented inline — Track R's `fuse_rankings` is shaped for the
source-level `(dense, kg)` `FusedResult`/`SignalProvenance` contract and does
not generalize cleanly to the two-list chunk/note item-dict shape. Dedup and
`text`/`vector`/`hybrid` tagging unchanged; added `_text_rank`/`_vector_rank`
provenance. The `fn::` functions and text/vector router paths are untouched.
