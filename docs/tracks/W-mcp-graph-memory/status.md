# Track W — status

## Phase W.1 — Expose hybrid search via the router (Feature 5a) — READY FOR REVIEW

**Branch**: `track/w1-hybrid-endpoint` (off `main`)
**Commits**:
- `982a135` feat(search): expose hybrid (BM25+vector) search via /search router
- `043ce43` test(search): router + service tests for hybrid /search

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
  + real-repo fusion test.

### Request schema change
`type` now accepts `"text" | "vector" | "hybrid"` (default `"text"`,
unchanged). New `text_weight` (0..1, default 0.5) used only by hybrid; the
vector weight is `1 - text_weight`. Existing `minimum_score`,
`search_sources`/`search_notes`, `limit` are reused for hybrid.

### Acceptance criteria
1. **Hybrid reachable / fused** — `POST /search {type:"hybrid"}` returns
   BM25+vector fused, deduped, tagged results. Router test
   `test_hybrid_happy_path_returns_fused_results`; service-level fusion
   `test_hybrid_search_fuses_text_and_vector` (real `SearchRepository`).
2. **text/vector byte-identical** — `TestSearchBackwardCompatible`
   (`test_text_branch_unchanged`, `test_default_type_is_text`,
   `test_vector_branch_unchanged`, `test_vector_without_embedding_model_still_400`).
   Full `apps/app-main/tests`: 1175 passed (3 pre-existing `docling`-missing
   failures unrelated to W.1, confirmed identical on `main`).
3. **Local embedding, graceful empty/no-model** — embedding resolved via DB
   default `mxbai-embed-large` (1024-dim, no hardcoded dim);
   `test_hybrid_empty_results`, `test_hybrid_no_embedding_model_does_not_error`.
4. **Tests** — router (happy + empty + no-model) and service (fusion +
   text_weight) added; retrieval pipeline suite green (24 passed).
5. **Staging probe** (read-only, `SURREAL_DATABASE=staging`) — `"Regio Deal"`
   returned 4 fused `Convenant Regio Deal …` sources (tagged `hybrid`);
   `"regiodeal samenwerking gemeente"` returned the same 4 (tagged `vector`).
   Embedding resolved to local `mxbai-embed-large` (verified 1024-dim).

### Known pre-existing issue (out of W.1 scope)
The repo-layer fusion reads `item.get("score", 0)`, but `fn::text_search`
returns `relevance` and `fn::vector_search` returns `similarity` — neither is
`score`. So `_combined_score` collapses to 0 on real data and ranking falls
back to insertion order. Dedup + `hybrid`/`vector`/`text` tagging still work.
This predates W.1 (instructions forbid touching fusion logic / `fn::`); flag
for W.2 (reranker reorders top-N and would mask this) or a dedicated fix.
