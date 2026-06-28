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

---

## Phase W.2 — Local cross-encoder reranker service (Feature 5b) — READY FOR REVIEW

**Branch**: `track/w2-reranker-service` (off `main`, has W.1 merged)
**Commits**:
- `1333966` feat(reranker): cross-encoder reranker microservice
- `24a235d` feat(compose): reranker service + RERANKER_SERVICE_URL on app-main
- `f41b818` feat(search): wire cross-encoder rerank into hybrid /search behind a flag

### Service shape (`services/reranker/`)
A small FastAPI app mirroring the `docling`/`whisperx` service pattern
(`api.py`, `Dockerfile`, `requirements.txt`, venv/PEP-668):
- `POST /rerank` — body `{query: str, passages: [str], top_k?: int}` →
  `{results: [{index: int, score: float}]}`, sorted by `score` descending;
  `index` is the position in the request `passages` list.
- `GET /health` — `{status, service, model, model_loaded}`.
- Loads a `sentence-transformers` `CrossEncoder` lazily (once), warmed at
  startup via a `lifespan` handler (`RERANKER_SKIP_WARMUP=1` to defer).
- **Model**: env `RERANKER_MODEL`, default `BAAI/bge-reranker-v2-m3`
  (multilingual — corpus is Dutch; configurable to tune size/latency).
  `RERANKER_DEVICE` (default empty → CPU/auto), `HF_HOME=/data/models` cache.
- `torch` + `sentence-transformers` live ONLY in
  `services/reranker/requirements.txt`.

### docker-compose
New `reranker` service (port `8105`, CPU by default, `reranker_models` volume
for the HF cache) next to `docling`/`whisperx`; `RERANKER_SERVICE_URL=
http://reranker:8105` added to the `open_notebook` service env. No hard
`depends_on` — app boots/searches even when the reranker is down (graceful
fallback). `docker compose config` validates.

### app-main wiring (plain HTTP, no ML libs)
- `services/reranker_http_client.py` — `RerankerHttpClient` (env-resolved URL,
  `httpx`); raises `RerankerServiceError` on any transport/service failure.
- `services/rerank_orchestrator.py` — `rerank_hybrid_results(...)`: takes the
  top-N fused results, sends passage texts (`content`→`title` fallback) to the
  service, reorders by returned scores, attaches `_rerank_score`. On
  `RerankerServiceError` → falls back to the heuristic `retrieval.reranker.
  Reranker` (logged, never raises).
- `SearchRequest` gains `rerank: bool = False` + `rerank_top_n: int = 50`.
- Router hybrid branch calls the orchestrator only when `rerank=true`.
  **Wiring point**: `apps/app-main/.../api/routers/search.py`, hybrid branch,
  AFTER `search_svc.hybrid_search(...)` returns the fused results.

### torch/sentence-transformers NOT in app-main (verified)
- `apps/app-main/pyproject.toml` declares neither (printed the full dep list;
  `torch in deps: False`, `sentence-transformers in deps: False`).
- The `retrieval` package (app-main's fallback dep) declares neither.
- No `import torch` / `import sentence_transformers` anywhere under
  `apps/app-main/src` or `pipelines/retrieval/src`.
- (They ARE importable from the *shared workspace venv* because sibling
  members like the whisperx/extraction pipelines pull them — that is the
  uv-workspace shared `.venv`, not app-main's own dependency closure, and not
  the reranker container, which has its own `requirements.txt`.)

### Acceptance criteria — evidence
1. **`/rerank` reorders + schema** — `services/reranker/tests/test_rerank_api.py`
   (4 tests, `CrossEncoder` mocked → no 2 GB download): asserts desc ordering,
   `index→passage` mapping, `top_k` truncation, empty-passages, `/health`.
2. **`rerank=true` reorders top-N; `rerank=false` byte-identical** —
   `test_search_rerank_router.py::TestRerankTrue` (reorder + `_rerank_score` +
   `rerank_top_n` tail preserved) and `::TestRerankFalseByteIdentical`
   (`resp.content == baseline.content`, reranker `rerank` patched to raise on
   call so any call would fail the test; no `_rerank_score` leaks).
3. **Service-down → heuristic fallback, never 500** —
   `::TestRerankServiceDownFallback` (client patched to raise
   `RerankerServiceError`; asserts 200 + heuristic score-desc reorder +
   `_rerank_score`) and the orchestrator-level
   `test_orchestrator_fallback_on_service_error`. Plus client transport tests
   in `test_reranker_http_client.py` (HTTP 503 / ConnectError → `RerankerServiceError`).
4. **torch/sentence-transformers only in `services/reranker/`** — see section above.
5. **Tests** — reranker service unit (4) + app-main integration with service
   mocked (6) + client transport (7) + orchestrator (2). Full `apps/app-main`
   suite: **1242 passed**, 2 skipped, **3 pre-existing `docling`-missing
   failures** (the known baseline, unrelated).
6. **Live model run — GATED (not run here).** The CPU `torch` +
   `sentence-transformers` wheel install made the image build exceed the inline
   timeout (~10 min); the ~2 GB `bge-reranker-v2-m3` download happens at first
   request on top of that. Dockerfile/requirements mirror the working
   docling/whisperx services and `docker compose config` validates. **Operator
   deploy + smoke test** (run by the USER):

   ```bash
   docker compose build reranker
   docker compose up -d reranker
   docker compose logs -f reranker          # wait for model load (~2 GB first run)
   curl -s localhost:8105/health            # expect model_loaded: true

   curl -s -X POST localhost:8105/rerank \
     -H 'content-type: application/json' \
     -d '{"query":"Wat zijn de doelstellingen van de regiodeal?",
          "passages":[
            "De regiodeal richt zich op economische versterking van de regio.",
            "Het weer is vandaag zonnig met een lichte bries.",
            "Doelstellingen van de regiodeal omvatten werkgelegenheid en leefbaarheid."]}'
   # Expect the two regiodeal passages (indices 0, 2) ranked above the weather one (1).
   ```

   Then end-to-end against the app: `POST /search {type:"hybrid", rerank:true}`
   on a Dutch query reorders the top-N; `docker compose stop reranker` →
   the same request still returns 200 via the heuristic fallback.
