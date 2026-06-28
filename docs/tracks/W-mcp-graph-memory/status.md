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

---

## Phase W.3 — MCP graph-tools (Feature 1) — READY FOR REVIEW

**Branch**: `track/w3-mcp-graph-tools` (off `main`, has W.1 + W.2 merged)
**Commits**:
- `d7e7c94` feat(graph): unified read-only load_all_edges across relation/mentions/cites
- `8f7bde0` feat(mcp): 5 graph tools (search/get_node/related/cite/add_note) + relation fix
- `519a87c` test(migration-67): scope cites assertion to the edge under test

### Embedding-source decision for `search` (KEY DESIGN QUESTION)
`search_similar` (the existing vector tool) takes a **precomputed embedding** as a
parameter — the surrealdb-service MCP layer has NO embedding model, and adding one
would pull a heavy dep into the package (explicitly out of scope). So the new
`search` tool **reuses that exact contract** (option (a)):
- `embedding` supplied → full **W.1 hybrid** search (BM25 ⊕ vector RRF).
- `embedding` omitted → **lexical BM25** fallback (`fn::text_search`), documented
  in the tool docstring as non-semantic.
This keeps the package dependency-free and consistent with `search_similar`; the
embed step stays in app-main's local-Ollama pipeline.

### `load_all_edges` shape
`GraphRepository.load_all_edges(node_id, edge_types?)` → list of uniform dicts:
```
{ "id": <edge id>, "source": <in>, "target": <out>,
  "edge_type": "relation"|"mentions"|"cites", "metadata": {<per-table fields>} }
```
Both endpoint directions (`in = $id OR out = $id`); per-table data fields rolled
into `metadata` (None-valued dropped); read-only; per-table failures isolated.

### `get_entity_graph` fix
`->relates_to->entity` / `<-relates_to<-entity` → `->relation->entity` /
`<-relation<-entity` (the real migration-39 edge). Falsifiability test asserts the
stale `relates_to` table has 0 rows.

### Per-tool test evidence (docker-backed, all green)
- `search`: lexical + hybrid(embedding) both return JSON lists.
- `get_node`: entity/source polymorphic fetch; missing id → `null`.
- `related`: returns relation(×2, both directions) + mentions + cites; edge_types
  filter + limit honoured.
- `get_entity_graph`: reads `relation` (outgoing contains the target); relates_to=0.
- `cite`: writes, re-cite same pair → `created:false` no duplicate; self-citation
  refused (`reason:self_citation`); only the `cites` edge written (`match_method:mcp`).
- `add_note`: writes note + `artifact` notebook link; embedding defaults to `[]`
  (note.embedding is non-optional array<float>).

### Shared-substrate demonstration (revised after review attempt 1)
The MCP tool fns take no config → they write through the GLOBAL connection pool.
The readback must use a GENUINELY INDEPENDENT connection to prove the
cross-session property (plan AC2), not merely durability. `execute_query(...,
config=cfg)` does NOT qualify: `get_pool(config)` returns the cached singleton
pool and ignores its `config` arg once the pool exists, so it reuses the writer's
own pool. The fix uses `db_connection(cfg)` (`connection.py:33`), which
constructs its OWN `AsyncSurreal` socket and never touches `_pool`. Evidence:
- `test_cite_writes_and_is_idempotent` / `test_add_note_writes_and_links_notebook`
  read the tool-written edge/note back via `_read_independent` (a
  `db_connection(cfg)` helper).
- `test_shared_substrate_across_two_independent_connections` makes it explicit:
  asserts the reader connection object is `!=` the global pool's pooled
  connection, then that this independent reader sees the just-written note.

### Staging `related` probe (read-only, `SURREAL_DATABASE=staging`)
`related(entity:00d3qmmq06mel8pmlyfz)` [canonical "Regio Deal", programme] →
**28 edges: 24 relation + 4 mentions**, both endpoint directions (the entity is
`source` in some, `target` in others). `get_entity_graph` on it: 19 outgoing + 5
incoming `relation` neighbours. SELECT-only; no writes to staging.

### Tests
- `test_graph_load_all_edges.py` — 6 unit tests (stubbed execute_query, no docker).
- `test_mcp_graph_tools_roundtrip.py` — 11 docker-backed tool tests (incl. the
  explicit two-connection shared-substrate test).
- Full surrealdb-service suite: **213 passed**.

### Running the MCP server
`uv run --project packages/surrealdb-service surrealdb-mcp --transport stdio`
(no auth — stdio-local only; add auth before any HTTP exposure).

### Follow-ups for W.4 / hardening (NOT in W.3 scope — reviewer minors)
1. The pre-existing `--transport {sse,streamable-http}` options would expose the
   new `cite`/`add_note` WRITE tools unauthenticated. Default is `stdio` (safe).
   Gate write-tool registration to stdio-only (or add auth) before any HTTP
   transport is used.
2. `cite`'s dedup is read-then-write (not atomic) — unreachable under the
   single-caller stdio transport, but would need a transaction/unique-edge guard
   if concurrent writers are ever introduced.

---

## Phase W.4 — Integration + docs + RETRO — READY FOR REVIEW (CLOSES Track W)

**Branch**: `track/w4-integration` (off `main`, has W.1 + W.2 + W.3 merged)
**Commits**:
- `2974bbd` test(mcp): W.4 e2e — search->related->get_node compose read-only, model-free
- (this section) docs(arch) §10 shared graph-memory; docs(track-w) operator runbook; docs(roadmap) Track W; docs RETRO + CLOSE

### Deliverables
1. **`ARCHITECTURE.md` §10 — shared graph memory (MCP graph-tools + reranker).**
   New section (existing §9 computed-vs-materialized untouched; old §10 Further
   reading → §11): the `surrealdb-mcp` shared substrate + the 5 graph tools and
   their repo backings (table); the embedding-source contract; the two
   deliberately-separate retrieval layers (chunk/note hybrid `search` vs the
   source-level Track R `/sources/{id}/related-hybrid`, cross-referenced to §9);
   the isolated default-off reranker service + heuristic fallback.
2. **Operator runbook** — `docs/tracks/W-mcp-graph-memory/OPERATOR_GUIDE.md`
   (mirrors the Track J `OPERATOR_GUIDE.md` location): (a) `.mcp.json` /
   `claude mcp add` shape for `surrealdb-mcp --transport stdio` + the `SURREAL_*`
   env table (`SURREAL_DATABASE=staging`); (b) the consolidated gated reranker
   bring-up (build → up → `/health` `model_loaded:true` → Dutch `/rerank` smoke →
   end-to-end `/search rerank:true` → `stop reranker` to verify the heuristic
   fallback); (c) safety notes (stdio/no-auth, cite read-then-write).
3. **Light E2E integration check** —
   `packages/surrealdb-service/tests/test_mcp_graph_tools_e2e_compose.py`
   (docker-backed, 2 tests, **green**). Composes the MCP tools MODEL-FREE:
   `search` (lexical BM25, no embedding) → `related` on the returned source node
   (mentions edge → entity) → `related` on that entity (relation edge →
   neighbour) → `get_node` on the neighbour. Every step consumes the previous
   step's id; read-only (seeds an isolated subgraph, then only SELECTs). The
   **rerank leg is explicitly NOT run** (gated operator smoke — see the runbook);
   the second test asserts the search leg takes no embedding (no `_vector_rank`
   provenance), proving it is genuinely model-free.
4. **RETRO** — below. **Track W marked CLOSED.**
5. **`docs/FEATURE_ROADMAP.md`** — Track W entry added (mirrors the Track U
   "(NEW)"/CLOSED pattern; additive).

### Acceptance criteria — evidence
1. **ARCHITECTURE §10 accurate, doesn't clobber §9, cross-refs both retrieval
   layers** — new §10 added; §9 byte-unchanged; `related`/`search` (chunk/note)
   vs Track R source-level retrieval explicitly distinguished and cross-linked.
2. **Runbook lets an operator (a) register the MCP server + (b) bring-up/smoke/
   fallback-test the reranker** — `OPERATOR_GUIDE.md` §1 (registration, env) + §2
   (gated reranker a–e). Commands consolidated from the W.2/W.3 status notes;
   app port `5055`, reranker port `8105`, both verified against `docker-compose.yml`.
3. **E2E composes search→related→get_node read-only, green; rerank leg gated** —
   `test_mcp_graph_tools_e2e_compose.py` 2 passed; rerank leg deferred to the
   runbook smoke (no 2 GB model run).
4. **RETRO written; Track W CLOSED; roadmap updated** — this section + the Track
   W roadmap entry.
5. **No regressions** — app-main search+rerank suites: 32 passed
   (`test_search_router` / `test_search_service` / `test_search_rerank_router` /
   `test_reranker_http_client`). surrealdb-service W graph suites: 26 passed
   (`test_search_hybrid_fusion` / `test_graph_load_all_edges` /
   `test_mcp_graph_tools_roundtrip` / `test_mcp_graph_tools_e2e_compose`).
   Reranker service: 4 passed. (Pre-existing top-level `tests/` import errors and
   the 3 known `docling`-missing failures are unrelated and out of scope.)

---

## Track W — RETROSPECTIVE

**What Track W set out to do**: build the *foundation* for shared graph memory —
Feature 1 (MCP graph-tools as the shared SurrealDB substrate) + Feature 5
(hybrid search 5a + a local multilingual reranker 5b). Citations / auto-link /
contradiction were explicitly deferred to later tracks.

### 1. "Most of this already existed" — the recurring finding
As with Track U (where the schema already defined the edges), the dominant W
finding was that the machinery was largely present and just **not wired**:
- The hybrid BM25⊕vector fusion chain existed end-to-end at the repo/service
  layer (`SearchRepository.hybrid_search` → `RetrievalService` → `SearchService`)
  — W.1 only had to add the router branch, the request field, and the
  `text_weight` thread-through. No new fusion logic, no `fn::` functions, no
  embedding model.
- The `surrealdb-mcp` server already existed with 5 low-level tools — W.3 added
  5 graph tools *alongside* them, not a new server.
- The repo backings for the graph tools mostly existed (`relate_cites`,
  `NoteRepository.create_with_embedding` + `add_to_notebook`, the specialized
  edge loaders) — the only genuinely missing piece was a *unified*
  `load_all_edges`.
The lesson (again): **survey before building.** The track stayed small because
each phase confirmed the existing surface first and added the thin missing seam.

### 2. The latent RRF fusion bug — caught by asking for honest findings (W.1)
The repo-layer fusion read `item.get("score", 0)`, but the FTS functions return
`relevance` (BM25) and `similarity` (cosine) — **never** `score`. So
`_combined_score` silently collapsed to 0 and the "ranked" hybrid output
degenerated to insertion order, *failing the very acceptance criterion W.1
claimed to satisfy*. It surfaced only because the phase was pushed to produce
real evidence (the staging probe showed all-`0.0000` scores). The fix replaced
the broken linear sum with Reciprocal Rank Fusion (scale-independent, k=60,
matching Track R's validated fusion) — a rank-based fuse, not a magnitude one, so
the unbounded BM25 signal can't drown the bounded cosine signal. Lesson:
**"returns fused results" is not the same as "returns *correctly* fused
results"** — demand the numbers, not the shape.

### 3. Dependency-isolation discipline — the reranker as its own service (W.2)
The reranker pulls `torch` + `sentence-transformers` (~2 GB model). Rather than
let that bleed into `app-main`, it lives in its **own container**
(`services/reranker/`), mirroring `docling`/`whisperx`; app-main reaches it over
plain HTTP (`RERANKER_SERVICE_URL`) with no ML libs, and a service outage falls
back to the zero-dep heuristic reranker (never a 500). Verified
`torch`/`sentence-transformers` are absent from app-main's dependency closure.
This kept a heavy, optional capability genuinely optional and the core app lean.
(They *are* importable from the shared uv-workspace `.venv` because sibling
pipelines pull them — that is the workspace venv, not app-main's own closure, and
not the reranker container. The distinction matters and is documented.)

### 4. The vacuous "shared-substrate" test — "demonstrated" must mean it (W.3)
The plan's AC2 was *"two separate sessions see the same nodes — demonstrated."*
The first attempt read the tool-written node back via `execute_query(...,
config=cfg)` and called it proof. It wasn't: `get_pool(config)` returns the
cached singleton pool and **ignores** its `config` arg once the pool exists, so
the "second session" was reusing the writer's own pool — the test proved
durability on one connection, not the cross-session property. The fix used
`db_connection(cfg)`, which builds its own `AsyncSurreal` socket and never
touches the global pool, and asserted explicitly that the reader connection
object is `!=` the pooled one. Lesson: **a test that would pass for the wrong
reason proves nothing** — when an AC says "demonstrate X", verify the test can
only pass *because* of X.

### Cross-cutting
The through-line of all four: **honest evidence is the deliverable.** A green
test, a "fused" result, a "demonstrated" property — each was initially vacuous
and only became real when forced to show the number / the independent
connection / the dependency closure. The track's value is as much in catching
those as in the (thin) new code.

### Status — TRACK W CLOSED
- **W.1** hybrid `/search` (RRF) — done, merged.
- **W.2** reranker microservice + `rerank` flag + heuristic fallback — done,
  merged; **the live ~2 GB model run is operator-gated** (runbook §2).
- **W.3** 5 MCP graph tools + `load_all_edges` + `relation` fix — done, merged.
- **W.4** integration check + ARCHITECTURE §10 + operator runbook + roadmap +
  this RETRO — done.

**Operator / follow-up (outside the closed track):**
1. Run the gated reranker live-model smoke (runbook §2 a–e).
2. Gate `cite`/`add_note` write-tool registration to stdio (or add auth) before
   any `sse`/`streamable-http` exposure (W.3 follow-up 1).
3. Make `cite` dedup atomic if concurrent MCP writers are ever introduced (W.3
   follow-up 2).

**Track W is CLOSED.**
