# Track W — Shared graph memory (MCP graph-tools + hybrid search + local reranker)

> **DRAFT for user review — nothing built yet.** Derived from the Constella feature-adoption
> research (`docs/constella-features-adoption-research.md`). This track builds the **foundation**:
> Feature 1 (MCP graph-tools = shared memory substrate) + Feature 5 (hybrid search 5a + local
> reranker 5b). Feature 1's `search` tool leans on the hybrid search, so they ship together.
> Features 4 (citations), 2 (auto-link), 3 (contradiction) are SEPARATE later tracks.

## Decisions (locked 2026-06-28)
1. **Reranker** → **local cross-encoder `bge-reranker-v2-m3`** (multilingual — load-bearing, the corpus is
   Dutch; `mxbai-rerank` is English-leaning), as an **own microservice** `services/reranker/` (FastAPI),
   in `docker-compose.yml` next to `docling`/`whisperx`; `app-main` calls it via `RERANKER_SERVICE_URL`
   (mirrors `DOCLING_SERVICE_URL`). Keeps `torch`/`sentence-transformers` OUT of `app-main`. The existing
   heuristic `Reranker` (`pipelines/retrieval/.../reranker.py`) stays as a zero-dep fallback (service-down).
2. **MCP** → extend the EXISTING `surrealdb-mcp` server (`packages/surrealdb-service/.../mcp/server.py`,
   FastMCP, stdio); add 5 graph tools. No new MCP server.
3. **SurrealDB only**; reuse the existing search/repo/embedding pipeline; embeddings stay local Ollama 1024-dim.

## Existing pieces this track wires (verified)
- Hybrid search EXISTS but is not router-exposed: `SearchRepository.hybrid_search` (`…/repositories/search.py`)
  ↔ `RetrievalService.hybrid_search` (`pipelines/retrieval/.../service.py`); router `/search`
  (`apps/app-main/.../api/routers/search.py`) only does `text|vector`. FTS = `fn::text_search`/`fn::vector_search`
  (`migrations/1.surrealql`,`4.surrealql`).
- MCP server EXISTS (`surrealdb-mcp`): tools `query_database`, `get_record`, `list_sources`, `search_similar`,
  `get_entity_graph` (NOTE: queries a stale `relates_to` edge; real edge is `relation`).
- Repo backings for the 5 graph tools mostly EXIST: `relate_cites` (cite), `NoteRepository.create_with_embedding`
  + `add_to_notebook` (add_note), `load_mentions_edges`/`load_cites_edges`/`find_related_by_embedding`/
  `get_entity_detail` (related — 4 specialized; a unified one is missing).

**Workflow**: track methodology — `implementer` → `adversarial-reviewer` (≤3 → `escalation-handler`).
Main tree, `uv run pytest`, no worktree. Live writes gated. Per-phase branches `track/w<n>-…`.

---

## Phase W.1 — Expose hybrid search via the router (Feature 5a) (Backend)
**Why**: hybrid (BM25 + vector) fusion already exists at the repo layer but isn't reachable via the API;
the MCP `search` tool (W.3) and any client needs it.
**Deliverables**: extend `/search` to support `mode=hybrid` (calls `SearchService.hybrid_search` →
`RetrievalService.hybrid_search` → `SearchRepository.hybrid_search`), threading the query embedding through;
keep `text`/`vector` modes byte-identical (backward-compatible). Expose `text_weight`/`minimum_score` params.
**Acceptance**
1. `GET/POST /search` with `mode=hybrid` returns BM25+vector-fused chunk/note results (dedup, weighted), ranked.
2. Existing `mode=text`/`mode=vector` output unchanged (backward-compat; regression suite green).
3. Embedding resolved via the local model (no hardcoded dim); empty/no-result handled gracefully.
4. Unit + router tests; a read-only staging probe on a known query.
**Branch**: `track/w1-hybrid-endpoint`. **Depends on**: none.

## Phase W.2 — Local cross-encoder reranker service (Feature 5b) (Backend / Infra)
**Why**: rerank the top-N hybrid hits with a quality-appropriate, local, multilingual cross-encoder.
**Deliverables**
- New microservice `services/reranker/` — small FastAPI loading `bge-reranker-v2-m3` (sentence-transformers
  `CrossEncoder`), endpoint `POST /rerank` `{query, passages[]} -> [{index, score}]`; a `services/reranker/Dockerfile`.
- `docker-compose.yml` entry `reranker` (own container, like `docling`/`whisperx`); env `RERANKER_SERVICE_URL` on `app-main`.
- `app-main` reranker client + wire it into the hybrid path behind a `rerank=true` flag on `/search`; on
  service-unavailable, fall back to the existing heuristic `Reranker` (no hard failure).
**Acceptance**
1. `rerank=true` reorders the top-N hybrid results; a Dutch-query sanity check shows sensible reordering.
2. `rerank=false` (default) path is byte-identical to W.1 (no reranker call).
3. Reranker-service-down → graceful fallback to heuristic `Reranker`, logged, never a 500.
4. `torch`/`sentence-transformers` are ONLY in `services/reranker/`, not in `app-main` deps (verify).
5. Service `/rerank` unit test + an `app-main` integration test with the service mocked + fallback test.
**Branch**: `track/w2-reranker-service`. **Depends on**: W.1.

## Phase W.3 — MCP graph-tools (Feature 1) (Backend)
**Why**: expose the graph as tools so every Claude Code session shares the same SurrealDB memory substrate.
**Deliverables**
- A unified repo method `load_all_edges(node_id, edge_types?)` (relation + mentions + cites) returning a
  common `{id, source, target, edge_type, metadata}` shape (the missing piece for `related`).
- Fix `get_entity_graph` (`relates_to` → `relation`).
- Add 5 FastMCP tools to `surrealdb_service/mcp/server.py`: `search` (→ hybrid search from W.1), `get_node`
  (→ `SELECT * FROM type::thing($id)`, polymorphic), `related` (→ `load_all_edges`), `cite` (→
  `SourceRepository.relate_cites`), `add_note` (→ `NoteRepository.create_with_embedding` + `add_to_notebook`).
**Acceptance**
1. All 5 tools callable over stdio (`surrealdb-mcp --transport stdio`); return JSON.
2. Two separate sessions hitting the same SurrealDB see the same nodes/edges (shared substrate) — demonstrated.
3. `related` returns relation+mentions+cites edges for a node; `cite`/`add_note` write correctly + idempotently;
   no fabricated edges; `get_entity_graph` now reads `relation`.
4. Tool tests (MCP client or direct-call); canonical data only written via the intended tools.
**Branch**: `track/w3-mcp-graph-tools`. **Depends on**: W.1 (the `search` tool).

## Phase W.4 — Integration + docs + RETRO (Integration)
**Deliverables**: end-to-end check (MCP `search` → hybrid+rerank → `related`/`cite`/`add_note`); `ARCHITECTURE.md`
section on the MCP graph-memory + the reranker service; operator note on running `surrealdb-mcp` + the reranker
container; RETRO. **Acceptance**: E2E green; docs updated; Track W CLOSED. **Depends on**: W.1–W.3.

---

## Risks & open decisions for the user
1. **New heavy dep (torch/sentence-transformers)** → isolated in `services/reranker/` (mirrors `whisperx`); risks:
   model download size + CPU latency. Mitigation: top-N only, a small/quantized model, GPU if available.
2. **MCP has no auth** — fine for stdio-local. If you ever expose it over HTTP for remote sessions, add auth.
3. **`get_entity_graph` staleness** (`relates_to`) — fixed in W.3.
4. **`search` tool scope** — W.1/W.3 search is the chunk/note hybrid search. The source-level Track R retrieval
   (`/sources/{id}/related-hybrid`) is a different question; decide whether the MCP `search`/`related` should also
   surface source-level relatedness, or keep them separate (recommend: keep separate; `related` = graph edges).
5. **Reranker model size** — `bge-reranker-v2-m3` (~568M, multilingual, best NL quality) vs a smaller/faster
   variant. Decide at W.2 based on latency on your hardware.

## Verification (end-to-end)
- W.1: `POST /search {mode:"hybrid"}` on a known Dutch query → fused ranked hits; `mode:text|vector` unchanged.
- W.2: same query with `rerank=true` → reordered top-N; stop the reranker container → graceful heuristic fallback.
- W.3: run `surrealdb-mcp --transport stdio`, attach as an MCP server in a Claude Code session; call all 5 tools;
  open a second session and confirm it sees the same nodes (shared substrate).
- Reuse `@requires_docker` roundtrips + `uv run --project <pkg> pytest`; live probes against `staging` with
  explicit `SURREAL_DATABASE=staging`.
