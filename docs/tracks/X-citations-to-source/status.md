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

**One rule (revised after review attempt 1): attach chunk-level provenance only when the
EXACT originating chunk is identifiable; otherwise leave the chunk keys `None` and attach the
`source` only.** The distinguisher is the hit's **own `id` prefix**, not `parent_id`
(the `fn::` id shapes — `migrations/4.surrealql`):

| hit class | own `id` | `parent_id` | chunk-backed? |
|---|---|---|---|
| source_embedding (vector/hybrid) | `source:Y` | `source:Y` | **yes** (cosine-argmax) |
| source_insight | `source_insight:X` | `source:Y` | no |
| source title/full_text (text leg) | `source:Y` | `source:Y` | no (page-less; text path excluded) |
| note | `note:N` | `note:N` | no |

`SearchRepository.hydrate_provenance(hits, embedding)`:

- **`source:`-own-id hit WITH embedding** (vector/hybrid): one batched `SELECT` over
  `source_embedding ⋈ chunk` picks the top-1 chunk per source by
  `vector::similarity::cosine`. Because `fn::vector_search` collapses with `math::max(cosine)`,
  this is **exactly** the row that produced the hit's score → `physical_page` is the page of
  the *actual chunk the hit came from*. Verified equal to the `fn::` collapsed max on staging
  to **1e-9**.
- **`source_insight:` hit**: a synthesized source-level summary with no single originating
  chunk → all chunk keys `None`, `source` set from `parent_id`. **Never** routed through the
  chunk lookup. (Blocker fix: previously `_hit_source_id` returned `parent_id`, so an insight
  got stamped with the source's top *embedding* chunk's page — a different row that did not
  produce the hit. Latent on staging only because `source_insight`=0 rows there, but
  `pipelines/embeddings/service.py` populates `source_insight.embedding` in the real pipeline.)
- **text-only path** (`embedding is None`): no chunk-level keys for ANY hit (BM25 score is not
  reproducible out of context, and a `source:` text hit may even come from the title/full_text
  leg with no chunk). `source` set, all chunk keys `None`. (The earlier "arbitrary first
  chunk `section_path`/`element_type`" was dropped per review — same principle.)
- **notes / non-source / lookup failure**: chunk keys `None`; never raises (DB failure degrades
  silently, existing keys intact).

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

---

## Phase X.2 — Cited answers in the ask + source_chat graphs (Backend / LangGraph)

**Branch**: `track/x2-cited-answers` (off `main` @ `1a2eaee`, X.1 merged)
**State**: complete — ready for review.

### Hydration opt-in (the X.1 follow-up; AC5)

X.1 left `SearchRepository.{text,vector}_search` hydrating **by default**, so the
generic `/search` (UI/MCP) path paid the extra chunk-provenance `SELECT`. X.2
flips the default **off** across all three repo entry points and threads a
`hydrate` flag through `RetrievalService.{text,vector,hybrid}_search` (also
default `False`). Only the answer-citation path opts in (`hydrate=True`):

- `ask.provide_answer` → `RetrievalService.vector_search(..., hydrate=True)`
  (text fallback also `hydrate=True`).
- `hybrid_search` now only hydrates the fused set when `hydrate=True`.
- `SearchService` (the `/search` router's delegate) passes **no** `hydrate`, so
  the hot path stays unhydrated — confirmed by `test_default_skips_hydration_on_hot_path`
  and the four `test_search_service` delegation asserts (`hydrate=False`).

### Where provenance is threaded + how citations are derived

**`ask` (cross-source, fan-out)** — `apps/app-main/src/app_main/graphs/ask.py`:
- `SubGraphState` unchanged in shape; `provide_answer` now formats the retrieved
  hits into the prompt via `_format_results_with_provenance`, prefixing each
  block with `[source: <id> | p.<page> | <section>]` (X.1 keys).
- Citation set per sub-answer = `citations_from_hits(results)` — the provenance
  of the context hits actually fed to the LLM (deterministic; X.3 adds the
  membership guard). Accumulated into `ThreadState.citation_groups`
  (`Annotated[list, operator.add]` across the `Send` fan-out).
- `write_final_answer` emits `citations = merge_citations(citation_groups)`
  (de-duplicated by `(source, page, chunk_id)`), additive to `final_answer`.

**`source_chat` (single-source)** — `graphs/source_chat.py` +
`services/context_service.py`:
- `ContextService` gains an **optional** `chunk_repo` and `build_source_chunks`,
  returning the source's content chunks (noise-filtered) with provenance
  (`chunk_id`/`physical_page`/`printed_page`/`section_path`/`element_type`/`source`),
  capped by `max_chunks`/`max_tokens`. Graceful empty without `chunk_repo` or on
  lookup failure.
- The graph node fetches chunk context, injects page-cited passages
  (`## SOURCE PASSAGES`, each tagged `[source | p.X | section]`) ahead of the
  source-level `full_text` fallback, and emits `citations` from the chunk
  context. No chunks (audio/plain text) → a single source-level citation
  (`page=None`).

**Citations helper** — new `graphs/citations.py`
(`hit_to_citation`/`citations_from_hits`/`format_citation_tag`/`merge_citations`),
the one place the `{source, page, chunk_id, section}` shape and the
provenance-tag string are defined. Shared by both graphs.

### Prompt-template changes (additive attribution)

- `prompts/ask/query_process.jinja` — explains the provenance tag on each result
  and asks the model to cite `[document_id, p.<page>]` when a page exists, only
  using pages that appear in a tag (no invented pages); page-less results cite
  `[document_id]` as before.
- `prompts/source_chat.jinja` — same page-grounded instruction for the
  `## SOURCE PASSAGES` block.
- Both are purely additive; the existing `[document_id]` citation format and
  answer shape are preserved.

### API surface (additive)

- `AskResponse` gains `citations: list[dict] = []`; `/search/ask/simple` populates
  it and the SSE `final_answer` event now carries `citations`.
- `source_chat` SSE stream emits an additive `citations` event.

### Test evidence (per acceptance criterion)

- **AC1** (ask citation page == chunk page, real source id): `test_ask_graph_citations.py`
  `TestProvideAnswerCitations::test_citations_match_chunk_provenance` — seeded
  multi-page source (`source:doc1` p.3 / `source:doc2` p.12); asserts citation
  page == hit `physical_page`, `chunk_id`, `section`, and `hydrate=True` opt-in.
  `test_provenance_tags_injected_into_prompt` proves the tags reach the prompt.
  `TestWriteFinalAnswerCitations::test_merges_citation_groups` — merged/dedup
  final citations.
- **AC2** (source_chat chunk-level citations): `test_source_chat_citations.py`
  `test_chunk_level_citations_emitted` (page+section per chunk, not just title)
  + `test_page_tags_injected_into_prompt`.
- **AC3** (existing behavior intact; no 500 on no-page): ask
  `test_no_page_source_graceful` + `test_empty_results_emit_empty_citations`;
  source_chat `test_no_chunks_falls_back_to_source_level_citation` (page=None,
  full_text fallback present) + `test_answer_text_still_produced`. Graph/context
  suites green.
- **AC4** (seeded-source assertions): the seeded `SEEDED_HITS`/`CHUNK_CONTEXT`
  fixtures above; citations asserted equal to the chunks' provenance.
- **AC5** (generic `/search` skips hydration): `test_search_provenance.py`
  `test_default_skips_hydration_on_hot_path` + the `test_search_service`
  delegation asserts (`hydrate=False`); `pipelines/retrieval/test_service.py`
  `test_hydrate_opt_in_forwarded` confirms the flag forwards only on opt-in.
- **Helper units**: `test_graph_citations.py` (16) — dedup, page-0 inclusion,
  page-less, uncitable, merge.
- **Staging probe** (read-only, opt-in path): `test_search_provenance_staging.py`
  now passes `hydrate=True` → **2 passed** on `staging` (real `physical_page`/
  `section_path` on `source:dndibxmjveoxk7tfqfsl`).

Suites run green: surrealdb-service non-docker **97 passed, 2 skipped**;
retrieval **27 passed**; app-main citations/ask/source_chat/context/search
**65 passed**. (Pre-existing & ignored: top-level `tests/` import errors and the
3 `test_source_processing_service` docling `to_docling_options` failures — both
predate X.2 and are untouched.)

### Test-loading note (env gap)

`ai_prompter` is a runtime dep absent from the test venv, and `ask.py`/
`source_chat.py` import it (and `source_chat` compiles a checkpointer-bound graph
at import — a pre-existing `SqliteSaver.from_conn_string` type mismatch on
`main`). The graph tests load the node modules via `importlib.util.spec_from_file_location`
with `ai_prompter` (and `SqliteSaver`) stubbed in `sys.modules`, bypassing the
package `__init__` — same discipline as `test_chat_routing_telemetry`.

### Files

- `packages/surrealdb-service/src/.../repositories/search.py` — hydrate default → off.
- `pipelines/retrieval/src/retrieval/service.py` — `hydrate` kwarg threaded.
- `apps/app-main/src/app_main/graphs/citations.py` — NEW citation helpers.
- `apps/app-main/src/app_main/graphs/ask.py` — provenance in state/prompt + citations.
- `apps/app-main/src/app_main/graphs/source_chat.py` — chunk context + citations.
- `apps/app-main/src/app_main/services/context_service.py` — `build_source_chunks`.
- `apps/app-main/src/app_main/dependencies.py` — `chunk_repo` into `ContextService`.
- `apps/app-main/src/app_main/api/{schemas.py,routers/search.py,routers/source_chat.py}` — additive citations surface.
- `prompts/ask/query_process.jinja`, `prompts/source_chat.jinja` — additive attribution.
- Tests: `test_graph_citations.py`, `test_ask_graph_citations.py`,
  `test_source_chat_citations.py` (new); `test_context_service.py`,
  `test_search_provenance.py`, `pipelines/retrieval/tests/test_service.py`,
  `apps/app-main/tests/test_search_service.py`, `test_search_provenance_staging.py` (updated).

### Commits (on `track/x2-cited-answers`)

- `f751676` refactor(search): make provenance hydration opt-in (X.2)
- `ea73f7f` feat(ask): cite exact source/page/chunk in answers (X.2)
- `8de9003` feat(source-chat): chunk-level page citations in single-source chat (X.2)
- `a819daf` test(search): opt into hydration in the staging probe (X.2)

### Notes for X.3 / memory

- Citations are derived from the **context hits fed to the LLM** (deterministic),
  not parsed back out of the model's prose. X.3's faithfulness guard should
  membership-check the *model-emitted* `[id, p.X]` references against this
  context-hit set (and/or against the retrieval set) — the emitted `chunk_id`
  is present only for vector/hybrid hits.
- `source_chat`'s `SqliteSaver.from_conn_string` → `compile(checkpointer=...)`
  type mismatch is a **pre-existing** latent issue (present on `main`); it does
  not affect X.2 but is worth a separate fix.
