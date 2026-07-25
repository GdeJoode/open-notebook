# Architecture

Single source of truth for the **internal structure** of open-notebook. Cross-references implementation files instead of restating them; favors recency over completeness.

## 1. Four-layer monorepo

Code is organized in four UV workspace tiers. Lower layers do not import from higher layers.

```
frontend/                       — React + TypeScript UI (31k LOC, Vite-based)

apps/                           — Deployable end-user applications
└── app-main                    — FastAPI backend + orchestration (~18k LOC)

packages/                       — Reusable domain libraries (importable by apps & pipelines)
├── shared                      — Domain models, type aliases, config primitives
├── surrealdb-service           — SurrealDB connection, repositories, migrations
├── llm-manager                 — LLM provider routing + usage tracking
├── ontology-manager            — Ontology schema, versioning, evolution
├── file-manager                — File / knowledge-base management
├── job-queue                   — Async background job execution
├── zotero-integration          — Zotero bidirectional sync
└── semantic-intelligence       — Graph algorithms, reasoning, decision tracking
                                  (config also exposes shared SurrealDB/Ollama plumbing)

pipelines/                      — RAG/processing workflows (composed from packages)
├── ingestion                   — Docling/WhisperX → chunks (routes to GPU service)
├── embeddings                  — Vector generation + persistence
├── retrieval                   — Vector + text + hybrid search; reranker
├── entity-filtering            — Entity/relation filtering, dedup, scoring,
                                  canonical-entity dedup (Phase 4 addition)
├── ontology-extraction         — Ontology-guided extraction via LLM
└── summarization               — RAPTOR / TreeKG / Naive / Refine / MapReduce
                                  (+ 6 documented stub strategies — see
                                  docs/SUMMARIZATION_APPROACHES.md)

services/                       — HTTP services (GPU-heavy, separate processes)
├── docling                     — GPU-accelerated document parser
├── extraction                  — Standalone extraction API
├── summarization               — Standalone summarization API
└── whisperx                    — Speech-to-text
```

Wiring: every workspace member declares its deps in its own `pyproject.toml`;
the root `pyproject.toml` lists `[tool.uv.workspace]` members and
`[tool.uv.sources]` pins.

## 2. Core abstractions

Per the most recent graphify analysis (commit `a2130b40`), the top hubs by
edge count are the genuine domain primitives — touching many modules because
they *are* the domain model:

| Symbol | Edges | Role | Lives in |
|---|---|---|---|
| `Ontology` | 144 | Schema definitions for entity/relation types | `packages/ontology-manager` |
| `ExtractionResult` | 140 | Output dataclass from extraction pipelines | `apps/app-main/.../services/source_extractor.py` |
| `NotebookService` | 120 | Notebook CRUD + counts orchestration | `apps/app-main/.../services/notebook_service.py` |
| `SourceService` | 117 | Source CRUD | `apps/app-main/.../services/source_service.py` |
| `execute_query()` | 108 | Canonical SurrealQL executor | `packages/surrealdb-service/.../connection.py` |
| `ChunkInput` | 106 | Summarization-pipeline input contract | `pipelines/summarization/.../models/result.py` |
| `KGResolver` | 82 | Entity resolution against the KG | `pipelines/entity-filtering/.../resolution` |

High fan-in here is *healthy* — these types are deliberately central. The
Phase 3 god-split addressed the *other* kind of god (high internal complexity,
not fan-in): `SourceProcessingService` (903 LOC) became four single-purpose
classes (extractor + processor + two orchestrators).

## 3. Dependency injection

`apps/app-main/src/app_main/dependencies.py` is the composition root.
Every service has a `get_<service>()` factory function that wires its
repositories and downstream services. Routers receive them through
FastAPI `Depends(...)`.

Naming convention inside `apps/app-main/.../services/`:

- **`*Service`** — owns a single resource and its CRUD/business logic
  (`NotebookService`, `SourceService`, `ChatService`).
- **`*Orchestrator`** — coordinates one externally-triggered pipeline run
  (`SourceEmbeddingOrchestrator`, `SourceSummarizationOrchestrator`).
- **`*Processor`** — coordinates an end-to-end resource lifecycle
  (`SourceProcessor`: extract → update record → persist chunks).
- **`*Extractor` / `*Builder`** — stateless content transformation
  (`SourceExtractor`, `chunk_builder` module, `IngestionConfigBuilder`).
- **`*Repository`** — single-table SurrealDB persistence, no business logic.

The `Service` vs `Orchestrator` vs `Processor` split is the outcome of the
Phase 3 refactor and the pattern is now expected for every domain in this
codebase.

## 4. Where work happens

| Concern | Primary location |
|---|---|
| HTTP API routes | `apps/app-main/.../api/routers/` |
| Job handlers (async) | `apps/app-main/.../handlers.py` |
| Business orchestration | `apps/app-main/.../services/` |
| Domain models | `packages/shared/.../models/` |
| DB queries | `packages/surrealdb-service/.../repositories/` |
| LangGraph workflows | `apps/app-main/.../graphs/` |
| RAG composition | `pipelines/*/src/*/` |
| GPU-bound work | `services/docling`, `services/whisperx` |
| UI | `frontend/src/` |

## 5. Tests

Tests live next to the code they test, in each workspace member's `tests/`
directory. Run from repo root via `uv run --project <member> pytest <member>/tests/`.

App-main is the test-coverage hot spot (27 testfiles, 235 tests after Phase 5)
because it owns the bulk of the orchestration. Pipelines have lower but adequate
coverage; the under-covered `pipelines/retrieval` got attention in Phase 5.

## 6. Storage layer additions (Track B — KG quality)

Track B (closed 2026-06-12) extended the SurrealDB layer with four new tables
and three new fields on `entity`. Migration files live in `migrations/`:

| Table / Field | Migration | Purpose | Owner |
|---|---|---|---|
| `entity.type_tags` / `entity.primary_type` | 44 | Multi-schema type-tagging on existing entities; `type_tags` accumulates via union, `primary_type` set from the highest-confidence pass | B.1a |
| `notebook_schema` | 45 | Per-notebook ontology-evolution state (base ontology, accepted/pending extensions, soft-delete `excluded_types`, review-required toggle) | B.1b |
| `pass1_results` | 45 | Append-only first-pass LLM schema-validation output per source (coverage_pct, alternative_schemas, proposed_extensions) | B.1b |
| `notebook_event` | 46 | Shared append-only domain-event log per notebook (`schema_changed`, `extension_suggested`, `schema_mismatch`) consumed by the soft-nudge banner and Track G5 webhooks | B.3b |
| `metrics` | 47 | Always-on extraction telemetry (composite index on `(event_type, created_at)`, FLEXIBLE payload, env-toggle opt-out) | B.4 |
| `entity.orphan_status` / `reconnect_attempts` / `first_orphaned_at` / `last_reconnect_attempt_at` | 48 | Prune-lifecycle metadata for B.5a orphan-connector retries; archived rows are kept (recoverable) | B.5b |

All Track-B migrations are idempotent (`IF NOT EXISTS` everywhere) and have
matching `_down.surrealql` counterparts.

### Track-B service modules

The B-track services follow the section-3 naming convention. They live in
`apps/app-main/src/app_main/services/` unless noted otherwise:

| Module | Role | Phase |
|---|---|---|
| `EntityExtractionService` (rewired) | Branches on `(multi_schema_enabled, notebook_id)`; dispatches to single-schema or multi-schema path; raises `SchemaReviewPendingError` when the notebook gate is active | B.1f |
| `SchemaEditService` | Pure business logic for accept/reject extension, rename, merge, split, delete type; idempotent via deterministic op-ids; emits exactly one `notebook_event` per op | B.3b |
| `ReextractService` | Enqueues `ENTITY_EXTRACT` jobs after schema edits, with paused-job dedup | B.3d |
| `NotebookMergeService` | Cross-notebook graph merge (semantic-content idempotency, archived-source guard) | B.6 |
| `shared.services.metrics.record_metric` (in `packages/shared/`) | Always-on `INSERT INTO metrics` helper (env-flag `OPEN_NOTEBOOK_DISABLE_METRICS`, exception-swallow contract) | B.4 |
| `entity_filtering.resolution.orphan_connector` (in `pipelines/entity-filtering/`) | Co-occurrence-based orphan reconnection with LLM-confirm; prune-lifecycle update writes `orphan_status` transitions | B.5a / B.5b |

### Extraction-pipeline subsection

The extraction pipeline (`pipelines/ontology-extraction/`) gained a
multi-schema mode. Dispatcher pattern:

```
EntityExtractionService.run_extraction(notebook_id, multi_schema_enabled=True)
        │
        ▼
ExtractionWorkflow.extract(mode="multi"|"single")
        │
        ├─ mode="single" → existing single-schema Pass-2 path (default off-notebook)
        │
        └─ mode="multi"  → multi_schema_orchestrator.run_multi_schema(...)
                 1. detect_applicable_schemas(...) → top-3 ontologies, conf ≥ 0.3
                 2. Sequential Pass-1 per schema → pass1_results rows
                 3. Cumulative SoftNudgeDecision (NONE / EXTENSION_SUGGESTED / SCHEMA_MISMATCH)
                 4. Sequential Pass-2 per schema with accepted extensions
                 5. In-process merge: entity dedup via normalize_entity_name,
                    type_tags union, primary_type ← highest-confidence pass,
                    relation dedup with max-confidence wins
```

Confidence is populated on every entity AND every relation (B.4
invariant); the merge step uses confidence-max semantics so the highest-
confidence pass wins. Multi-schema is **on by default** when `notebook_id`
is provided; flip `multi_schema_enabled=false` per-request to fall back.

### Model-aware context packing (Track M)

The extraction failover chain is **heterogeneous** — each candidate model has a
different context window (Gemini ~1M, Ollama Cloud ~32K, llama3.1:8b ~8K). Rather
than hand the model the fixed ~2000-char ingestion chunks one call at a time,
`apps/app-main/src/app_main/services/extraction_chunking/context_packer.py`
(`pack_chunks_for_model`) RE-PACKS the persisted ingestion chunks into windows
sized to the ACTIVE candidate's `context_window`:

```
input_budget = (context_window − max_output_tokens − prompt_overhead) × 0.85
window_budget = min(input_budget, EXTRACTION_MAX_WINDOW_TOKENS)   # M.3 cap, default 6000
```

- **M.3** derives the window from the model's context (a big-context model packs
  a document into a few calls instead of ~28) and adds a tunable window-size CAP
  (`EXTRACTION_MAX_WINDOW_TOKENS`, default 6000) — measured to preserve exhaustive
  recall (one full-context call collapses recall to a few themes) while still
  cutting call count.
- **M.4** re-splits any single chunk that alone exceeds the budget
  (`_split_oversized_text`) so no window ever overflows the smallest candidate —
  the no-overflow invariant. Provenance becomes a `constituent_chunk_ids`
  window-of-chunks list (Decision M-D4), not a single id.

The M.5 regression gate (`test_heterogeneous_chain_extraction.py` +
`chunking_metrics.py`) packs the same document for the whole chain and asserts
`overflow_count == 0` per candidate plus `est_calls` shrinking as context grows.
The full per-document failover re-architecture (packing folded INTO the failover
attempt, M-D3 (a)) is a deferred follow-up; the shipped path packs per-candidate
up-front with the M.4 oversized guard.

## 7. Knowledge graph export surfaces (Track D — output richness)

Track D (closed 2026-06-16) added three HTTP export surfaces over the
notebook-scoped KG projections from D.0. All three share a single
filter pipeline so the "what you'll export" preview the UI shows
matches the actual export byte-for-byte.

### Endpoints + services

| Endpoint | Service module | Metric event | Notes |
|---|---|---|---|
| `POST /api/notebooks/{id}/export-obsidian` | `apps/app-main/src/app_main/services/obsidian_export_service.py` | `export.obsidian` | Two modes: `mode="zip"` streams an in-memory zip (one `.md` per entity + `README.md`); `mode="vault_path"` writes the same files directly to `Settings.vault_path / Settings.vault_entities_folder` via per-file `tempfile + os.replace` (POSIX atomic rename). The vault-path branch is the sole async export surface, dispatched through `JobType.EXPORT_OBSIDIAN` in `apps/app-main/src/app_main/handlers.py` (Q-D-2). |
| `POST /api/notebooks/{id}/export-jsonl` | `apps/app-main/src/app_main/services/jsonl_export_service.py` | `export.jsonl` | Streaming zip of `entities.jsonl` + `relations.jsonl`. Build-then-stream (Q-D-7): `model_dump(mode="json", exclude={"embedding"})` per row, written into the open ZIP member then yielded in 16KB chunks. Per-line keys are Neo4j-`apoc.load.json`-compatible (`source_entity`/`target_entity` not `in`/`out`). **Track U.5**: with `filter.include_document_layer=True`, adds `sources.jsonl` + `mentions.jsonl` + `cites.jsonl` (the document graph; `cites.jsonl` present-but-empty until Track V). |
| `POST /api/notebooks/{id}/export-networkx` | `apps/app-main/src/app_main/services/networkx_export_service.py` | `export.networkx` | Builds a `networkx.DiGraph` then serialises to one of 7 formats: **GraphML**, **GEXF**, **GML**, **JSON-tree**, **edge-list**, **adjacency-list**, **pickle**. Attribute flattening contract: `type_tags` → CSV string, `properties` → JSON-encoded string; round-trip tests confirm flatten/unflatten preserves data (Risk 5). **Track U.5**: with `filter.include_document_layer=True`, adds `source` nodes + `mentions`/`cites` edges, tagged `node_kind`/`edge_kind` so the two layers stay separable. See §9. |
| `GET /api/notebooks/{id}/export-preview?filter=…` | inlined `_export_preview` fn in `apps/app-main/src/app_main/api/routers/exports.py` | (no metric — read-only) | Counts-only surface used by the Obsidian dialog + JSONL popover before the user submits an export. Applies the **same** filter pipeline (see below) so dialog counts and actual export counts cannot drift. |

All routers live in `apps/app-main/src/app_main/api/routers/exports.py`;
DI wiring in `apps/app-main/src/app_main/dependencies.py`
(`get_obsidian_export_service`, `get_jsonl_export_service`,
`get_networkx_export_service`).

### Shared filter pipeline

The three exporters + the preview endpoint apply the same four-stage
pipeline in this exact order:

1. **SurrealQL gate** in
   `EntityRepository.list_entities_for_notebook` /
   `list_relations_for_notebook` (D.0) — applies `min_confidence`,
   `entity_types`, `include_orphans` at the DB layer.
2. **Status post-filter** —
   `EXCLUDED_ENTITY_STATUSES = frozenset({"archived", "merged"})`
   defined in `obsidian_export_service.py`; imported (not duplicated)
   by `jsonl_export_service.py`, mirrored locally in
   `networkx_export_service.py` + `exports.py`. Drops tombstones the
   SurrealQL gate currently doesn't filter on `status`.
3. **`_apply_min_connections_filter`** — static method on
   `ObsidianExportService`; computes per-entity degree on the
   status-filtered relations then drops entities below threshold.
   Re-used (not re-implemented) by JSONL + preview so future tuning
   lands on both paths simultaneously.
4. **Q-D-4 endpoint intersection** — relations whose source or target
   didn't survive steps 1–3 are silently dropped (never emitted as
   broken wikilinks / dangling JSONL relation lines / broken NetworkX
   edges).

Parity is the load-bearing UX invariant: the Obsidian dialog and
JSONL popover both display "Will export: E entities, R relations"
before submit, and the live export must produce exactly E `.md` files
/ E lines in `entities.jsonl` / E nodes in the NetworkX file. This is
why the preview endpoint shares the pipeline rather than approximating
it.

### Shared models + filter

Pydantic models live in `packages/shared/src/shared/models/export.py`:
`ExportFilter` (the slider bundle), `ObsidianExportRequest`,
`JsonlExportRequest`, `NetworkxExportRequest`, `NetworkxFormat`
(`Literal` of the 7 names), `ExportReport`, `ExportPreviewCounts`.
TypeScript mirrors in `frontend/src/lib/types/exports.ts`.

`shared.utils.external_ids.resolve_external_ids` ships as a V1 stub
that returns `[]`; the Obsidian frontmatter therefore carries
`external_ids: []` until Track M4 (Q9) lands TOOI + Crossref
resolution. Swap is a single-file change with no caller migration
(matches the B.1c `name_normalizer` stub pattern).

### Telemetry

The three `export.*` events join the B.4 `metrics` table. Payloads
are **counts only** (Q-D-8): no entity IDs, no relation IDs, no raw
filesystem paths. The Obsidian vault-path branch carries
`mode: "vault_path"` + `vault_path_redacted: True` in the payload —
the raw vault path NEVER lands in `metrics`. Recursive-walk
assertions in the test suite confirm this for all three services.

### Open Knowledge Format interchange (Track OKF)

Track OKF (closed 2026-07-24) adds a fourth, **vendor-neutral** export
target — Google Cloud's **Open Knowledge Format** (OKF v0.1) — plus its
inverse import. OKF is an interchange adapter at the *edges* of the
system: it makes a curated notebook projection portable to arbitrary
OKF-aware agents without coupling them to SurrealDB or the MCP tools. It
complements, never replaces, the graph/search/provenance substrate.

The track is **mostly a conformance layer** over the Track D projection:
it reuses the same `ExportFilter`, the same shared entity/relation/note
projection, and the same `ExportReport` accounting; the genuinely new
work is the SPEC-conformant frontmatter/link mapping and the bundle
tree + manifest.

| Surface | Location | Notes |
| --- | --- | --- |
| `POST /api/notebooks/{id}/export/okf` | `apps/app-main/src/app_main/services/okf_export_service.py` | Maps the shared projection → an OKF v0.1 Knowledge Bundle (one Markdown concept per entity + `index.md`, standard-markdown links between concepts). Normal notebooks build + zip inline and stream the archive; notebooks over `OKF_ASYNC_ENTITY_THRESHOLD` defer to the job queue (`JobType.EXPORT_OKF`) and return `202` with a pollable `job_id`. The `ExportReport` (incl. the omitted-field ledger) is surfaced in the **`X-OKF-Export-Report`** response header so the zip body stays a clean archive. Byte-stable output (sorted keys, caller-injected timestamp). |
| `POST /api/notebooks/{id}/import/okf` | `apps/app-main/src/app_main/services/okf_import_service.py` | Multipart-uploaded zip → entities/relations/notes/sources, upserted via deterministic `(name, type)` dedup + the injection-safe RELATE primitive, all provenance-tagged `okf-import` and idempotent under re-import. A malformed bundle *container* is a `422`; a malformed *concept* inside a valid bundle is skipped non-silently and reported in the `200` body — never a half-written graph. `apply_dedup` runs the K.5 propose + K.3 auto-merge pass so an imported entity is matched to the existing graph. |
| MCP `export_okf` / `import_okf` | `packages/surrealdb-service/src/surrealdb_service/mcp/server.py` | The agent-facing surface (Track OKF.4), following the Track W tool conventions. These are the **one deliberate exception** to surrealdb-service's repo-direct, no-app-main rule: the OKF services are app-main orchestrators, so the tools reach them via a lazy import inside the tool body (`get_okf_export_service` / `get_okf_import_service`) and degrade to a clean `import_error` when app-main is absent, keeping module import app-main-free. |
| OKF export dialog | `frontend/src/components/notebooks/exports/OkfExportDialog.tsx` + `frontend/src/lib/hooks/use-okf-export.ts` | "Export OKF" in the notebook header opens a zip-only dialog mirroring the Obsidian dialog (shared `ExportFilter` knobs + the live `/export-preview` parity widget). It states the lossy-by-design loss up-front and, after export, itemises the omitted-field ledger parsed from the `X-OKF-Export-Report` header. |

**Lossy-by-design ledger.** Embeddings, chunk-level provenance
(Track X), verdict/contradiction edges (Track Z), and hybrid-search
signal (Track W) have no OKF representation and are intentionally
dropped. `OKF_OMITTED_FIELDS` in `okf_export_service.py` is the single
source of truth: every dropped substrate field maps to a human-readable
reason, is copied verbatim into each `ExportReport.metadata.omitted_fields`,
and is surfaced through both the REST header and the UI so the loss is
explicit, never silent (OKF-D4: drop-and-report, not `x-` extensions).

The OKF export reuses `ObsidianExportRequest` for its filter contract
(it only ever emits a zip, so `mode` is fixed to `"zip"`), and shares
the Track D four-stage filter pipeline above — dialog counts and the
actual bundle cannot drift.

## 8. Cloud/local model routing (Track J)

The three LLM pipeline stages — **entity extraction**, **summarization**,
**chat** — route through a privacy-aware ordered provider chain with
per-document failover and a local last-resort fallback. The service lives in
`apps/app-main/src/app_main/services/model_routing/`.

### The three-stage routing path

```
                       resolve_privacy_mode(source.private, notebook.privacy_mode, global)
                                            │  (document > notebook > global, sticky-private)
                                            ▼
        LLMTask {ENTITY_EXTRACTION │ SUMMARIZATION │ CHAT}  +  PrivacyMode {CLOUD │ PRIVATE}
                                            │
                                            ▼
                         RouteResolver.resolve(task, mode)  →  ResolvedRoute
                          (model_route row, else hard-coded default chain;
                           drops keyless providers; CLOUD appends local tail;
                           PRIVATE = local-only)
                                            │
                          ┌─────────────────┴──────────────────┐
            extraction / summarization                       chat
                          │                                    │
        FailoverExecutor.execute_with_failover         provision_langchain_model
        (circuit-breaker skip, rate-limit + backoff,   (construction-time failover
         is_failover_eligible whitelist, advance        only — J-D4; no mid-stream
         on transient error → next candidate →          re-issue)
         local last resort)                                    │
                          │                                    │
                          └──────────────► record_routing_event ◄──────────────┘
                                     (routing.served telemetry: served_provider,
                                      was_failover, fallback_from, task, source, notebook)
```

- **Resolver** (`route_resolver.py`): generalizes B.8's `resolve_default_model_id`
  from one model id to an ordered `ResolvedRoute`. `LLMTask` has **exactly three
  members** — there is deliberately no `EMBEDDING`/`PARSING` (see the invariant
  below). `resolve_default_model_id` is preserved as a thin shim over
  `ordered_candidates[0]` so the B.8 contract stays intact.
- **Failover executor** (`failover_executor.py`) + **circuit breaker**
  (`circuit_breaker.py`) + **rate limiter** (`rate_limiter.py`): execute a route
  with per-document failover. A transient provider error (whitelisted by
  `error_mapping.is_failover_eligible` — 429/5xx/timeout/connect/401/403) trips
  the breaker and advances to the next candidate; a non-eligible error
  (programming bug, 400) propagates rather than masking it. The cloud rate
  limiter (conservative default, `RateLimitTimeout`-on-saturation → failover to
  the throttle-exempt local without a breaker penalty) is the fair-use guard.
  Breaker/limiter state is **in-process** (single-worker V1, J-D3).
- **Privacy resolver** (`privacy_resolver.py`): document `private` → notebook
  `privacy_mode` → global `default_privacy_mode` (default `cloud`), most-specific
  wins, with a single early-return **sticky** rule so a `private` document can
  never resolve to CLOUD.
- **Telemetry** (`telemetry.py`): one `routing.served` `metrics` row per routed
  stage carrying the served provider/model + failover trail. The
  `GET /api/model-routes/health` + `GET /api/model-routes/summary` endpoints read
  it back for the UI's per-provider health chips and the per-notebook
  cloud-vs-local routing summary.

### Invariant — embeddings + parsing stay LOCAL and FIXED (§1.2)

> **Blocking, not a preference.** Embeddings are pinned to 768-dim
> `nomic-embed-text` (Track I.G HNSW index). Routing an embedding through the
> cloud chain would change the vector space (OpenAI `text-embedding-3` is
> 1536/3072-dim) and silently corrupt vector search.

`get_embedding_service` resolves embeddings via `DefaultModels` and imports
**nothing** from `model_routing`. Parsing (docling/MinerU) is GPU-service-bound,
never an LLM routing call. The enforcement is mechanical: `LLMTask` has no
embedding/parsing member, and `test_embedding_local_guardrail.py` is the
falsifiable canary (asserts the embedding-service source contains no
`model_routing` import and that `resolve_route("embedding")` raises).

### Operator surface

`docs/tracks/J-model-routing/OPERATOR_GUIDE.md` documents the provider env keys
(`NVIDIA_API_KEY`), the layered privacy model, the provider-chain config UI, the
fair-use caution (local for high-volume extraction, cloud for summarization), and
how to enable/disable providers.

## 9. Document relatedness: computed vs materialized (Tracks R + U)

There are two ways the system knows that two documents are related, and they
are deliberately kept separate because they answer different questions.

### The computed layer (Track R — search)

R.1/R.2/R.3 compute document relatedness **on the fly**, at query time, and
store no edges:

- **R.1 content similarity** — cosine over the source aggregate embeddings
  (`SourceRepository.find_related_by_embedding`). A continuous signal; turning
  it into an edge would need a threshold, which throws information away.
- **R.2 shared-entity salience** — `packages/shared/src/shared/retrieval/kg_source_scorer.py`
  scores "these two documents share entity X" by `salience(X) × rarity(X)` so a
  rare, on-topic entity counts far more than a generic one shared by everything.
- **R.3 hybrid fusion** — `packages/shared/src/shared/retrieval/hybrid_fusion.py`
  blends the two into the ranked "related documents" the UI shows.

This layer is cheap, always current (it reads canonical data directly), and
needs **no upkeep** — re-extract a document and the next query already reflects
it. It is the right tool for *search* and *"show me related documents"*.

### The materialized layer (Track U — navigate / draw / export / traverse)

The materialized layer persists real edges so the document graph can be
*walked, drawn, exported, and run through graph algorithms* — things a
per-query score cannot do:

- **`mentions` (source → entity)** — `packages/shared/src/shared/retrieval/mentions_projection.py`
  + `MentionsProjectionService`. A **regenerated projection** of
  `entity.source_documents`, carrying the SAME R.2 weight and the SAME R.6
  entity filtering as the search signal, so the drawn/exported graph matches
  search by construction. Adds **nothing** to search — its entire value is the
  graph being real.
- **`cites` (source → source)** — `packages/shared/src/shared/retrieval/cites_matching.py`
  + `CitesMaterializationService`. The genuinely **new** information: discrete
  citation facts that live nowhere else. **Persisted once** from confident
  intra-corpus reference matches (precision-first), fed by Track V.

### When to use which

| Need | Use | Why |
|---|---|---|
| "Documents related to this one" (search/UI) | **Computed** (R.1–R.3) | No threshold loss, always current, no upkeep |
| Draw the document graph (U.4 viz) | **Materialized** `mentions` (+`cites`) | A canvas needs nodes + edges, not a score |
| Export the graph (U.5 NetworkX/JSONL) | **Materialized** | A file is a static snapshot of edges |
| Graph traversal / algorithms (centrality, paths) | **Materialized** | Algorithms run on an edge set |
| Genuinely-new citation facts | **Materialized** `cites` only | Computed layer never had this information |

The guiding rule: **don't double-store what you already cheaply compute.**
`mentions` is materialized only because viz/export/traversal need a real graph,
not because it adds a signal (it doesn't). `cites` is materialized because it
*is* new data with nowhere else to live.

### The sync model

The two materialized tables have **different** maintenance contracts:

- **`mentions` = regenerated projection (stateless).** It is a derived view of
  `entity.source_documents`; the canonical data is the array + the extraction.
  `MentionsProjectionService.regenerate()` clears and rebuilds it idempotently
  on demand (a re-run yields the identical edge set, no duplicates). It must be
  re-run after a re-extraction or entity merge changes the array, but it carries
  no state of its own — losing the table loses nothing.
- **`cites` = persisted once (stateful).** The matched citation edges ARE the
  data; there is no array to re-project from. Track V parses references and
  hands `CitesMaterializationService` a `{source_id: [ParsedReference]}` map;
  the service matches them against in-corpus sources (precision-first) and
  RELATEs confident matches with their confidence + matched reference text. On
  the current corpus this is a clean **0-edge no-op** (no parseable intra-corpus
  citations until Track V feeds references), but the mechanism is built + tested.

Both materialized tables are **notebook-agnostic** global edge tables. The U.5
exporters scope them to one notebook's source set
(`SourceRepository.load_notebook_source_ids`, via the `reference` edge) so the
exported document layer describes the same corpus slice as the entity layer, and
prune `mentions` to entity nodes that survived the export filter (no dangling
endpoints). The export document layer is **additive and gated** behind
`ExportFilter.include_document_layer` (default off), so entity-only exports are
byte-for-byte unchanged. See `apps/app-main/src/app_main/services/document_layer_export.py`.

> **SurrealDB `RELATE` is not idempotent on its own** — a repeated `(in, out)`
> writes a *second* row rather than collapsing. Both materializers therefore
> **clear before they relate** (and `cites` additionally de-dups `(origin,
> target)` within a run). Idempotency is a property of the regenerator, not of
> `RELATE`.

## 10. Shared graph memory: MCP graph-tools + reranker (Track W)

Track W turns the SurrealDB knowledge graph into a **shared memory substrate**
that any Claude Code session can read and write through a small set of MCP
tools, and adds an optional, isolated **reranker** to sharpen hybrid search.
Most of the machinery already existed (hybrid fusion at the repo layer, an MCP
server, the repo backings) — Track W exposed it and tied it together.

### The MCP graph-tools layer (one DB, many sessions)

The existing `surrealdb-mcp` server
(`packages/surrealdb-service/src/surrealdb_service/mcp/server.py`, FastMCP,
stdio) is the shared substrate: every session that attaches it talks to the
**same** SurrealDB (one database, not per-session state), so notes/citations one
session writes are immediately visible to another. Track W.3 added five graph
tools alongside the original low-level ones (`query_database` / `get_record` /
`list_sources` / `search_similar` / `get_entity_graph`):

| Tool | Backed by | What it does |
|---|---|---|
| `search` | `SearchRepository.hybrid_search` / `.text_search` | Ranked chunk/note search. With a caller-supplied `embedding` → full W.1 hybrid (BM25 ⊕ vector RRF); without → lexical BM25 only |
| `get_node` | `SELECT * FROM type::thing($id)` | Polymorphic fetch of any node (`entity:` / `source:` / `note:` / `notebook:`) |
| `related` | `GraphRepository.load_all_edges` | Every edge incident on a node — `relation` (entity↔entity) + `mentions` (source→entity) + `cites` (source→source), both directions, uniform `{id, source, target, edge_type, metadata}` shape |
| `cite` | `SourceRepository.relate_cites` | Write a `source→cites→source` edge, idempotently (re-cite = no-op; self-cite refused) |
| `add_note` | `NoteRepository.create_with_embedding` + `add_to_notebook` | Write a note and link it into a notebook |

**Embedding-source contract.** The surrealdb-service package has no embedding
model on purpose (keeps a heavy dep out of it), so `search` mirrors the existing
`search_similar` contract: the *caller* supplies any query vector. The embed
step stays in app-main's local-Ollama pipeline. No `embedding` → lexical BM25.

**Auth.** None — correct for the default `stdio` transport (the only caller is
the local process). The `sse`/`streamable-http` transports would expose the
`cite`/`add_note` WRITE tools unauthenticated; gate or add auth before any HTTP
exposure. See `docs/tracks/W-mcp-graph-memory/OPERATOR_GUIDE.md`.

### Two retrieval layers, deliberately separate

The MCP `search` tool is the **chunk/note hybrid search** (W.1): BM25 over
source/note text fused with vector similarity via Reciprocal Rank Fusion
(`SearchRepository.hybrid_search`, exposed on `/search` `type=hybrid`). It
answers *"which chunks/notes match this query?"*.

This is **not** the source-level Track R retrieval
(`/sources/{id}/related-hybrid`, §9 computed layer), which answers *"which whole
documents are related to this one?"* by fusing content-similarity with
shared-entity salience. Track W keeps them separate: `related` returns **graph
edges**, `search` returns **ranked passages**, and Track R returns **related
documents**. (See §9 for the computed-vs-materialized document story the MCP
tools read from.)

### The reranker service (optional, isolated, default-off)

`services/reranker/` is a small FastAPI microservice loading a local
multilingual cross-encoder (`BAAI/bge-reranker-v2-m3`, configurable via
`RERANKER_MODEL`) in its **own container** next to `docling`/`whisperx`. It
exposes `POST /rerank {query, passages[]} -> [{index, score}]` and
`GET /health`. This keeps `torch`/`sentence-transformers` **out of `app-main`**
entirely.

app-main reaches it over plain HTTP (`RERANKER_SERVICE_URL`), no ML libs:

- `services/reranker_http_client.py` — `RerankerHttpClient` (raises
  `RerankerServiceError` on any transport/service failure).
- `services/rerank_orchestrator.py` — `rerank_hybrid_results(...)` reorders the
  top-N fused hits by the service's scores. On `RerankerServiceError` it **falls
  back** to the zero-dep heuristic `retrieval.reranker.Reranker` (logged, never a
  500).

The rerank leg is **off by default**: `/search` runs it only when
`rerank=true`, so the W.1 path stays byte-identical without the flag, and a
reranker outage degrades gracefully. The model run (~2 GB download, CPU latency)
is **operator-gated** — see the runbook for the bring-up + smoke + fallback test.

## 11. Provenance → citation flow (Track X)

Answers cite the **exact source/page/chunk** a claim came from. The provenance
(file/page/section) is already stored on the `chunk` table by the Docling
ingest; Track X threads it through retrieval into the answer graphs and guards
the result. Three stages:

**X.1 — provenance hydration (retrieval layer).** The `fn::vector_search` /
`fn::text_search` SurrealDB functions collapse a source's many matching
embedding rows to a *single source-level hit* (`GROUP BY source` with
`math::max(similarity)`), so the originating chunk's page is lost inside the
function. `SearchRepository.hydrate_provenance` re-attaches it **without
touching the `fn::`**: for a `source:`-own-id vector/hybrid hit it runs one
batched `SELECT` over `source_embedding ⋈ chunk` picking the top-1 chunk by
`vector::similarity::cosine` — exactly the row that produced the collapsed
`math::max`, verified equal to 1e-9 on staging. Each hit then carries
`chunk_id`/`physical_page`/`printed_page`/`section_path`/`element_type`/`source`
(value-or-`None`). Hydration is **opt-in** (`hydrate=True`) so the generic
`/search` hot path does not pay the extra `SELECT`; only the answer-citation
path opts in. Text-only hits and `source_insight` hits get no chunk page (the
BM25 score is not reproducible out of context, and an insight has no single
originating chunk) — they cite the `source` page-less.

**X.2 — context-derived citations (answer graphs).** `ask.provide_answer` and
`source_chat` inject each hit's provenance into the prompt as a tag
`[source: <id> | p.<page> | <section>]`, and emit a structured `citations:
[{source, page, chunk_id, section}]` array. The array is built **from the
context hits actually fed to the LLM** (`graphs/citations.py::citations_from_hits`),
*not* parsed back out of the model's prose — so it is deterministic and `⊆` the
retrieval set by construction. The prompt templates additionally ask the model
to write inline `[document_id, p.<page>]` attribution markers in the answer prose
(what the user sees), citing only pages that appear in a provenance tag.

**X.3 — faithfulness guard (`graphs/citations.py`).** Because the `citations`
array is context-derived, the genuine hallucination risk is the **inline prose
markers** in the user-visible answer: the model can write `[source:x, p.99]` for
a page it never saw. `guard_answer_citations` parses those markers and
membership-checks each against the retrieval set's `(source, page)` pairs /
record ids; a marker citing a source/page not in the context is flagged
(recorded; non-destructive to the answer text by default — `strip=True` removes
only the offending marker token). The guard runs on the **text the user actually
reads**: in `source_chat` that is the single agent answer; in `ask` it is the
`final_answer` synthesized by `write_final_answer` (NOT the intermediate
sub-answers, which are never surfaced), validated against the **union of every
sub-answer's retrieval hits** (accumulated in `ThreadState.retrieval_hits`).
`guard_citation_array` is a defensive membership safety-net on the chunk-bearing
entries of the array — a no-op on current output, regression insurance if
citations ever stop being context-derived (it short-circuits to a no-op when the
retrieval set carries no chunk_ids, so an empty/unthreaded hit set never drops a
valid citation).

> **Limitation — membership, not semantic support.** The guard verifies the
> cited source/page/chunk *was retrieved* (it was in the model's context); it
> does **not** verify the cited passage actually *supports* the claim. A
> semantic-support check would need a second LLM pass and is out of scope. This
> is the same precision-first discipline as the Track U.3 `cites` edges:
> drop/flag the unverifiable, never fabricate.

## 12. Note auto-link: embedding → similarity → `related_note` edges (Track Y)

When a note is created, its most-related notes are found by embedding similarity
and the links are persisted as `related_note` graph edges — the same
embedding-similarity mechanism the source layer already used
(`SourceRepository.find_related_by_embedding`), lifted to the note level.

**The flow** (three layers, each tested in isolation):

1. **Similarity (Y.1)** — `NoteRepository.find_related_by_embedding(note_id, k)`
   ranks other notes by `vector::similarity::cosine` over `note.embedding`,
   excludes self and empty-embedding notes, returns the top-`k`
   `[{id, title, score}]`. An unembedded note (the strict-but-empty `[]`
   embedding — see [[note-embedding-non-optional]]) yields `[]`, never a crash.
2. **Edge + idempotent RELATE (Y.1)** — `related_note` is a
   `TYPE RELATION FROM note TO note` table (migration 68, fields
   `similarity_score`, `method`, `created_at`). `NoteRepository.relate_note`
   **clears-before-relates** each `(in, out)` pair so re-linking yields the
   identical edge set (`RELATE` is not idempotent — same lesson as the §9
   materializers). Both endpoints are strict-validated against `_RECORD_ID_RE`
   **before** interpolation, because `RELATE`'s graph syntax cannot bind a
   `$param` in the in/out position (a parameterized `RELATE $from->...->$to`
   silently writes nothing).
3. **Orchestrator + precision gate (Y.2)** —
   `NoteAutoLinkService.auto_link(note_id, k, min_similarity)` ensures the note
   is embedded, ranks candidates, keeps only those at/above `min_similarity`
   (conservative defaults `min_similarity=0.75`, `k=5` — no graph explosion),
   and writes idempotent edges. It lives in **app-main**, not surrealdb-service,
   because the embed step (`EmbeddingService.embed_note`) is app-main's
   local-Ollama pipeline; surrealdb-service stays embedding-free (its MCP
   `auto_link_note` tool is the embedding-free sibling that requires a
   pre-embedded note — the same split as the W.3 `search`/`add_note` tools).

**The trigger — two halves (the phased decision):**

- **On-demand (Y.2)** — `POST /notes/{id}/auto-link` and the MCP `auto_link_note`
  tool drive the orchestrator directly.
- **Background (Y.3)** — the job queue. A note can only be ranked by similarity
  once it *has* an embedding, so the **embed job is the trigger**:
  `_handle_embed_single_item` best-effort enqueues a `NOTE_AUTO_LINK` job after a
  note is successfully embedded (mirroring the R.0 `DOCUMENT_PARSE` →
  `embed_source` chaining). `handle_note_auto_link` then runs the orchestrator.

**Isolation (Y.3).** Auto-link is a **separate job** from the embed, deliberately:
the note + its embedding are persisted by the upstream embed job before the
auto-link job runs, so a linking failure can never corrupt the note. The enqueue
seam is best-effort (a queue hiccup never fails the embed); the auto-link job
itself raises on a hard failure so the worker records it `FAILED`, but it only
ever writes `related_note` edges (no note CRUD), and `relate_note` is idempotent,
so a re-run is safe and a failed run leaves no half-written graph state.

**The sync model.** The job handles **new** notes (create → embed → link).
**Re-linking on note EDIT** (content/embedding change → stale edges) is a noted
follow-up, *not* Track Y core — the same honest framing as §9: the mechanism is
built and tested; the upkeep loop on mutation is the next increment. Because
`auto_link` is idempotent, an edit-triggered re-link would be a clean drop-in
(re-run after the edit's embedding settles) rather than new machinery.

**Data reality.** The current corpus is source-heavy with few notes, so auto-link
is a built-and-tested **mechanism** that lights up as notes accumulate — like the
§9 `cites` edges (a clean no-op today, correct the moment the data arrives).

### 12a. Note→source (`note_about`): the second pass over the same embedding (Track NS)

Track Y core is note↔note. **Track NS** extends it to **note→source** — a note
auto-links to the *sources it is about* — by reusing Y's machinery rather than
adding a parallel pipeline. The key insight is that one note has **one
embedding**, so the orchestrator embeds the note **once** and runs **two passes**
over that single vector, each a `rank → gate → idempotent RELATE`:

1. **notes** → `find_related_by_embedding` → `relate_note` (the Y path, byte-for-byte unchanged);
2. **sources** → `find_related_sources_by_embedding` → `relate_note_source` (NS) → `note_about` edges.

Both passes share the **same** `min_similarity` + top-`k` gate (`k` bounds each
pass independently), and `needs_embedding` / `not_found` short-circuit **both**
symmetrically. The source pass is **purely additive** — appended after the note
pass — so Y's behaviour is untouched. `AutoLinkResult` gained a parallel,
distinctly-named source counter family (`source_links_created`,
`source_skipped_existing`, `source_below_threshold`,
`source_candidates_considered`, `linked_source_ids`) so the two link types never
conflate; one `auto_link` call now reports both. Every trigger inherits this for
free: the endpoint flattens `result.to_dict()`, the job returns
`**result.to_dict()`, and the MCP `auto_link_note` tool runs the same second
pass (repo-direct — both source primitives live in surrealdb-service, so the
embedding-free layer covers sources without an embedding model).

**The cross-type same-space requirement.** Ranking a note against sources only
means anything if `note.embedding` and `source.embedding` live in the **same
vector space**. They do, by construction: a `source.embedding` is the R.0
mean-pool of that source's chunk vectors, each produced by the *same*
`EmbeddingService` → `embedding_model.aembed` call that embeds a note
(mxbai-embed-large, 1024-dim — [[embedding-model-pin-1024]]). cosine(note vector,
source mean-pool) is then the standard query↔document retrieval signal. The
`array::len(embedding) = array::len($q)` guard in the ranking query is the hard
net: any dimension mismatch is *excluded*, never crashed — so even a future
model swap degrades to "no candidates," not a corrupt edge.

**The edge.** `note_about` (migration 70) is a
`TYPE RELATION FROM note TO source` table — lowercase snake_case like
`related_note`/`source_verdict`, schema.org/about-flavoured, and deliberately
**distinct** from the ontology's uppercase entity-relation `ABOUT` predicate (a
different layer/grain). It carries the same strict-with-defaults fields
(`similarity_score`, `method`, `created_at`) and is fresh-container-verified
(S.4). `relate_note_source` mirrors `relate_note`: strict `_validate_record_id`
on **both** raw ids before interpolation (the cross-track `relate_cites`
injection lesson — `RELATE`'s in/out positions cannot bind a `$param`),
enforcing `note:` (in) / `source:` (out), clear-before-relate per `(in, out)`.

**The no-embedding guard is load-bearing on the source side.** Both ranking
queries share the predicate `embedding != NONE AND array::len(embedding) > 0`.
In this SurrealDB version `array::len(NONE)` **raises** (it does not return
`NONE`), so the bare `> 0` form the plan first suggested crashes on a source
whose aggregate embedding is genuinely `NONE` (e.g. unaggregated). The `!= NONE`
clause short-circuits before the length check and is therefore mandatory for the
source path — correctness-neutral for the note path (strict `[]` embeddings),
one shared form.

**Data reality.** Unlike the note↔note layer in a note-sparse corpus, the
note→source layer lights up **immediately** in the current source-heavy corpus —
a note links to the convenanten / papers it resembles the moment it is embedded.
That is the explicit reason NS was chosen as Y's first extension.

> See **§12** for the shared orchestrator, trigger, and idempotency model that
> NS reuses. Source: `apps/app-main/src/app_main/services/note_auto_link_service.py`
> (the two-pass `auto_link`),
> `packages/surrealdb-service/.../repositories/notebook.py`
> (`find_related_sources_by_embedding` + `relate_note_source`), migration
> `70.surrealql`, and `docs/tracks/NS-note-source-autolink/status.md`
> (per-AC evidence + RETRO).

> See `apps/app-main/src/app_main/services/note_auto_link_service.py`,
> `apps/app-main/src/app_main/handlers.py` (`handle_note_auto_link` +
> `_handle_embed_single_item`'s chaining), migration `68.surrealql`, and
> `docs/tracks/Y-auto-link/status.md` (per-AC evidence + RETRO).

## 13. Contradiction detection: related pairs → LLM judge → `source_verdict` edge (Track Z)

Track Z asks an LLM to judge whether two *already-related* sources **reinforce**,
**contradict**, or are **neutral**, and persists only confident verdicts as a
`source_verdict` edge. It is the highest-risk graph-write feature — a false
contradiction pollutes the graph — so the whole pipeline is **precision-first**:
*a fabricated contradiction is worse than a missed one* (the same discipline as
the §9 `cites` membership gate).

**The flow (candidate → judge → gate → edge).**

1. **Candidates from the related substrate, not O(n²).** For a source, the judge
   pulls its top-`k` related sources from the **Track R** substrate
   (`find_related_hybrid` / `find_related`, §9 — dense + KG fusion, already
   topically-ranked) and forms `(source, related)` pairs (self-excluded, deduped,
   bounded by top-`k`). The corpus is never enumerated; LLM cost is O(top-`k`),
   not O(pairs).
2. **Pairwise LLM judge.** Each pair is judged by the **Track J** routed LLM
   (`RoutedLLMCaller` / `make_default_llm_caller(default_chat_model, json_mode=True)`)
   over a compact context (both titles + bounded `full_text` snippets). The system
   prompt is a conservative fact-checker that is repeatedly biased toward
   `neutral` — it must point at a specific, mutually-exclusive factual conflict to
   say `contradicts`, never infer one from topic overlap.
3. **Robust parse (no-false-edge invariant).** The response is parsed to a
   `{verdict, confidence, reasoning}` accepting **only a top-level JSON object**.
   Any failure mode — non-JSON, missing keys, an unknown label, a string/NaN
   confidence, or a top-level JSON *array* (a common "respond with JSON" failure
   that could smuggle `[{"verdict":"contradicts",...}]`) — degrades to
   `neutral`/`0.0`. A parse failure can only ever **suppress** an edge, never
   fabricate one.
4. **Precision gate → edge.** An edge is written **only** when the verdict is
   `contradicts`/`reinforces` **and** `confidence >= min_confidence` (default
   0.7, conservative). `neutral` and below-threshold verdicts are counted but
   written nowhere. Persistence goes through Z.1's idempotent, injection-safe
   `relate_verdict` (clear-before-relate per `(in,out)`; strict id validation
   before interpolation — the cross-track `relate_cites` injection lesson), so
   re-judging is a no-op for unchanged verdicts. Canonical `source` rows are
   never mutated.

**Schema.** Migration 69 asserts `source_verdict` as `TYPE RELATION FROM source
TO source`, strict fields with defaults (`verdict`/`confidence`/`reasoning`/
`judge_model`/`created_at`), fresh-container-verified. It is kept **distinct** from
the app-side `claim`/`contradicts` scaffolding (a *source→claim* edge — different
unit, different shape) and from the triage `VERSTERKT` entity-relation predicate
(a different layer/grain).

**On-demand trigger.** The judge is driven by `POST
/sources/{id}/judge-contradictions` (params `k`, `min_confidence`; route-layer
validation → 422 on a bad id / out-of-range bounds, 404 on a missing source). An
MCP `judge_contradiction` tool is **deferred** (documented): the judge needs the
app-main LLM layer, and the surrealdb-mcp server is a thin repo-direct layer with
no app-main dependency, so an MCP tool would invert that layering — it is a
follow-up for once an app-main base URL is cleanly available to that server (see
the server module docstring).

**Cost + the deferred background job.** Judging is O(pairs) of LLM calls, so it is
**on-demand only** in Z core — automate (a background job over new/edited sources)
only once the judge is trusted, the same staging the §12 auto-link job took. The
related-substrate bound + top-`k` cap keep each run cheap.

**Extensions (documented, not built in Z).** (a) A **background job** that judges
new/edited sources on ingest, gated behind the same precision threshold. (b)
**Claim-level** contradiction — judging a source against a specific `claim`
(the app-side *source→claim* `contradicts` edge) rather than source↔source — a
finer grain that lights up as claims accumulate (sparse today, like §9's `cites`).

**Data reality.** Few sources today, so this is a built-and-tested **mechanism**
that lights up as the corpus grows — like the §9 `cites` edges.

> See `apps/app-main/src/app_main/services/contradiction_judge_service.py`,
> `apps/app-main/src/app_main/api/routers/sources_processing.py`
> (`judge_contradictions`), migration `69.surrealql`,
> `SourceRepository.relate_verdict`, and `docs/tracks/Z-contradiction/status.md`
> (per-AC evidence + RETRO).

## 14. Source-ingestion pipeline: auto source→KG with gates (Track PL)

Before Track PL, ingest auto-chained only `parse → chunk → embed(per-chunk)` and
**stopped**. Everything analytical — the mean-pool `source.embedding` aggregate,
entity/relation extraction, the `mentions` document-graph, insights — was manual
or orphaned. Track PL makes ingest a **fully automatic, gated, idempotent**
pipeline from a raw source to its KG end-result, and consolidates the wiring
into one declarative definition with a per-source status model.

### The auto-chain

```
DOCUMENT_PARSE ─► EMBED ─┬─► EXTRACT ─► GRAPH ─► complete
  (ingested)   (embedded) │  (extracted) (graphed)
                          │      ▲
                          │      └─ gate: schema_review → awaiting_schema_review
                          └─► INSIGHTS  (parallel, toggle: notebook.auto_insights;
                                         does NOT advance the spine)
```

- **EMBED** writes per-chunk vectors **and** the mean-pool `source.embedding`
  aggregate (PL.1 — previously only a backfill script wrote the aggregate, so
  fresh sources were invisible to "Verwante"/document-graph/contradiction, all
  of which read `source.embedding`). The aggregate logic lives in a reusable
  module (`pipelines/embeddings/.../aggregate.py`), not a script.
- **EXTRACT** (PL.2) auto-runs the existing ontology-guided `run_extraction`
  chain (typing + relations + triage). It is **gated**: an unreviewed
  `NotebookSchema` raises `SchemaReviewPendingError` → the source parks at
  `awaiting_schema_review` and the job goes `PAUSED_FOR_REVIEW` (no entities
  written until reviewed).
- **GRAPH** (PL.3) refreshes the source-scoped `mentions` projection **inline**
  (cheap, no LLM). Invariant: it runs the **full-corpus** projection (so each
  edge keeps its global R.2 weight × R.6 IDF/df — the weighting is inherently
  cross-source) then writes **only this source's** edges. Source-scoped writes,
  global projection — not a per-source recomputation.
- **INSIGHTS** is a **parallel** enrichment branch off EMBED, behind the
  per-notebook `notebook.auto_insights` toggle (default on). It produces
  `source_insight` rows and never gates `complete`.
- **contradiction (Z)** and **`cites` (U.3)** stay **on-demand/deferred** — not
  part of the auto-chain.

### One pipeline definition + `advance_source` (PL.4)

The chaining used to be scattered across handlers (each handler wrote its stage
AND held an ad-hoc next-stage enqueue inline — the "R.0 forward-fix" pattern).
PL.4 consolidates it into ONE declarative `SOURCE_PIPELINE`
(`apps/app-main/.../services/source_pipeline.py`): an ordered list of
`PipelineStage` dataclasses (`name` / `produces` / `auto` / `gate` /
`depends_on` / `enqueue_command` / `parallel`). The single driver
`advance_source(source_id)` reads the source's `processing_stage`, finds "where
it is", and dispatches the next allowed stage(s) — the spine successor plus any
parallel branch — honouring `auto`, `depends_on`, the gates, and the
`embedded_chunks > 0` guard. Handlers became **thin**: do the stage's work,
write the produced stage, call `advance_source`. (The schema-review gate stays
in the extract handler because it must *reraise* for the worker.)

### The `processing_stage` status model

Each source carries a strict `processing_stage` string (migration 71,
`DEFAULT "ingested"`): `ingested → embedded → extracted → graphed → complete`,
plus the parked/terminal `awaiting_schema_review` and `failed`. Stages are
idempotent and **resumable** — `advance_source` can pick up from the current
stage; parked/terminal stages need an explicit resume (a review, a re-run).
The stage is surfaced on the source read API (`SourceResponse.processing_stage`)
so the UI can show per-document progress. `complete` means "the KG chain RAN",
not "the KG is non-empty": a source whose entities share no df>1 concept with
the corpus produces 0 `mentions` edges (R.6 drops df==1 singletons) yet
correctly reaches `graphed → complete`.

### Gates

- **Schema review** (EXTRACT): unreviewed schema → park at
  `awaiting_schema_review`, zero entities written.
- **`auto_insights` toggle** (INSIGHTS, per-notebook, default on): off → skip
  the parallel insights branch; the KG spine still runs.
- **Global LLM-cost reality**: full-auto extraction means every new doc triggers
  extraction; the schema-review gate + the per-notebook toggle are the brakes.

### Worker + sync-path (PL.5)

- **Bounded worker concurrency** (`JobWorker`, default **1** == serial /
  historical behaviour; raise via `JOB_WORKER_CONCURRENCY`). Safe because stage
  ordering is enforced by the **job chain**, not by worker serialization — a
  successor stage is only ever enqueued *after* its predecessor completes, so
  concurrent workers cannot reorder a source's stages. The DB layer is
  per-connection-pooled (no shared session state) and the LLM path is already
  `asyncio.Lock`-coordinated (Track J rate limiter + circuit breakers).
- **Sync ingest decoupled from the shared worker**: the sync upload path runs
  the `DOCUMENT_PARSE` handler **in-request** (via the registry) instead of
  enqueuing it onto the shared queue and polling 300s — so a sync ingest no
  longer occupies the single worker slot for the whole parse and no longer
  starves other queued jobs. The handler's internal `advance_source` still
  enqueues the lightweight downstream chain as background jobs.
- **Cleanups**: the dead `USE_MINERU_SERVICE` env removed (routing is decided
  per-source by the `parser_engine` ContentSetting); the parser-`auto` decision
  unified into one source of truth (`engine_dispatcher.resolve_parser_route`,
  consumed by `SourceExtractor`), eliminating the drift between the dispatcher
  and the extractor's separate raw-`"auto"` re-check.

> See `apps/app-main/src/app_main/services/source_pipeline.py`
> (`SOURCE_PIPELINE` + `advance_source`),
> `pipelines/embeddings/src/embeddings/aggregate.py`,
> `apps/app-main/src/app_main/handlers.py` (thin stage handlers),
> `apps/app-main/.../services/mentions_projection_service.py`
> (`refresh_source`), `packages/job-queue/src/job_queue/worker.py` (concurrency),
> `apps/app-main/.../api/routers/sources_upload.py` (sync decouple),
> migrations `71`/`72`, and `docs/tracks/PL-source-pipeline/` (plan, status,
> RETRO; per-AC evidence per phase).

### Frontend: `processing_stage` as the pipeline spine (Track UX)

The Next.js frontend surfaces `source.processing_stage` as the **single source
of truth** for pipeline progress across every source surface — the list cards,
the source-detail header, and the creation flow. A pure, React-free state
machine (`frontend/src/lib/pipeline/pipeline-stages.ts` →
`derivePipelineNodes`) maps the stage plus the job axis and output counts into
the ordered spine `Ingest → Embed → Extract → Graph → Complete` (Embed **before**
Extract) with Insights as a parallel branch. One canonical `PipelineStatus`
component renders it in three variants (`live` / `card` / `detail`). Rules
mirror the backend: `processing_stage` decides which node is `done`, the job
axis (`/sources/{id}/status`) only drives the current node's spinner, and output
counts are enrichment on `done` nodes (never the status source). `graphed`/
`complete` marks the Graph node `done` (its "linked" badge is keyed off the
stage, not entity-relation counts). `awaiting_schema_review` parks Extract as a
gated node; `failed` renders the failed node with a Retry action.

Because GRAPH / EXTRACT / INSIGHTS now run automatically in the backend
auto-chain, the leftover manual "Run X" runners on the detail view are
**recovery-only**: each is enabled only when the source is on a recovery stage
(`failed` / `awaiting_schema_review`) or its own output is genuinely missing,
and otherwise renders disabled with a "Runs automatically" hint. Reprocess stays
always available as the parser/Docling config home.

> See `frontend/src/lib/pipeline/` (`processing-stage.ts`, `pipeline-stages.ts`,
> `source-counts.ts`), `frontend/src/components/sources/pipeline/PipelineStatus.tsx`,
> `frontend/src/lib/hooks/use-sources.ts` (`useSourcePipeline`), and
> `docs/tracks/UX-pipeline-alignment/` (plan, status; per-AC evidence per phase).

## 15. Further reading

- `docs/SUMMARIZATION_APPROACHES.md` — design + status of all 11 summarization strategies
- `docs/KNOWLEDGE_GRAPH_IMPLEMENTATION_PLAN.md` — KG architecture and roadmap
- `docs/SURREALDB_KNOWLEDGE_GRAPH_PROPOSAL.md` — DB schema rationale
- `docs/WORKSPACE_REFACTOR_PLAN.md` — the original monolith → workspace migration
- `docs/REFACTOR_PLAN.md` — the recent god-split + structural cleanup refactor
- `docs/GRAPH_FEATURES_IMPLEMENTATION_GUIDE.md` — UI/UX implementation of KG features
- `docs/tracks/A-mineru/RETRO.md` — Track A retrospective (parser-engine routing)
- `docs/tracks/B-kg-quality/RETRO.md` — Track B retrospective (multi-schema KG)
- `docs/tracks/D-output-richness/RETRO.md` — Track D retrospective (export surfaces)
- `docs/tracks/U-document-graph/status.md` — Track U retrospective (document graph: `mentions`/`cites`, computed-vs-materialized)
- `docs/tracks/J-model-routing/OPERATOR_GUIDE.md` — cloud/local routing: env keys, privacy model, fair-use, enable/disable
- `docs/tracks/W-mcp-graph-memory/status.md` — Track W retrospective (MCP graph-tools shared substrate + hybrid search + reranker)
- `docs/tracks/W-mcp-graph-memory/OPERATOR_GUIDE.md` — registering the `surrealdb-mcp` server + gated reranker bring-up/smoke/fallback
- `docs/tracks/X-citations-to-source/status.md` — Track X retrospective (provenance → citation flow + faithfulness guard; membership-not-semantic)
- `docs/tracks/Y-auto-link/status.md` — Track Y retrospective (note auto-link: embedding → similarity → `related_note`; on-demand + background-job trigger; note↔source extension)
- `docs/tracks/NS-note-source-autolink/status.md` — Track NS retrospective (note→source auto-link: the second pass over one embedding → `note_about`; cross-type same-space; the `array::len(NONE)` gotcha)
- `docs/tracks/Z-contradiction/status.md` — Track Z retrospective (contradiction detection: related pairs → LLM judge → `source_verdict`; precision-first; background-job + claim-level extensions)
- `docs/tracks/PL-source-pipeline/RETRO.md` — Track PL retrospective (auto source→KG pipeline: orphaned-aggregate bug, foundational auto-extract gap, the source-scoped-mentions-must-be-global-projection invariant, one `SourcePipeline`/`advance_source` consolidation, the worker-concurrency decision)
- `docs/troubleshooting/exports.md` — failure-mode diagnostics for the three export formats
