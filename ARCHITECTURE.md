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

## 7. Knowledge graph export surfaces (Track D — output richness)

Track D (closed 2026-06-16) added three HTTP export surfaces over the
notebook-scoped KG projections from D.0. All three share a single
filter pipeline so the "what you'll export" preview the UI shows
matches the actual export byte-for-byte.

### Endpoints + services

| Endpoint | Service module | Metric event | Notes |
|---|---|---|---|
| `POST /api/notebooks/{id}/export-obsidian` | `apps/app-main/src/app_main/services/obsidian_export_service.py` | `export.obsidian` | Two modes: `mode="zip"` streams an in-memory zip (one `.md` per entity + `README.md`); `mode="vault_path"` writes the same files directly to `Settings.vault_path / Settings.vault_entities_folder` via per-file `tempfile + os.replace` (POSIX atomic rename). The vault-path branch is the sole async export surface, dispatched through `JobType.EXPORT_OBSIDIAN` in `apps/app-main/src/app_main/handlers.py` (Q-D-2). |
| `POST /api/notebooks/{id}/export-jsonl` | `apps/app-main/src/app_main/services/jsonl_export_service.py` | `export.jsonl` | Streaming zip of `entities.jsonl` + `relations.jsonl`. Build-then-stream (Q-D-7): `model_dump(mode="json", exclude={"embedding"})` per row, written into the open ZIP member then yielded in 16KB chunks. Per-line keys are Neo4j-`apoc.load.json`-compatible (`source_entity`/`target_entity` not `in`/`out`). |
| `POST /api/notebooks/{id}/export-networkx` | `apps/app-main/src/app_main/services/networkx_export_service.py` | `export.networkx` | Builds a `networkx.DiGraph` then serialises to one of 7 formats: **GraphML**, **GEXF**, **GML**, **JSON-tree**, **edge-list**, **adjacency-list**, **pickle**. Attribute flattening contract: `type_tags` → CSV string, `properties` → JSON-encoded string; round-trip tests confirm flatten/unflatten preserves data (Risk 5). |
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

## 9. Further reading

- `docs/SUMMARIZATION_APPROACHES.md` — design + status of all 11 summarization strategies
- `docs/KNOWLEDGE_GRAPH_IMPLEMENTATION_PLAN.md` — KG architecture and roadmap
- `docs/SURREALDB_KNOWLEDGE_GRAPH_PROPOSAL.md` — DB schema rationale
- `docs/WORKSPACE_REFACTOR_PLAN.md` — the original monolith → workspace migration
- `docs/REFACTOR_PLAN.md` — the recent god-split + structural cleanup refactor
- `docs/GRAPH_FEATURES_IMPLEMENTATION_GUIDE.md` — UI/UX implementation of KG features
- `docs/tracks/A-mineru/RETRO.md` — Track A retrospective (parser-engine routing)
- `docs/tracks/B-kg-quality/RETRO.md` — Track B retrospective (multi-schema KG)
- `docs/tracks/D-output-richness/RETRO.md` — Track D retrospective (export surfaces)
- `docs/tracks/J-model-routing/OPERATOR_GUIDE.md` — cloud/local routing: env keys, privacy model, fair-use, enable/disable
- `docs/troubleshooting/exports.md` — failure-mode diagnostics for the three export formats
