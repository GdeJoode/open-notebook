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

## 6. Further reading

- `docs/SUMMARIZATION_APPROACHES.md` — design + status of all 11 summarization strategies
- `docs/KNOWLEDGE_GRAPH_IMPLEMENTATION_PLAN.md` — KG architecture and roadmap
- `docs/SURREALDB_KNOWLEDGE_GRAPH_PROPOSAL.md` — DB schema rationale
- `docs/WORKSPACE_REFACTOR_PLAN.md` — the original monolith → workspace migration
- `docs/REFACTOR_PLAN.md` — the recent god-split + structural cleanup refactor
- `docs/GRAPH_FEATURES_IMPLEMENTATION_GUIDE.md` — UI/UX implementation of KG features
