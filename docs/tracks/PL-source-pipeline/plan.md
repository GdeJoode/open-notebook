# Track PL — Source-ingestion pipeline: auto source→KG (with gates)

> From the pipeline review (`~/.claude/plans/jiggly-forging-dusk.md`). The current ingest only
> auto-chains `parse → chunk → embed(per-chunk)` and stops; the entire KG/analytical layer
> (entities, mentions, document-graph) is manual/orphaned. Goal (user-chosen): a **fully automatic**
> pipeline (with explicit gates) source→KG-eindresultaat, plus architectural cleanup (one pipeline
> definition, a per-source status model, worker/sync fixes). Contradiction + cites stay deferred.

## Decisions (locked 2026-06-29)
1. **Full-auto with gates:** ingest auto-chains `embed(+aggregate) → EXTRACT → GRAPH(mentions)` (+ INSIGHTS toggle). Schema-review pauses EXTRACT. Contradiction (Z) + cites (U.3, blocked on Track V) stay on-demand/deferred.
2. **Architecture in scope:** one declarative pipeline definition (no scattered handler-chaining), a per-source `processing_stage` status model, worker-concurrency / sync-path fixes, and the dead-code/duplication cleanups.
3. **Idempotent + resumable:** every stage idempotent, resumable from the current `processing_stage`; live writes gated; `staging` explicit.

## Verified current state (from 3 Explore agents)
- Auto chain: `DOCUMENT_PARSE` (`handlers.py:104` → `SourceProcessor.process_source`) → `embed_source` (`handlers.py:144`, enqueued in the HANDLER) → `EMBEDDING_GENERATE` (`_handle_embed_source:361` → `SourceEmbeddingOrchestrator.embed_source` → per-chunk `source_embedding`). **Ends there.**
- 🔴 `source.embedding` (mean-pool aggregate) written ONLY by `scripts/backfill_chunk_embeddings.py` (`populate_source_embedding`/`set_aggregate_embedding`/`mean_pool`); read by R.1/R.2/`find_related_by_embedding`/hybrid/contradiction. Orphaned.
- 🔴 `ENTITY_EXTRACT` (`handlers.py:246`) only via manual `POST /sources/{id}/run-entities` (`sources_processing.py:161`). Internally coherent: `entity_extraction_service.run_extraction:982` → `_embed_entities` → `FilteringWorkflow` → `persist_filtered_result` (typing L + relations O) → `_run_triage` (status/queue/backbone).
- `mentions` (U.2) manual regenerate (`MentionsProjectionService.regenerate`); `INSIGHT_EXTRACT` manual; single in-process one-at-a-time worker (`api/app.py` lifespan); sync ingest path polls the shared worker (`sources_upload.py:482`); dead `USE_MINERU_SERVICE`; duplicated parser-`auto` decision.

**Workflow**: track methodology — `implementer` → `adversarial-reviewer` (≤3 → `escalation-handler`). Main tree, `uv run pytest`, no worktree. Additive/idempotent; live writes gated; `SURREAL_DATABASE=staging`.

---

## Phase PL.1 — Fix the orphaned `source.embedding` aggregate (Backend) 🔧 near-bug, highest leverage
**Why**: a freshly-ingested source has per-chunk vectors but a NULL aggregate → invisible to "Verwante"/document-graph/contradiction (all read `source.embedding`). Unblocks the live test immediately.
**Deliverables**: in the EMBED step (`SourceEmbeddingOrchestrator.embed_source` or the embeddings service it calls), after writing per-chunk vectors, compute + write the mean-pool `source.embedding` aggregate. Reuse `mean_pool` (`packages/shared/.../utils/vectors.py`) + `SourceRepository.set_aggregate_embedding`; promote the `populate_source_embedding` wrapper out of `scripts/backfill_chunk_embeddings.py` into a shared/service location (don't import from a script). Idempotent (recompute on re-embed). Plus a one-time backfill of existing sources on `staging` (gated).
**Acceptance**
1. After the embed job for a source, `source.embedding` is a populated 1024-dim vector (not NONE/empty); confirmed by a `@requires_docker` test (new source → embed → aggregate present, correct dim).
2. Idempotent: re-embedding recomputes the aggregate, no error on a source with 0/empty chunks (graceful).
3. The `populate_source_embedding` logic lives in a reusable, non-script location; the backfill script imports it (no behavior change to the script).
4. Existing embed/source tests stay green; a read-only staging probe shows a freshly-embedded source carries the aggregate.
5. (Live) backfill existing `staging` sources so the current corpus is related-retrievable.
**Branch**: `track/pl1-source-aggregate-embedding`. **Depends on**: none.

## Phase PL.2 — Auto-chain EXTRACT after EMBED + `processing_stage` (Backend) 🔧 foundational
**Why**: the KG never builds automatically. Auto-chain entity extraction (the existing coherent `run_extraction` chain), respecting the schema-review gate, and add a visible per-source stage.
**Deliverables**: after a successful source embed (`_handle_embed_source`), enqueue `run_entities`/`ENTITY_EXTRACT` (best-effort, mirroring the embed-chain pattern). Respect the schema-review gate (`PAUSED_FOR_REVIEW` / `awaiting_schema_review`). Add a `source.processing_stage` field (verify/extend any existing `source.status`): `ingested → embedded → extracted [/awaiting_schema_review] → graphed → complete` (+ `failed`); each stage transition sets it. Idempotent (re-extract dedups/merges per `(in,out,type)`).
**Acceptance**
1. Adding a source (with chunks) results — with NO manual call — in `entity` + `relation` rows after the chain runs; verified by a `@requires_docker` test driving ingest→embed→extract.
2. The schema-review gate still pauses extraction (unreviewed schema → `awaiting_schema_review`, no entities written until reviewed); test it.
3. `processing_stage` advances `ingested→embedded→extracted` (and `awaiting_schema_review` on the gate); persisted on the source; idempotent re-runs.
4. A triage failure does not fail extraction (best-effort preserved); existing extraction/triage tests green.
**Branch**: `track/pl2-autochain-extract`. **Depends on**: PL.1.

## Phase PL.3 — Auto-chain GRAPH (mentions refresh) + INSIGHTS toggle (Backend) 🔧
**Why**: even with entities, the document-graph stays empty until a manual mentions regenerate.
**Deliverables**: after a successful EXTRACT, auto-refresh `mentions` scoped to this source (reuse `MentionsProjectionService.regenerate`, source-scoped/incremental), setting `processing_stage = graphed`. Add an INSIGHTS stage (`INSIGHT_EXTRACT`/`run_summaries`) auto-chained after EMBED, behind a per-notebook toggle (default on). `processing_stage = complete` when EMBED+EXTRACT+GRAPH(+INSIGHTS) are done.
**Acceptance**
1. After ingest of a source with entities, `mentions` edges for it exist with NO manual regenerate; the document-graph endpoint returns them; `@requires_docker` test.
2. The mentions refresh is source-scoped + idempotent (re-run → same edge set, no dups); R.6 noise normalization preserved.
3. INSIGHTS auto-runs when the notebook toggle is on, is skipped when off; `source_insight` rows appear; test both.
4. `processing_stage` reaches `complete`; suites green.
**Branch**: `track/pl3-autochain-graph-insights`. **Depends on**: PL.1, PL.2.

## Phase PL.4 — One `SourcePipeline` definition + status surfaced (Backend) 🏗
**Why**: the chaining is scattered across handlers ("R.0 forward-fix"); consolidate into one declarative, testable pipeline + expose the stage.
**Deliverables**: a declarative `SourcePipeline` (ordered stages with `depends_on` + `gate` + `auto`) and one `advance_source(source_id)` that reads `processing_stage` and runs/enqueues the next allowed stage. Handlers become thin (call `advance`). Expose `processing_stage` on the source read API (so the UI can show per-document progress). No behavior change vs PL.1–PL.3 — pure consolidation.
**Acceptance**
1. The chain is driven by `advance_source`; handlers contain no ad-hoc next-stage enqueues; unit tests on the stage-transition table (each stage → next, gates respected).
2. `processing_stage` is returned by the source read endpoint(s); a test asserts it.
3. End-to-end ingest still produces the same result as PL.3 (no regression); suites green.
**Branch**: `track/pl4-pipeline-definition`. **Depends on**: PL.1–PL.3.

## Phase PL.5 — Worker concurrency / sync-path + cleanups (Backend) 🏗
**Why**: the single one-at-a-time worker + the 300s sync-poll are bottlenecks; plus dead/duplicated code.
**Deliverables**: bounded concurrency in `JobWorker` (N parallel) OR queue-lanes (light ingest vs heavy extraction/LLM) — verify the `packages/job-queue/.../worker.py` model first and pick the smaller safe change; decouple the sync-ingest path from the shared worker (don't block the request 300s on the shared queue). Cleanups: remove the dead `USE_MINERU_SERVICE` env; de-duplicate the parser-`auto` decision (single source of truth in `engine_dispatcher`).
**Acceptance**
1. Two queued jobs no longer strictly serialize (or are lane-separated): a heavy extraction doesn't block a light ingest — demonstrated by a worker test.
2. The sync ingest path doesn't hold the shared worker for the whole parse (verify the decoupling).
3. `USE_MINERU_SERVICE` removed (compose + any refs); parser-`auto` decided in one place; behavior unchanged (docling/mineru/auto-fallback routing tests green).
4. No regression across the ingest/extraction/job suites.
**Branch**: `track/pl5-worker-cleanup`. **Depends on**: PL.4 (so the pipeline is consolidated first).

---

## Risks & open decisions
1. **LLM cost of full-auto extraction** — every new doc now triggers extraction. Mitigation: the schema-review gate + a per-notebook auto-KG toggle could gate it; default per the user's full-auto choice. (Surface a global off-switch.)
2. **Worker change blast radius** — concurrency touches the shared worker; PL.5 comes last + picks the smallest safe change after verifying the worker model.
3. **`processing_stage` vs existing `source.status`** — verify whether a status field exists to extend rather than add a parallel one (PL.2).
4. **Schema-review gate UX** — auto-extraction parking on `awaiting_schema_review` must be visible (the `processing_stage` surfaces it; UI is out of this backend scope but the field enables it).

## Verification (end-to-end)
- PL.1: new doc → `source.embedding` populated (1024-dim); "Verwante"/document-graph show it.
- PL.2/PL.3: new doc → entities + relations + `mentions` appear with no manual step; `processing_stage` runs `ingested→…→complete`; unreviewed schema parks on `awaiting_schema_review`.
- PL.4: one `advance_source` drives the chain; stage exposed on the API.
- PL.5: concurrent jobs don't serialize; sync path doesn't block the queue; cleanups verified.
- `@requires_docker` roundtrips + `uv run --project <pkg> pytest`; `SURREAL_DATABASE=staging`.
