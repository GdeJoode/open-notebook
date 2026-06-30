# Track PL — Retrospective (Source-ingestion pipeline: auto source→KG with gates)

> Closing date: 2026-06-30
> Branches merged: `track/pl1-source-aggregate-embedding`,
> `track/pl2-autochain-extract`, `track/pl3-autochain-graph-insights`,
> `track/pl4-pipeline-definition`, `track/pl5-worker-cleanup`
> Final state: see `docs/tracks/PL-source-pipeline/status.md`.

This retrospective draws on the per-phase `status.md` entries and the
adversarial-review files (`reviews/`). Goal of the track (user-chosen, locked
2026-06-29): turn ingest from a chain that stopped at `embed` into a
**fully-automatic, gated, idempotent** pipeline from a raw source to its KG
end-result, plus the architectural cleanup (one pipeline definition, a
per-source status model, worker/sync fixes, dead/duplicated code).

## What we shipped (the spine)

`DOCUMENT_PARSE → EMBED(+aggregate) → EXTRACT → GRAPH(mentions) → complete`,
with INSIGHTS as a parallel toggle-gated branch off EMBED, the schema-review
gate parking EXTRACT, a per-source `processing_stage` status model, and one
declarative `SOURCE_PIPELINE` + `advance_source` driver. Contradiction (Z) and
`cites` (U.3) stayed on-demand/deferred per the locked decision.

## The bugs / gaps this track closed

### 1. The orphaned `source.embedding` aggregate (PL.1) — a near-silent data bug

A freshly-ingested source had per-chunk vectors but a **NULL** mean-pool
`source.embedding` aggregate, because the aggregate was written ONLY by a
backfill *script* (`scripts/backfill_chunk_embeddings.py`). Every reader of
relatedness — "Verwante", the document-graph, contradiction, hybrid search —
reads `source.embedding`. So a new doc was effectively **invisible** to the
entire relatedness/analytical layer until someone manually ran a script. This
was the highest-leverage fix in the track: a one-line-of-intent gap (write the
aggregate in the live embed step, reusing the now-promoted
`pipelines/embeddings/.../aggregate.py` helper) that unblocked the whole layer.
Lesson: **a value written only by a script is a latent orphan** — if a
production reader depends on it, the write belongs in the live path, and the
script becomes a backfill-only convenience.

### 2. The foundational auto-extract gap (PL.2) — the KG never built itself

The entire KG/analytical layer (entities, relations, mentions, document-graph)
was internally coherent but **only reachable by manual API calls**
(`POST /sources/{id}/run-entities`, a manual mentions regenerate). Ingest
auto-chained `parse→embed` and stopped. So the product's headline capability —
a knowledge graph — never materialized without a human poking each source. PL.2
(extract) + PL.3 (graph + insights) closed this by chaining the *existing*
coherent `run_extraction` / `refresh_source` paths off a successful embed.
Lesson: **"the capability exists" ≠ "the capability runs."** A well-built
subsystem with no automatic trigger is, from the user's seat, missing.

## The load-bearing invariants we had to preserve

### 3. Source-scoped `mentions` must be a GLOBAL projection, filtered on write

The naive read of "refresh mentions for this source" is "compute this source's
edges in isolation". That is **wrong**: each `mentions` edge weight is global
(R.2 weight × R.6 IDF/df) — the IDF/df term is inherently cross-source (a
concept shared by 1 source vs 50 sources weighs differently). So
`MentionsProjectionService.refresh_source` runs the **full-corpus projection**
(to get correct global weights), then keeps/writes only the edges whose
`source_id == this source`. Source-scoped *write*, global *projection*. R.6
noise normalization is preserved (df==1 singletons never become edges, even
under the scoped refresh). PL.4 relocated only the *call site* of this
projection (handler → `advance_source._run_graph_inline`), never the projection
itself. Lesson: **scoping a write is not the same as scoping the computation** —
when a value is defined relative to the whole corpus, you must compute globally
and filter the write, or you silently corrupt the weights.

### 4. `complete` means "the chain ran", not "the KG is non-empty"

A source whose entities share no df>1 concept with the rest of the corpus
produces **0** `mentions` edges (R.6 drops df==1 singletons), yet its extraction
succeeded and its graph "refreshed to empty". That source correctly reaches
`graphed → complete`. We made this explicit in code + docs so the UI reads
"complete with 0 edges" as a healthy terminal state, not a failure.

## The consolidation (PL.4)

The chaining had grown as a scattered "R.0 forward-fix": each handler wrote its
own `processing_stage` AND held an inline ad-hoc next-stage enqueue. That is
fine for one link and a liability at four — the chain's shape lived nowhere you
could read it, and adding/gating a stage meant editing N handlers. PL.4 moved
ALL of it into one declarative `SOURCE_PIPELINE` (ordered `PipelineStage`
dataclasses with `depends_on`/`gate`/`auto`/`parallel`) + a single
`advance_source` driver; handlers became thin. The schema-review gate stays in
the extract handler (it must *reraise* `SchemaReviewPendingError` for the worker
→ `PAUSED_FOR_REVIEW`; `advance_source` is only on the success path). Lesson:
**consolidate a chain the moment it has more than two links** — the declarative
table is the thing reviewers and the UI can actually reason about.

## The worker-concurrency decision (PL.5) — verify-first on a load-bearing path

The single one-at-a-time worker was flagged as a bottleneck. The instruction
(and the right instinct) was **conservative**: a worker regression destabilizes
*all* ingestion. The hazard analysis we ran before touching it:

- **Stage ordering is enforced by the job CHAIN, not by worker serialization.**
  A successor stage (e.g. EXTRACT) is only ever enqueued *after* its predecessor
  (EMBED) completes — via `advance_source`. So even N concurrent workers cannot
  run a source's stages out of order; the queue never holds EXTRACT-before-EMBED
  for the same source. This is the crux: we confirmed the pipeline does **not**
  depend on the worker being serial for correctness.
- **The DB layer is per-connection-pooled** (`surrealdb_service` acquires a
  distinct connection per call; no shared session `LET` state, named `$params`),
  so concurrent queries don't collide.
- **The LLM path is already concurrency-coordinated** — the Track J rate limiter
  and the circuit breakers are `asyncio.Lock`-guarded shared singletons built to
  throttle parallel calls.
- **The only designed same-source parallelism** (INSIGHTS ∥ EXTRACT off EMBED)
  operates on disjoint, idempotent data (`source_insight` vs
  `entity`/`relation`/`mentions`).

Given that, we shipped the **smallest safe change**: a bounded
`asyncio.Semaphore` in `JobWorker`, **default 1 (serial — behaviour-identical to
today)**, raised via `JOB_WORKER_CONCURRENCY` / a constructor arg. The default
preserves the conservative deployment exactly (all 10 pre-existing worker tests
pass unchanged); an operator opts into parallelism. Tests prove both directions:
two independent jobs overlap at concurrency=2 (not serialized), the default
stays strictly serial, and the bound caps peak parallelism. Lesson:
**verify the hazard model before changing a load-bearing primitive, and make the
risky behaviour opt-in with a behaviour-preserving default** — that converts a
scary change into a safe, reversible one.

## The cleanups (PL.5)

- **Dead `USE_MINERU_SERVICE` env** — set in compose, read **nowhere** in Python
  (grep-confirmed; only `MINERU_SERVICE_URL` is read). Routing is decided
  per-source by the `parser_engine` ContentSetting. Removed.
- **Duplicated parser-`auto` decision** — `engine_dispatcher.select_parser_engine`
  resolved `auto→docling` while `SourceExtractor._process_file` independently
  re-checked the raw `"auto"` string to arm the A.1c confidence fallback (and
  kept its own copy of the docling-parseable extension set). Two decision sites
  that could drift. Unified into one `resolve_parser_route()` (the single source
  of truth, folding both the concrete engine and `use_auto_fallback`), consumed
  by the extractor. Behaviour unchanged (routing tests green). Lesson:
  **a decision made in two places is a decision that will drift** — give it one
  home and have every caller read it.

## Sync-path decouple (PL.5)

The sync upload path enqueued `process_source` onto the **shared** single-worker
queue then polled its status for up to 300s — blocking the request AND starving
every other queued job behind the one busy worker slot. We run the same
`DOCUMENT_PARSE` handler **in-request** via the registry instead, so the heavy
parse executes in the request's own coroutine and never occupies the shared
worker; the handler's internal `advance_source` still enqueues the lightweight
downstream chain as normal background jobs. What remains **by design**: a *sync*
caller still blocks for its own parse duration (that is the contract;
`async_processing=true` is the non-blocking path). The shared-queue starvation is
gone.

## Deferred (intentionally)

- **Contradiction (Z)** and **`cites` (U.3, blocked on Track V)** stay
  on-demand/deferred — not auto-chained.
- **Worker concurrency > 1 in production** is shipped but **opt-in** (default 1).
  Turning it on for the live deployment, and choosing a tuned default, is a
  follow-up that wants a load test against the live corpus — the mechanism is
  built, tested, and reversible (set the env back to 1).

## What worked

- **Phase decomposition (PL.1→PL.5, safe→risky).** Landing the data-bug fix
  first (PL.1), then the foundational gap (PL.2/PL.3), then the pure
  consolidation (PL.4), then the riskiest worker change last (PL.5) meant each
  phase built on a green base and the scariest change inherited a consolidated,
  well-tested pipeline.
- **`@requires_docker` round-trips per phase** caught the live schema realities
  (the S.4 SCHEMAFULL writability hazard on migrations 71/72, the gate parking,
  the source-scoped graph) that pure-mock tests would have missed.
- **Best-effort posture, preserved consistently.** Every status write / enqueue
  / inline refresh is guarded so a side-effect failure never fails the
  already-persisted producing step. The pipeline degrades to "stage didn't
  advance, re-run recovers" rather than "ingest failed".

## What to carry forward

- **Idempotent + resumable from `processing_stage`** is the property that makes
  the whole chain safe to re-drive. Keep every new stage idempotent (delete-then-
  write / dedup per key) so a double-dispatch is a no-op.
- **The 3 docling baseline failures** (`TestBuildIngestionConfig`) are
  environmental (docling not installed in CI) and pre-date this track — they
  stayed out of scope and remain the known baseline.
