# Track R — Hybrid source retrieval (embeddings + KG + cluster summaries)

> **DRAFT for user review.** Born from `docs/reviews/extraction-kg-embeddings-coherence-review.md`.
> Goal: make "find relevant sources by content" a first-class **hybrid** retrieval function.
> User steer (load-bearing): **the KG must play a LARGE role in search** — not de-emphasized.
> So retrieval fuses three signals: dense embeddings + KG entity/graph proximity + embedded
> cross-source cluster summaries.

**Status of inputs**
- The embedding layer is dormant in staging (0 chunk embeddings, no source-level embedding field).
  → **R.0 depends on the root-cause investigation (in progress).** R.0 specifics are finalized once that lands.
- The KG exists but is noisy (8.7% active, 81% of active = `topic`, duplicate EN/NL predicates).
  → Kept and re-scoped (R.2 uses it for search; R.6 trims noise without breaking exports/resolution).

**Workflow**: track methodology — `implementer` → `adversarial-reviewer` (≤3 → `escalation-handler`).
Main tree, `uv run pytest`, no worktree. Live writes gated by checkpoint.

---

## Phase R.0 — Embedding foundation: forward-fix + backfill (Backend) — root-cause RESOLVED
**Root cause (confirmed against live staging)**: the embedding layer is **not missing — it's decoupled-and-unrun**.
- The `source_embedding` table (`migrations/1.surrealql:16-20`, chunk back-ref `migrations/27.surrealql:16`) **IS** the chunk-content embedding store (1 row/chunk: `source`, `order`, `content`, `embedding`) and is already searched by `fn::vector_search` (`migrations/3.surrealql`). It's **empty**, not absent. (The earlier review's "no source-level embedding field" was wrong.)
- `SourceProcessor.process_source` deliberately does NOT call the (fully working) embed step; embedding only runs via the **manual** `POST /sources/{id}/run-embed` → `embed_source` job → `source_embedding_orchestrator.embed_source` (`source_embedding_orchestrator.py:41`). So chunks are created without vectors.
- **Reset drift**: 9,053 of 10,501 chunks are **orphaned** (their sources were deleted; no source→chunk cascade). Only **1,448 chunks under the 6 live sources** are usefully embeddable.
- **Model**: `mxbai-embed-large:latest` (Ollama), **1024-dim** (verified — NOT the "768-dim pin" the docs claim; entity vectors are already 1024). Reachable now. See [[embedding-model-pin-1024]].

**Deliverables**
- **Forward-fix**: chain `source_embedding_orchestrator.embed_source` into `process_source` after chunk creation (the orchestrator already exists — just unwired into ingest), so new ingests auto-embed.
- **Backfill** `scripts/backfill_chunk_embeddings.py` (Track-P pattern: idempotent, resumable, batched, `--dry-run`) iterating the live sources, calling `embed_source` (which itself reads chunks + batches).
- **Orphan GC** of the 9,053 dead chunks (GATED on user decision — intentional reset vs accidental; see open Q5).
- **Source-level aggregate vector**: add `embedding: option<array<float>>` to the `source` table/model (genuinely absent today) + populate by **mean-pooling** the chunk vectors (no extra LLM calls once chunks are embedded) — this is what enables source↔source cosine in R.1.

**Acceptance**
1. After the forward-fix, a fresh ingest auto-populates `source_embedding` rows for its chunks (no manual run-embed).
2. Backfill populates `source_embedding` for the 1,448 live-source chunks; idempotent (post-run `--dry-run` → 0 missing); logs observed dimension == **1024**.
3. `source.embedding` aggregate field exists + is populated (mean-pool) for the 6 live sources.
4. Orphan chunks resolved per user decision (count before/after); no churn to live-source data beyond the embedding fields.
5. Stale `768` assertions/comments in touched code reconciled to the real 1024 pin (or made model-derived).
**Branch**: `track/r0-embedding-foundation`. **Live steps** (backfill + GC + source.embedding populate) = gated checkpoint.

## Phase R.1 — Source-level kNN retrieval + `/sources/{id}/related` (Backend + API)
**Why**: the simplest content-linking primitive — nearest sources by dense similarity.
**Deliverables**: a retrieval service computing cosine kNN over source embeddings; `GET /sources/{id}/related?k=` endpoint; router tests.
**Acceptance**
1. Endpoint returns top-k related sources ranked by cosine, excluding self.
2. Deterministic ordering; bounded k; empty/no-embedding handled gracefully.
3. Unit + router tests; measured on ≥2 known-related staging sources.
**Branch**: `track/r1-source-knn`. **Depends on**: R.0.

## Phase R.2 — KG retrieval signal (Backend) — *the "KG plays a large role" piece*
**Why**: rank related sources by **shared knowledge**, not just surface text similarity.
**Deliverables**: a KG-proximity scorer between sources:
- Shared **active** canonical entities (weighted by type salience + inverse entity frequency — a rare named org/programme counts more than a generic `topic`).
- Optional 1-hop relation expansion (sources connected via an entity→relation→entity path).
- Excludes archived/noise entities and the duplicate-predicate noise (coordinate with R.6).
**Acceptance**
1. Given a source, produces a ranked related-source list from shared active entities + relation proximity, with per-pair explanation (which entities/edges drove the score).
2. Generic-bucket entities (`other`, bare `topic`) down-weighted vs named/typed entities — measurable in the score.
3. Tested on staging pairs with known entity overlap; scorer is pure + unit-tested.
**Branch**: `track/r2-kg-signal`. **Depends on**: KG present (no R.0 dependency — can run parallel to R.1).

## Phase R.3 — Hybrid ranker / fusion (Backend)
**Why**: combine dense + KG (+ cluster-summary, R.4) into one ranking the user controls.
**Deliverables**: a fusion ranker (reciprocal-rank-fusion or tunable weighted sum) over the R.1/R.2 signals; config-surfaced weights so the **KG weight can be set high** per the user's steer; ablation harness.
**Acceptance**
1. Hybrid endpoint returns a fused ranking from dense + KG signals.
2. Ablation shows each signal independently changes the ranking (neither is dead weight).
3. Weights are config-tunable (incl. a KG-heavy preset); change in weight provably changes order.
**Branch**: `track/r3-hybrid-ranker`. **Depends on**: R.1, R.2.

## Phase R.4 — Cross-source cluster summaries, embedded + reused (Backend)
**Why**: the user's "notebooks make overarching summaries to avoid rebuilding info." Today RAPTOR is
per-source and summaries are neither cross-source nor embedded nor reused.
**Deliverables**: repoint/extend clustering to run **across sources within a notebook**; generate
notebook-level cluster summaries; **embed them** (768-dim); persist + cache (recompute only on
membership change); feed summaries into the hybrid ranker and as reusable context for chat/summarization.
**Acceptance**
1. Cluster summaries computed across ≥2 sources in a notebook; persisted with membership + embedding.
2. Re-run with unchanged membership = cache hit (no recompute); changed membership recomputes.
3. Cluster-summary embedding contributes a signal to the R.3 ranker (ablatable).
**Branch**: `track/r4-cluster-summaries`. **Depends on**: R.0 (embedding infra), R.3 (fusion seam).

## Phase R.5 — Search-function integration + UI (Integration + UI)
**Why**: wire hybrid retrieval into the actual search the user cares about, KG-forward.
**Deliverables**: search endpoint/UI uses the hybrid ranker; related-sources surfaced on the source view; result rows show *why* (matched entities / similar chunks / shared cluster); E2E.
**Acceptance**
1. Search returns hybrid-ranked sources with per-result provenance (dense vs KG vs cluster contribution).
2. a11y + loading/empty/error states; E2E covers a known query.
**Branch**: `track/r5-search-ui`. **Depends on**: R.3 (R.4 optional-but-preferred).

## Phase R.6 — Extraction noise re-scope (Backend) — *keep the KG, trim the noise*
**Why**: the KG is noisy; but it's load-bearing for search (R.2), exports, and resolution. Trim
noise for the linking path **without** a teardown.
**Deliverables**: down-rank/suppress `other` + singleton + duplicate-predicate noise in the
search-facing graph projection; dedupe EN/NL predicate variants + typos (`ACEPTS`→`ACCEPTS`,
`IS_PIJLER_VAN`/`LEIDT_TOT` canonicalization); keep typed entities + exports + triage intact.
**Acceptance**
1. Measurable noise reduction in the search-facing projection (e.g. % generic-bucket in ranked results drops) with before/after counts.
2. Export (Obsidian/NetworkX/JSONL), entity resolution (K), and triage (Q) regression suites stay green — nothing torn out.
3. Predicate-canonicalization is reviewable + reversible.
**Branch**: `track/r6-noise-rescope`. **Depends on**: R.2 (so noise rules are evaluated against the search signal).

## Phase R.7 — Integration, telemetry, docs, RETRO (Integration)
**Deliverables**: end-to-end hybrid-retrieval telemetry (per-signal contribution), ARCHITECTURE
section, roadmap entry (Track R), status/RETRO; final audit.
**Acceptance**: E2E hybrid search green; per-signal telemetry emitted; docs updated; final audit clean.
**Depends on**: R.5, R.6.

---

## Open questions for the user (before execution)
1. **KG weight default** — should the shipped default ranker be KG-heavy, or balanced-with-a-KG-preset? (You said KG should play a *large* role — confirm "large by default" vs "tunable, high preset available".)
2. **Cluster-summary scope** — per-notebook only, or also global/cross-notebook?
3. **Embedding model** — stay on the local 768-dim pin for chunk/source/cluster embeddings, or is a higher-dim cloud embedding (via Track J routing) on the table for retrieval quality?
4. **Sequencing vs O/P** — run Track R after O+P close, or interleave (R.2/R.6 are KG-only and need no live writes)?
5. **Orphan chunks (gates R.0 backfill)** — 9,053 of 10,501 staging chunks are orphaned from deleted sources. Were those 40 source deletions intentional (a reset) → GC the orphans; or accidental → attempt re-association? Need your call before any GC.
6. **768→1024 reconciliation** — the "768-dim I.G pin" is referenced across `route_resolver.py`, `model_routing/`, `entity.py`, `jsonl_export_service.py`, `backfill_entity_embeddings.py`, and guardrail tests, but live reality is **1024** (mxbai-embed-large). Confirm whether 768 was ever a real constraint or always a doc error — R.0 reconciles the touched references; a full sweep may warrant its own small task.
