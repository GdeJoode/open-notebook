# Phase I.F — adversarial review (attempt 1)

> Branch: `track/i-f-structure-graph`, diff `main...HEAD` (6 commits)
> Reviewer: adversarial-reviewer agent
> Plan: `docs/tracks/I-docling-studio/plan.md` §I.F (7 ACs)
> Date: 2026-06-19

## VERDICT: REVISIONS_NEEDED → resolved

Solid, honest work: the chunk-derived design is well-justified (docling JSON is transient
LangGraph state, never persisted), the migration matches the plan, cross-track B collision is
genuinely PASS, tests pass for real. Two MAJORs fixed.

## Findings

### 1. MAJOR — duplicate/null `order` collides leaf self_ref → silent total graph loss
`doc_graph_builder.py:246` used `#/chunks/{order}`. Two chunks with the same (or missing) `order`
→ same self_ref → violates the UNIQUE `(source, self_ref)` index → `_insert_nodes` raises → the
best-effort hook swallows it → the source gets NO structure graph, silently. (The happy regrouper
path yields unique order, so latent, but zero defense + invisible failure.)
*Fixed:* leaf ref now derives from the deterministic loop index (`#/chunks/{seq}`); the sort gained a
stable id tiebreaker; added `test_duplicate_order_yields_distinct_leaf_refs`.

### 2. MAJOR — node click rebuilds the entire Sigma WebGL graph
`StructureGraphView.tsx:159` had `selectedChunkId` in the build effect deps; the effect does
`sigma.kill()` + full reconstruction. Every highlight tore down the canvas — O(nodes) + flicker,
worst exactly on large graphs.
*Fixed:* highlight moved to a separate lightweight effect (`setNodeAttribute` + `sigma.refresh`)
against a `graphRef`; build effect no longer depends on selection.

### Minors (deferred as follow-ups)
1. AC5 click→bbox highlight has no headless assertion (wiring verified by inspection).
2. `total_nodes` = returned count, not pre-truncation total (UI understates on truncation). A count
   query would ripple into mocked tests for a non-blocking nit — deferred.
3. Idempotency test is in-process only; ordering is deterministic by construction (first-appearance
   over an id-tiebroken order sort), so it holds, but a hard-coded signature would catch regressions.

## High-risk spot assessment
1. Chunk-derived tree — PASS (section dedup correct; parent_of nests; next_node is reading-order chain).
2. Idempotency — PASS w/ caveat (deterministic + delete-then-rebuild; MAJOR 1 was the risk, now fixed).
3. derived_from coverage — PASS (every chunk with an id gets an edge; flat sources don't drop coverage).
4. Migration 49 — PASS (forward matches plan; down `REMOVE TABLE` ×4 fully reverses; mirrors 39/48).
5. Orchestrator hook — PASS (runs only if chunks exist; try/except logs + continues; best-effort).
6. API router — PASS (422 above cap before reader; bounded LIMIT; params bound; empty graph not 500).
7. Frontend wiring — PASS on wiring/tsc; MAJOR 2 perf (now fixed).
8. Tests real — MOSTLY PASS (assert concrete structure; gaps were duplicate-order [now added] + in-process idempotency).

## AC scorecard
- AC1 migration apply/revert: live-deferred (PASS by inspection; no SurrealDB in sandbox).
- AC2 200+ nodes/depth≥3/chains: computed-verified on fixture (255 nodes, depth 4); live counts deferred.
- AC3 derived_from ≥90%: computed-verified (100% on fixture; flat sources covered).
- AC4 API <200ms: live-deferred (bounded query exists).
- AC5 click→bbox highlight: wired (verified by inspection); no headless assertion.
- AC6 re-ingest idempotency: computed-verified + now collision-safe; live delete-rebuild deferred.
- AC7 page_limit cap 500→422: fully verified.

## Cross-track B collision: PASS (doc_node/parent_of/next_node/derived_from only in migration 49).

## Validation (post-revision)
27 builder/router tests pass (was 26); ruff clean; tsc exit 0.

---

## Attempt 1 — revisions

| # | Severity | Resolution | Commit |
|---|---|---|---|
| 1 | MAJOR | Deterministic `#/chunks/{seq}` leaf ref + id-tiebroken sort + duplicate-order test. | `c228539` |
| 2 | MAJOR | In-place highlight effect (setNodeAttribute + refresh), build effect no longer rebuilds on select. | `c228539` |
| 3-5 | Minor | Deferred as documented follow-ups (AC5 headless assertion, total_nodes semantics, in-process idempotency test). | — |

> Live-DB ACs (1/4/6 row counts + latency) remain deferred to a run with a running SurrealDB — no DB in
> this sandbox. The builder computation, migration SQL, router guards, and frontend wiring are verified.
