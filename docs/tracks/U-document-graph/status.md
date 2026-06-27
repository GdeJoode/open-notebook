# Track U — status

## Phase U.2 — `mentions` edge projection (Backend) — APPROVED

**Branch**: `track/u2-mentions` (off `main` @ `006b953`)
**Date**: 2026-06-27
**Review**: APPROVED (`reviews/phase-U.2-attempt-1.md`) — 0 blockers, 0 majors.
The reviewer falsification-tested the migration-66 safety claim (patched in a
blanket DELETE → confirmed edge loss → the preservation test genuinely catches
it). Three minor follow-ups were filed and have since been fixed in
`559722f`: (1) `relate_mention` docstring now states it raises on transport
error; (2) `regenerate` telemetry (`emitted_concepts`/`active_entities`) now
comes from the projection stats — accurate under a `min_weight` cutoff;
(3) `regenerate` loads the active entities once (via `_project`) instead of
twice. Live regenerate remains correctly gated (staging `mentions` still 0 rows,
still `TYPE ANY` — migration 66 not yet applied to staging).

### What was built
The document↔entity bipartite graph is now real: `mentions` (source→entity)
edges are a regenerated, idempotent projection of `entity.source_documents`,
carrying the SAME R.2 weight and keeping the SAME R.6-filtered entities as the
search signal — so the drawn graph matches search by construction. No LLM, no
canonical-data mutation.

1. **Pure projection** — `packages/shared/src/shared/retrieval/mentions_projection.py`
   (`project_mentions_edges`). Reuses R.2 `entity_weight` (salience × rarity) and
   R.6 `normalize_entities_for_signal` (case/type unify + df==1 singleton drop).
   Each surviving concept maps back to its max-salience **representative entity
   id** so an edge anchors a real `entity` row while the weight uses the unified
   concept's type + df. `min_weight` (default 0.0) + 0.3 `named_only` preset.
   Pure, deterministic, no I/O. 20 unit tests.
2. **Regenerator** — `MentionsProjectionService`
   (`apps/app-main/src/app_main/services/mentions_projection_service.py`) +
   `EntityRepository` seam (`clear_mentions`, `relate_mention`, `count_mentions`,
   `load_mentions_edges`). `regenerate()` clears + RELATEs idempotently;
   `project_edges()` is a write-nothing dry-run. 7 `@requires_docker` tests.
3. **Migration 66** — `migrations/66.surrealql` (+ `_down`, no-op). Defines
   `mentions` as `TYPE RELATION FROM source TO entity` (it had drifted to
   `TYPE ANY SCHEMALESS` on staging). Non-destructive OVERWRITE (migration-62
   strategy); null-endpoint-only DELETE (a blanket DELETE wiped healthy edges —
   caught by the test). Empty table ⇒ no strict-field backfill needed (S.4 note
   in-migration). 4 `@requires_docker` tests.
4. **Endpoints** — `knowledge_graph.py`:
   `GET /knowledge-graph/document-graph` (fetch edges for U.4 viz; `min_weight`
   + `source_id` scope) and `POST /knowledge-graph/document-graph/regenerate`
   (`dry_run`, `named_only`, `min_weight`, `drop_singletons`). Wired into DI.

### `mentions` table state
Staging probe (`INFO FOR DB`, 2026-06-27): `mentions` was
`DEFINE TABLE mentions TYPE ANY SCHEMALESS` with **0 rows** — NOT a relation
table (siblings `cites`/`discusses`/`authored_by` were already correct
`TYPE RELATION`). **Migration 66 added** to fix the drift. SurrealDB 2.6.5 on
both staging and the testcontainer.

### Projection weights
`weight = type_salience(concept_type) × IDF(df, N_sources)` — the R.2
`entity_weight` verbatim, computed on the UNIFIED concept (so case/type
duplicates count as one). Staging-measured: min 0.200 / median 0.234 / max 1.336
("Regio Deal" programme & "Regio" area, df=4). Per-edge `concept_name` /
`concept_type` / `document_frequency` carry the "why".

### Per-criterion evidence (staging DRY-RUN, read-only)
| AC | Evidence |
|---|---|
| AC1 edges from array, count = U.1 estimate | **67** edges (R.6 on, default), 25 entities, 4 source nodes — exact U.1 match. 455 with singleton-drop off (≈ the 466 active raw). Filtering is the default. |
| AC2 each edge weighted | Top weights "Regio Deal"/"Regio" = 1.336; every edge > 0. |
| AC3 idempotent, no dup edges | Container test: 2nd regenerate clears exactly what 1st created; identical (source,entity,weight) set; no duplicate pairs. |
| AC4 singleton/generic noise handled | df==1 spoke dropped (466→67); generics down-weighted (~6×), not torn out. |
| AC5 canonical rows untouched | Container test snapshots entity+source rows before/after regenerate — byte-identical. |
| AC6 traversal returns K4; papers isolated | Container test `->mentions->entity<-mentions<-source` reaches all convenanten; entity-less source never reached. Named-only preset = 8-edge "Regio Deal/Regio" skeleton. |

### Live regenerate: LEFT GATED
Performed a **read-only dry-run only** against staging (67 edges projected,
0 written). The `mentions` table on staging is still at **0 persisted edges**.
The live regenerate (additive/idempotent on the empty table — safe) is left for
the user to run as the gated step:
`POST /knowledge-graph/document-graph/regenerate` (or the dry-run script
`scratchpad/u2_dryrun.py`, which writes nothing).

### Tests
- `packages/shared` — 490 passed (incl. 20 new projection unit tests).
- `packages/surrealdb-service` — migrations roundtrip + 62 + 66 = 29 passed.
- `apps/app-main` — 7 new `@requires_docker` regenerate tests + existing KG
  router/service (25) green.
- mypy clean on the new modules.

### Notes for U.4
`GET /knowledge-graph/document-graph` returns `{edges, count}` where each edge is
`{id, source, target, weight, concept_name, concept_type, document_frequency}` —
ready to render with weight→thickness/opacity and `concept_name` as the per-edge
"why". The 0.3 `min_weight` slider value is the named-only overview.
