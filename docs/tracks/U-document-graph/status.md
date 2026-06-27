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

---

## Phase U.4 — Document graph in the KG visualization (UI) — READY FOR REVIEW

**Branch**: `track/u4-document-graph-viz` (off `main` @ `f07d6b4`)
**Commits**: `1339d08` → `35b04df` → `62d272b` → `6dd0d9e`
**Date**: 2026-06-27

### Component approach — sibling, not a toggle on SigmaGraphView
A new **`DocumentGraphView`** component renders under a new **"Document Graph"**
tab on the existing KG page, alongside the entity-only "Graph" tab. It reuses
the same Sigma + forceAtlas2 + dynamic-import (`ssr:false`) machinery, design
tokens, and click-to-side-panel wiring, but is a *sibling* rather than a mode on
`SigmaGraphView`. Rationale: the document graph is **bipartite** (two node kinds,
distinct sizing/labels), needs a **source-title join** the entity graph never
does, drives **edge thickness from weight**, and supports an **entity-layer
collapse** — overloading the entity-only view with all of that would muddy a
working component. The pure graph-construction logic lives in a separate,
node-testable module so the React/WebGL layer stays thin.

### What was built
1. **API + types** — `getDocumentGraph({min_weight, source_id})` typed to the
   U.2 `{edges, count}` shape (`DocumentGraphEdge`/`DocumentGraphResponse`);
   `useDocumentGraph(minWeight)` + `useAllSources()` hooks (the latter joins
   `source:` ids → titles, since the edge payload carries only ids).
   `frontend/src/lib/api/knowledge-graph.ts`, `.../hooks/use-knowledge-graph.ts`.
2. **Pure graph builders** —
   `frontend/src/lib/knowledge-graph/document-graph.ts`:
   `buildBipartiteGraph` (documents + concepts, one `mentions` link/edge),
   `buildCollapsedGraph` (entity layer hidden → doc↔doc links, one per shared
   concept, summed weight + concept list as the "why"), `summarizeGraph` (the
   a11y summary). Isolated documents are injected from the source list so the
   two papers (0 active entities) still appear. **11 vitest unit tests.**
3. **`DocumentGraphView`** —
   `frontend/src/app/(dashboard)/knowledge-graph/components/DocumentGraphView.tsx`.
   Documents = indigo nodes prefixed with a ◆ glyph; concepts = amber nodes.
   Edge thickness ∝ weight; edge label = shared concept(s). Controls: "Show
   shared concepts" toggle (bipartite↔collapsed), a **min-weight slider**
   (0–1.4) + a **Named-only preset** button (snaps to 0.3), Reload.
4. **States** — loading skeleton (`document-graph-loading`), friendly empty
   (`document-graph-empty`, hints to `POST …/regenerate`), error+retry
   (`document-graph-error`). Empty/error never crash the canvas.

### WebGL a11y fallback
Sigma renders to a WebGL canvas that is neither screen-reader nor keyboard
navigable, so the canvas is paired with **non-canvas equivalents**:
- an `aria-live` text **summary** ("N documents, M shared concepts, K links")
  with an isolated-document callout, wired as the `figure`'s `aria-labelledby`;
- a keyboard-reachable **"Links as a list"** `<details>` enumerating every link
  with its shared-concept "why" (the same per-edge basis the canvas shows);
- the document-vs-concept distinction is **glyph + label + size**, not colour
  alone — the legend names the ◆ glyph and the colour. All controls are native
  Switch/Slider/Button (focusable, aria-labelled / aria-pressed).

### Per-criterion evidence
| AC | Evidence |
|---|---|
| AC1 docs as nodes via shared entities; layer toggleable; weight visible; isolated shown | Screenshot + E2E: 2 convenanten linked via "Regio Deal"/"brede welvaart", thickness ∝ weight; toggle collapses to a single doc↔doc link; the paper renders isolated (callout), not dropped. |
| AC2 per-edge basis surfaced | Edge labels + the link-list show "via Regio Deal" / "via Regio Deal, brede welvaart". |
| AC3 loading/empty/error | Skeleton; empty → regenerate hint; error → retry alert. E2E asserts empty + error don't crash. |
| AC4 a11y fallback, keyboard controls, not colour-only | aria-live summary + link-list + ◆ glyph/label; native focusable controls. |
| AC5 E2E + build/typecheck/lint/vitest clean | See below. |

### Gate results
- **vitest**: 98 passed (12 files), incl. 11 new document-graph unit tests.
- **typecheck** (`tsc --noEmit`): clean.
- **lint** (`next lint`): no new errors/warnings in U.4 files (pre-existing
  warnings elsewhere untouched).
- **build** (`next build`): success.
- **E2E** (`e2e/track-u/document-graph.spec.ts`, 4 tests): all pass against a
  built **standalone** server on :8599, fully route-mocked (no backend/DB).

### Notes / follow-ups
- The build uses `output: standalone`; the E2E harness must serve via
  `node .next/standalone/server.js` (after `npm run standalone-prep`), **not**
  `next start`, or the app renders blank.
- **Pre-existing, unrelated**: `e2e/track-q/triage.spec.ts:138` fails on a
  strict-mode selector ("unsure — operator decides" matches a heading *and* a
  cell) on the `/knowledge-graph/triage` route — untouched by U.4. Not fixed
  here (out of scope); flagged for the track-q owner.
- `cites` layer deferred with U.3 (0 intra-corpus citations per U.1) — the view
  is mentions-only by design. The optional draw-only `related_to` embedding
  layer for the isolated papers was not built (out of U.4 acceptance scope).

---

## Phase U.3 — `cites` extraction (Backend, INFRASTRUCTURE) — READY FOR REVIEW

**Branch**: `track/u3-cites-infra` (off `main` @ `eac6fa2`)
**Date**: 2026-06-28
**Nature**: infrastructure — builds the `cites` (source→source) materialization
mechanism + the Track V input interface. Per U.1 there are **0 intra-corpus
citations** on this corpus and Track V (which feeds references) is unbuilt, so
this ships **tested on SYNTHETIC references with 0 live edges**. On staging the
mechanism runs and produces a clean **0-edge no-op** (verified read-only).

### The Track V boundary — `ParsedReference`
The single documented contract between Track V (produces) and U.3 (consumes),
a frozen dataclass in `packages/shared/.../retrieval/cites_matching.py`:

```
ParsedReference(raw_text: str, title: str = "",
                authors: tuple[str, ...] = (), year: int | None = None,
                doi: str | None = None, venue: str | None = None)
```

`raw_text` is always populated (it is what a created `cites` edge records as the
human-readable citation). All other fields are best-effort; the matcher uses
whatever is present. V hands U.3 `{source_id: [ParsedReference]}`.

### Matching precision rules (a wrong match fabricates a citation → worse than a miss)
- **DOI exact** (normalized: lowercased, `doi:`/URL prefix stripped) → certain
  match, confidence `1.0`, method `doi`. Two empty DOIs never match.
- **Title + author** (no-DOI fallback) → requires BOTH `title_sim ≥ 0.85`
  (normalized Levenshtein) AND author-surname agreement `> 0`, with combined
  confidence `≥ 0.80` (blend `0.7·title + 0.3·author`), method `title_author`.
  Title-only or author-only is rejected.
- **Ambiguity guard** — a fuzzy winner must beat the runner-up by `≥ 0.05`; two
  viable fuzzy targets → no edge. A DOI winner is blocked only by another DOI hit.
- **No self-citation** — `from_source_id` excludes the origin source; a final
  `relate_cites` backstop refuses any `src == tgt` RELATE.

### Migration 67 — needed (the U.1 contingency fired)
U.1 reported `cites` was already `RELATION source→source` **on staging** — true,
but it was defined there at runtime by the relation write path, NOT by the
migration suite. On a **fresh migration-only container** `cites` comes up as the
default `TYPE ANY SCHEMALESS` (proven by the container test) — the same drift
class migration 66 fixed for `mentions`. So **migration 67 was required**: it
asserts `DEFINE TABLE OVERWRITE cites TYPE RELATION FROM source TO source`
non-destructively (null-endpoint `DELETE` + `OVERWRITE`, healthy edges preserved,
mirroring 66) AND adds the U.3 fields (`confidence`, `reference_text`,
`match_method`, `created_at`) the SCHEMAFULL table would otherwise silently drop
on RELATE-with-SET. S.4 prevention: every added field is strict-typed WITH a
DEFAULT; the table is empty; the regenerator supplies values explicitly.

### Per-criterion test evidence
| AC | Evidence |
|---|---|
| AC1 confident intra-corpus → edges w/ confidence + ref text | `test_confident_intra_corpus_refs_become_cites_edges` (DOI + title_author edges, both carry confidence + reference_text + match_method). Pure: `test_doi_exact_match_*`, `test_fuzzy_title_author_match_above_threshold`. |
| AC2 external → no edge; no self-citation | `test_external_and_self_citation_produce_no_edge` (created=0, self_skipped=1, external=1). Pure: `test_no_self_citation`, `test_external_reference_no_match`. |
| AC3 near-miss rejected (precision over recall) | `test_near_miss_reference_rejected` (right author, different title → created=0). Pure: `test_near_miss_title_below_threshold_no_edge`, `test_title_match_but_wrong_author_is_rejected`, `test_ambiguous_two_viable_targets_no_edge`. |
| AC4 idempotent regenerator | `test_materialize_is_idempotent` (re-run: identical edge set, cleared==created, no dup pairs). Pure: `test_matching_is_deterministic`. |
| AC5 cites is `RELATION source→source` | `test_migration_67_discovered_and_cites_is_source_to_source` (asserts `IN source OUT source`); `test_drifted_any_cites_converted`; `test_healthy_cites_edges_preserved`. |
| AC6 ParsedReference documented; canonical rows untouched | `ParsedReference` docstring + this status; `test_canonical_source_rows_untouched`. |
| AC7 staging no-op, no crash | Read-only staging probe: 6 sources load, `match_corpus_references({}, sources)` → 0 matches, no writes. `test_empty_input_is_clean_no_op`. |

### Test runs
- **Pure matcher** (`packages/shared/tests/test_cites_matching.py`): **16 passed**.
- **Migration 67** (`packages/surrealdb-service/tests/test_migration_67_cites_fields.py`): **5 passed** (`@requires_docker`).
- **Materialization** (`apps/app-main/tests/test_cites_materialize_db.py`): **7 passed** (`@requires_docker`).
- **No regression**: `test_mentions_projection.py` + `test_cites_matching.py` → 36 passed.
- **mypy**: pure module clean; service has only the pre-existing workspace
  `import-untyped` notes (shared/surrealdb_service lack `py.typed`), identical to
  the U.2 mentions service baseline — no new type errors.

### Honest framing / follow-ups
- **0 live edges by design.** The mechanism is correct and tested but draws
  nothing until Track V feeds real references AND source bibliographic metadata
  (`authors`/`doi`) is populated. The `source` table is SCHEMAFULL with no
  top-level `authors`/`doi`; the loader reads them from the FLEXIBLE `metadata` /
  `type_metadata` / `external_ids` objects (a Zotero/Docling enrichment home),
  empty on this corpus → the title-only precision guard correctly declines.
- **External references** are counted + logged, not edged (kept simple per the
  brief). Track V may later stub them as external nodes — the `CorpusMatchResult`
  already separates `external` / `ambiguous` / `self_citations_skipped` counts.
- No router/endpoint was added (out of the infrastructure brief); the service is
  DI-wired (`get_cites_materialization_service`) and ready for U.4/U.5 to consume.
