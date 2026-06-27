# Track U — Document graph (documents as first-class nodes)

> **DRAFT for user review — nothing built yet.** From the user's proposal: make the KG
> contain DOCUMENTS as nodes, connected (a) **via the entities they share** and (b)
> **directly** (e.g. citations). Key finding: the schema for this ALREADY EXISTS — the
> ontology (`packages/ontology-manager/ontologies/schema_core.yaml`) defines `cites`,
> `mentions`, `discusses`, `authored_by`, `implements`, `leads_to`, `supersedes` edges —
> but they are EMPTY (0 rows). Only `relation` (entity→entity, 1466) is populated. Track U
> fills the document-centric layer the design already anticipates.

## Current state (staging, measured 2026-06-27)
- **Nodes**: `source` (6, documents), `entity` (4870), `doc_node` (structure), `topic`/`person`/… .
- **Populated edges**: `relation` entity→entity (1466); `parent_of`/`next_node` doc_node→doc_node (~1400, Track I structure); `reference` source→notebook (6).
- **EMPTY but defined** (the target): `mentions` source→entity, `cites` source→source, `discusses` source→topic, `authored_by` source→person, `implements`/`leads_to`/`supersedes` source→source.
- Documents↔entities are linked TODAY only implicitly via the array `entity.source_documents` (e.g. "Regio Deal" → the 4 convenanten), not via edges.
- A KG visualization already exists: `frontend/.../knowledge-graph/` (`SigmaGraphView.tsx`) + `apps/app-main/.../routers/knowledge_graph.py`.

## The three document-connection types (different nature → different treatment)
1. **Via shared entities** (`mentions` paths: source→entity←source). DERIVED — **R.2 already
   computes this** as the search signal (salience × rarity). Materializing `mentions` edges
   makes it a real, traversable/drawable/exportable graph, but adds NOTHING to search. Value =
   visualization, export (NetworkX/Obsidian), graph algorithms, traversal.
2. **Content similarity** (embeddings). CONTINUOUS signal — **R.1 already**. An edge needs a
   threshold (throws information away). Keep computed; optionally materialize top-k `related_to`
   edges PURELY for the drawing.
3. **Direct citations** (`cites`: source→source). The genuinely NEW information — discrete facts
   that live nowhere else. Highest content value. Needs citation extraction.

## Core trade-off (decide at review)
Two complementary layers now exist: the **computed** layer (R.1/R.2/R.3 — efficient for search,
on-the-fly, no upkeep) and the **materialized** layer (the empty edge tables — needed for navigate/
draw/export/algorithms, but you pay sync upkeep: edges must be refreshed when entities/documents
change/re-extract). Materialize only where viz/export/traversal justifies it, or where the info is
genuinely NEW (citations) — don't double-store what you already cheaply compute.

**Workflow**: track methodology — `implementer` → `adversarial-reviewer`. Main tree, `uv run pytest`,
no worktree. Additive/reversible where possible; live writes gated; canonical entity/relation data untouched.

---

## Phase U.1 — Design + measurement (Discovery, read-only) — DECISION GATE
**Why**: pin the exact projection source, citation data reality, edge volume, and the sync model before writing edges.
**Deliverables**: a report — confirm `mentions` projects 1:1 from `entity.source_documents` (count the edges it would create; on staging "Regio Deal"→4 alone, so estimate total + the over-connection risk from generic/singleton entities — coordinate with R.6's salience/singleton rules); confirm what citation data the sources actually carry (references/bibliography in `full_text`; the 2 papers vs the 4 policy convenanten differ — measure); decide the sync model (regenerated projection vs persisted+maintained); pick edge-weight/threshold rules so the graph doesn't explode.
**Acceptance**: edge-volume estimates per type; citation-data feasibility per source class; a recommended sync model + thresholds; go/no-go per later phase (U.1 can recommend dropping a phase if data doesn't support it — e.g. if no parseable citations exist).
**Depends on**: none.

## Phase U.2 — `mentions` edge projection (Backend)
**Why**: make the document↔entity bipartite graph real (traversable/drawable/exportable) — cheap, no LLM.
**Deliverables**: project `mentions` (source→entity) edges from `entity.source_documents`, carrying a weight (reuse R.2's salience × rarity so the graph is pre-weighted). Idempotent regenerator (rebuild on demand; reversible — it's a derived view, canonical data is the array/extraction). Apply R.6's normalization (drop singletons / down-weight generics) so the projected graph isn't dominated by noise.
**Acceptance**: `mentions` edges created from the array (count matches the U.1 estimate); each carries a weight; regenerating is idempotent (no dup edges); singleton/generic noise handled; canonical `entity`/`source` rows untouched; a `document → entity → document` traversal returns the convenant cluster.
**Depends on**: U.1.

## Phase U.3 — `cites` extraction (Backend) — the new information
**Why**: documents that cite each other within the corpus — discrete facts not captured anywhere.
**Deliverables**: extract references from each source's `full_text` (the bibliography/reference section), match them against the in-corpus sources (title/author/DOI fuzzy match — reuse Track K resolution discipline + a precision guard so a wrong match doesn't fabricate a citation), and create `cites` (source→source) edges only on a confident match. Confidence-scored; reviewable; only intra-corpus citations (external refs noted but not edged, or edged to a stub — U.1 decides).
**Acceptance**: `cites` edges created only on confident intra-corpus matches (precision over recall — a fabricated citation is worse than a missing one); each carries confidence + the matched reference text; measured on the 2 papers (most likely to carry formal citations); no false self-citation; exports/triage unaffected.
**Depends on**: U.1.

## Phase U.4 — Document graph in the KG visualization (UI)
**Why**: show the document graph the user described — sources as nodes, linked via `mentions` paths + direct `cites`.
**Deliverables**: extend the existing `SigmaGraphView`/`knowledge_graph.py` to render a document-centric view: source nodes, `mentions` (source↔entity) and `cites` (source↔source) edges, with the entity layer toggleable; optionally a top-k `related_to` embedding layer (thresholded, draw-only). Per-edge "why" (shared entity name / citation). a11y + states.
**Acceptance**: the KG viz can show documents as nodes connected via shared entities and via citations; entity layer + embedding layer are toggleable; edges show their basis; renders the convenant cluster; E2E covers the document view.
**Depends on**: U.2 (+ U.3 if it lands).

## Phase U.5 — Integration: exports + telemetry + docs + RETRO
**Deliverables**: richer NetworkX/JSONL/Obsidian exports including the document↔document + document↔entity edges; ARCHITECTURE note on the computed-vs-materialized layering; RETRO. **Depends on**: U.2–U.4.

---

## Risks & open decisions for the user
1. **Sync upkeep** — materialized `mentions`/`cites` must be refreshed on re-extraction/merge. Regenerated-projection (cheap, stateless) vs persisted-and-maintained (needs triggers). U.1 recommends; lean regenerated for `mentions`, persisted for `cites` (it's real data).
2. **Graph explosion** — `mentions` over generic/singleton entities would create a hairball. Reuse R.6 (drop singletons, down-weight generics) + a weight threshold. Decide the default.
3. **Citation precision** — fuzzy reference-matching can fabricate edges. Precision-first (confident matches only); is intra-corpus-only acceptable, or also link to external/stub nodes?
4. **Does U add search value, or only viz/export?** Honest framing: search already works (R.1–R.3). U's value is navigation/visualization/export/graph-algorithms + the NEW citation facts. If those aren't priorities, U could stop at U.3 (citations as data) without the viz.
5. **Sequence** — U.1 first (measure), then decide which of U.2/U.3/U.4 to build.
