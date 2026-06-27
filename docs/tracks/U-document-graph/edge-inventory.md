# Track U.1 — Edge inventory + decision gate

> **Phase U.1 deliverable (DECISION GATE).** Read-only discovery — no code changed,
> no DB writes. All numbers measured against **staging** (`ws://localhost:8000`,
> ns `open_notebook`, db `staging`, root/root) on **2026-06-27**.
> Sanity check passed: **6 sources · 4870 entities · 1466 relations**
> (`SELECT count() FROM source|entity|relation GROUP ALL`).
>
> Measurement scripts (scratchpad, not committed) reused the production scoring
> code verbatim: `packages/shared/src/shared/retrieval/kg_source_scorer.py`
> (R.2 salience × rarity) and `…/kg_signal_normalizer.py` (R.6 case/type grouping
> + singleton drop). Numbers below therefore reflect exactly what U.2/U.4 would draw.

---

## Corpus at a glance

| Source id | Title | full_text chars | Class | Active entities |
|---|---|---:|---|---:|
| `source:hjrb1cjvzv86oyaojblj` | Convenant Regio Deal Noord-Holland Noord | 71 376 | convenant | 151 edges |
| `source:dndibxmjveoxk7tfqfsl` | Convenant Regio Deal Midden-Limburg | 68 083 | convenant | 110 edges |
| `source:052dtl7jrwu1czlpnui4` | Convenant Regio Deal Zuidwest-Friesland | 67 341 | convenant | 118 edges |
| `source:0o96pew25fovpoom5n6r` | Convenant Regio Deal Het Hogeland | 48 203 | convenant | 87 edges |
| `source:bc6xax5piw8cntg5nomc` | JCMS 2025 — Ali, *Cohesion Policy & Institutional Quality* | 127 060 | paper | **0** |
| `source:1k3c41i456l7ywu6y7iv` | Brakman & Garretsen, *Economics without equilibrium* | 37 605 | paper | **0** |

Entity status mix (whole DB): **423 active · 3313 archived · 1055 reference · 79 merged**.
The R.2/R.6 signal — and therefore the `mentions` projection — runs on the **423
active** rows only (archived/reference/merged are excluded upstream, see the
`kg_source_scorer` module docstring "Only `status='active'` entities are scored").

**Load-bearing structural fact:** the **two academic papers carry 0 active
entities** (the Ali paper has 421 entity rows, all archived; Brakman has 0 rows at
all). So on this corpus the document↔entity layer is **entirely the four
convenanten** — the papers are isolated nodes in any `mentions` view.

---

## Measurement 1 — `mentions` edge volume (source → entity)

`mentions` projects 1:1 from `entity.source_documents` (one edge per
(entity, source) membership). Counts:

| Projection | Edges | Notes |
|---|---:|---|
| **All entity rows** (incl. archived/reference) | **5 398** | 147 rows have empty `source_documents` |
| **Active only** (the real candidate set) | **466** | 0 active rows with empty `source_documents` |
| **Active, R.6-normalized, drop df==1 singletons** | **67** | the drawable graph |

### Active-edge breakdown by entity type (raw, pre-singleton-drop)

| entity_type | edges | entities | TYPE_SALIENCE |
|---|---:|---:|---:|
| `topic` | 377 | 344 | 0.15 (generic) |
| `administrative_area` | 37 | 35 | 1.0 (named) |
| `programme` | 29 | 24 | 1.0 (named) |
| `government_organization` | 17 | 17 | 1.0 (named) |
| `concept` | 6 | 3 | 0.15 (generic) |

**81 %+ of the active edges are generic `topic`/`concept` (383 of 466).** This is the
exact "LLM-noise majority" the R.2 salience table is built to down-weight
(`kg_source_scorer.py` lines 67–112), and it is precisely the hairball risk
flagged in plan.md Risk 2.

### R.6 filtering (the legibility step)

Running `normalize_entities_for_signal(active, drop_singletons=True)`:

```
input_entities    = 423
grouped_concepts  = 413   (10 merged_groups — case/type duplicates unified)
singletons_dropped= 388   (df==1 concepts that can't link two sources)
emitted_concepts  =  25
FILTERED mentions edges (concept→source, df>=2) = 67
```

So **388 of 413 concepts appear in only one source** and are dropped (they would be
leaf spokes touching a single document — pure visual noise). The drawable graph is
**25 concepts / 67 edges across the 4 convenanten** — eminently reasonable to draw.

### Over-connection / generic-dominance (top concepts by df, post-R.6)

| df | type | salience | weight | concept |
|---:|---|---:|---:|---|
| 4 | programme | 1.00 | 1.336 | **Regio Deal** |
| 4 | administrative_area | 1.00 | 1.336 | **Regio** |
| 4 | topic | 0.15 | 0.200 | brede welvaart |
| 3 | topic | 0.15 | 0.234 | gezondheid, energietransitie, energie, samenredzaamheid, … |
| 3 | concept | 0.15 | 0.234 | jongeren |
| 2 | topic | 0.15 | 0.277 | circulariteit, wonen, vergrijzing, … |

Only **2 high-salience concepts** ("Regio Deal" programme, "Regio" area) anchor the
cluster; the rest are low-salience topics that the weight already discounts ~6×
(0.20–0.28 vs 1.34). The graph is **not** dominated by a single super-hub — the
densest node ("Regio Deal") touches 4 sources, which is the whole corpus.

**Verdict: the filtered `mentions` graph is reasonable to draw** (25 nodes / 67
edges / 4 source nodes), and R.6's singleton-drop is what makes it so (466 → 67).

---

## Measurement 2 — Citation data reality (the load-bearing U.3 measurement)

Parsed each source's `full_text` for a references/bibliography section and for
intra-corpus citations.

| Source | Class | Ref-section header | DOIs | `(YYYY)` cites | Reality |
|---|---|---|---:|---:|---|
| Zuidwest-Friesland | convenant | — none — | 0 | 0 | no bibliography |
| Het Hogeland | convenant | — none — | 0 | 2 | no bibliography |
| Midden-Limburg | convenant | — none — | 0 | 0 | no bibliography |
| Noord-Holland Noord | convenant | — none — | 0 | 0 | no bibliography |
| Brakman & Garretsen | paper | **"References" ✓** | **20** | 53 | real bibliography |
| Ali (JCMS 2025) | paper | **"References" ✓** | **12** | 105 | real bibliography |

The **4 convenanten have no reference section** — they are parallel policy
agreements that cite legislation/programmes inline, not a bibliography. The **2
papers have genuine academic bibliographies** (Brakman cites Kaldor, Dixit-Stiglitz,
Dixon-Thirlwall…; Ali cites Acemoglu-Robinson, Barca, Rodríguez-Pose…).

### Intra-corpus citations: **0**

The decisive measurement: **does any of the 6 sources cite any other of the 6?**

- Title-containment probe (does src A's `full_text` contain src B's title?):
  **0 hits across all 30 ordered pairs.**
- Manual inspection of both bibliographies: **every cited work is an external
  academic work.** Neither paper cites the other, and neither cites (nor could —
  wrong genre/language/date) any of the four Dutch policy convenanten.

**∴ intra-corpus citation density on this corpus = 0.** U.3 (`cites`, source→source)
would build a correct extraction+matching mechanism with **no edges to draw on this
corpus**. The DOIs/references are real and parseable (the mechanism is buildable and
testable), but every confident match would resolve to an *external* work outside the
6-source corpus — and U.3's own acceptance criterion is "only intra-corpus matches"
(plan.md U.3). So the honest finding: **mechanism-with-no-data right now.**

---

## Measurement 3 — Sync model per edge type

| Edge type | Source of truth | Recommendation | Rationale |
|---|---|---|---|
| `mentions` (source→entity) | `entity.source_documents` array | **Regenerated projection** — stateless, rebuild on demand, no persisted upkeep | It is a pure derived view of the array; re-extraction/merge already rewrites the array, so a rebuild is always correct and cheap (67 edges). Persisting it would only add a sync burden (R.6 Risk 1) with zero benefit. |
| `cites` (source→source) | `full_text` bibliography → fuzzy match | **Persisted + maintained** — *if/when built*; refresh on re-extraction | Genuinely new, discrete facts that live nowhere else and are expensive to recompute (LLM/parse + fuzzy match). Worth storing with confidence + matched-reference text. **But: 0 intra-corpus edges on this corpus** (see M2) — nothing to persist yet. |
| `related_to` (embedding top-k) | R.1 embeddings | **Computed, draw-only** (optional) | Continuous signal; an edge needs a threshold that throws away information. Materialize at most a top-k draw-only layer if U.4 wants it. Out of U.1's required scope; deferred. |

This matches plan.md's lean (regenerated for `mentions`, persisted for `cites`),
now confirmed by the data: `mentions` is cheap & derived; `cites` is real-but-absent.

---

## Measurement 4 — Thresholds / weights (keep the graph legible)

Reuse R.2's per-edge weight verbatim: `weight = type_salience(type) × IDF(df, N=6)`
(`kg_source_scorer.entity_weight`). Apply R.6 first (group case/type duplicates;
**drop df==1 singletons**). Measured weight distribution over the 67 filtered edges:

```
edge weight  min=0.200  median=0.234  max=1.336
```

Min-weight cutoff sweep (edges retained):

| min_weight | edges kept | effect |
|---:|---:|---|
| 0.0 / 0.1 / 0.2 | 67 | full filtered graph |
| **0.3** | **8** | keeps only the 2 named anchors (Regio Deal / Regio) |
| ≥ 0.5 | 8 | same — drops every generic topic |

**Recommended defaults for U.2/U.4:**
1. **Mandatory:** R.6 normalization + **drop df==1 singletons** (466 → 67 edges). This
   alone makes the graph drawable and is non-negotiable (it's what kills the spokes).
2. **Default min-weight cutoff = 0.0** (keep all 67 df≥2 edges) — the graph is already
   small; carry the weight as an edge attribute for rendering (thickness/opacity)
   rather than as a hard filter.
3. **Provide a UI weight slider** defaulting low, with **0.3** as a "named-only" preset
   (collapses to the 8-edge "Regio Deal / Regio" skeleton) for a clean overview.
4. Weight every edge with `entity_weight` and surface the contributing concept name as
   the per-edge "why" (R.2 lineage already produces this).

---

## Per-phase go / no-go

### U.2 — `mentions` projection — **GO**
- **Feasible and cheap.** Projects 1:1 from `entity.source_documents`; 466 active raw
  edges → **67 after R.6** across 25 concepts and the 4 convenant source nodes.
- The bipartite projection yields a **fully-connected K4 of the convenanten** — all 6
  source-pairs share ≥7 df≥2 concepts (Zuidwest-Friesland↔Het Hogeland share 12).
  A `document → entity → document` traversal returns the convenant cluster exactly as
  the U.2 acceptance criterion requires.
- **Caveat to honor:** the 2 papers have 0 active entities → they are **isolated** in
  the `mentions` view. That's a faithful reflection of the data (their entities were
  triaged to archived), not a bug. U.2 should not special-case them.
- **Thresholds:** as Measurement 4 — R.6 + singleton-drop mandatory; min-weight 0.0
  default with a 0.3 named-only preset; sync = regenerated projection.

### U.3 — `cites` extraction — **DEFER (mechanism-with-no-data)**
- The mechanism is buildable: both papers have real, parseable bibliographies with
  DOIs (20 + 12) and structured author-year entries.
- **But intra-corpus citation density is 0** — no source cites any other source in the
  corpus. Every confident match resolves to an *external* work, which U.3's own
  "intra-corpus only" criterion excludes. Building U.3 now produces **0 drawable
  edges** while paying full extraction + fuzzy-match + precision-guard cost.
- **Recommendation: defer U.3** until the corpus contains documents that cite each
  other (e.g. a paper + the works it cites are both ingested, or a policy doc that
  references another in-corpus convenant). The design is valuable for future corpora;
  it is premature for *this* one. If built speculatively, gate it behind a test corpus
  with known intra-corpus citations so the precision guard is actually exercised.

### U.4 — document graph in the KG viz — **GO (mentions-only)**
- Would show: **4 convenant source-nodes** in a clique, linked via shared-entity
  (`mentions`) paths, with the **25-concept entity layer toggleable** and per-edge
  "why" = the shared concept name (e.g. "Regio Deal", "brede welvaart"). The 2 papers
  appear as isolated source-nodes (honest).
- The existing `SigmaGraphView` + `knowledge_graph.py /graph` endpoint already render
  an edge-first active-only graph, so adding a document-centric view is incremental.
- **No `cites` layer to draw** (0 edges) — U.4 ships the document↔entity view only.
  The optional top-k `related_to` embedding layer (draw-only, thresholded) could add a
  document↔document layer for the 2 isolated papers if a visual link is desired, but
  that's an embedding similarity edge, not a citation.

---

## Decision-gate verdict

| Phase | Verdict | One-line reason |
|---|---|---|
| **U.2 mentions** | **BUILD NOW** | 67 clean edges, K4 convenant cluster, cheap regenerated projection |
| **U.3 cites** | **DEFER** | 0 intra-corpus citations on this corpus — correct mechanism, no data |
| **U.4 viz (mentions)** | **BUILD NOW** | small, legible document↔entity view; incremental on existing SigmaGraphView |
| **U.4 cites layer** | **DEFER with U.3** | nothing to draw |
| **U.5 exports/RETRO** | **scope to U.2+U.4** | export the `mentions` + document↔entity layer only |

**Headline:** Track U's document↔entity layer is real, small, and worth building
(**U.2 + U.4 GO**). The document↔document **citation** layer (**U.3**) is a sound
design with **zero applicable data on the current 6-source corpus** — defer it until
a citing-document corpus exists rather than ship a mechanism that draws nothing.
