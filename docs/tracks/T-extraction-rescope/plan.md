# Track T — Extraction re-scope (LLM where it earns its keep)

> **DRAFT for user review — nothing built yet.** Born from the question "now that
> embeddings + the KG hybrid search work (Track R), where does LLM entity/relation
> extraction still belong?" Answer: it shifts from "expensive, blanket, everything, at
> every ingest" to "scarce, targeted, named-entities + relations — the things only an LLM
> can do." Themes move to the cheap embedding/clustering cascade.

## The role split (the design principle)
| Job | Best tool | Why |
|---|---|---|
| "Do these documents resemble each other?" | embeddings (R.1) | cheap, catches paraphrase, no LLM |
| "What themes run through them?" | embeddings + clustering | reproducible; an LLM inventing per-doc topics is costly + inconsistent |
| "What *specific* things do they share?" (Regio Deal, ministries) | **LLM extraction → KG** (R.2) | only extraction yields discrete, countable, linkable named entities |
| "What is the *relation*? (A funds B)" | **LLM relation extraction** | embeddings/overlap say "related", not *how* |

**Conclusion**: embeddings do similarity-linking, KG entities do shared-fact linking, and the
LLM is only still needed for the two things the rest can't: **recognise named entities** and
**name relations**. The ~75% generic `topic`/`concept` the LLM produces today is largely wasted
for linking (R.2 down-weights it, R.6 de-fragments it) and belongs in the embedding/clustering cascade.

## Measured baseline (the waste, from Track R reviews on staging)
- 4,870 entities, **8.7% active** (423); **75% of active are generic buckets** (`topic`/`concept`/`other`).
- **388 of 423 active entities are singletons** (df==1, appear in ONE source → cannot link anything).
- So the expensive blanket extraction spends most of its budget producing rows that the search signal
  then discards. That budget is the target.

## Existing pipeline this re-scopes (do not break)
- `apps/app-main/.../services/entity_extraction_service.py` — Track B two-pass orchestration (`_embed_entities`, `_run_triage`, schema workflows).
- Extraction prompts: `packages/ontology-manager/.../prompts.py`, `pipelines/ontology-extraction/.../prompts/pass2.py`.
- Generic-type vocabulary: `packages/ontology-manager/.../schema.py`, `shared/models/notebook_schema.py`.
- Track L typing, Track K resolution, Track Q triage, the exports (Obsidian/NetworkX/JSONL), and R.2's KG scorer all CONSUME the extraction output — must stay green.

## Reference
`docs/reviews/extraction-kg-embeddings-coherence-review.md`, `docs/tracks/R-hybrid-search/design-thematic-classification.md` (the classifier-cascade you approved), memory `extraction-model-typing-tradeoff` (rich typing needs an ontology-following model — use it for named entities, not loose keywords).

**Workflow**: track methodology — `implementer` → `adversarial-reviewer`. Main tree, `uv run pytest`,
no worktree. Additive/reversible where possible; measure before/after; live writes (re-extraction) gated.

---

## Phase T.1 — Extraction-economy baseline (Discovery, read-only)
**Why**: measure the cost + waste precisely before changing the load-bearing extraction path.
**Deliverables**: a measured report — per-document LLM call count + token cost of current extraction; of the produced entities/relations, the % that is named vs generic, singleton vs linking, kept-active vs archived by triage; which prompt/schema fields drive the generic-topic output. Identifies where the spend is wasted.
**Acceptance**: numbers for cost-per-document, named/generic split, singleton %, and the prompt/schema origin of generic topics; a concrete "what to cut" list. **Depends on**: none.

## Phase T.2 — Named-entity-focused extraction (Backend) — START REVERSIBLE
**Why**: stop the LLM inventing generic topics; focus it on named entities + relations (signal).
**Approach (two steps, measure between)**:
- **T.2a (additive, safe)**: a *projection filter* — keep the LLM output as-is but stop persisting/feeding generic singleton `topic`/`concept` into the KG (config-flagged, reversible). Measures the quality impact of dropping them with ZERO prompt risk.
- **T.2b (prompt change, if T.2a justifies)**: tighten the extraction prompt/schema so the LLM produces fewer/no loose generic topics — named entities (org/programme/person/place/legislation/dataset) + relations only, ontology-followed. This is where the LLM-cost saving lands.
**Acceptance**: (1) measurable drop in generic/singleton extraction output; (2) KG search signal (R.2/R.3) holds or improves (convenant cluster intact); (3) exports/K/Q regression suites green; (4) reversible via config; (5) T.2b shows a real LLM-call/token reduction per document. **Depends on**: T.1.

## Phase T.3 — Themes via embedding-clustering, not the LLM (Backend)
**Why**: replace the LLM's generic-topic job with the cheap, reproducible cascade.
**Approach**: derive theme labels from clusters of the R.0 source/chunk embeddings (zero-shot/kNN on a small corpus; one LLM summary per *cluster* at most, not per document). This is the `design-thematic-classification.md` cascade and overlaps Track R.4's clustering — **reuse R.4's clustering if R.4 lands first; otherwise T.3 builds the minimal clustering it needs**.
**Acceptance**: theme labels generated from embeddings (no per-document LLM topic pass); labels are reproducible; they feed the same search/triage surfaces the old topics did; cost per document for themes drops to ~0 LLM calls. **Depends on**: R.0 (done); coordinates with R.4.

## Phase T.4 — Confidence-gated LLM refinement (Backend)
**Why**: the LLM returns ONLY for the hard margin, not blanket.
**Approach**: low-confidence/ambiguous entities + theme assignments are escalated to an LLM refine pass (richer prompt) — gated on a confidence threshold; high-confidence cheap-path results are accepted as-is. Wire into Track Q triage as the review queue.
**Acceptance**: only sub-threshold items hit the LLM (measured escalation rate); a calibration shows precision/recall vs the threshold; high-confidence path makes zero LLM calls. **Depends on**: T.2, T.3.

## Phase T.5 — Integration, cost/quality measurement, docs, RETRO
**Deliverables**: end-to-end before/after — LLM cost per document, noise (generic %, singleton %), KG-signal quality (R.2 cluster), exports intact; ARCHITECTURE note on the new role split; RETRO. **Depends on**: T.2–T.4.

---

## Decisions (locked 2026-06-27)
1. **KG role** → **rich export artifact too — be careful.** Prune ONLY the search-facing projection (R.6-style, in-memory); do NOT mutate the canonical `entity`/`relation` rows; exports (Obsidian/NetworkX/JSONL) keep their topics. This means T.2 is search-projection-only by default.
2. **Prompt appetite** → **T.2a filter first, then re-decide.** Ship the reversible search-facing filter; the prompt change (T.2b) is CONDITIONAL on T.1 + T.2a showing it's worth the extraction-core risk. No LLM-cost saving lands until T.2b/T.3/T.4 are explicitly approved later.
3. **Sequence** → **start now with T.1** (read-only). T.2+ are gated on T.1's findings.
4. **Model** (named-entity extraction local vs NIM ontology-following) → deferred; only relevant if/when T.2b is approved.

**Consequence (honest):** R.6 already de-fragments + drops singletons + down-weights generics in the
search projection. With the "search-projection-only, careful" course, **T.1 is the decision gate** — it
measures whether meaningful cost/noise reduction remains BEYOND R.6, and whether the higher-risk
T.2b/T.3/T.4 (which touch the extraction core / add the cascade) are justified. If T.1 shows R.6 already
captured most of the available win, Track T may correctly stop at T.1 + a thin T.2a.

## Risks & trade-offs (decide at review)
1. **Touches the load-bearing extraction pipeline** (Track B/L). Mitigation: T.2a is an additive reversible filter first; the prompt change (T.2b) only after the filter proves the value; every phase keeps exports/K/Q green.
2. **Export richness vs leanness** — the Obsidian/NetworkX exports currently carry the generic topics too. If those exports matter, dropping topics from the KG affects them. **Product decision**: is the KG primarily a *search signal* (snip aggressively) or also a *rich export artifact* (keep more)?
3. **Small-corpus theme quality** — embedding-clustering themes need enough data; on 6 sources they're thin (same caveat as R.4). May need zero-shot theme descriptions until the corpus grows.

## Open decisions for the user
1. **Export role** (risk 2) — KG as search-signal-only (lean) vs also rich-export (keep topics)? Sets how aggressive T.2 is.
2. **Sequence vs R.4/R.5** — run Track T now, or after R.4 (so T.3 reuses its clustering) / R.5 (UI)?
3. **Prompt change appetite** — stop at T.2a (safe filter, no LLM-cost saving) or go to T.2b (prompt change, real saving, more risk)?
4. **Model for named-entity extraction** — keep local, or use the ontology-following NIM model (memory `extraction-model-typing-tradeoff`) for sharper typed named entities?
