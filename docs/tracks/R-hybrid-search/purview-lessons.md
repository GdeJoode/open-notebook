# Lessons from Microsoft Purview — applied to Track R

Research input (2026-06-25, WebSearch + reasoning) on what to borrow from Microsoft
Purview's document labeling + linking approach. Purview is **enterprise governance/
compliance** (sensitivity labels, DLP, retention, cloud); open-notebook is **local +
retrieval-focused**. The *purpose* of labeling differs (Purview = risk/sensitivity;
us = content/theme), so we cherry-pick mechanisms, not the architecture.

## Transferable (and where it lands in our tracks)

1. **Trainable classifiers instead of expensive per-document LLM extraction** — Purview
   trains a light classifier on examples (≥50 positive / ≥150 negative) to recognize a
   concept, then auto-applies it. For us: the chunk/source embeddings R.0 turns on ARE the
   feature vectors such a classifier needs. Instead of the noisy per-document LLM pass
   (75% `other`/`topic`), train recurring labels (e.g. "Regio Deal convenant", a policy
   theme) on existing labeled examples. → **R.6** (noise re-scope) + review step **S4**
   (let embeddings/clusters generate topics). Treat embeddings as a classifier feature
   basis, not only kNN.

2. **Managed taxonomy / "active glossary" instead of free labels** — Purview labels are
   curated, not free-form; glossary terms are active objects driving discoverability. Our
   pain is free LLM typing. We already have the canonical bridge (L.1); the lesson is to
   strengthen it toward a small managed label ontology that both the LLM and the classifier
   map into. → **Track L**.

3. **Simulation mode before applying** — Purview dry-runs an auto-label policy, shows the
   result, then applies. That is exactly our dry-run/checkpoint discipline and the **Track Q**
   triage review-queue. Confirmation the approach is right; lesson: give re-extraction /
   auto-labeling an explicit preview/simulation in-product.

4. **Explicit lineage as a linking AND explanation signal** — Purview makes lineage
   (source → transformation → output) visible. We have `provenance_chain`/`source_documents`
   but barely surface it. → **R.5**: show the chain source → chunk → entity → cluster-summary
   as "why is this linked" — which is the "explain why a result matched" already in R.5.

5. **Two layers: raw metadata vs curated catalog** — Purview separates the Data Map (raw
   scan) from the Unified Catalog (curated, searchable). Analogous to our KG (raw extraction)
   vs a clean search-facing projection in **R.6**. Confirms: keep the KG, project a cleaned
   layer for search.

## Deliberately NOT adopted
- Governance/compliance machinery (DLP, retention, insider risk, regulatory sensitivity) — not our goal.
- Cloud/enterprise architecture — open-notebook stays local/SurrealDB-only (cf. the I.H3 rejection of external stores).
- The training-data requirement (≥50 examples/label) — too much for a small personal notebook;
  prefer embedding-kNN or LLM-with-ontology where labeled examples are scarce. Use classifiers
  only where enough labeled examples exist (scale judgment).

## Net effect on the plan
Confirms the Track R direction (embeddings + curated layer) and adds two concrete things:
(a) treat embeddings as a feature basis for light classifiers, not only kNN (→ R.6 / Track L);
(b) surface provenance/lineage as linking explanation (→ R.5). Plus reassurance that triage (Q)
+ dry-run/checkpoint already are the "simulation mode + review" best practice.

Sources: Microsoft Learn — Trainable classifiers; Auto-apply sensitivity labels; Unified
Catalog; Data lineage; Data governance overview.
