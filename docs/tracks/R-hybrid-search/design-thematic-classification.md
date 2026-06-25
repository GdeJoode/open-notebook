# Design note — thematic classification via embeddings (cascade), Track R

Captures the design the user and the assistant converged on (2026-06-25): link
documents/chunks to one or more **themes** primarily from **embeddings + a light
classifier**, using the LLM only to verify/refine — not to label every document.
Builds on `purview-lessons.md`. This is design intent for R.4/R.6/R.5 + Track L,
not yet a committed phase.

## Core idea (the cascade)
1. **Embed** chunks/sources (R.0 — done) → semantic feature vectors (local, 1024-dim).
2. **Cheap labeler** assigns 0..N themes per document with a confidence score, from the
   embeddings — NO per-document LLM call.
3. **Confidence gate**: high-confidence labels are accepted as-is; only **low-confidence /
   ambiguous** cases (or a sample for QA) are escalated.
4. **LLM refiner** runs a richer prompt ONLY on the escalated set — to confirm, correct,
   or add nuance. This is where the expensive model earns its cost.

Net effect: the bulk is labeled cheaply, deterministically, reproducibly; the LLM budget
goes to the hard margin. This is Purview's "auto-classify + review", and our Track Q triage,
applied to *thematic* labeling.

## Two clarifications that make it accurate
- **Semantic, not literal.** The labeler matches on *meaning* (embedding similarity to
  examples/theme descriptions), not "this exact phrase recurs". It catches paraphrases and
  synonyms ("krimpregio" ↔ "bevolkingsdaling landelijk gebied") that keyword/regex matching
  (Purview's Sensitive Information Types) would miss. Trade-off: less literally explainable
  → surface lineage/"why matched" (R.5) to compensate.
- **Multi-label.** A document can carry several themes, each with its own confidence — not a
  single forced class.

## Technique grows with the corpus (no training needed early)
The labeler in step 2 should be chosen by how much labeled data exists:

| Corpus state | Labeler (no/low training) | Notes |
|---|---|---|
| Small / cold-start | **Zero-shot**: embed the *theme description* itself, assign by nearest theme | 0 training examples; quality tracks how well themes are described |
| Some labeled examples | **kNN over embeddings**: label by nearest already-labeled docs | works with a handful per theme |
| Unlabeled but clustered | **Clustering** (R.4): group by embedding similarity, label the *cluster* with ONE LLM call instead of one-per-doc | cheap bulk labeling; ties directly into R.4 cluster summaries |
| Mature / enough labels | **Trained classifier** (Purview-style, ≥~50 pos/theme) | fastest + most stable once data exists |

Always, regardless of stage: **LLM as confidence-gated refiner**, not as the primary labeler.

## Where it lands in the plan
- **R.4 (cluster summaries)** — clustering is also the cold-start labeler: one LLM summary/label
  per cluster instead of per document. Embed the cluster label/summary so it feeds retrieval.
- **R.6 (noise re-scope)** — replace/augment the noisy per-document LLM entity/topic pass with
  the embedding labeler; the LLM only refines low-confidence cases. Directly attacks the 75%
  generic `other`/`topic` noise.
- **Track L (entity typing)** — the themes/labels map into a **managed taxonomy** (extend the
  L.1 canonical bridge) so labels are curated, not free-form LLM output.
- **R.5 (search UI)** — show *why* a theme/source matched (matched examples / nearest cluster /
  contributing chunks) — the lineage/explanation Purview surfaces.
- **Track Q (triage)** — the existing review-queue is the human-in-the-loop for the escalated /
  low-confidence labels; confidence drives queue priority.

## Open decisions (resolve when this becomes a phase)
1. **Theme taxonomy source** — hand-curated seed list, or bootstrapped from clusters (R.4) then curated?
2. **Confidence gate threshold** — where does "accept" end and "escalate to LLM" begin? Needs a
   small labeled validation set to calibrate (precision/recall trade-off).
3. **Labeler granularity** — label at source level, chunk level, or both (chunk votes → source label)?
4. **Refiner prompt scope** — confirm-only, or allowed to add/split themes (affects the taxonomy)?
5. **Re-label triggers** — when a theme is added/renamed or the taxonomy changes, what gets re-labeled
   (simulation-mode preview first, per Purview / our dry-run discipline)?
