# D6 — Edge prediction, measured

**Question** (from `component-decisions.md`): edge prediction (filtering stage 13)
is on, creates relations, and nobody has looked. When it writes a relation, does
the source text support it?

**Answer.** Rarely as a relation of its own. About one predicted edge in six is a
relation the text states (population-weighted 16.8%, bootstrap 95% CI 8.5–26.8%).
Almost all the rest are **two-hop shortcuts**: two entities that already share an
LLM-extracted neighbour, typically two parties in the same list of one Regio Deal,
written as a direct `RELATED` edge and stamped `extraction_method = 'llm'`.

**Where each figure comes from.** Everything is under `claudedocs/d6-edge-prediction/`
and produced by `scripts/d6_edge_prediction_measurement.py` (`census`, `sample`,
`report`), unless another file is named:
- `census.json`: counts over all relations, pairs, hubs, the largest source's
  subgraph, and the re-score.
- `results.json`: verdict rates, intervals, Fisher test, indirect categories and
  quote fidelity.
- `indirect_categories.json`: the per-item category of each indirect verdict.
- `sample_key.jsonl`: per-item evidence counts.
- `judge_prompt.md`: the judge prompt, verbatim.
- `verdicts.jsonl`: the judge's verdicts.

## Scope — read this before the numbers

- **Snapshot.** A frozen copy of `staging` taken 2026-09-25 (backup
  `surreal_data_backup_2026-09-25`, served read-only on a second port). Only the
  edges that survive in it are counted; deleted ones cannot be seen.
- **Configuration.** All 538 stored confidences equal the constructor's
  **fallback** weights (cosine 0.5, Adamic-Adar 0.2, co-occurrence 0.1, Jaccard
  0.2) (`census.json` → `rescore`). They were written 2026-06-23 to 2026-09-02.
  PC.6 (2026-09-05) made the workflow pass `EdgePredictionConfig` (cosine 0.6,
  Adamic-Adar 0.3), so the weights now sum to 1.2 against the same 0.5 threshold.
  **For the same input**, main today scores every candidate at least as high as
  what was measured, and so emits a superset. From graph signals alone the
  maximum rises from 0.5 to 0.6. Upstream stages have changed since (PC.3, PC.6),
  so today's actual output is not measured here.
- **Judge.** A blind LLM judge, not a human: Sonnet, one instance per batch (5
  instances for 7 documents), each given the whole document as context. The
  owner's spot check is in the last section.

## Identification

Predicted edges carry `properties.prediction_method = 'edge_predictor'`, and
nothing else distinguishes them:

| | rows | distinct pairs | `relation_type` | `extraction_method` |
|---|---|---|---|---|
| predicted | **538** (28.4% of 1,892) | 456 | all `RELATED` | all `llm` |
| LLM-extracted | 1,354 | 1,240 | none `RELATED` (18 `RELATED_TO`, 2 empty) | all `llm` |

The review checked the property's completeness:
- It has been set since the predictor's first commit.
- `_upsert_relation` overlays properties and never drops a key.
- The only other writer of `RELATED` (the persistence default) has fired 0 times.
- No other table or database in the namespace holds predicted edges.

**Concentration.** 342 of the 538 edges come from one document, Convenant Regio
Deal Noord-Holland Noord. It ran twice on 2026-06-23: 171 edges at 19:46, and
159 + 12 at 20:52/20:53.

## Result

Sample: 64 predicted edges (target 60, stratified by signal pattern with a floor
of 5, so 50/9/5) and 40 LLM-extracted edges as a baseline, seed 6, active edges
only. Origins were mixed per document, and the judge saw only the two names and
their types.

| | n | direct | indirect | none | direct rate (Wilson 95% CI) |
|---|---|---|---|---|---|
| predicted | 64 | 14 | 47 | 3 | 21.9% (13.5–33.4) |
| LLM baseline | 40 | 36 | 2 | 2 | 90.0% (76.9–96.0) |

**Population-weighted** over the three patterns, the predicted direct rate is
**16.8%** (stratified bootstrap 95% CI 8.5–26.8%). The
`co-occurrence+adamic+jaccard` pattern (80 edges) scored 0/9 direct, and
`cosine+co-occurrence` (10 edges) scored 5/5.

**By document, the figure is corpus-dependent.** The Noord-Holland Noord document
is 2/36 (5.6%). All other documents pooled are 12/28 (43%). A single judge
instance handled the Noord-Holland Noord document, so document effects and judge
effects cannot be separated. Even the pooled other documents stay well below the
LLM baseline.

**What the edges are.**
- **528 of 538** predicted edges join two entities that already share an
  LLM-extracted neighbour.
- The shared hubs are the deals themselves: Regio Deal Noord-Holland Noord (353),
  Regio Zwolle II (97), Zuidwest-Friesland (50) and Het Hogeland (34).
- Of the 47 predicted `indirect` verdicts:
  - 36 are co-listing: two items in the same list (parties, defined terms,
    countries).
  - 9 are cross-group Rijk–Regio links.
  - 2 join a party to an attribute of the deal: its date, or its subtitle.
- The category is a rule over the judge's stated reason, published per item.

A predicted edge is rarely false: direct-or-indirect is 95.3%, against 95% for
the LLM baseline. It **restates a path the graph already has, as an edge the text
does not state**, with a confidence that looks like an extracted relation's.

**Confidence does not track support.** Within the dominant pattern, the direct
rate is 2/25 at ≥0.75 against 4/15 below 0.6 (Fisher two-sided p = 0.17, not
significant). So the only safe claim is that a high score is no sign of a
supported edge. Reading the pooled figures (2/25 against 9/29) as an inverse
relation would be wrong, because the floor-5 stratum always scores low and is
always direct.

## Mechanism (`edge_predictor.py`)

- **343 of 538** clear the 0.5 threshold on graph signals alone. On all 343,
  co-occurrence, Adamic-Adar and Jaccard are exactly 1.0. Under the fallback
  weights that sums to exactly the threshold. Structurally identical neighbours
  (siblings of one hub) score the maximum.
- **Co-occurrence is 1.0 on all 538 edges**, so it adds a constant 0.1 and carries
  no information here.
- Co-occurrence and Adamic-Adar are normalised to the batch maximum, so 1.0 means
  "highest in this document". Cosine and Jaccard are not normalised.
- Cosine only compares the first 20 entities that have an embedding. Adamic-Adar
  and Jaccard only compare the first 20 entities with any relation. Which pairs
  are considered at all therefore depends on list order.

## Side findings

1. **A pair can persist in both directions.**
   - 82 of the 456 pairs occur more than once: 45 in the opposite direction and 37
     in the same direction.
   - The same-direction copies predate the relation upsert of 2026-06-24.
   - The opposite-direction copies can still happen today, and one pair has a copy
     written after 06-24. The predictor removes reversed duplicates within one
     run, but which direction it keeps depends on set iteration order (Python
     string hashing is randomised per process), while `_upsert_relation` keys on
     `(in, out, type)`.
2. **The baseline is an upper bound, and not ground truth.**
   - The judge never saw the LLM edges' types, so the 90% says the text links the
     pair, not that the type (`FINANCIERT`, `VERSTERKT`, …) is correct.
   - Two LLM verdicts are lenient:
     - E054 is judged direct through a mediator ("via gezondheid"), and its quote
       skips a sentence without marking it.
     - E069's quote does not mention Krachtige Kernen.
   - LLM edge E046 (confidence 1.0) names two entities, neither of which occurs in
     any chunk of its recorded source document (`sample_key.jsonl`: 0 shared, 0
     A-only, 0 B-only). So `source_documents` does not prove where an edge came
     from.
3. **Quote fidelity.**
   - 94 of the 99 non-empty quotes occur in the judged document after normalising
     whitespace and case: 91 verbatim, and 3 only after splitting on "…".
   - The 5 that don't are all direct verdicts on LLM edges (E052, E054, E069,
     E087, E091).
   - Checked by hand: four differ only in typography or footnote markers, and E054
     is the stitched one.

## Recommendation

This asks for a decision; nothing has been changed.

1. **Stop enabling edge prediction in the extraction path.**
   - `FilteringConfig.edge_prediction_enabled` already defaults to False.
   - What turns it on is the hard-coded `edge_prediction_enabled=True` at
     `entity_extraction_service.py:1928`.
   - What the stage writes is almost entirely derivable from the graph at query
     time: a two-hop path through the deal. Materialised, it is indistinguishable
     from an extracted relation, and it thickens around hubs. The 25 entities
     around Noord-Holland Noord are joined by 264 of the 300 possible pairs. They
     are both Partijen lists plus the deal's date and subtitle.
   - What that does to graph-signal ranking (R), and to the predictor's own
     Adamic-Adar and Jaccard on the next run, is **not measured here**.
2. **If it stays, it proposes and does not write.** Apply the rule
   `component-decisions.md` sets for scores: stamp its own `extraction_method`,
   and route its output to the curator queue as a relation proposal, not to
   `relation`.
3. **The 538 existing edges.** They can be identified exactly by the property.
   Retiring them (by status, reversibly) is the owner's call. Nothing here
   requires it before (1).
4. **Separately, whatever is decided:** the both-directions duplication in side
   finding 1 is a defect of the relation upsert, not of prediction.

## Owner's spot check

Fifteen verdicts, stratified by origin and verdict (8 predicted, 7 LLM), are in
`claudedocs/d6-edge-prediction/spotcheck.json`, with the owner's mark per item.
**The owner agreed with the judge on 15 of 15** (2026-09-25). That includes E054,
which side finding 2 calls lenient: the owner accepted it as `direct`, so that
reservation is the reviewer's, not a disagreement with the judge.

Fifteen items bound the judge's error rate only loosely. Even 15/15 is consistent
with a disagreement rate of up to about 20% (Wilson upper bound). The conclusion
does not depend on that margin: 47 of 64 predicted edges are `indirect`.
