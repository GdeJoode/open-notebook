# Phase N.4 — attempt 1 — VERDICT: REVISIONS_NEEDED

- **Branch**: `feature/track-n4-concept-alignment` (commit `da0bda6`) — **PARKED, not merged**
- **Date**: 2026-09-01
- **Outcome**: user decision → park the branch, re-plan N.4 before writing more code.

## Verification the reviewer ran

| Check | Result |
|---|---|
| `pipelines/entity-filtering` → `pytest -q` | 1 failed, 544 passed, 1 skipped |
| The 1 failure (`test_llm_matcher.py::TestMatchPair::test_calls_ollama_for_unknown_pair`) | **Pre-existing on `main`** — independently confirmed by checking out main. NOT from N.4. |
| `pipelines/ontology-extraction` → `pytest -q` | 316 passed — the `FilteredResult.concept_alignment_report` addition is backward compatible |
| `ruff` on all 5 changed files | clean |
| `ExtractedRelation(**seeded)` | does not raise; keys map exactly, confidence in range |

## 🔴 Blockers

### B1 — lexical containment does not imply `is_a`
`lexical_subsumption` infers `NARROWER_THAN` from "name A contains name B", at the
module's **highest** confidence (0.90). Run over realistic Dutch corpus names, 6 of
13 cases are wrong-typed:

| Pair | Emitted | Actual relation |
|---|---|---|
| `Tweede Kamer der Staten-Generaal` ⊃ `Tweede Kamer` | NARROWER_THAN | same_as (alias) |
| `Den Haag Zuidwest` ⊃ `Den Haag` | NARROWER_THAN | part_of |
| `Universiteit Utrecht Bibliotheek` ⊃ `Universiteit Utrecht` | NARROWER_THAN | part_of |
| `Van Gogh Museum` ⊃ `Van Gogh` | NARROWER_THAN | named_after |
| `Willem Alexander Claus George Ferdinand` ⊃ `Willem Alexander` | NARROWER_THAN | same_as (person) |
| `Ministerie van Financien` ⊃ `Ministerie van` | NARROWER_THAN | noise stub |

The token matcher itself is sound (genuinely token-bounded; `Gemeente Den Haag` vs
`Gemeente Den Bosch` → `None`; diacritics do not false-link). The defect is the
**inference from containment to subsumption**.

Worse, the input population is structurally biased toward the failure mode:
`KGResolver`'s fuzzy tier *rejects* long-form/short-form alias pairs because the
length delta tanks Levenshtein (`"tweede kamer der staten-generaal"` vs
`"tweede kamer"` ≈ 0.39 < 0.85), so those pairs arrive at the aligner marked
`is_new` and containment fires. The tier's primary diet is **aliases and meronyms,
not subtypes**. `min_inner_tokens=2` is not the guard the docstring claims.

### B2 — seeded `is_a` edges are silently discarded, while the report counts them
Stage 10c runs **before** the ontology filter (stage 11). That filter drops any
relation whose endpoints are not in the batch's entity texts; a seeded edge's
target is by construction an **existing KG node**, never in the batch. Empirically:
`surviving relations: []`, `invalid_relations: 1`, no diagnostic trace — while
`concept_alignment_report["seeded_is_a"]` still reports `1`. With
`ontology_validation.enabled=True` the feature's headline output is 100% lost,
silently, and the report is wrong.

## 🟡 Majors

3. **Endpoint disambiguation thrown away** — the seeded dict sets no
   `source_type`/`target_type`, and the target is off-batch so `type_by_name`
   misses too → persistence falls back to name-only resolution
   (`SELECT ... WHERE canonical_name = $name LIMIT 1`), the exact cross-type
   homograph mis-binding Track O.1 exists to prevent. The aligner *knows* the
   target record id but buries it in `properties`.
4. **"Non-destructive" is false at workflow level** — `graph_analyzer._build_graph`
   auto-creates a phantom node for the off-batch target; PageRank is normalised over
   all nodes, so enabling alignment shifts every `centrality_score` and can change
   which entities stage 12 **removes**. Reachable when `ontology_validation` is off
   and `graph_centrality_enabled` is on.
5. **False evidence, in an evidence-first track** —
   (a) `find_by_type` filters the canonical **`entity_type`** column while
   `entity["label"]` at this stage is the raw rich extraction label; under Track-L
   rich typing (`Gemeente`, `RegioDeal`) the fetch returns `[]` **by construction**,
   so the stage degrades to a no-op that stamps *"no existing concepts of a
   comparable type"* — an assertion never established — on every new entity.
   (b) An item the judge stayed *silent* on is stamped `method=llm_judge` +
   *"judge found no link"* because some *other* batch item got a ruling, making
   `method_counts["llm_judge"]` exceed `judged_count`.
6. **Zero integration coverage for stage 10c** — all 27 tests are unit-level against
   `ConceptAligner`. B2 and M4 would have been caught by one workflow-level test.
7. **Unreachable in production + silent misconfiguration** — no production caller
   supplies `entity_repo`, `ontology`, or the new `alignment_llm_caller`
   (`entity_extraction_service` constructs `FilteringWorkflow(config=...)` bare),
   and there is no env flag (contrast N.3's `EXTRACTION_NOT_A_CONCEPT`). Enabling it
   with a missing repo or with `kg_resolution` disabled logs nothing — the
   orphan-connector block right below does warn, a pattern adopted after an earlier
   review.

## 🔵 Minors

1. `_candidates` has no per-batch cache: up to 3 `find_by_type` calls per novel
   entity, on top of `KGResolver`'s own per-entity call for the same type.
2. `type_chain_ancestors` resolved twice per entity.
3. `parse_judge_response` keys `allowed` by text; two pending entities sharing a
   surface form collide (latent for standalone use; unreachable via the workflow).
4. `nearest_by_embedding` — `cand.get("properties", {}).get("embedding")` raises on
   a row with `properties: None`, outside the `try`.
5. `align` reads `e.get("properties", {}).get("is_new")` unguarded — `properties:
   None` raises out of the whole `process()`.
6. `_normalize` does not strip diacritics while `EntityNormalizer` may →
   asymmetric batch-vs-DB normalisation (recall-only).
7. `_candidate_name`'s in-batch branch is dead — `_candidates` never supplies batch
   entities, so two novel concepts in the same document are never aligned to each
   other.
8. `_CONF_LEXICAL = 0.90` is the highest confidence in the module, assigned to the
   least reliable tier.

## Planner error found during this review

The N.4 chapter listed the **`evolution` agent's gap analysis** as an input. The
implementer searched `apps/` and `pipelines/` for a *directory* named `evolution*`,
found nothing, and recorded in the commit that "no such agent exists". That is
**wrong**: `packages/ontology-manager/src/ontology_manager/evolution.py` contains a
working `OntologyEvolutionAgent` (`record_gap` → frequency threshold →
`SchemaProposal` → approve/reject/implement, plus `list_gaps` /
`get_gap_statistics`). It is wired into `ontology_manager/manager.py` but **not**
into the filtering pipeline. The re-plan must use it.

## Acceptance-criteria status (original N.4 chapter)

| AC | Status |
|---|---|
| Subsumption derived deterministically via `parent_type` + embedding neighbourhood | ⚠️ parent_type tier present; embedding never used for subsumption |
| Uses the `evolution` agent's gap analysis | ❌ not implemented (wrongly declared non-existent) |
| LLM-judge (default ON) arbitrates the RELATED/NOVEL split | ⚠️ correctly fenced (no fabricated targets, no subsumption verdicts, failure → NOVEL) but unreachable — no caller wires it |
| Every classification carries its evidence | ⚠️ always populated, demonstrably false in two paths |
| …and is reversible | ❌ at module level yes; at workflow level no (M4) |
| Tests: narrower lands under broader / true-novel stays NOVEL / related links not merges | ✅ present and asserting real behaviour |

## What survives for the re-plan

Not everything is wrong — these parts held up under scrutiny and should be carried
forward rather than rebuilt:

- The **four-verdict taxonomy** and the `Alignment` evidence record.
- The **judge fencing**: it may not answer subsumption, a target outside the
  concept's own neighbour list is downgraded to NOVEL, garbage/silence → NOVEL,
  sync/async caller handling is correct.
- The **token-boundary matcher** `_is_token_subsequence` (correct; only the
  inference drawn from it was wrong).
- `find_by_type` as the **only** repo method used — no new repository surface, no
  migration.
- The three plan-mandated behavioural tests.
