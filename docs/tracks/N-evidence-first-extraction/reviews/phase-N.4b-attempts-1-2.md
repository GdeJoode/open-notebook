# Phase N.4b — attempts 1 & 2 — VERDICT: APPROVED (attempt 2)

- **Branch**: `feature/track-n4b-placement-seeding` — merged as `19add9c`
- **Commits**: `f7aeb1b` (attempt 1) → `85e9719` (revision) → `f8f1bf6` (residual)
- **Date**: 2026-09-01

N.4b's job was to make the N.4a verdicts SURVIVE the pipeline, and to seed an
`is_a` for a `NARROWER_THAN` with a materialised target. Its substance is
placement, not new classification logic.

## What placement had to fix

Both blockers of the parked N.4 attempt were properties of WHERE the stage ran,
which is why unit tests on the classifier could never have seen them. Both were
re-measured directly rather than reasoned about:

| Failure mode | Measurement |
|---|---|
| The ontology constraint filter drops any relation with an off-batch endpoint, and a seed points at an existing graph node by construction | offering a seed to the real filter yields `surviving relations: []`, `invalid_relations: 1` |
| `GraphAnalyzer` auto-creates a node for an unknown endpoint, and PageRank normalises over all nodes | the same two entities score `0.5/0.5` without the seed and `0.25974/0.25974` with it |

Stage 15 therefore runs after ontology validation, centrality, edge prediction and
the orphan-connector. The Stage-11 bypass follows the precedent Stage 14 set, with
one honest caveat added during review: the filter also validates the predicate's
declaration and domain/range, so under `strict_mode=True` a Hearst-mined `is_a`
(which enters before Stage 11) can be dropped while a seeded one survives.
Harmless at the shipped default, but not zero.

## Attempt 1 — REVISIONS_NEEDED (1 blocker, 4 majors, 8 minors)

**B1 — `build_is_a_seeds` emitted a self-referential `is_a`.** Source and target
were compared nowhere, and `type_chain_subsumption` never received the entity's
own text, so no layer could catch it. Reproduced end-to-end as
`Deal - is_a -> Deal`. Reachability is structural: `KGResolver` fetches by the
RICH label while alignment fetches by the CANONICAL type, so an entity that
already exists under its canonical type still arrives `is_new`, and the tier then
matches it against its own row whenever its surface form coincides with an
ancestor type name — and `Deal` / `Gemeente` / `Provincie` are all plausible Dutch
surface forms.

It mattered *at this phase specifically*: before N.4b every seed was discarded by
Stage 11, so seed soundness did not matter. Once seeds reach persistence, both
endpoints resolve to the same record via `(canonical_name, entity_type)`, writing
a 1-cycle into the subsumption hierarchy that N.4c's descendant sweep (D-N4-11)
and inverse-chain reachability (D-N4-10) are both specified to traverse.

**M1** `build_is_a_seeds` had zero direct tests — including the safety-critical
"RELATED_TO is never seeded" that the plan names as an AC.
**M2** The misconfiguration WARNING was untested and the `caplog` fixture was
dead: loguru does not propagate into it. The suite already solved this once for
the orphan-connector with a `_PropagateHandler` bridge; the commit had mirrored
Stage 14's *code* but not its *test*, while carrying an argument that read as if
it had.
**M3** The WARNING text was FALSE in two of three branches — with a missing repo
or ontology the stage RUNS and records NOVEL verdicts, so "will not classify
anything" over-claimed. The same defect class this track keeps correcting in its
evidence, this time in a log line.
**M4** The carried C2/C4 disclosures shipped with no assertions.

## Attempt 2 — APPROVED

B1 was fixed at BOTH levels — the tier no longer matches an entity against itself
(and skips an ancestor whose name IS the concept's own name), and the seeding
boundary refuses it again. Fixing the verdict as well as the edge was deliberate:
a self-referential `NARROWER_THAN` would be wrong even with seeding off, and N.4c
reads verdicts rather than edges.

The reviewer verified by **mutation testing**: disabling each new guard makes a
specific test fail (four non-equivalent guards, four failures), and disabling the
warning branches fails all three warning tests. One mutant proved equivalent — the
`by_name` comprehension filter is fully subsumed by the loop-level guard — which
is belt-and-braces, not a test gap. Both placement falsifications were re-run and
still fail the placement tests. mypy delta against `main` is zero.

Minors closed in the same revision: a seed without a canonical type is now
REFUSED rather than emitted untyped (an untyped edge falls back to the name-only
resolution Track O.1 exists to prevent); seeds are de-duplicated on
`(source, target)` as N.2's Hearst miner already does; `target_name` is stripped
symmetrically with `source`; the stale band-tuple annotation is corrected; and the
centrality-survivor test — which could not fail — now sets its floor at 0.4,
between the measured 0.26 (misplaced) and 0.5 (correct), so it fails loudly.

## Residuals carried to N.4c

| # | Item |
|---|---|
| **R1** | ~~Compound warning branch contradicted itself when `kg_resolution` was off AND a DI input missing~~ — **fixed** in `f8f1bf6` before merge. |
| **R2** | **Cross-pass duplicates.** The dedup is intra-pass. An `is_a` on the same ordered pair already contributed by N.2's Hearst miner is not suppressed; at persist the two collapse on `(in, out, relation_type)` and the later write re-tags `relation_source`, weakening the "one `WHERE relation_source = …` drops the pass" reversibility claim. Natural home is N.4c alongside the sweep. |
| **R3** | C2's two ruled-judge branches and `EV_INCOMPARABLE_VECTORS` remain unasserted — all four consume the identical `sampled` element, so negligible. |
| **R4** | `float(props.get("alignment_confidence") or 0.0)` raises on a non-numeric value and forwards an out-of-range one to Pydantic. Unreachable from `_enrich`; recorded so it is not rediscovered. |
| **R5** | Under the SHIPPED defaults the pipeline seeds **zero** edges, because `type_chain_enabled` is off and is the only `NARROWER_THAN` producer. Documented in module and plan; D-N4-10's verifiable subsumption is what turns seeding on for real. |

## Standing lesson, extended

N.4a's report noted that every failure in this phase came from *a claim that
outran what had been established*. N.4b repeated it in a third register: not in
evidence, not in a verdict, but in a **log line** telling the operator the stage
would do nothing while it was recording verdicts. The counter-measure that keeps
working is the same one: measure the mechanism instead of reasoning about it —
running the real ontology filter and the real graph analyser is what turned two
plausible arguments into two numbers, and mutation-testing the guards is what
turned "the tests pass" into "the tests would notice".
