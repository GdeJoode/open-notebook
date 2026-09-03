# Track N — Retrospective (evidence-first extraction & abstention)

> Closing date: 2026-09-03
> Branches merged: `feature/track-n1-candidate-anchors`,
> `feature/track-n2-hearst`, `feature/track-n3-not-a-concept`,
> `feature/track-n4a-alignment` … `feature/track-n4d4-gap-loop`,
> `feature/track-n5a-counters-survive-merge`.
> Final state: `docs/tracks/N-evidence-first-extraction/status.md`.

Draws on the per-phase review reports in `reviews/` and the live pipeline review
in `claudedocs/extraction-pipeline-review.md`.

## Summary

Track N set out to make extraction evidence-first: deterministic candidates
before the LLM (N.1), a deterministic hierarchy miner (N.2), a gate that drops
what is not a concept and abstains rather than guessing (N.3), alignment against
what the graph already holds (N.4), and a regression gate to keep it honest
(N.5).

The track's real product turned out to be less the individual mechanisms than a
way of checking them. Across N.4d alone, four review rounds closed three blockers
and eight majors, and the majority were one shape: **a test that reports as a
guard while being unable to fail.** N.5 then found the same shape in the
production code rather than in the tests — measurements that read as statements.

## What each phase actually cost

| Phase | Review rounds | The finding that mattered |
|---|---|---|
| N.1 | 2 | — |
| N.2 | 2 | Hearst precision needs both endpoints extracted; the gate that gives it also makes the miner produce nothing (found in N.5b, not here) |
| N.3 | 2 | The counters it added were discarded by the merge (found in N.5a, not here) |
| N.4a–c | 2–4 each | Subsumption is a relation between TYPES while the table stores MENTIONS (D-N4-12) |
| N.4d.1–4 | 3–5 each | Three binding rules about test doubles (D-N4-14) |
| N.5a–d | — | Two measurements that read as statements |

The pattern in the right-hand column is worth naming: **three of the five phases'
most important findings were made by a LATER phase, not by the phase's own
review.** N.2's and N.3's defects were both invisible until something ran the
pipeline against real documents and asked what the numbers said.

## The three lessons that cost the most

**1. A test that cannot fail is worse than no test, because it reports as one.**
Established in N.4d.3 (a whole-vocabulary sweep that killed zero mutants; a
production call site whose deletion left 1632 tests green) and re-confirmed in
every later phase. The standing practice that came out of it: verify a guard by
mutation, not by a green suite. It found a dead control-character rule, a
rejection test containing no rejection, and a correct helper with no seam behind
it — the last one in this track's own final phase.

**2. A double must reproduce the real method's failure RETURN, not a raise; a
fixture must build the PRODUCTION argument set; a guard reading a collaborator's
value must be exercised against the real collaborator at least once** (D-N4-14).
Each of the three was paid for by a separate blocker in N.4d.4.

**3. A measurement that is absent reads as a measurement that is zero.** N.5's
contribution. `_merge_results` discarded N.3's counters, so a run that culled 14
entities to 5 reported `over_generation_rate` 0.00 — not "unknown", but "nothing
was culled". The same shape appeared in the gate design and was designed against:
a dimension with no baseline is SKIPPED, never PASSED.

## What we would do differently

**Run it earlier.** The pipeline review that re-planned N.5 was requested after
N.4d, and it invalidated assumptions four phases had been built on — most
sharply that applicability detection worked at all (it fired for 2 of 14
documents, scoring each against its cover page). Every one of those findings was
available from the first day there were documents in the database. A phase that
ships a producer should measure what it produces on real data before the review,
not four phases later.

**Be suspicious of a producer that ships enabled and changes nothing.** N.2's
miner ran in production for eight documents and put zero edges in the graph.
Nothing failed, no test noticed, and the only reason its output would have
survived at all is that a validator downgrades unknown predicates outside strict
mode. "It works and it is on" and "it contributes" are different claims.

**Separate the expensive measurement from the cheap decision.** N.5d's gate is
useful because its arithmetic runs on every suite execution. An earlier instinct
was to write it as one integration test behind `@requires_docker`, which would
have made the rule itself untested in practice.

## Numbers

These come from three different measurements and are listed separately on
purpose — an earlier draft ran them together into one sentence, which a review
caught:

- **The harness runs** (`scripts/n_pipeline_review_run.py`, capped at 10 chunks
  per document): 8 records over 7 distinct PDFs, 124 entities.
- **The database** at the time of the review: 14 sources, 3823 chunks, 1895
  relations across 100 relation types.
- **Scans over that database**: the Hearst miner yields 220 raw pairs (138
  distinct) and the graph holds **0** `is_a` edges; applicability detection fires
  for 2 of 14 sources before the sample fix and 13 of 14 after (the fix itself
  landed in Track PC.1).
- Attrition through the fifteen filtering stages after the LLM: effectively zero
  — which is itself a finding, and one the review's own table cannot fully
  explain, because it starts AFTER the N.3 gate.

## What left the track

Most of the pipeline review's findings were not Track N's. The curator-queue
writer, cross-document identity, canonicalisation stability, the alias-policy
contradiction, the gap/proposal read path and default-configuration coherence
moved to **Track PC — pipeline coherence**. PC.1 has since shipped and took five
review rounds of its own, including one escalation to the user; its own report
is in `docs/tracks/PC-pipeline-coherence/reviews/`.
