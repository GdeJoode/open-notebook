# Phase N.5 (a–d) — attempts 1–3 — VERDICT: APPROVED (attempt 3)

- **Branch**: `feature/track-n5a-counters-survive-merge` — merged as `a4bb9d1b`
- **Commits**: `e8ae249c` → `2942cb5b` → `8d9937fc` → `47f424dd` → `5c1d6060` →
  `f8db7bb4` → `2702a89a` → `14a65106` → `f640b1bf` → `3ba2f9bf` → `0457ffb1` →
  `fdd957e3`
- **Date**: 2026-09-03

The phase that closes Track N. One blocker per round for three rounds, and all
three were the same shape: **a measurement that reads as a statement.**

## What shipped

1. **N.5a** — the per-pass counters survive the multi-schema merge, summed, with
   `per_schema` keeping the unmerged view and `merged_duplicates_collapsed`
   separating duplication from gate rejection.
2. **N.5b** — `is_a` declared in `schema_core`, `base` and `policy_themes`
   (covering all eleven ontologies through `extends`); the Hearst miner ships
   default-off.
3. **N.5c** — R2 fixed at both relation collapses; C2/C3/C4/C5 verified closed.
4. **N.5d** — a regression gate split into a unit-tested comparison
   (`shared.regression.extraction_gate`) and an opt-in runner
   (`scripts/n_extraction_gate.py`), with a checked-in baseline.

## The three blockers, in order

**B1 (attempt 1) — the gate read a key nothing wrote.** `summarise_run` read
`result["counters"]`; the repository had two hits for that key — that line, and
the test helper that invented it. `run_extraction` returned entity and relation
counts plus filtering stats, so the counters N.5a had just rescued from the merge
never crossed out of the service. Two of four dimensions were structurally unable
to fail while three documents claimed re-measuring would populate them.

The lesson is not "we forgot a key". **N.5a and the gate were each internally
correct, and the thing between them was never built.** A producer and a consumer
can both pass their own tests indefinitely.

**B2 (attempt 2) — the fix for B1 introduced the same defect.**
`_observability_counters` lifts whatever counters a result carries, and the legacy
single-schema path carries exactly one: `chunk_count`, which is `abstain_rate`'s
DENOMINATOR. `summarise_run` set a `saw_counters` flag from whole-dict truthiness
and guarded each rate on one input, so a legacy run reported `abstain_rate: 0.0`
for something never counted. `over_generation_rate` escaped only because its guard
happened to test its own numerator.

Fixed at the claim rather than the source: the legacy path drives a pluggable
extractor rather than `run_pass2` and genuinely cannot count abstention. Each rate
now requires ALL of its own inputs, from EVERY document — stricter than requiring
one document to supply them, because a mixed corpus would otherwise sum legacy
`chunk_count` into the denominator while only multi-schema documents reached the
numerator, understating the rate by exactly the legacy share.

**B3 (attempt 3, self-caught) — the test for B2 could not fail.** For the same
reason the code had the bug: the fixture supplied no extraction inputs, so
`extracted` summed to 0 and the accidental `extracted > 0` guard refused anyway.
The test was passing on the luck it was written to remove. Separating the guards
needs a corpus where one document DOES supply the extraction inputs.

## Findings the reviewer contributed

- **The measurement disappearing was reported as a missing baseline, and passed.**
  Both comparators skipped whenever either side was `None`, with the reason
  hard-coded to "no baseline value". The rule as written covered "this metric did
  not exist when the baseline was taken"; its mirror image — "this metric existed
  and this run lost it" — is the case that catches a regression in this track's
  own code. Now FAILS.
- **A zero baseline passed anything.** `0 × (1 − tolerance)` is 0.0, so a baseline
  recorded from a corpus that yielded nothing made the gate permanently green with
  `inconclusive=False`.
- **R2 was fixed on the wrong path.** The residual defined it at PERSIST; the
  first fix addressed the in-memory merge. Both are now fixed, and the union reads
  back the accumulated list so a third write stacks.
- **ARCHITECTURE.md asserted a query that does not work.** The union lives under
  `relation_sources`, so `WHERE relation_source = 'hearst'` still misses a
  collapsed edge — following that sentence would under-delete.
- **`3823 chunks` mixed two measurements.** An 8-record harness run and a
  database-wide scan, run together into one sentence in six places. The conclusion
  held; the attribution did not.

## Numbers, stated as the separate measurements they are

An earlier draft ran these together, which is what the review caught.

- **Harness runs** (capped at 10 chunks/document): 8 records over 7 distinct PDFs,
  124 entities.
- **The database** at review time: 14 sources, 3823 chunks, 1895 relations across
  100 relation types.
- **Scans over it**: 220 raw Hearst pairs (138 distinct), **0** `is_a` edges;
  applicability detection 2/14 before PC.1's sample fix, 13/14 after.

## Follow-ups closed before merge rather than filed

The reviewer approved with three, and two were one-line defects introduced by the
fixes; the third was the argument this phase itself makes.

- `scripts/n_extraction_gate.py` had **no tests**, while two of the review's fixes
  lived only there. Not a guard that cannot fail — a behaviour a mutation cannot be
  applied to, which is the same gap from further away, and it sat directly against
  the reason the comparison module is unit-tested.
- A refusal exited **1**, the code reserved for "regression", so a CI wrapper could
  not tell the two apart. Refusals now exit 3.
- `--write-baseline` superseded a stale note while carrying the stale metadata
  around it — the same class of defect one field over.

## Carried out

- **PC.2a**: the user's reading of the whole pattern — derived state dropped at
  handoffs, per document, per notebook and across the graph — with the six known
  boundaries tabulated and the observation that each was a producer with no
  consumer.
- A **pre-existing entity-filtering failure** (verified at `b8a5238f`), whose broad
  `except Exception` converts an `AttributeError` into the business verdict "not a
  match, confidence 0.0". Track K's code; recorded in `status.md`.
- The baseline **predates** PC.1's sample fix and N.5a's counters, so its two cost
  dimensions skip until someone re-measures — and its provenance note carries three
  cautions for whoever does.
