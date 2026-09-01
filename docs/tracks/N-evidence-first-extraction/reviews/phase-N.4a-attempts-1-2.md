# Phase N.4a — attempts 1 & 2 — VERDICT: APPROVED (attempt 2)

- **Branch**: `feature/track-n4a-ontology-subsumption` — merged as `fad6833`
- **Commits**: `e0929ad` (attempt 1) → `27e5c64` (revision) → `2355cd4` (plan D-N4-11)
- **Date**: 2026-09-01

N.4a is the first sub-phase of the re-planned N.4 (v2), after attempt 1 of the
phase was parked (`reviews/phase-N.4-attempt-1.md`). Scope: **verdicts + evidence
only** — no relation seeding, no workflow stage, no config wiring.

## Attempt 1 — REVISIONS_NEEDED (4 blockers, 4 majors)

Every blocker was **one defect class**: evidence reporting an OBSERVATION as the
inference it would license — precisely the claim (D-N4-7) the phase is built on.

| # | Defect |
|---|---|
| **B1** | A raised candidate fetch was stamped *"the graph holds no concepts of type X"*, and cached — poisoning every remaining entity of that type in the batch. Aggravated by the real `EntityRepository.find_by_type` catching its own exceptions and returning `[]`, so query-failure and zero-rows are genuinely indistinguishable from here — yet the strong form was asserted unconditionally. |
| **B2** | A missing embedding on the NEW entity was blamed on the graph (*"N concepts exist but none could be compared"*), reproduced against two candidates that both had embeddings. Same class as the sentinel bug the author had already self-found — the ambiguity had merely moved up a level. |
| **B3** | `EmbeddingResolver._cosine_similarity` returns `0.0` for mismatched-length and zero-norm vectors — an out-of-band sentinel indistinguishable from a genuinely orthogonal pair. A 1024-vs-768 pair was reported as *"compared … at cosine 0.000 < 0.75"*. This repo has documented 768/1024 dimension drift. |
| **B4** | Any internal exception was stamped `no_candidates_fetched`, including crashes where candidates HAD been fetched successfully. No `EV_ERROR` existed. |
| **M1** | Judge batch keyed by surface form: two `is_new` entities named "Den Haag" (different types, different neighbours) → one ruling produced two RELATED_TO verdicts, `judged_count` diverged from `method_counts[llm_judge]`, and `target_id`/`target_name` in the **same evidence record pointed at different graph nodes** (item A validated against item B's neighbour list). |
| **M2** | `resolve_types` — the function fixing attempt 1's root cause — had **zero tests**, and the D-N4-3 regression test monkeypatched it away, then asserted something true for *any* implementation. |
| **M3** | The only subsumption tier matched an ancestor TYPE name against a candidate INSTANCE name, unverifiable (the row has no type column). Empirically near-inert: `canonical_bridge` terminates at the first mapped schema.org base, so `ancestors` is typically one English identifier that will not be a node name in a Dutch graph. |
| **M4** | `NOVEL` was a claim about an arbitrary `LIMIT 100` unordered sample, presented as a claim about the graph — and N.4c turns every NOVEL into a `record_gap`. |

## Attempt 2 — APPROVED

The fix was a restructure, not four patches: every `EV_*` code now names an
observation with exactly one cause.

- `_Fetch(rows, ok)` separates a raised fetch (`EV_FETCH_FAILED`, *"nothing was
  established"*) from an empty result (`EV_NO_ROWS`, whose evidence states that
  the repository reports a failed query as an empty result, so it cannot prove the
  graph is empty).
- `NeighbourProbe` + `probe_neighbours` separate the three "nothing was compared"
  causes: `EV_NO_QUERY_VECTOR` / `EV_NO_CANDIDATE_VECTORS` /
  `EV_INCOMPARABLE_VECTORS`.
- A local `_cosine` returns `None` for dimension mismatch and zero norm instead of
  inheriting the out-of-band `0.0`.
- `EV_ERROR` added.
- `JudgeItem = (item_id, text, neighbours)` — index-keyed, so a ruling touches one
  item and a borrowed target is downgraded to NOVEL for real.
- `resolve_types` tested against the REAL `canonical_bridge` with a real
  `Ontology`, plus an end-to-end fetch test with no stub.
- The type-chain tier is OFF by default, discloses its unverifiable match in the
  evidence string itself, and its near-inertness is documented.
- The LIMIT cap is disclosed in the evidence and reported as `candidate_cap` /
  `capped_type_fetches`.

The reviewer re-ran all four blocker probes and the M1 scenario against the new
code independently rather than trusting the summary. `564 passed, 1 skipped` plus
the pre-existing, unrelated `test_llm_matcher` failure; 47 tests in
`test_concept_alignment.py`; ruff clean.

## Non-blocking minors → carried forward as C1–C6

Recorded as binding line items in the N.4 chapter. The one to watch:

> **C1** — `EV_NONE_CLOSE` still has three causes (below-floor, judged-and-rejected,
> **unadjudicated band**), against D-N4-7's own "exactly one cause" contract. The
> below-floor and unadjudicated-band cases share both the code and `method=none`.
> Must be split before N.4c filters gap-recording on `reason_code`, or a concept
> nobody adjudicated gets recorded as a confirmed ontology gap.

C2 (cap disclosure in 2 of 5 paths), C3 (`type_chain_subsumption` omits
`canonical_type`, needed exactly where N.4b seeds), C4 (`EV_NO_REPO` drops
`canonical_type`), C5 (`assert` as a production control guard), C6 (plan drift —
fixed).

## Standing lesson

Both N.4a attempts, and the parked N.4 attempt before them, failed on the same
axis: **a claim that outran what had been established.** First lexical containment
read as subsumption; then a query result read as a fact about the graph. The
guard that worked was neither review nor intuition but *grounding every assumption
in the real code before building on it* — checking `find_by_type`'s actual
projection is what exposed two dead tiers, and testing `resolve_types` against the
real bridge is what turned a self-referential assertion into a real one.
