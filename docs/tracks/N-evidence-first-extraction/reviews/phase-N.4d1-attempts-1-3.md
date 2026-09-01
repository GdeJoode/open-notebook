# Phase N.4d.1 — attempts 1–3 — VERDICT: APPROVED (attempt 3)

- **Branch**: `feature/track-n4d1-type-placement` — merged as `ab526fc`
- **Commits**: `ca49de9` → `6dfff32` (naming) → `dd77d4f` → `08f0e2e` → `901137e`
- **Date**: 2026-09-01

The deterministic half of D-N4-12: validate the parent a proposal DECLARES, and
enumerate the bounded candidate set for BROADER_THAN. First phase in which
subsumption sits where the question is well-posed.

## The finding that matters most

Attempts 1 and 2 were both rejected, and the reviewer named the cause in a single
sentence that is worth more than either fix:

> the two revisions that changed behaviour each introduced one instance of the
> failure this track keeps correcting, which is an argument for measuring the
> *fix* on the real vocabulary the same way the *defect* was measured.

That is exactly what had happened. Twice.

## Attempt 1 — REVISIONS_NEEDED (2 blockers, 3 majors)

| # | Defect |
|---|---|
| **B1** | `sibling_types` read only `parent_type`, but `canonical_bridge` PREFERS `schema_org_type` — and `general` (DEFAULT_ONTOLOGY) plus `base` declare **zero** `parent_type`. On the default ontology the candidate set was empty for every parent while the evidence said "therefore the only candidates". |
| **B2** | The "REAL shipped ontologies" fixture called the private `_load_from_file`, not `registry.get`, so it never resolved `extends`. Raw `deals` has 8 types; the applied vocabulary has 53. A load-bearing assertion was false on what production sees — the N.4a M2 lesson one level down. |
| **M1** | `known_schema_org_base` claimed the bridge and it "cannot drift apart" while drifting three ways: case, the `schema:` prefix `base.yaml` writes, and aliases in the parent slot. |
| **M2** | The CYCLIC branch had zero coverage, and its docstring claimed cycles unreachable for a new type — false, because the module's own model has a third state: referenced as a parent, defined nowhere. |
| **M3** | A schema.org base counted as an existing type in the PARENT slot but was invisible in the NAME slot: one string, both "existing" and "new", within a call. |

## Attempt 2 — REVISIONS_NEEDED (1 new blocker, 2 majors)

Every prior finding was closed and mutation-verified. But the fixes introduced:

- **B3** — stripping the `schema:` prefix made a type declaring
  `schema_org_type: schema:Person` root at ITSELF, so the declared PARENT was
  handed to the judge as a candidate to become the proposal's CHILD. Accepting it
  closes a two-cycle via a path that never consults `would_cycle`. Four of
  `general`'s eight types.
- **M4** — the M3 fix reused `EV_NAME_TAKEN` for a mapped base: one code, two
  causes, in the module asserting one cause per code, on the field N.4d.4 gates
  gap recording on.
- **M5** — the test written to close M2 asserted `PLACED` and the helper, never a
  CYCLIC placement. The docstring was corrected; the coverage gap was not.

## Attempt 3 — APPROVED

7 of 8 mutants die, including the two that had survived twice (delete the CYCLIC
branch; case-sensitive sibling dedup). The reviewer reproduced the numbers rather
than trusting them, and independently measured the *stronger* property the test
did not assert — that no candidate is a deeper ancestor of the parent — finding
zero violations across all eleven ontologies.

What earned approval was not a green suite (it was green in all three attempts)
but the **shape of the guard**: `test_no_candidate_can_close_a_cycle` states a
property over the whole vocabulary — 262 candidates, zero violations — instead of
pinning one hand-picked case, and it fails when the fix is reverted.

The three approved-with minors were closed rather than carried: `roots_at`'s
docstring lagging its own body by three lines; the both-fields precedence being
documented, true and unguarded (the last surviving mutant); and "necessary and
sufficient" holding only under an acyclicity assumption the sweep did not assert.

## Standing lesson, fifth entry — and it moved

The four prior phases failed with prose overreaching correct code. Here it moved
into the code itself: B1 and M1 were prose *and* implementation disagreeing with
the bridge, and B2 was a test that could not establish what its own docstring
claimed.

The counter-measure sharpened accordingly. "Measure the mechanism" is not enough
if only the defect is measured; **the fix has to be measured the same way**, and
the guard has to be a property over the real vocabulary rather than an example.
A test that pins agreement with a collaborator (`test_agrees_with_the_bridge`)
does what a docstring promising it cannot.
