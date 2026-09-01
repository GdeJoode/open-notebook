# Phase N.4d.2 — attempts 1–3 — VERDICT: APPROVED (attempt 3)

- **Branch**: `feature/track-n4d2-placement-judge` — merged as `3b02d76`
- **Commits**: `a1e40d9` → `c1f5396` → `97cebc2` → `02b156c`
- **Date**: 2026-09-01/02

The judgement half of D-N4-12: which of a proposal's siblings actually belong
under it. Pure by design — builds a prompt, parses a reply, never calls a model —
matching the package's own convention and keeping the judgement testable without
one. The LLM call belongs to N.4d.3.

## The three rounds, and why they are one lesson

Each attempt was rejected for a different SHAPE of the same failure. Stating them
together is the point:

| Attempt | Shape |
|---|---|
| 1 | **A fence claimed but removed.** The docstring said the fences were "each inherited from a specific earlier failure"; in fact both ancestors (`not_a_concept.py:247`, `concept_alignment.py:408`) open their loop with an `isinstance` guard that this module dropped. |
| 2 | **A guard measuring the direction that was never at risk.** The vocabulary-wide sweep fed "every valid id plus invented ones", which can only detect widening. The reviewer's `chosen = list(offered)` mutant — move everything regardless of the reply — passed it. |
| 3 | **A guard asserted where it could not fail.** The same-name test loaded ONE ontology per iteration, and within one `Ontology` `entity_types` is a name-keyed dict, so the property held by the dict rather than by the mechanism. Deleting `sibling_types`' de-duplication left all 27 tests green. |

## The blocker (attempt 1)

`for item in data.get("move_under_proposal", []) or []` never checked that the
value is a **list**. Python iterates a `str` by character and a `dict` by key, so:

```
deals, parent CreativeWork, 11 siblings, ids 0-10
  {"move_under_proposal": "10"}              -> moved 'Article' + 'Report'   (model meant 'Project')
  {"move_under_proposal": {"0":false,"1":true}} -> moved both                (an explicit NO became a yes)
  {"move_under_proposal": 2}                 -> TypeError out of the parser
```

`widened` was `False` in each case, so the invariant the phase leaned on never saw
it — and it moved in the **over-move** direction, the one the prompt and system
prompt both exist to suppress.

## What closed it

* The parser requires a list, skips non-scalar elements, refuses a top-level array
  by which **structural** character comes first (so bare, ```json-fenced and
  prose-prefixed are one shape), and fails closed everywhere.
* The sweep became **two-sided**: eleven refusal shapes must select nothing at
  every parent, alongside the widening check.
* The sweeps run **production-shaped applied sets** — groups of three, matching
  `detect_applicable_schemas(top_k=3)`, plus the whole vocabulary. Measured: 64
  parents, 274 candidates, and the de-duplication fires 250 times where a
  single-ontology load fired it zero times.
* `test_a_same_named_pair_really_exists_to_be_deduplicated` guards the property
  against becoming vacuous if the vocabulary changes — attempt 3's failure mode
  encoded as a test rather than as a resolution.

Final mutation state: every non-equivalent mutant dies. One survives and is
provably equivalent (with positional ids, `str()` of a dict/list/bool can never
equal a decimal string).

## Found while fixing, and worse than the finding

The sweeps computed their parent set from `schemas[0]` alone. Under a
single-ontology load that was invisible; the moment applied sets grew past one
ontology it covered 24 parents while reporting the shape of 64. Only the floor
assertion caught it — which is the argument for floors that looked cosmetic when
they were written.

## Standing lesson, sixth entry

N.4d.1's approval established: measure the FIX the way the DEFECT was measured.
This phase adds two refinements, both learned the hard way:

1. **Ask which direction the danger is in.** A sweep guarding the safe direction
   feels exactly as rigorous as one guarding the dangerous direction.
2. **Assert in the shape production uses.** A property asserted in a
   configuration the system never assembles can be true, swept, and still
   unfalsifiable.
