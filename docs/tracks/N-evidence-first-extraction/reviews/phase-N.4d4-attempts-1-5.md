# Phase N.4d.4 — attempts 1–5 — VERDICT: APPROVED (attempt 5)

- **Branch**: `feature/track-n4d4-gap-loop` — merged as `948416bf`
- **Commits**: `6b59b939` → `57a2e0b2` → `27f60e34` → `3da176ef` → `41e6eab8` → `2175e70e`
- **Date**: 2026-09-02

The gap loop, and C1 as its precondition. Three blockers and seven majors across
five rounds; roughly sixty mutants run by the reviewer, all dead at the end.

## What shipped

1. **C1** — `EV_NONE_CLOSE` had three causes. Split into `EV_NONE_CLOSE`
   (compared, nearest below the floor), `EV_JUDGE_NO_LINK` (the judge ruled on
   this item), `EV_BAND_UNADJUDICATED` (the band, nobody ruled). Two of the three
   also shared `method=none`, separable only by comparing `similarity` to a floor
   the consumer had to know out of band.
2. **The gap loop** — `record_gap` on a NOVEL verdict, gated on the REASON CODE.
   Only the two codes that establish something license a gap; everything else is
   NOVEL because NOVEL is the safe default.
3. **D-N4-8** — `ENABLE_CONCEPT_ALIGNMENT` plus DI for all four collaborators.
   Without it the stage was unreachable in every real run.

## The telling fact about C1

After the split, **all 61 existing tests passed unchanged**. Nothing in the suite
distinguished the three sites, which is exactly why the collapse survived three
prior sub-phases. Four tests now pin each outcome to its own code and one asserts
all three are mutually distinct.

## The blockers

**B1 (attempt 1) — gaps were named after a per-document ranking.** The name came
from `applicable_schemas[0]`, and `detect_applicable_schemas` ranks by content
overlap with the document in hand. Gaps are keyed on
`(entity_text, ontology_name)`, so the same concept in two documents of one
notebook landed in two rows at frequency 1 instead of one at frequency 2 —
defeating the cross-document accumulation the `source_id` plumbing exists for,
and contradicting `_schema_name`'s own docstring. Two halves: the name is now the
notebook's DECLARED `base_ontology`, and alignment receives ALL applied schemas
(with `top_k=3`, a type declared in the second or third failed to resolve and
produced a code licensing no gap).

**F1 (attempt 4) and its residual (attempt 5) — the same lesson, twice missed.**
This phase's own plan bullet says: *`record_gap` swallows its own exceptions and
returns a gap with `id=None`; treat a null id as "not recorded", never as
success.* Its sibling `get_gap_statistics` does the same thing — swallows and
returns `{"ontology_name": …, "error": str(e)}` — and the first version watched
only for a raise, so a store that was down reported `ok` with the error payload
as the standing totals, persisted into `extraction_result.metadata`. The fix then
read that key by TRUTHINESS, and `str(e)` is `""` for any exception raised without
arguments. Measured against the real agent:

```
RuntimeError('db is down') -> unavailable      # the only one that worked
RuntimeError() / KeyError() / TimeoutError() / Exception() -> ok
```

A bare `TimeoutError` is what an asyncio timeout on that query raises — a slow gap
store under load, precisely the condition the status exists to surface.

## The standing lesson, eighth entry — and the first time it broke

N.4d.3 established: *a guard that cannot fail is worse than no guard, because it
reports as one.* It recurred here three times, twice **inside the fix for a
finding about exactly that**:

| Attempt | The assertion that could not fail |
|---|---|
| 1 | The stub left `_applicable_schemas` at `None`, so the ontology assertion was vacuous. |
| 2 | The stub performed the stash itself, so the assertion round-tripped the stub's own literal. |
| 3 | The helper always supplied `ontology=`, so both the correct and the reverted condition were false — in the fix for finding (2). |

Attempts 4 and 5 contain none. What broke it was not more care but three
structural rules, each of which killed a mutant that had survived a prior round:

> 1. **A test double must reproduce the real method's failure RETURN**, not a raise
>    it never performs.
> 2. **At least one fixture must build the PRODUCTION argument set**, not a
>    superset. A helper that always supplies an argument the app has stopped
>    supplying makes the discriminating configuration unreachable.
> 3. **A guard that reads a collaborator's value must be exercised against the
>    REAL collaborator at least once.** Both blockers were found this way and by
>    no other means; neither was findable by reading the diff.

These are recorded as binding in D-N4-14.

## Process note worth keeping

One correction in attempt 3 was committed and did not exist: a string replace on
`plan.md` silently matched nothing, because the anchor had drifted and nothing
asserted it. Every plan edit since asserts its anchor. A decision record that
quietly fails to update is worse than one never written, because the commit
message says it was.

## Operational note for N.5

The entity-filtering suite runs in an environment with a **reachable live
SurrealDB**. Any future test that calls a repository or agent method without
patching will silently exercise the real database and can pass for the wrong
reason. Both real-agent tests here were checked unpatched and do fail without the
patch, so the sweep is not green because a store happens to be absent.
