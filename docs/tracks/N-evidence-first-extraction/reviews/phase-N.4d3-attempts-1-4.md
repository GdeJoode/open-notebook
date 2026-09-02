# Phase N.4d.3 — attempts 1–4 — VERDICT: APPROVED (attempt 4)

- **Branch**: `feature/track-n4d3-reparent-type` — merged as `801f84ae`
- **Commits**: `e733443` → `349135f` → `a7d3420` → `31e60c7d` → `c8f8d285`
- **Date**: 2026-09-02

Applying an accepted placement as a schema edit, and showing the curator the
placement in the first place. Three blockers and eight majors across four rounds,
every one verified dead by mutation rather than by a green suite.

## What the phase turned out to be

The plan reads like one service method. It is four pieces, because the AC promises
that an entity of a re-parented type resolves through its new ancestor via
`canonical_bridge` — and the bridge reasons over `Ontology` objects and never sees
`notebook_schema`. Recording the intent in `accepted_extensions` alone would have
shipped a feature that does nothing.

1. `SchemaEditService.reparent_type` + `POST /schema/reparent`. One entry per moved
   type, not one entry listing N: separately visible, separately reversible,
   rendered by existing tooling, and nothing for a caller to mis-iterate.
2. `ontology_manager.schema_projection` applies accepted edits to DEEP COPIES.
3. A symmetry fix in the bridge's parent walk.
4. `TypePlacementService` — the read half, running the placement and N.4d.2's judge
   at accept time and reporting what it found. It writes nothing.

## Measured before writing, not reasoned about

Two facts decided the design, and assuming either would have shipped a no-op:

* **The registry hands out shared objects.** A second `get('deals')` returns an
  identical instance, `entity_types` dict included. In-place editing would give the
  next notebook in the process this one's vocabulary.
* **The bridge prefers `schema_org_type` over `parent_type`.** 13 of 277 applied
  type entries declare one, and for 10 a `parent_type` rewrite alone leaves the
  canonical untouched. Every one is a root a curator plausibly re-parents. So a
  re-parent clears the field its placement overrides.

A third came out of a failing test rather than a probe: the walk honoured
`schema_org_type` only at step 1, never mid-chain. On the `(general, deals,
government)` set that decided 90 of 92 outcomes — a move under `Location` was
refused 90 times while a move under `Person` applied 89 — purely because `Person`
is spelled like a mapped base and `Location` is spelled differently from `Place`.
The fix moves **zero of 277** canonicals today; it is reachable only through a
re-parent.

## The four rounds

| Attempt | Verdict | What was wrong |
|---|---|---|
| 1 | 2 blockers, 5 majors | An alias-shadowing regression; AC 2 dropped with a reason narrower than the omission; three claimed mechanisms with no guard; a sweep that killed zero mutants. |
| 2 | 1 blocker, 2 majors | All in the NEW surface: a discriminator that did not discriminate, a "both endpoints" claim guarded on one, and a numbered decision that was itself falsifiable. |
| 3 | 1 major (+ procedural escalation flag) | A guard that could not fail, and a commit sentence claiming it could. |
| 4 | **APPROVED** | — |

## B1 (attempt 1) — the alias shadowing

`_materialise` refused a NAME collision but not an ALIAS one, while
`canonical_bridge._find_definition` matches names AND aliases in
ontology-then-insertion order. A definition materialised onto `schemas[0]`
outranked an alias owner in a later applied ontology. On `(base, deals, general)` —
the first group the module's own sweep iterates — six real aliases were affected:

```
BEFORE: topic / Topic          # label "Subject" -> canonical topic, via general.Topic
AFTER : concept / Subject      # after accepting an extension named "Subject"
```

`Subject`, `Theme`, `Category` shadow `general.Topic`; `Framework`, `Protocol`,
`Standard` shadow `general.Technology`. The guard tool — `type_placement.alias_owner`
— was already in the module this one imports from. It is the same harm
`_reparent_one`'s orphan check exists to prevent.

## The standing lesson, seventh entry

N.4d.1: measure the FIX the way the DEFECT was measured. N.4d.2: ask which
direction the danger is in, and assert in the shape production assembles. This
phase adds the sharpest form yet, because it recurred **inside a guard written to
prevent exactly this**:

> **A guard that cannot fail is worse than no guard, because it reports as one.**

It appeared three times, in three registers:

1. The whole-vocabulary sweep re-parented everything under `Person`, a mapped base
   NAME, so "resolved before → resolves after" held by construction. It killed
   **zero** mutants — not the `schema_org_type` clear, not the orphan check, not
   the cycle check. Rewritten to give each branch a move constructed to exercise
   it, with a floor per branch, it kills three.
2. Deleting the projection's ONLY production call site left all 1632 tests green.
   So did removing either re-parent consumer filter, both of which carried long
   docstrings about defence-in-depth.
3. The test written to stop D-N4-13 from drifting asserted
   `service._apply_notebook_schema_default is not None` — true of any object with
   that attribute — and re-asserted its own fixture. Removing the gate left all
   1670 green.

Each fix moved the claim to the seam where the mechanism runs, and added a vacuity
guard so the negative means something.

## D-N4-13, and being wrong twice about the same sentence

The first attempt argued AC 2 away: no applied set exists at accept time. The
review was right that this proves less than it claims — the notebook-level FORCED
set does exist. The second attempt wrote that down and added a new overstatement:
that a placement can never contradict the runtime verdict, since the forced set is
a subset. Disproved two ways, both reproducible:

* **Verdicts are not monotone.** `ScholarlyArticle` under `Deal` is `PLACED`
  against `(deals, policy_themes)` and `DUPLICATE` once auto-detection adds
  `scholarly`. A superset premise carries only monotone conclusions — and
  `PARENT_UNKNOWN` and missing siblings, the two consequences that attempt listed,
  happen to be the monotone ones.
* **The forced set is not always a subset.** `_apply_notebook_schema_default` is
  gated on a truthy `base_ontology`, which the Regio-Deal notebooks leave empty.
  There the runtime forces nothing at all.

What makes the disagreement tolerable is that the placement is ADVISORY: it writes
nothing, and a re-parent is applied only by an explicit POST. Both limitations are
now pinned as tests on both sides.

## Four states, because three of them look identical

`parse_judge_response` never returns `None`; every refusal yields an empty
selection. So a reply the parser REFUSED reported identically to a judge that
looked and moved nothing, while four docstrings and the API contract said
otherwise. Fixed at the parser — `JudgeSelection.decided`, False on every
`_nothing` path — so the distinction is available to every future consumer, not
just this caller. `judge_status` is `not_asked` / `unavailable` / `refused` /
`decided`; three carry an empty selection, which is why emptiness cannot be the
discriminator. **N.4d.4's gap loop must gate on this**, per C1: a concept nobody
adjudicated must not be recorded as a confirmed gap, and a refused reply is nobody
adjudicating.

## Binding for later phases

* Assert at the seam where the mechanism runs, and add the vacuity guard that makes
  a negative assertion meaningful. A test that "nothing was forced" is worth
  nothing unless a sibling test shows the forcing still fires.
* A behaviour change stated in a docstring is not thereby measured. Materialising
  accepted extensions makes them resolvable by the bridge where they were not —
  that is real, and it needed the alias guard nobody would have looked for.
* The re-parent discriminator is spelled in six places that cannot import one
  another; a test pins them together.
