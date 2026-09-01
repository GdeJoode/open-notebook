# Phase N.4d.0 — attempts 1 & 2 — VERDICT: APPROVED (attempt 2)

- **Branch**: `feature/track-n4d0-retire-instance-tier` — merged as `d347474`
- **Commits**: `c7f65f3` (removal) → `b87e08e` (claims corrected) → `a426443` (residuals)
- **Date**: 2026-09-01

N.4d.0 is the removal that D-N4-12 implies: the instance-level subsumption tier
was already on main (shipped in N.4a/N.4b), so relocating subsumption to the type
boundary would otherwise have left machinery that looks alive but can never be
correct.

## The shape of this review

**The code removal was mechanically clean from the first attempt**, and the
reviewer verified that rather than assuming it: no orphaned exports (checked by
import), no unwired config field, no helper gone dead, `ConceptAligner.__init__`
keyword-only so no positional caller breaks, `resolve_canonical_type`'s narrowing
behaviourally identical, and no consumer anywhere in `apps/`/`packages/` reading a
removed symbol. Coverage loss was verified one-for-one: collect-only 596 → 579, and
the 17 removed test functions account for exactly that. Both new guards were
mutation-tested and both fail when a producer is reintroduced.

**Every issue was in what the commit said about itself.** That is the fourth
consecutive phase in this track with the same failure mode, and it is worth naming
as the standing pattern: the code converges, the prose runs ahead of it.

## 🔴 B1 — a guarantee disproved by mutation, then withdrawn

The commit kept N.4b's placement assertions and claimed they still guard against a
future producer reintroducing either blocker. The reviewer reintroduced a producer
in **the historically correct shape** — an edge into an existing, **off-batch**
graph node, exactly what the retired tier seeded — placed before Stages 11 and 12,
and all thirteen workflow tests stayed green: the ontology filter discards the edge
before centrality ever sees it. The tests fire only for ON-batch endpoints, so they
catch the safe reintroduction and miss the dangerous one.

**Resolution: the claim was dropped, not propped up.** With no producer in the
tree, a guard test must construct its own producer, so the only thing it can verify
is that the fixture it just wrote sits where its author put it — and its green
status would then read as coverage it never had, which is the same
"looks-tested-while-testing-nothing" pattern that let both N.4b blockers through
originally. The reviewer agreed explicitly and said he would have pushed back on a
guard. The requirement now lives in the plan as binding on whoever adds a producer:
re-establish it with an **off-batch** endpoint test.

## 🟡 The majors, all of one kind

| # | Claim that outran what was established |
|---|---|
| M1 | `resolve_canonical_type` still documented the deleted two-tuple contract — the one signature this commit changed, misdescribed for whoever writes N.4d.1 |
| M2 | Present tense for machinery that does not exist: "subsumption now lives at the type boundary". N.4d.1–.3 are unwritten; today this system decides subsumption **nowhere** |
| M3 | "every `entity` row is written by `EntityPersistenceService`" is false — `vault_sync_service` writes rows directly — and the same docstring refuted its own absolute three paragraphs later. Narrowed to what was observed: no writer creates a row DENOTING an ontology type |
| M4 | "earlier versions produced `NARROWER_THAN`/`BROADER_THAN`" reversed an established fact: BROADER_THAN was declared but never producible, and seeding produced zero edges under shipped defaults |
| M5 | The shared `FilteredResult` docstring still advertised the removed `seeded_is_a` key |
| M6 | The test baseline was guessed, not measured: main is **594**, not 599, so the delta is **17** — exactly the deleted test functions, which is the stronger claim |
| M7 | `canonical_type`'s justification named a consumer this commit deleted; kept as audit provenance and now says so |
| M8 | A test docstring kept an N.4b measurement (0.5 vs 0.25974) it can no longer reproduce |

## A claim of a fix that was not made

The re-review's first minor was a correction to the implementer's own revision
report: it told the reviewer M2's tense fix covered `config.py` "same as
`workflow.py`", and `config.py` was not in the diff at all. Recorded here because
it is the same category as the majors above, one level up — and because the
reviewer's decision to grade the artifact rather than the report is what caught it.

## Standing lesson, fourth entry

N.4a: evidence that reported an observation as the inference it would license.
N.4b: the same, in a log line telling the operator the stage would do nothing while
it recorded verdicts. N.4c: the same, in a commit message claiming the pipeline now
seeded under shipped defaults when the stage was `enabled=False`. N.4d.0: the same,
in a guarantee that a ten-line mutation disproved.

What keeps working is unchanged and now well-evidenced: **measure the mechanism
instead of reasoning about it**, and **mutate the guard to find out whether it
would notice**. Both of the reviewer's decisive findings in this phase came from
running the real thing, not from reading it.
