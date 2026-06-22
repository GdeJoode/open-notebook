# Review — Track K Phase K.1 attempt 4 (rev3, Option A)

**Branch**: `track/k1-nl-normalizer`
**Fix commit**: `359d363` (range `e09e34a..359d363`)
**Decision**: REVISIONS_NEEDED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-22

## Summary

The name-only false-merge gate is genuine and binding, the strip-list edits are
correct for the curated corpus, the relation-endpoint regression test is real,
and the full suite is green (296 passed). However, criterion 3 fails: the
retained `ministerie van` strip still produces a realistic cross-type NAME
collision in the project's own live fixture — `Ministerie van Onderwijs` (org)
and `Onderwijs` (concept/topic) both normalize to `onderwijs`. This is the same
root-cause mechanism as the attempt-3 blocker (org-leader strip lands on a bare
word independently extracted as a different type), and the canary is green only
because the `must_not_merge` corpus omits that pair. The code comment's claim
that `ministerie van` is "same type, no cross-type collision" is empirically
false against `convenant_entities.jsonl`.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | Strip list no longer creates cross-type name collisions for the curated cases | ✅ | `Minister van BZK`!=`Ministerie van BZK`; `Gemeente/Provincie Groningen`!=`Groningen` — verified by execution |
| 2 | Name-only over-merge gate is real + binding | ✅ | `count_false_merges` compares names ignoring type; `test_name_only_collision_is_a_false_merge` proves it fires (1) where the type-aware variant returns 0; canary asserts 0 over full corpus; corpus contains the cross-type pairs + originals |
| 3 | No realistic remaining cross-type collision from the strip list | ❌ | `Ministerie van Onderwijs`(other) → `onderwijs` collides with `Onderwijs`(concept/topic). Present in live fixture, caused by the strip. Same shape as the removed gemeente/minister rules |
| 4 | Relation endpoints route to distinct type-correct buckets | ⚠️ | The new test is genuine for the BZK case (asserts both distinct surface forms appear as RELATE sources, relations_created==2). But the underlying name-only mechanism still collapses the `onderwijs` case |
| 5 | Fragmentation honesty (−19, more correct than rev2's −46) | ✅ | Verified empirically: 1360→1341 over 1402 entities. BZK cluster size 4 collapses only same-type `other` org forms (2 spelling variants + 2 ministerie forms) |
| 6 | No regression / B.8 contract | ✅ | 296 passed; hash_id derive-rule untouched (appears only in a status doc); ruff clean except one pre-existing I001 (confirmed present at base `e09e34a`) |

## Test status

```
packages/shared/tests/test_resolution_metrics.py packages/shared/tests/test_nl_normalization.py: 61 passed
apps/app-main/tests/test_notebook_merge_service.py: 8 passed
packages/shared/tests/ apps/.../test_notebook_merge_service.py test_entity_persistence_service.py
  packages/surrealdb-service/tests/test_entity_repository_roundtrip.py: 296 passed in 8.78s
ruff (changed files): 1 pre-existing I001 in test_notebook_merge_service.py (present at base commit)
```

## Issues found

### 🔴 Blockers (must fix)

1. **`ministerie van` strip still creates a cross-type NAME collision** — `packages/shared/src/shared/utils/nl_normalization.py:55` (`_ROLE_ORG_PREFIXES`)
   - Issue: `normalize_entity_name("Ministerie van Onderwijs") == normalize_entity_name("Onderwijs") == "onderwijs"`. In `tests/fixtures/entity_resolution/convenant_entities.jsonl`, `Ministerie van Onderwijs` is tagged `other` while bare `Onderwijs` is independently extracted as `concept`, `topic`, `organization`, and `other`. Under the rev3 name-only criterion (relations resolve endpoints by `WHERE canonical_name = $name` with no type filter), a relation to the education-concept and a relation to the ministry-org collapse onto one endpoint. This is the identical failure shape to the attempt-3 blocker — an org-leader strip lands on a bare word that is a different-typed real entity — just with `ministerie van` instead of `minister van`. The strip CAUSES it: without stripping, `ministerie van onderwijs` != `onderwijs`.
   - Same pattern also affects `Ministerie van Volkshuisvesting en Ruimtelijke Ordening` → collides with `Volkshuisvesting en Ruimtelijke Ordening`(topic), and `Ministerie van Volksgezondheid, Welzijn en Sport` → collides with the org/other bare forms (4 induced keys total in the live corpus).
   - The code comment at `nl_normalization.py` asserting `ministerie van` is "KEPT ... same type, no cross-type collision" is contradicted by the project's own frozen dump.
   - Recommendation: out of scope to prescribe the fix, but the claim that `ministerie van` is collision-free must be reconciled with the corpus — either the rule is shown safe (it isn't, for these keys) or the criterion-3 guarantee does not hold.

### 🟡 Major (must fix)

1. **Canary corpus omits the realizable cross-type pair, so it passes while the corruption exists** — `tests/fixtures/entity_resolution/must_not_merge.jsonl`
   - Issue: The corpus pairs `Ministerie van Onderwijs` only against `Ministerie van Onderwijs en Arbeid` (a tail-differs same-shape pair). It does not include `Ministerie van Onderwijs` vs bare `Onderwijs`(concept), which is the realizable cross-type collision present in the live fixture. The name-only gate is sound, but its corpus does not exercise the one collision the retained strip rule still produces, so `count_false_merges == 0` is green by omission rather than by correctness. Criterion 2(b) asks the corpus to include the cross-type pairs the strip can produce; the `ministerie-van`-induced one is missing.
   - Recommendation: the adversarial corpus should cover the collisions the kept rule can realistically generate.

### 🔵 Minor (optional)

1. **Stale module docstring** — `packages/shared/src/shared/utils/name_normalizer.py:16-18, 27-29`
   - The module docstring still says the K.1 stage strips "(`ministerie van`, `minister van`, `gemeente`, …)" and that "`Minister van BZK` and `Minister van Financiën` stay distinct ... `Gemeente` on its own is left unchanged rather than stripped." This contradicts rev3 (minister/gemeente are no longer stripped at all). The function-level docstrings in `nl_normalization.py` were updated; this top-level one was not.

## Decision rationale

Criteria 1, 2, 5, 6 hold and the relation test (criterion 4) is genuine for the
BZK case. But the review brief set the bar explicitly: "If you find a real
remaining collision, be specific... don't pass a real one." `Ministerie van
Onderwijs` vs `Onderwijs`(concept) is not a manufactured/marginal example — it
is in the project's frozen live corpus, it is caused by the retained strip rule,
it is the same class of cross-type collision the user chose Option A to
eliminate, and it routes a relation to a wrong shared endpoint by the very
name-only mechanism rev3 adopted. One blocker forces REVISIONS_NEEDED.

## Escalation recommendation

This is attempt 4 (the 3rd+ revision) and a blocker remains, so per the
adversarial-reviewer protocol this is flagged for escalation.

Rationale: the cycle has now caught person/org (`minister`), org/location
(`gemeente`/`provincie`), and now org/concept (`ministerie van X` where bare `X`
is also a concept/topic) collisions. The recurring root cause is that ANY
org-leader prefix strip can land on a bare token that the extractor independently
emits as a different-typed entity, because the relation layer resolves endpoints
by name alone. Removing prefixes one class at a time (rev2 → rev3) keeps
surfacing the next instance. A durable resolution likely needs a structural
decision the implementer cannot make unilaterally inside K.1: either (a) make
`ministerie van` stripping conditional/safe against the bare-token-as-other-type
case, accepting reduced consolidation, or (b) bring forward the K.7 type-safe
relation-endpoint resolution (already planned as Option B) so relations stop
resolving cross-type by name alone, at which point name-level collisions stop
corrupting the graph. The user should choose the path; that is an escalation
decision, not an implementer fix.

## Next steps

REVISIONS_NEEDED with escalation flag. The escalation-handler agent should
surface the org/concept collision class and the (a)-vs-(b) structural choice to
the user before another K.1-local revision attempt.
