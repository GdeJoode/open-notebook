# Review — Track B Phase B.1e attempt 1

**Branch**: `track/b-multi-schema-orchestrator` (HEAD `22371a8`, last code `5e4fa93`)
**Decision**: **APPROVED** (with major tracked as B.4 follow-up)
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-06

## Summary

B.1e is the largest, most load-bearing phase in Track B. All six plan acceptance criteria met. 47 new tests / 234 total in `pipelines/ontology-extraction`; 99% coverage on `multi_schema_orchestrator.py` (178/180 lines, two genuinely defensive guards). Shared config is a single source of truth (`is`-identity verified at runtime). Failure-isolation paths pinned. Token-budget guard holds.

0 blockers · 1 major (spec-literal, tracked for B.4) · 5 minors (test-tightening + plan-text drift).

## Acceptance criteria check

| # | Criterion | Status |
|---|---|---|
| 1 | Entity in ≥ 2 passes → `type_tags` ≥ 2 + `primary_type` from highest-conf pass | PASS — `test_multi_tagged_entity_from_three_passes` (3 passes Alice; Researcher@0.95 wins) |
| 2 | Pass-2 prompt ≤ 3000 tokens | PASS — `test_each_pass2_prompt_under_3000_tokens` + Pass2's own `Pass2TokenBudgetExceeded` is the real guard |
| 3 | `pass1_results` row per schema attempted | PASS — `test_two_schemas_persist_pass1_records` + 3 failure-isolation tests |
| 4 | Single-schema input bit-identical to direct `run_pass2` | PARTIAL test, full implementation — architectural identity via `model_copy(deep=True)`; test only pins 5 entity fields (minor 4) |
| 5 | Shared config single source of truth | PASS — `is`-identity at runtime; test uses `==` only (minor 2) |
| 6 | ≥ 85% line coverage | PASS — 99% (missing lines: budget-edge break + empty-relation-type guard, both defensive) |

## Major (track as B.4 follow-up — do NOT block merge)

**Merge step does not re-link relation endpoints to canonical merged-entity surface form** — `multi_schema_orchestrator.py:682-696`

Relations are deduped by `(normalize(src), normalize(tgt), rel_type)` with max-confidence winning, but the surviving relation retains raw, un-normalized `source_entity`/`target_entity` text. When the entity merger and relation merger pick different pass-winners, the relation's endpoint string no longer matches the merged entity's `text` field.

Concrete: Pass-A `Alice@0.9` + `Alice→MIT@0.6`; Pass-B `alice@0.6` + `alice→MIT@0.9` → entity `text="Alice"` (Pass-A wins), relation `source_entity="alice"` (Pass-B wins). Downstream `entity_by_text[rel.source_entity]` misses.

This is spec-literal (plan §272-275 didn't mandate endpoint rewriting), but the downstream KG-quality consequence is real. Track as a B.4 follow-up.

## Minors (5, non-blocking)

1. AC #1 dedup-collision path (same label, same entity) covered for mechanic but not for AC text.
2. Shared-config test asserts `==` rather than `is` (verified by reviewer that `is`-identity holds; tighten test).
3. Tie-breaking on equal-confidence entities not tested — first-seen wins; deterministic but unpinned.
4. AC #4 "bit-identical" test only pins 5 entity fields; widen with `model_dump()` round-trip.
5. Plan §281 says `mode = "multi"` default; implementation uses `"single"` (correct for staged rollout). Update plan text or add `# B.1f flips this` comment.

## Test status (independently verified)

- `pipelines/ontology-extraction`: 234 passed (187 baseline + 47 new), 99% on orchestrator
- `packages/shared`: 145 passed
- `apps/app-main`: 368 passed (no regressions)
- `packages/surrealdb-service`: 52 + 20 docker-skipped (env)

## Kudos

- Failure isolation is exemplary (3 distinct failure-mode tests).
- `MergedEntity` non-Pydantic plain class with `__slots__` is the right transient-bookkeeper call.
- Telemetry naming convention matches B.1c/B.1d (greppable, dashboard-ready).
- Conservative name normalizer REUSED via single import — Q9 swap will be one-file change.
- Per-schema `accepted_extensions_by_schema` routing tested.

## Next steps

APPROVED — ready for merge. Implementer to file relation-endpoint re-linking as B.4 follow-up issue.
