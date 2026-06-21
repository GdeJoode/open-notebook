# Review — Track B Phase B.6 attempt 2

**Branch**: `track/b-notebook-merge` (HEAD `6894d97`) (PR #25)
**Decision**: **APPROVED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-12

## Summary

Attempt 1's two blockers (B1 timestamp-idempotency + B2 self-merge guard) and two majors (M3 docstring lie + M4 mock infidelity) are all fixed. The idempotency fix is **honest**: I independently reverted `_entity_matches` to include `updated_at` and ran the idempotency scenario — `second.entities_merged` jumped from 0 to 1 (counter inflates), confirming the strict-advance mock + semantic comparison genuinely catches the regression. Router 422 guards verified by 8 new tests pinning the branches. 508/508 app-main tests + 154/154 shared + tsc clean + Playwright 1/1 pass. One **minor** discovered (provenance_chain missing from `_entity_matches`), one **minor** flake-risk (float identity on confidence). Both deferrable to B.7 retro or a follow-up.

## Acceptance criteria

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | Synchronous merge endpoint | PASS | `POST /api/notebooks/merge`, returns `NotebookMergeReportResponse`. |
| 2 | Idempotent re-run → 0 entities, 0 relations | **PASS** | Now driven by semantic-content comparison; honestly tested with strict-advance mock. |
| 3 | OECD union type_tags | PASS | unchanged from attempt 1. |
| 4 | Relations dedup max-confidence | PASS | unchanged. |
| 5 | UI dialog multi-select + dry-run | PASS | unchanged. |
| 6 | Playwright happy path | PASS | 1/1 on port 8503. |

## B1 verification — independently reproduced inversion failure

**Claim**: `_entity_matches` excludes `updated_at` and `id`; idempotency holds via semantic comparison.

**Verification**:

1. Confirmed the comparison fields by code inspection (`notebook_merge_service.py:660-678`). Comparison set = `{canonical_name, entity_type, primary_type, confidence, type_tags(sorted), source_documents(sorted), properties}`. `updated_at` and `id` absent.
2. **Mental inversion empirically reproduced**: I monkey-patched `_entity_matches` to also compare `updated_at`, then ran the same scenario as `test_idempotent_re_run`. Result:
   ```
   first.entities_merged=1, second.entities_merged=1
   ```
   The second-run counter inflated to 1 (expected 0). This proves the strict-advance mock + semantic comparison combo is the only reason the real test passes — reverting either piece would break it. The regression test is honest.
3. Edge probes on `_entity_matches`:
   - `type_tags` reorder `['A','B']` vs `['B','A']` → matches (sorted)
   - `source_documents` reorder → matches (sorted)
   - `properties = {'k':'v1'}` vs `{'k':'v2'}` → does NOT match (writes)
   - Nested dict `{'a':{'b':1}}` vs same → matches (Python dict ==)
   - Row Nones for list/dict fields → handled via `or []` / `or {}`
   - Row has extra `properties` key → does NOT match (asymmetric, correct)

## B2 verification — router guards

8 router tests cover:
- `test_merge_happy_path` (200, baseline)
- `test_merge_rejects_overlapping_target_and_source` → 422 + service NOT called
- `test_merge_rejects_target_in_mixed_source_list` → 422 (target appears alongside other sources)
- `test_merge_rejects_empty_source_ids` → 422
- `test_merge_404_when_target_missing` → 404
- `test_merge_404_when_source_missing` → 404 with "source" in detail
- `test_merge_rejects_archived_target` → 422 + "archived" in detail (Minor 3)
- `test_merge_rejects_archived_source` → 422 + "archived" in detail (Minor 3)

Each rejection-path test asserts `merge_svc.merge_notebooks.assert_not_called()`, so the guards genuinely short-circuit before the service is touched.

## M3 verification — rename complete

`grep -rn '_find_target_entity'` returns ZERO hits in `apps/app-main/`. Only `_find_canonical_entity` remains. The docstring now correctly describes the global-key lookup ("Migration 39 keys entity rows globally by `(canonical_name, entity_type)`"). The unused `notebook_id` parameter has been removed.

## M4 verification — mock fidelity

`_make_service_with_fake_db._upsert` now:
1. Persists the FULL post-write row (not just `updated_at`).
2. Strictly advances via `fake.upsert_call_count`, producing `2026-06-12T00:00:01Z`, `…:02Z`, … on successive calls.
3. The idempotency test asserts `second_call_ts == "2026-06-12T00:00:02Z"` to PROVE the mock is wired (catches a future refactor that accidentally stops advancing).

## Quality gates (independently re-run)

- `apps/app-main` pytest: **508 passed** in ~67s
- `packages/shared` pytest: **154 passed** in ~2.4s
- `frontend` `npx tsc --noEmit`: clean
- `frontend` `npm run lint`: only pre-existing warnings (ExtractionTab `useEffect` deps, `<img>` warnings, unused `ModelStatus` type). Zero new warnings.
- Playwright `notebook-merge.spec.ts`: **1/1 pass** in 3.7s

## Issues found

### Blockers (must fix)

**None.**

### Major (must fix)

**None.**

### Minor (optional, can be deferred to follow-up)

1. **`_entity_matches` does not compare `provenance_chain`** — `notebook_merge_service.py:646-678`. `EntityRepository.upsert_entity` DOES union `provenance_chain` (entity.py:206-208 + 224), so a follow-up merge that adds a NEW source-notebook to the source list will silently extend `provenance_chain` while `_entity_matches` reports `match=True` → `entities_merged=0`. Not an issue for the strict "identical re-run" idempotency contract (AC #2) since same sources → same provenance. Becomes wrong only for the "merge subset, then merge superset" workflow. Recommend adding `sorted(existing_row.get('provenance_chain') or []) != sorted(merged.provenance_chain)` to the comparison, OR explicitly documenting that provenance extension is intentionally counted as zero. Either choice acceptable; leaving the silent path is a future-debugging hazard.

2. **Float identity on `confidence`** — `notebook_merge_service.py:666`. `float(existing) != float(merged)` is exact equality. SurrealDB round-trip CAN lose precision (typical JSON `0.85` → IEEE `0.84999…`). Today's confidence values are usually fixed-point at the source (`0.85`, `0.9`), so unlikely to flake, but the contract is brittle. Recommend `math.isclose(..., abs_tol=1e-9)` or document the brittleness in the docstring. Defer to B.7 retro lessons.

3. **Mock counter format-string `02d` overflows at 100 upserts** — `tests/test_notebook_merge_service.py:228`. Cosmetic; current tests use ≤ 5 upsert calls. Worth a one-line comment ("supports up to 99 calls before the timestamp string format becomes invalid").

4. **Sources existence + archived check costs N extra round-trips** — `notebooks.py:262-276`. For the Minor 3 guard, we now make N `notebook_service.get(...)` calls before calling the merge. For typical N≤5 source notebooks this is negligible (dominated by the entity scan that follows). For larger source lists, consider folding into a single `WHERE id IN $ids` lookup. Cost-acceptable today.

5. **Minor 4 deferred (dialog component tests)** — implementer rationale (Playwright covers the same code paths) is acceptable. Not a blocker.

## Kudos

- **Honest regression test**. The strict-advance mock + semantic comparison combo is the right shape: the mock now matches production `time::now()` semantics, and the only way the test passes is the correct fix. I independently verified that reverting `_entity_matches` to compare `updated_at` makes the test fail.
- **Module docstring updated to explain the abandoned approach** (lines 36-52) — future readers see WHY the timestamp signal was rejected and don't accidentally reintroduce it. This is exactly the kind of "why" comment PRINCIPLES.md calls for.
- **B2 guard wired at router boundary** (not just UI). Symmetry with pydantic `min_length=1` means the API contract is bulletproof from the OpenAPI surface inward.
- **`_create_relation` Minor 1 fold** done cleanly: dry-run still uses the standalone `_relation_exists` probe (correct — must not write), commit path saves one round-trip per relation.
- **Test coverage of guards is exhaustive**: every 422/404 branch has an explicit `merge_svc.merge_notebooks.assert_not_called()` assertion, which would have been easy to skip but catches "what if a future refactor accidentally short-circuits the guard into the service?".
- The renamed `_find_canonical_entity` docstring correctly cites Migration 39 as the source of truth — readers can trace why the lookup is global.

## Decision rationale

Zero blockers. Zero majors. Five minors, all deferrable. The attempt-1 blockers were the production-correctness anchors (B1) and the API-contract anchor (B2); both are now fixed with HONEST tests (verified by inversion). M3 and M4 are clean code-quality fixes with no remaining ambiguity.

The minor on `provenance_chain` is the closest to "should be fixed before merge" — but the idempotency contract (AC #2) explicitly says "same merge twice", and same sources → same provenance → no asymmetry triggered. Mark it as a B.7 retro note + follow-up issue.

**This phase is APPROVED for merge.** Track B is feature-complete pending the B.7 RETRO.

## Independent verification of idempotency honesty (per reviewer charter)

Confirmed: I monkey-patched `NotebookMergeService._entity_matches` to ALSO include `updated_at` in the comparison, then re-ran the `test_idempotent_re_run` scenario manually. Without my patch: `first=1, second=0` (passes). With my patch: `first=1, second=1` (fails). The regression test is therefore a faithful guard against the attempt-1 bug — a future refactor that reintroduces timestamp comparison will be caught.

## Next steps

- **Track B implementer**: merge PR #25 after human sign-off. File a follow-up issue (or B.7 retro entry) for `provenance_chain` in `_entity_matches`.
- **Track B**: proceed to B.7 RETRO. B.6 is the LAST feature phase per plan.
- **Future-B note**: float identity on `confidence` should be addressed before this codebase serves merges across heterogeneous source systems (JSON round-trip risk).
