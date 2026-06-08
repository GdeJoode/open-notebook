# Review — Track B Phase B.1f attempt 1

**Branch**: `track/b-extraction-service-wiring`
**Decision**: REVISIONS_NEEDED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-07

## Summary

The wiring is mostly correct — the multi-schema flip happens in the right place (service, not workflow primitive), the kill-switch propagates through three call layers, the LLMExtractor's broken import is genuinely removed and pinned by an AST regression test, and the B.4 relation re-link logic correctly handles the entity/relation pass-winner divergence. Quality gates are all green (240+388+36+145=809 tests, 99% coverage on the orchestrator, 98% on the LLMExtractor).

However, three of the seven advertised acceptance criteria are claimed "PASS" without test backing. The 409 API contract, the worker's `JobPausedForReviewError → PAUSED_FOR_REVIEW` translation, and the `get_notebook_id` SurrealDB query are all asserted in the self-review with no covering test. For a PR that flips multi-schema mode ON by default, the absent worker-translation test is the most concerning: if a future refactor changes the worker's exception handling, no test catches it and every paused review job will look like a hard failure (retry, dead-letter).

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | `run_extraction(source_id, ontology_name="general")` (no `notebook_id`) → single-schema | PASS | `test_no_notebook_id_uses_single_schema` checks `mode != "multi"` in extract kwargs |
| 2 | `run_extraction(source_id="x", notebook_id="n")` → `_run_multi_schema` invoked | PASS | `test_notebook_id_routes_to_multi_schema` spies on `_run_multi_schema` directly |
| 3a | `review_required=True` + no accepted extensions → `SchemaReviewPendingError` raised by service | PASS | `test_review_required_raises` pins exception fields |
| 3b | API returns 409 Conflict on review-pending | UNVERIFIED | router pre-check exists in code (`sources_processing.py:151-183`) but no `test_sources_processing.py` covers `/run-entities` at all |
| 3c | Worker translates exception → `JobStatus.PAUSED_FOR_REVIEW` | UNVERIFIED | code path exists (`worker.py:154-167`) but no test in `packages/job-queue/tests/test_worker.py` covers this branch |
| 4 | `multi_schema_enabled=False` forces single-schema | PASS | `test_multi_schema_enabled_false_forces_single_schema` asserts `_run_multi_schema.assert_not_awaited` |
| 5 | New B.1e merge output flows through `persist_filtered_result` | PASS (indirect) | seam unchanged; the multi-schema return type is still `ExtractionResult` so the downstream pipeline is type-equivalent |
| 6 | LLMExtractor wired to `ModelManager`, no silent empty | PASS | `test_pre_b1f_import_path_no_longer_referenced` walks AST, `test_extract_dispatches_to_injected_caller` proves end-to-end wire |
| 7 | B.4: relation source/target rewritten to canonical entity text | PASS | `test_relation_endpoints_relinked_to_canonical_entity_text` pins concrete `Alice@0.9` vs `alice@0.9` scenario |

## Test status

```
packages/shared              : 145 passed
packages/job-queue           : 36 passed
pipelines/ontology-extraction: 240 passed (multi_schema_orchestrator 99% / llm_extractor 94% / total 98%)
apps/app-main                : 388 passed (entity_extraction_service 46% — see Major #2)
```

All workspace members pass. No regressions.

## Issues found

### 🔴 Blockers (must fix)

None. The code as written is consistent and the critical paths (multi-schema flip, LLMExtractor fix, B.4 relink) are covered with real assertions.

### 🟡 Major (must fix)

1. **Worker pause-translation untested** — `packages/job-queue/tests/test_worker.py`
   - Issue: The worker's `JobPausedForReviewError → JobStatus.PAUSED_FOR_REVIEW` branch (`worker.py:154-167`) has zero test coverage. Existing tests cover `COMPLETED`, `RETRYING`, `FAILED`, `CANCELLED`, and `missing` job paths but not the pause path. The self-review claims this as PASS via `test_paused_error_subclasses_job_paused`, but that test only checks `isinstance(err, JobPausedForReviewError)` — it does NOT exercise the worker.
   - Impact: AC #3c is unverified. If any future refactor of `_execute_job` removes or reorders the `except JobPausedForReviewError` clause (it sits above the generic `except Exception` — order matters), paused jobs will be treated as failures, retried, and dead-lettered. Silent regression.
   - Recommendation: add `test_paused_for_review_no_retry_no_dead_letter` analogous to `test_failed_job_exhausted_retries`, asserting `update_status(..., PAUSED_FOR_REVIEW, error_message=...)` and that `enqueue` / `add_to_dead_letter` are NOT called.

2. **`/run-entities` router untested** — `apps/app-main/tests/`
   - Issue: There is no `test_sources_processing.py` file at all. The router has non-trivial logic — the 409 pre-check (`sources_processing.py:151-183`), the kill-switch payload field forwarding, and the langextract option forwarding — none of which are exercised. AC #3 explicitly says "the API returns 409 Conflict"; this is asserted PASS in the self-review with no covering test.
   - Impact: The UI in B.3c keys off the response body `{"code": "schema_review_pending", "notebook_id": ..., "pending_count": ...}`. If a future change to the router silently drops a field, no test catches it.
   - Recommendation: add `test_sources_processing_router.py` covering at minimum (a) 409 + body shape when `review_required=True` + empty accepted, (b) 200/queued when extension accepted, (c) `multi_schema_enabled=False` payload pass-through to the queued job.

3. **`_run_multi_schema` orchestrator-glue untested** — `apps/app-main/src/app_main/services/entity_extraction_service.py:219-355`
   - Issue: Coverage on `entity_extraction_service.py` is 46% across the entire app-main suite. Lines 269-346 (the entire `_run_multi_schema` body — schema discovery, applicability detection, accepted-extensions broadcasting, LLM-caller construction, fallback-to-single-schema when no applicable schemas) are completely uncovered. The single test that touches this code (`test_notebook_id_routes_to_multi_schema`) patches `_run_multi_schema` itself, mocking out the very logic we need to verify.
   - Impact: The "no applicable schemas → fall back to single-schema" branch (line 303-315), the broadcast-extension logic (lines 321-331), and the lazy notebook_schema_repo construction (line 248) are all paths an end-to-end production flow will hit but no test pins. Specifically: the AC #1 regression guard says single-schema CLI must still work, and there's a SEPARATE single-schema fallback inside `_run_multi_schema` (no-applicable-schemas) which has different semantics — neither path is tested in isolation.
   - Recommendation: at least one test per branch — schemas-found vs schemas-empty (forces fallback), extension-with-schema-name vs extension-without (forces broadcast), llm-caller-success vs llm-caller-raises (forces lazy default).

### 🔵 Minor (optional follow-up)

1. **Orphan relations: behavior documented but untested** — `pipelines/ontology-extraction/src/ontology_extraction/multi_schema_orchestrator.py:715-749`
   - The code says "If [the entity] not found, leave [the relation] as-is — it'll be filtered out downstream — better than synthesising an entity." Sound design, but no test pins this contract. A future maintainer could decide to drop orphans here and break the silent-downstream-filter assumption with no test failing. Add a `test_relation_with_orphan_endpoint_passes_through_unchanged`.

2. **Per-call model override warning untested** — `apps/app-main/src/app_main/services/entity_extraction_service.py:115-119`
   - The conditional `if _model and _model not in ("default", model_record.id)` logs a warning to flag "caller asked for per-call override we ignored." The test `test_caller_routes_through_model_manager_and_achat_complete` passes `"default"` so the warning never fires. Implementer flagged this in the self-review §Known follow-ups #2. Add a test passing an unrelated model id and using `caplog` (or whatever loguru-test harness is in use) to pin that the warning fires.

3. **`SourceRepository.get_notebook_id` untested** — `packages/surrealdb-service/src/surrealdb_service/repositories/source.py:39-74`
   - Thin SQL wrapper, but the docstring claims "deterministic ordering" via SurrealDB's insertion-order guarantee on `reference` edges. The `LIMIT 1` query has no explicit `ORDER BY`. If insertion order is NOT actually guaranteed across the SurrealDB version we target, sources in multiple notebooks could route to a non-deterministic notebook-schema. Test or strengthen the query with an explicit `ORDER BY created`.

4. **Unused import in worker** — `packages/job-queue/src/job_queue/worker.py:9`
   - `from datetime import datetime, timezone` is imported but never used in this module (the actual datetime calls live in `repository.py`). Cosmetic.

5. **LLMExtractor "silent empty + warning" path is still a footgun** — `pipelines/ontology-extraction/src/ontology_extraction/extractors/llm_extractor.py:91-97`
   - The B.1f fix replaces a silent ImportError with a logged warning + empty result. Production-wise this means a wiring failure in `make_default_llm_caller` (caught at service line 478-482) produces an empty extraction with a logged warning — extraction "succeeds" with 0 entities, the job completes, the UI shows a finished extraction with no entities. Better than pre-B.1f, but still surprising. Consider raising in production wiring vs. test wiring once the lazy-default-empty path is no longer needed by any test.

## Decision rationale

- 0 Blockers
- 3 Majors (M1: worker pause-translation untested; M2: `/run-entities` router untested incl. 409 contract; M3: `_run_multi_schema` body untested, 46% coverage)
- 5 Minors

Per the decision matrix, ≥1 Major forces REVISIONS_NEEDED. All three Majors are about **test coverage of explicitly-claimed acceptance criteria**: AC #3 says "the API returns 409" and "the source's job goes to `paused_for_review` state" — both are claimed PASS in the self-review without supporting tests. AC #2 and the multi-schema branch are similarly only spy-mocked at the boundary, with the orchestrator-glue logic uncovered.

This is the production-mode-flip PR. The threshold for "trust without tests" is lower than usual. The code I read looks correct, but I cannot stamp APPROVED on three "PASS" claims that the test suite does not verify.

## Kudos

- B.4 relation re-link fix: the test `test_relation_endpoints_relinked_to_canonical_entity_text` is exemplary — it pins the exact `Alice@0.9 + Alice→MIT@0.6` vs `alice@0.6 + alice→MIT@0.9` scenario from the B.1e review, with explicit assertions on both `text == "Alice"` AND `source_entity == "Alice"`. The companion no-op test (`test_relation_endpoint_unchanged_when_already_canonical`) prevents the over-rewrite regression.
- LLMExtractor AST regression test: `test_pre_b1f_import_path_no_longer_referenced` uses `ast.walk` to verify the broken import path is structurally gone, not just absent from a string match. Higher bar than a typical import-test.
- Exception hierarchy: putting generic `JobPausedForReviewError` in `job-queue` (where the worker catches it) and specific `SchemaReviewPendingError` in `app-main.services` (where it's raised) is correct domain decomposition. The worker stays domain-agnostic.
- Defense-in-depth review gate: identical predicates in router pre-check and service-level enforcement (`nb_schema.review_required and not nb_schema.accepted_extensions`). The architectural comment about "if they drift, the worker check wins because it sees the freshest notebook state" is exactly right.
- Coverage on the critical pipelines is high: multi_schema_orchestrator 99%, llm_extractor 94%.

## Next steps

Implementer should address the three Majors (worker pause-test, router 409 test, `_run_multi_schema` branch tests) and re-submit. The Minors can be filed as follow-ups for B.3c or beyond — they don't affect the production-mode flip.
