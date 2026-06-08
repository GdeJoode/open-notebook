# Review — Track B Phase B.1f attempt 2

**Branch**: `track/b-extraction-service-wiring`
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-08

## Summary

All three attempt-1 Majors are properly closed by real tests that exercise the
right code paths with assertions that would fail on plausible regressions
(not just smoke "code runs" checks). M1 pins the worker's ordered-except
invariant (both for the base exception and a user-defined subclass). M2 pins
the 409 response body shape the B.3c UI keys off, including a 404-precedence
test that prevents leaking review state for nonexistent notebooks. M3 covers
six branches inside `_run_multi_schema` end-to-end including the schemas-empty
fallback, factory-failure non-fatal path, and the schema-tagged-vs-broadcast
extension routing. Test counts (38/241/401/145) and coverage delta (46% → 64%
on EES) match the self-review claims exactly. Two minor deferrals
(M3 SQL `ORDER BY`, M5 LLMExtractor raise-on-empty) are documented with
acceptable rationale. Production-mode flip is ready to merge.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | `run_extraction(source_id, ontology_name="general")` (no `notebook_id`) → single-schema | PASS | `test_no_notebook_id_uses_single_schema` checks `mode != "multi"` |
| 2 | `run_extraction(source_id="x", notebook_id="n")` → `_run_multi_schema` invoked | PASS | `test_notebook_id_routes_to_multi_schema` + new `TestRunMultiSchemaBody` exercises body end-to-end (was UNTESTED in attempt 1) |
| 3a | `review_required=True` + no accepted extensions → `SchemaReviewPendingError` raised by service | PASS | `test_review_required_raises` pins exception fields |
| 3b | API returns 409 Conflict on review-pending | PASS | NEW `test_returns_409_with_contract_body_when_review_required` pins the exact body shape `{code, notebook_id, pending_count, message}` |
| 3c | Worker translates exception → `JobStatus.PAUSED_FOR_REVIEW` | PASS | NEW `test_paused_for_review_no_retry_no_dead_letter` + `test_paused_for_review_subclass_routes_correctly` |
| 4 | `multi_schema_enabled=False` forces single-schema | PASS | `test_multi_schema_enabled_false_forces_single_schema` + NEW `test_multi_schema_enabled_false_forwarded_to_job_payload` |
| 5 | New B.1e merge output flows through `persist_filtered_result` | PASS (indirect) | Seam unchanged; multi-schema return type is still `ExtractionResult` |
| 6 | LLMExtractor wired to `ModelManager`, no silent empty | PASS | `test_pre_b1f_import_path_no_longer_referenced` (AST) + `test_extract_dispatches_to_injected_caller` + NEW `test_invokes_default_llm_caller_factory_for_multi_path` |
| 7 | B.4: relation source/target rewritten to canonical entity text | PASS | `test_relation_endpoints_relinked_to_canonical_entity_text` + NEW `test_relation_with_orphan_endpoint_passes_through_unchanged` (closes minor 1) |

## Test status

Executed all workspace members on the detached `track/b-extraction-service-wiring` head:

```
packages/shared              : 145 passed in 2.87s
packages/job-queue           : 38 passed in 39.54s (was 36, +2 worker pause)
pipelines/ontology-extraction: 241 passed in 49.96s (was 240, +1 orphan-relation)
apps/app-main                : 401 passed in 60.31s (was 388, +13: 6 router + 6 _run_multi_schema body + 1 model-override warning)
```

Coverage on `entity_extraction_service.py` (full app-main suite):
```
src/app_main/services/entity_extraction_service.py     213     76    64%
```
Confirmed delta 46% → 64%. Uncovered lines (502-560, 589-670, 674-702) are the
filtering workflow, `run_filtering_only` standalone entry point, and
`_save_result` DB write — all orthogonal to the multi-schema critical path.

No regressions across any workspace member. Test count math checks out
exactly (388 + 13 = 401, 36 + 2 = 38, 240 + 1 = 241).

## Issues found

### 🔴 Blockers (must fix)

None.

### 🟡 Major (must fix)

None.

### 🔵 Minor (optional follow-up)

1. **`get_notebook_id` SQL still lacks explicit `ORDER BY`** —
   `packages/surrealdb-service/src/surrealdb_service/repositories/source.py:60-63`
   - The query `SELECT VALUE out FROM reference WHERE in = $source LIMIT 1`
     relies on SurrealDB's documented-but-version-dependent insertion-order
     behaviour to pick "a" notebook when a source belongs to more than one.
     The docstring documents this contract, but a future version bump could
     break the ordering silently. In the common 1-notebook case this is a
     non-issue; in multi-notebook sources it could route to the wrong
     notebook-schema (wrong review gate, wrong applicable schemas).
   - Acceptable as a follow-up since (a) multi-notebook sources are
     uncommon, (b) the docstring already documents the assumption, and
     (c) the downstream notebook-schema repo just needs *a* notebook.
     File as a B.3c follow-up or pin to the next version-pin update.

2. **`LLMExtractor` silent-empty + warning path is still a footgun** —
   `pipelines/ontology-extraction/src/ontology_extraction/extractors/llm_extractor.py:91-97`
   - When `llm_caller is None`, the extractor warns and returns
     `ExtractionResult()`. In production this is observable via loguru
     warnings but the job still completes "successfully" with 0 entities
     — surprising for an operator. Implementer's deferral (would ripple
     through several test fixtures) is justified given the dedicated
     factory-failure-non-fatal test now pins the upstream guard at line
     337-344.
   - File as a production-hardening follow-up (B.3c or later).

3. **Single-schema LLM caller wiring exception branch uncovered** —
   `apps/app-main/src/app_main/services/entity_extraction_service.py:478-479`
   - The legacy single-schema path has its own
     `make_default_llm_caller(...)` call wrapped in try/except, but the
     except branch (legacy path) is uncovered. Mirrors the multi-schema
     test pattern; would be a one-line addition. Doesn't affect the
     multi-schema critical path so OK to defer.

## Decision rationale

- 0 Blockers
- 0 Majors
- 3 Minors (all documented or filed as follow-ups)

Per the decision matrix, 0 Blockers + 0 Majors → APPROVED. The three
attempt-1 Majors are all closed by real tests:

- **M1 (worker pause-translation)**: Two new tests in
  `packages/job-queue/tests/test_worker.py` directly raise
  `JobPausedForReviewError` (and a subclass) from a registered handler
  and assert `update_status(..., PAUSED_FOR_REVIEW, error_message=...)`,
  `RETRYING`/`FAILED` were NEVER called, `add_to_dead_letter` was NEVER
  called, and the queue is empty (no re-enqueue). The docstring explicitly
  documents the ordered-except invariant. Mentally inverting the except
  order in `worker.py:152-167` would cause both tests to fail loudly on
  the `JobStatus.PAUSED_FOR_REVIEW` assertion. The subclass test is
  important because the real-world raiser (`SchemaReviewPendingError` in
  app-main) is a subclass.

- **M2 (router 409)**: NEW `test_sources_processing.py` with 6 tests.
  `test_returns_409_with_contract_body_when_review_required` pins the
  exact contract body shape `{detail: {code: "schema_review_pending",
  notebook_id: "notebook:xyz", pending_count: 2, message: <str>}}` that
  the B.3c UI keys off. `test_no_409_when_accepted_extensions_present`
  proves the gate opens once an extension is accepted.
  `test_404_when_source_not_found` proves the source-not-found 404 fires
  BEFORE the schema-review pre-check (no information leak about review
  state for nonexistent notebooks). `test_multi_schema_enabled_false_*`
  proves the kill-switch reaches the job payload.
  `test_langextract_options_forwarded` pins all five langextract option
  fields. `test_langextract_path_skips_review_gate` proves the gate is
  bypassed for non-llm extractors with `mock_source_repo.get_notebook_id
  .assert_not_called()` — verified bypass is intentional given
  langextract runs an entirely separate code path.

- **M3 (`_run_multi_schema` body)**: NEW `TestRunMultiSchemaBody` class
  with 6 tests. Each test patches at a different seam to pin one branch.
  Schema-discovery test asserts `document_type`, `document_text`
  (truncated ≤ 2000 chars), and both candidate ontologies are forwarded
  to `detect_applicable_schemas`. Empty-applicable test asserts the
  fallback to single-schema (positional `workflow.extract(chunks)`, no
  `mode="multi"` kwarg). Factory-success test asserts the sentinel caller
  is threaded through to `workflow.extract`'s `llm_caller` kwarg.
  Factory-failure test asserts the service does NOT raise AND
  `llm_caller=None` is forwarded (Pass-1/Pass-2 fall back to their lazy
  defaults). Schema-tagged routing test asserts a schema-named extension
  goes ONLY to that schema's bucket (not broadcast). Schema-untagged test
  asserts broadcast to all applicable schemas. Coverage on EES rose
  exactly as claimed (46% → 64%) and the uncovered remainder is
  filtering / standalone-filtering-entrypoint / persistence — orthogonal
  to the multi-schema flip.

The four minors that were addressed (orphan-relation test, model-override
warning test, unused-import removal, and the corrected-attempt-1-claims
table in the self-review) are all real fixes. The two deferred minors
(SQL ORDER BY, LLMExtractor raise-on-empty) have honest justifications
and don't block the production-mode flip.

The implementer's "Corrections to attempt-1 PASS claims" section in the
self-review (lines 138-148) is exactly the right professional move —
explicitly acknowledging the three rows that were over-claimed before
fixing them. That table is a stronger guard against repeat over-claiming
than just the new tests.

## Kudos

- **Subclass routing test (M1)**: `test_paused_for_review_subclass_routes_correctly`
  with a user-defined `FancyPausedError` proves the worker's catch is on
  the BASE class — which is what the real production raiser
  (`SchemaReviewPendingError`) relies on. This is the kind of integration
  guard that catches "I refactored the exception hierarchy and forgot the
  worker checks `isinstance` vs `type ==`" bugs.
- **404 ordering test (M2)**: `test_404_when_source_not_found` is small
  but important — it proves that a nonexistent source returns 404 BEFORE
  any review-state check runs, preventing both unnecessary DB calls and
  a potential information leak.
- **Langextract bypass test (M2)**: `mock_source_repo.get_notebook_id
  .assert_not_called()` is the right way to pin a negative — verifies
  the bypass is intentional, not accidental.
- **Factory-failure non-fatal test (M3)**: The "test it doesn't crash
  AND verify the fallback value is forwarded" two-assertion structure is
  exemplary defensive testing. A single "doesn't raise" assertion would
  miss the case where the exception is swallowed but the result is
  garbage.
- **Self-review correction table (lines 138-148)**: Honest documentation
  of the attempt-1 over-claims. Healthy engineering culture.
- **Docstrings document the ordered-except invariant**: The test
  docstrings (worker.py tests:194-208, 217-224) explicitly call out that
  reversing the except clause order in `worker.py:154-167` will cause
  the test to fail loudly. Future-maintainer-friendly.

## Next steps

APPROVED — ready for human approval / merge. On merge, multi-schema mode
goes live by default for any `run_extraction(source_id, notebook_id=...)`
call. The three remaining minors should be filed as follow-up tickets
(B.3c or later) but do not block this PR.
