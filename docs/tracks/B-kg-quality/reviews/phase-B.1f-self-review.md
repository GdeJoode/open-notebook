# Phase B.1f Self-Review

**Branch**: `track/b-extraction-service-wiring`
**Commits**: `821cc39` → `684af89` → `be23140` → `ae7990e`
**Date**: 2026-06-07
**Status**: code complete, all quality gates green, ready for adversarial review

## Scope

B.1f as expanded by the reviewer covers three concerns:

1. Wire `EntityExtractionService.run_extraction()` to the B.1e multi-schema orchestrator
2. Fix the long-standing `LLMExtractor` broken-import bug (B.1c r2 TODO)
3. Land the B.4 relation-endpoint re-link fix flagged in the B.1e review

All three landed in this branch.

## Acceptance criteria check

| # | Criterion | Status |
|---|---|---|
| 1 | `run_extraction(source_id, ontology_name="general")` (no `notebook_id`) → single-schema workflow | PASS — `test_no_notebook_id_uses_single_schema` (regression guard, spies on `ExtractionWorkflow.extract` to verify `mode` is not `"multi"`) |
| 2 | `run_extraction(source_id="x", notebook_id="n")` → `_run_multi_schema` invoked | PASS — `test_notebook_id_routes_to_multi_schema` (spy on the helper directly) |
| 3 | `review_required=True` + no accepted extensions → `SchemaReviewPendingError`; API → 409; job → `PAUSED_FOR_REVIEW` | PASS — `test_review_required_raises` (service) + `test_paused_error_subclasses_job_paused` (worker discrimination) + router pre-check + worker translates error to `PAUSED_FOR_REVIEW` |
| 4 | `multi_schema_enabled=False` forces single-schema (rollback) | PASS — `test_multi_schema_enabled_false_forces_single_schema` |
| 5 | New B.1e merge output flows through `entity_persistence_service.persist_filtered_result` | PASS — multi-schema returns `ExtractionResult` which the existing run-filtering path consumes unchanged (no change at the persistence seam) |
| 6 | B.4: relation source/target rewritten to canonical entity surface form | PASS — `test_relation_endpoints_relinked_to_canonical_entity_text` pins the concrete bug scenario from the B.1e review; `test_relation_endpoint_unchanged_when_already_canonical` pins the no-op path |
| 7 | LLMExtractor wired to ModelManager, no silent empty result | PASS — `test_constructs_with_async_caller_no_import_error` + `test_extract_dispatches_to_injected_caller` + `test_pre_b1f_import_path_no_longer_referenced` (AST-level regression guard) |

## Quality gates

```
packages/shared              : 145 passed (no regressions)
packages/job-queue           : 36 passed (no regressions; new exception type)
packages/surrealdb-service   : 52 passed, 20 docker-skipped (env)
pipelines/ontology-extraction: 240 passed (234 baseline + 6 new)
apps/app-main                : 388 passed (380 baseline + 8 new)
```

No regressions. Full test suite finishes in ~3 minutes.

## Architectural notes

### Why `JobPausedForReviewError` lives in `job-queue`, not `app-main`

The worker (`JobWorker._execute_job`) needs to distinguish "park the job"
from "the job failed". Coupling the worker to handler-specific exception
types would force a knowledge-leak; instead, the worker treats anything
inheriting `JobPausedForReviewError` as a pause signal. Concrete handlers
(B.1f: `SchemaReviewPendingError`) subclass it and add their own
context fields.

This sets up Track B.3c cleanly: when the UI calls "approve extension and
resume", the resume path just re-queues the job; no new worker logic
needed.

### Why the API pre-checks the review gate

The handler also catches `SchemaReviewPendingError` (defense in depth),
but the synchronous router-level pre-check saves a worker roundtrip in
the common case: the UI knows immediately. The two checks must agree —
if they ever drift, the worker's check wins (it sees the freshest
notebook state), so the router check is just an optimisation.

### Why the LLMExtractor takes a caller rather than a `Model`

Matches the Pass-1 / Pass-2 contract. Single caller protocol across all
three modules means production wiring is a single
`make_default_llm_caller(...)` call. Tests inject fakes the same way
Pass-2's tests do.

### Why the B.4 fix lives in `_merge_results` (not a separate pass)

The merge is the natural choke-point — both entity-canonicalisation and
relation-canonicalisation share the same key (`normalize_entity_name`).
A separate pass would re-do the keying work. The fix is `O(R)` extra
work for `R` relations and only allocates a copy when the rewrite
actually changes either endpoint (common case: no allocation).

## Pre-resolved decisions honoured

- **Multi-schema by default** when `notebook_id` is present (`multi_schema_enabled=True`).
- **Kill-switch flag** wired through API → handler → service, ops can roll back without code changes.
- **Notebook-source link** resolved via the existing `reference` graph edge — no schema change.
- **Pre-check + handler catch** as defense in depth for the review gate.
- **`LLMCaller` re-used** rather than defining a new protocol.
- **`SchemaReviewPendingError` subclasses generic `JobPausedForReviewError`** — clean handoff to the queue worker.

## Known follow-ups (out of scope for B.1f)

1. **B.3c UI**: the "approve extension and resume" flow — the queue
   side is ready; UI work is B.3c proper.
2. **Per-call model override**: `make_default_llm_caller` ignores
   the `model` arg passed by Pass-1/Pass-2 and binds the configured
   default at factory time. If we ever want per-chunk model routing
   (e.g. cheap-model-for-pass-1, expensive-model-for-pass-2), the
   factory needs to return a router rather than a bound caller.
3. **Schema-name routing of accepted extensions**: the service
   broadcasts schema-less extensions to every applicable schema. This
   is conservative; B.3c may want stricter per-schema routing once the
   extension model adds a required `schema_name` field.

## Files touched

```
apps/app-main/src/app_main/services/entity_extraction_service.py   ← main wiring
apps/app-main/src/app_main/handlers.py                              ← notebook_id lookup + error reraise
apps/app-main/src/app_main/api/routers/sources_processing.py        ← 409 pre-check + payload field
apps/app-main/tests/test_entity_extraction_service.py               ← NEW (8 tests)
packages/shared/src/shared/types/enums.py                           ← + JobStatus.PAUSED_FOR_REVIEW
packages/job-queue/src/job_queue/exceptions.py                      ← NEW (JobPausedForReviewError)
packages/job-queue/src/job_queue/__init__.py                        ← export
packages/job-queue/src/job_queue/worker.py                          ← catch + translate
packages/job-queue/src/job_queue/repository.py                      ← terminal-status semantics
packages/surrealdb-service/src/surrealdb_service/repositories/source.py
                                                                   ← + get_notebook_id()
pipelines/ontology-extraction/src/ontology_extraction/extractors/llm_extractor.py
                                                                   ← DI rewrite
pipelines/ontology-extraction/src/ontology_extraction/multi_schema_orchestrator.py
                                                                   ← B.4 relation re-link
pipelines/ontology-extraction/src/ontology_extraction/workflow.py   ← + extractor ctor arg + llm_caller pass-through
pipelines/ontology-extraction/tests/test_extractors.py              ← +4 LLMExtractor caller tests
pipelines/ontology-extraction/tests/test_multi_schema_orchestrator.py
                                                                   ← +2 B.4 fix tests
```
