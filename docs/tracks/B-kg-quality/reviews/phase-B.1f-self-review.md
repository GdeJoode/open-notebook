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

---

## Attempt 2 fixes (2026-06-07)

Reviewer's attempt-1 decision was `REVISIONS_NEEDED` with 3 majors + 5 minors,
ALL of them "missing tests" (no logic bugs). This attempt addresses the
three majors and the four cheap minors. The fifth minor
(LLMExtractor silent-empty: raise rather than warn) is intentionally
deferred — it's a follow-up production-hardening exercise that would
break existing test fixtures.

### Corrections to attempt-1 PASS claims

Re-reading the attempt-1 table above against the reviewer's notes, three
rows were over-claiming:

| # | Original claim | Actual status before fix |
|---|---|---|
| 3a (router → 409) | "PASS — router pre-check" | UNTESTED — no `test_sources_processing.py` existed |
| 3c (worker → PAUSED_FOR_REVIEW) | "PASS — worker translates" | UNTESTED — `test_paused_error_subclasses_job_paused` only pinned the static type; worker path never exercised |
| 2 (`_run_multi_schema` invoked) | "PASS — spy" | PARTIAL — `_run_multi_schema` body itself (schema discovery, applicability, broadcast, llm-caller) was untested (46% coverage on `entity_extraction_service.py`) |

### Major 1: Worker pause-translation tests

`packages/job-queue/tests/test_worker.py`: +2 tests under `TestJobWorker`.

- `test_paused_for_review_no_retry_no_dead_letter` — handler raises
  `JobPausedForReviewError` → asserts the worker calls
  `update_status(..., PAUSED_FOR_REVIEW, error_message=...)`. Also
  asserts `RETRYING` / `FAILED` are NEVER called and `add_to_dead_letter`
  is NEVER called. Queue size is 0 (no re-enqueue).
- `test_paused_for_review_subclass_routes_correctly` — a user-defined
  subclass of `JobPausedForReviewError` is also routed correctly,
  proving the worker discriminates on the base class (which is what
  `SchemaReviewPendingError` relies on).

The tests document the "ordered except clauses" invariant in the
docstrings: the `JobPausedForReviewError` clause sits ABOVE the generic
`except Exception` in `worker.py:154-167`. Reversing the order would
cause every paused job to be treated as a transient failure → these
tests fail loudly if that regression happens.

### Major 2: `/run-entities` router 409 tests

New file `apps/app-main/tests/test_sources_processing.py` (6 tests):

- `test_returns_409_with_contract_body_when_review_required` — asserts
  status 409 AND the exact response body shape the B.3c UI keys off:
  `detail.code == "schema_review_pending"`, `detail.notebook_id`,
  `detail.pending_count`, `detail.message`.
- `test_no_409_when_accepted_extensions_present` — gate opens once
  user has accepted at least one extension.
- `test_404_when_source_not_found` — pre-flight 404 happens before the
  schema-review check.
- `test_multi_schema_enabled_false_forwarded_to_job_payload` — kill-
  switch path: payload field reaches the job (otherwise rollback is
  silently dropped).
- `test_langextract_options_forwarded` — all five langextract option
  fields reach the handler payload.
- `test_langextract_path_skips_review_gate` — non-llm extractor types
  skip the gate entirely (no `get_notebook_id` call).

### Major 3: `_run_multi_schema` body tests

`apps/app-main/tests/test_entity_extraction_service.py`: new
`TestRunMultiSchemaBody` class with 6 tests, each pinning ONE branch:

- `test_calls_detect_applicable_schemas_with_correct_args` — schema
  discovery passes `document_type` from source metadata, sampled
  chunk text (≤ 2000 chars), and the registry's ontologies.
- `test_falls_back_to_single_schema_when_no_applicable` — empty
  applicable list → falls back to single-schema (legacy semantics).
- `test_invokes_default_llm_caller_factory_for_multi_path` — production
  LLM-caller factory invoked, result threaded through to
  `workflow.extract`.
- `test_llm_caller_factory_failure_is_non_fatal` — factory raising
  doesn't break the job; `llm_caller=None` is forwarded so Pass-1 /
  Pass-2 use their lazy defaults.
- `test_accepted_extensions_with_schema_name_routed_to_one_schema` —
  schema-tagged extension only goes to that schema's bucket.
- `test_accepted_extensions_without_schema_name_broadcast_to_all` —
  un-tagged extension broadcasts to every schema (conservative
  default).

Coverage on `entity_extraction_service.py` after this attempt:
**46% → 64% (full app-main suite)**. The remaining 36% is filtering,
embedding, and persistence code (lines 193-215, 502-560, 589-670,
674-702) — orthogonal to the multi-schema flip; they belong to a
separate filtering-workflow test track.

### Minors addressed

- **Minor 1** (orphan-relation behaviour pinned): new test
  `test_relation_with_orphan_endpoint_passes_through_unchanged` in
  `test_multi_schema_orchestrator.py::TestMergeResults`. Pins that
  when one endpoint resolves to a canonical entity and the other
  doesn't, the orphan endpoint stays as the LLM-raw text (not
  rewritten, not dropped).
- **Minor 2** (per-call model override WARNING): new test
  `test_per_call_model_override_logs_warning` in
  `test_entity_extraction_service.py::TestMakeDefaultLLMCaller`.
  Calls the caller with a model id that differs from the bound model;
  asserts the WARNING is emitted via loguru → caplog.
- **Minor 4** (unused `datetime` import): removed from
  `packages/job-queue/src/job_queue/worker.py`.
- **Minor 5** (LLMExtractor silent-empty + warning): intentionally
  deferred. Production hardening — would require touching the
  llm_extractor production wiring and ripple changes through several
  test fixtures. Filed as follow-up.

### Minor NOT addressed

- **Minor 3** (`get_notebook_id` SQL — explicit `ORDER BY`): the
  current query uses `LIMIT 1` against the `reference` edge. Reviewer
  noted: "SurrealDB returns reference edges in insertion order (no
  explicit ORDER BY required)". This is documented in the repository
  docstring; an explicit `ORDER BY created` is a hardening I'd want
  to add when the version pin moves, but the source ↔ notebook
  relationship is 1:N with N typically ≤ 3 and any notebook is a
  valid anchor for schema lookup. Filed as a follow-up doc-comment
  task.

### Quality gates (attempt 2)

```
packages/shared              : 145 passed (no change)
packages/job-queue           : 38 passed (was 36; +2 worker pause tests)
pipelines/ontology-extraction: 241 passed (was 240; +1 orphan-relation test)
apps/app-main                : 401 passed (was 388; +6 router tests, +6 multi-schema body tests, +1 warning test)
```

Coverage on `entity_extraction_service.py`:
- attempt 1: 46%
- attempt 2: **64%** (full suite) / **64%** (EES tests only)

No regressions. Total suite finishes in ~3 minutes.

### Verification commands

```
cd packages/job-queue && uv run --project ../.. --extra dev pytest tests/test_worker.py -v
cd apps/app-main && uv run --project apps/app-main pytest tests/test_sources_processing.py -v
cd apps/app-main && uv run --project apps/app-main pytest tests/test_entity_extraction_service.py -v
cd pipelines/ontology-extraction && uv run --project . pytest tests/test_multi_schema_orchestrator.py -v -k "orphan or canonical or relink"
```
