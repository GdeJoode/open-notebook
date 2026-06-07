# Phase B.1e — self-review

> Author: implementer agent, 2026-06-06
> Branch: `track/b-multi-schema-orchestrator`
> Commits: `2e23dfe` (orchestrator skeleton) → `5e4fa93` (test suite)

## Acceptance criteria check

| # | Criterion | Verified? |
|---|---|---|
| 1 | 3 mock ontologies + synthetic document with entities split across 2 schemas → single `ExtractionResult` where each entity in ≥ 2 passes has `type_tags` length ≥ 2 and `primary_type` matches highest-confidence label | YES — `TestRunMultiSchema::test_multi_tagged_entity_from_three_passes` (3 ontologies, Alice in all three passes with conf 0.95/0.6/0.8) asserts `len(type_tags) == 3`, `primary_type == "Researcher"` (the 0.95 pass). `TestMergeResults::test_entity_in_two_passes_gets_type_tags` covers the 2-of-3 case directly. |
| 2 | Token budget per Pass-2 call ≤ 3000 tokens | YES — `TestTokenBudgetGuard::test_each_pass2_prompt_under_3000_tokens` asserts ≤ 3000 for every chunk in the standard fixture. The B.1d-internal 2400-token cap (20% safety margin) is also pinned by `test_internal_budget_target_holds`. |
| 3 | `pass1_results` rows written for each schema attempted | YES — `TestRunMultiSchema::test_two_schemas_persist_pass1_records` uses a fake repo and asserts `len(repo.records) == 2`, with `schema_attempted` set to `{"scholarly", "policy"}`. Each row carries `source` / `notebook` IDs forwarded from the orchestrator call. |
| 4 | Single-schema input `[(scholarly, 0.92)]` is bit-identical to B.1d `run_pass2` direct call | YES — `TestRunMultiSchema::test_single_schema_input_bitidentical_to_pass2` runs the same chunks + ontology through both paths, asserts per-entity `text`/`label`/`confidence` equality, and pins `type_tags == []` + `primary_type is None` (the merge step is a no-op for one pass). `TestMergeResults::test_single_schema_input_is_passthrough` adds a unit-level confirmation. |
| 5 | Soft-nudge thresholds configurable via constants in `packages/shared/.../config.py` so UI imports same source | YES — `packages/shared/src/shared/config.py` declares `SOFT_NUDGE_COVERAGE_HIGH = 0.95`, `SOFT_NUDGE_COVERAGE_LOW = 0.80`, `MIN_APPLICABLE_CONFIDENCE = 0.3`. The orchestrator re-imports them (no local copies). `TestSharedThresholdsRoundTrip` (3 tests) asserts the orchestrator-side names match the shared-config originals. |
| 6 | ≥ 85 % line coverage on `multi_schema_orchestrator.py` | YES — `99 %` (178 / 180 lines). The 2 uncovered lines are defensive belt-and-braces branches: the `remaining <= 0` early break in `_sample_text_for_pass1` (only reachable on a very specific budget-edge fixture not worth simulating) and the `not key[2]` empty-relation-type guard in `_merge_results` (drops a malformed LLM payload). |

## Pre-resolved decisions honoured

| Q | Decision | Implementation |
|---|---|---|
| Q-B-2 | Heuristic token budget, NO `tiktoken` | `_sample_text_for_pass1` uses a char budget (`PASS2_SAMPLE_CHAR_BUDGET = 6000`); Pass-1 / Pass-2 internal guards still use `len(text) // 4`. No new deps added to `pyproject.toml`. |
| Q-B-4 | Re-use the B.1c conservative `normalize_entity_name` stub, don't reimplement | The orchestrator imports `from shared.utils.name_normalizer import normalize_entity_name` exactly once — used in both entity-key dedup and relation-key dedup. Q9 (Track M4) can swap the body without touching the orchestrator. |
| Q-B-6 | Telemetry always-on | `pass1_attempt` logged per schema with applicability, `multi_schema_pass1_complete` with best coverage + decision + extension count, `multi_schema_run skipped` on the empty-applicable short-circuit, `pass1_failed` / `pass1_record_failed` / `pass2_failed` warnings per schema. Plus the existing Pass-1 / Pass-2 telemetry from B.1c / B.1d flows through unchanged. |
| Q-B-7 | Stub name-normalizer reused (already on main), Q9 deferred | The B.1c-shipped `normalize_entity_name` is reused as-is — no changes, no new normalizer module. |

## Design decisions worth flagging for the reviewer

### Single-source-of-truth thresholds (RETRO lesson #2)

`packages/shared/src/shared/config.py` is a **new** module that owns the three policy constants. Both the orchestrator and the future B.3c UI slider must import from there. This is enforced by the `TestSharedThresholdsRoundTrip` group: if the orchestrator re-declared the values locally and the shared values drifted, those tests would fail and surface the duplication.

### `detect_applicable_schemas` combines two signals

The plan asked for "document-type mapper + keyword-overlap fallback". I implemented both as parallel signals combined by `max(...)`:

- **Document-type mapper**: when `get_ontology_for_document_type(document_type)` matches an ontology's `metadata.name`, the score floor is `0.92` — matches the AC #4 single-schema snapshot value.
- **Keyword overlap**: count entity-type-name substring matches in the document text, normalize by `max(1, len(types))`, cap at `0.9` so a keyword-heavy hit never beats the document-type signal.

Either signal alone can push an ontology over the `MIN_APPLICABLE_CONFIDENCE` floor. The `max(...)` combiner avoids double-counting and keeps the score interpretable. A custom mapper can be injected for tests (used by `test_custom_mapper_injected`).

### Sequential, not parallel

Per the plan and FEATURE_ROADMAP §621-639, schemas are processed **sequentially** through Pass-1 and Pass-2. The orchestrator holds a single LLM caller as a shared resource and we want deterministic `primary_type` resolution: two parallel passes finishing in different orders on rerun would change merge semantics. Parallelism is a B.4 / B.5 optimization.

### Failure isolation per schema

Pass-1 raising (token budget, transport, malformed JSON path) for one schema MUST NOT abort the run — log a warning, skip that schema, continue. Same for `pass1_repo.record(...)` failures and Pass-2 failures. Tests `test_pass1_failure_skips_schema_does_not_abort_run` / `test_pass1_repo_record_failure_does_not_abort` / `test_pass2_failure_skips_schema` pin all three branches. The reasoning: a single-schema crash should not blackout extraction for the user; the operator gets ledger logs to investigate.

### Merge uses `MergedEntity` as an in-process aggregator

`MergedEntity` is a plain Python class with `__slots__` — deliberately NOT a Pydantic model. It never crosses a serialization boundary; the merge loop converts back to `ExtractedEntity` at the end. Two benefits: allocation-cheap per-entity bookkeeping, and the test fixtures stay readable (no Pydantic boilerplate for transient state).

### Highest-confidence entity instance supplies provenance

The merge step searches the per-schema results for the *exact* entity instance whose confidence matches `best_confidence` and uses it as the basis for `source_chunk_id`, `source_grounding`, and `extraction_context`. This means the merged entity's provenance points to whichever pass was most confident — which is the pass downstream tooling will most likely want to follow when debugging.

### Workflow defaults to `mode="single"`

Per the plan's explicit footnote, `ExtractionWorkflow.extract(chunks, mode="single")` is the new signature with `mode` defaulting to `"single"`. This preserves zero behaviour change for every existing caller. B.1f flips the default once `EntityExtractionService` is wired and `notebook_id` is plumbed through.

## Test summary

```
pipelines/ontology-extraction tests/
  47 new tests for test_multi_schema_orchestrator.py
  234 total, all passing
  coverage on multi_schema_orchestrator.py: 99 %

packages/shared:
  145 tests passing (no regressions; +0 new)

packages/surrealdb-service (non-docker):
  52 tests passing

apps/app-main (non-integration):
  pending — see report
```

## Files touched

- **NEW** `packages/shared/src/shared/config.py` — shared policy constants.
- **NEW** `pipelines/ontology-extraction/src/ontology_extraction/multi_schema_orchestrator.py` — orchestrator + merger + helpers.
- **NEW** `pipelines/ontology-extraction/tests/test_multi_schema_orchestrator.py` — 47-test suite.
- **MODIFIED** `packages/shared/src/shared/models/extraction.py` — `type_tags` + `primary_type` on `ExtractedEntity`.
- **MODIFIED** `pipelines/ontology-extraction/src/ontology_extraction/workflow.py` — `mode` parameter on `extract`.
- **MODIFIED** `pipelines/ontology-extraction/src/ontology_extraction/__init__.py` — public exports.

## Outstanding follow-ups

- B.1f wires `EntityExtractionService.run_extraction()` to call `run_multi_schema(...)` when a `notebook_id` is supplied; flips `ExtractionWorkflow.extract`'s default to `"multi"`.
- B.3c surfaces `SoftNudgeDecision` in the notebook UI and presents `metadata["proposed_extensions"]` for review.
- B.4 layers `confidence` display on the KG page using the now-non-null `primary_type` / `type_tags`.
- The two uncovered orchestrator lines could be exercised by adding fixtures that build sample texts at exactly the budget boundary and emit relations with empty types — done deliberately as low-value tests.
