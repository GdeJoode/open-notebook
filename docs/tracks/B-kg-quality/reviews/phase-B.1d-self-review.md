# Phase B.1d — self-review

> Author: implementer agent, 2026-06-06
> Branch: `track/b-pass2-module`
> Commits: `a3fd1e5` (Pass-2 prompt) → `293ef1b` (run_pass2 module) → `0ce55b4` (tests) → `8292cb3` (LLMExtractor docstring) → `e65316a` (docs + self-review)

## Acceptance criteria check

| # | Criterion | Verified? |
|---|---|---|
| 1 | `run_pass2(chunks, ontology, accepted_extensions=[])` produces `ExtractionResult` for a 3-chunk fixture | YES — `TestRunPass2::test_three_chunks_with_no_extensions` sends a 3-chunk fixture, mocks the LLM to return the canonical Pass-2 shape, asserts entity/relation count and metadata. Shape is equivalent to the current `LLMExtractor.extract` output. |
| 2 | With an `EarlyCareerResearcher` extension, prompt body carries the extension AND mock LLM returns `ExtractedEntity(label="EarlyCareerResearcher", confidence=0.85)` | YES — `TestRunPass2::test_extension_injected_and_used` asserts `"EarlyCareerResearcher"` and `"parent: Researcher"` are visible in the user prompt, then mocks the LLM to return that exact label and verifies the resulting `ExtractedEntity`. |
| 3 | Confidence present and non-null on EVERY entity AND EVERY relation | YES — `TestRunPass2::test_confidence_present_on_every_element` runs a 2-chunk fixture (3 entities + 2 relations total), iterates each, and asserts `confidence is not None`, `isinstance(confidence, float)`, and the value is in `[0.0, 1.0]`. `_clamp_confidence` defaults to `0.0` for missing/unparseable inputs, so the model invariant holds even on partial LLM output. |
| 4 | Malformed LLM JSON → graceful degradation (empty ExtractionResult + WARNING log) | YES — `TestRunPass2::test_malformed_response_per_chunk_degrades` sends one malformed chunk mid-run; the other two chunks succeed; per-chunk failure does NOT interrupt the run. `TestParseChunkResponse::test_malformed_json_emits_warning_log` confirms the WARNING. The pattern mirrors B.1c (`Pass1SchemaValidator._parse_response`). |
| 5 | Token budget enforced — large chunk raises `Pass2TokenBudgetExceeded` BEFORE LLM call | YES — `TestTokenBudget::test_budget_guard_fires_pre_llm` sends a 40 000-char chunk (~10 000 tokens), confirms `Pass2TokenBudgetExceeded` is raised AND the LLM caller's `called` flag remains `False`. |
| 6 | ≥ 90% line coverage on both `pass2_typed_extraction.py` and `prompts/pass2.py` | YES — `prompts/pass2.py` 100% (68/68 stmts), `pass2_typed_extraction.py` 96% (5 stmts missed: the lazy default LLM-caller fallback at lines 342-353, intentionally unreachable from unit tests since it requires the unit-test-incompatible `llm_manager.manager` import — mirrors the Pass-1 convention). Combined: **98%**. |
| 7 | Imports work from `from ontology_extraction import run_pass2, Pass2TokenBudgetExceeded` | YES — `TestPublicAPI::test_run_pass2_exported`, `test_token_budget_exceeded_exported`, `test_pass2_parse_error_exported`. The package `__init__.py` exports `run_pass2`, `Pass2TokenBudgetExceeded`, `Pass2ParseError`. |
| 8 | `apps/app-main` continues to pass (no regressions; pass2 not wired into service yet — that's B.1f) | YES — 368 tests pass, identical to baseline. No app-main file touched. |

## Pre-resolved decisions honoured

| Q | Decision | Implementation |
|---|---|---|
| Q-B-2 | Coarse `len(text) // 4` heuristic, target ≤ 2400 tokens (3000 budget with 20% margin), NO `tiktoken` | `_estimate_tokens` is exactly `len(text) // 4`. `TOKEN_BUDGET_TARGET = 2400`. No new deps in `pyproject.toml`. |
| Q-B-6 | Telemetry always-on | Three structured log lines per run (`pass2_chunk_start`, `pass2_chunk_complete`, `pass2_run_complete`) plus a WARNING on budget breach and a WARNING on parse failure. When the metrics table lands (B.4) these become counters/histograms. Until then, the loguru ledger is the source of truth. |

## Design decisions worth flagging for the reviewer

### LLM caller is injected, no `LLMExtractor` modifications

The plan offered two options for the back-compat path: (a) extract `LLMExtractor`'s current per-chunk prompt-build into `run_pass2` and rewire `LLMExtractor.extract` to call it, or (b) leave `LLMExtractor` as-is and provide `run_pass2` as a parallel path. I went with (b) because:

1. The plan's caveat is explicit: don't touch the `LLMManager → ModelManager` TODO marker.
2. `LLMExtractor.extract` uses `OntologyPromptGenerator.generate_combined_extraction_prompt`, which builds a much richer prompt (entities + relationships + concepts + claim types). Pass-2's prompt is **focused** — entities + relations + accepted extensions only. Folding the two together would conflate two distinct contracts.
3. AC #1 says "equivalent in shape" — testable via mocks, no need for behavioural co-occurrence.
4. B.1f owns the wiring decision (which path `EntityExtractionService` actually calls). Keeping both paths in place gives B.1f a clean choice.

Net effect: `LLMExtractor.extract` is untouched (one docstring addition pointing at `run_pass2`), and the pre-existing `LLMManager` TODO is in the same state as the end of B.1c.

### `_parse_chunk_response` accepts the legacy `subject/predicate/object` relation shape

Pass-2's canonical relation shape is `{source, target, type}`. But an LLM trained on the older `LLMExtractor` prompt would return `{subject, predicate, object}`. To stay robust during the transition, `_parse_chunk_response` accepts both — `source` falls back to `subject`, `target` to `object`, `type` to `predicate`. Tests `test_legacy_subject_predicate_object_shape` and `test_well_formed_response` pin both shapes. The canonical shape is the one the prompt asks for.

### Confidence defaults to `0.0`, not "dropped"

The B4 contract is "confidence on every element". `_clamp_confidence(None)` returns `0.0` rather than discarding the element. The reasoning: an extraction with low confidence is still useful downstream (the dedup + edge-scoring pipeline can filter by threshold), while losing the surface form altogether is irreversible. The model field has `ge=0.0`, so the value is always sensible. Test `test_missing_confidence_defaults_to_zero` pins this.

### Compression threshold mirrors Pass-1 (30 types / 30 relations)

`COMPACT_TYPE_THRESHOLD = 30` and `COMPACT_RELATION_THRESHOLD = 30` — symmetric to Pass-1's `COMPACT_SUMMARY_THRESHOLD`. The threshold was justified in B.1c attempt 2 (100-type ontology + 1500-token sample fits comfortably). The stress test `TestStressTokenBudget::test_realistic_headroom` re-verifies that envelope for Pass-2 with **3 extensions** added on top.

## Token-budget headroom

I measured on a synthetic 100-type ontology + 1500-token chunk + 3 extensions:

```
~ build_pass2_prompt with 100 entity types + 0 relations + 1500-tok chunk + 3 ext:
~1929 tokens / 2400 budget (~ 19.6% headroom)
```

Test `TestStressTokenBudget::test_realistic_headroom` is the regression net.

For a typical chunk + the `scholarly` ontology (8 entity types, 2 relationship types) + 3 extensions, the prompt is well under 2000 tokens — comfortably below the 2400 cap with the 20% safety margin against the 3000 plan ceiling.

## Test summary

| Suite | Before | After | Delta |
|---|---|---|---|
| `pipelines/ontology-extraction` | 124 | 187 | **+63** new tests, 0 regressions |
| `apps/app-main` | 368 | 368 | unchanged (no regression) |
| `packages/shared` | 145 | 145 | unchanged |

## Coverage

```
pipelines/ontology-extraction/src/ontology_extraction/pass2_typed_extraction.py: 141 stmts, 5 missed (96%)
pipelines/ontology-extraction/src/ontology_extraction/prompts/pass2.py:           68 stmts, 0 missed (100%)
Combined: 209 stmts, 5 missed (98%)
```

The missed lines (`342-353`) are the lazy default LLM caller's stub body, intentionally unreachable from unit tests (requires the `llm_manager.manager` import that B.1f owns the rewire for).

## Files added / modified

| Path | Change |
|---|---|
| `pipelines/ontology-extraction/src/ontology_extraction/pass2_typed_extraction.py` | NEW |
| `pipelines/ontology-extraction/src/ontology_extraction/prompts/pass2.py` | NEW |
| `pipelines/ontology-extraction/tests/test_pass2.py` | NEW |
| `pipelines/ontology-extraction/src/ontology_extraction/__init__.py` | MODIFIED — exports `run_pass2`, `Pass2TokenBudgetExceeded`, `Pass2ParseError` |
| `pipelines/ontology-extraction/src/ontology_extraction/prompts/__init__.py` | MODIFIED — exports Pass-2 prompt helpers |
| `pipelines/ontology-extraction/src/ontology_extraction/extractors/llm_extractor.py` | MODIFIED — module-level docstring pointer to `run_pass2`; no behaviour change |

## Open follow-ups for downstream phases

- **B.1e (multi-schema orchestrator)**: import `run_pass2` from `ontology_extraction`. The single-schema path is stable; orchestrator dedup + name normalization live in B.1e.
- **B.1f (service integration)**: wire `EntityExtractionService` to call `run_pass2` with an LLM caller supplied via DI. Decide whether `ExtractionWorkflow` migrates over from `LLMExtractor` or stays on the legacy path until langextract is also addressed.
- **B.4 (telemetry)**: when the metrics table lands, replace the three `loguru.info` lines with counter increments (`pass2_chunks_total`, `pass2_entities_total`, etc.) plus a `pass2_token_budget_exceeded` counter.
