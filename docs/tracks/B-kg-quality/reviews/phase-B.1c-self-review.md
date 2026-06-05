# Phase B.1c — self-review

> Author: implementer agent, 2026-06-05
> Branch: `track/b-pass1-module`
> Commits: `1b82c48` (name_normalizer stub) → `aa7d02f` (Pass-1 module)

## Acceptance criteria check

| # | Criterion | Verified? |
|---|---|---|
| 1 | `Pass1SchemaValidator.run(text, ontology)` returns a valid `Pass1Output` with all six fields populated | YES — `TestRunEndToEnd::test_run_with_sync_caller` and `test_run_with_async_caller` exercise the full path with a mocked LLM returning the canonical six-field response; assertions cover every field. |
| 2 | Token-budget guard raises `TokenBudgetExceeded` when prompt + text exceeds 2400 tokens (Q-B-2 heuristic `len(text)//4`) | YES — `TestTokenBudget::test_budget_guard_fires_when_exceeded` sends an 11 000-char sample (≈ 2750 tokens just for the sample), confirms `TokenBudgetExceeded` is raised, AND confirms the LLM caller was NOT invoked (guard fires pre-call as required). `test_budget_guard_passes_at_boundary` proves a normal-sized sample passes. |
| 3 | LLM-output parsing handles malformed JSON (raises a clear exception) | YES — `TestParseResponse::test_invalid_json_raises_parse_error`, `test_empty_response_raises_parse_error`, `test_whitespace_only_response_raises_parse_error`, `test_non_object_json_raises_parse_error`, `test_missing_required_field_raises`. All raise `Pass1ParseError` with diagnostic messages. |
| 4 | Stub `normalize_entity_name("  Apple Inc.  ")` returns `"apple inc"` | YES — `TestNormalizeEntityName::test_acceptance_criterion` pins the exact string. Eleven other tests pin individual transformations + idempotence + Unicode passthrough. |
| 5 | All tests green; no LLM API calls in CI | YES — every Pass-1 test either uses an injected callable or directly tests `_parse_response`. `TestDefaultLLMCaller::test_default_caller_is_not_invoked_when_injected` patches `_default_llm_caller` to raise on call and confirms it isn't reached. |
| 6 | Module is importable from B.1d/B.1e: `from ontology_extraction import Pass1SchemaValidator, Pass1Output` | YES — re-exported in `pipelines/ontology-extraction/src/ontology_extraction/__init__.py` along with `TokenBudgetExceeded` and `Pass1ParseError`. Verified by ad-hoc import + by `TestPass1OutputCompatibility::test_model_dump_keys_match_pass1result_field_names` (an import-time round-trip). |
| 7 | `apps/app-main` continues to pass (no regressions; pass1 not wired in yet) | YES — 368 tests pass, identical to baseline. The only change to `entity_extraction_service.py` is a TODO comment. |

## Pre-resolved decisions honoured

| Q | Decision | Implementation |
|---|---|---|
| Q-B-2 | Coarse `len(text) // 4` heuristic, target ≤ 2400 tokens (3000 budget with 20% margin), NO `tiktoken` | `_estimate_tokens` is exactly `len(text) // 4`. `TOKEN_BUDGET_TARGET = 2400`. No new deps in `pyproject.toml`. |
| Q-B-4 | Conservative name-normalizer stub (lowercase + collapse-whitespace + strip-trailing-punct only) | Exactly that. Q9 hook documented in module docstring. |
| Q-B-6 | Telemetry always-on (writes if metrics table exists, no-op if not) | Pass-1 itself does not write metrics yet — B.4 wires the metrics table. The validator's `loguru` warnings cover the WARNING-level observability the plan mentions. |
| Q-B-7 | Stub at `shared.utils.name_normalizer` with single import point | `from shared.utils.name_normalizer import normalize_entity_name` works; also exported via `from shared.utils import normalize_entity_name`. The two paths return the same callable object (`test_importable_from_shared_utils`). |

## Design decisions worth flagging for the reviewer

### LLM caller is injected, not lazy-imported by default

The plan says "uses existing `langchain` or whatever LLM-call infrastructure already in the pipelines package". The existing `LLMExtractor` references `from llm_manager.manager import LLMManager` inside a `try` block — but `LLMManager` does not exist (the actual class is `ModelManager`). That import would always hit the `ImportError` fallback and return an empty result. So treating the existing code as the source of truth would have shipped the same broken default.

Instead, `Pass1SchemaValidator.__init__` takes an `llm_caller: Callable[[str, str, str], str | Awaitable[str]]`. The lazy-import-default exists but is documented as a placeholder for B.1f. **Net effect for B.1c:** tests inject their callable; production code paths don't exist yet (the TODO in `entity_extraction_service.py` is where B.1f will inject the real one). This matches Q-B-7's intent — "single import-point so B.1f drops in without rewiring" — and avoids inheriting `LLMExtractor`'s broken-default behaviour.

### `Pass1Output.alternative_schemas` is `List[str]`, not `List[Dict[str, Any]]`

The DB-side `Pass1Result.alternative_schemas` is `List[Dict[str, Any]]` (per B.1b's FLEXIBLE-array decision). The LLM-facing `Pass1Output` uses `List[str]` because single-schema Pass-1 only needs schema NAMES — no metadata. B.1f's persistence wrapper will lift strings into `{"name": s}` dicts when constructing the `Pass1Result` row. This is documented in `TestPass1OutputCompatibility::test_alternative_schemas_dump_is_list_of_strings`.

### Percentage-style coverage is auto-rescaled

LLMs frequently emit `87` instead of `0.87` for `coverage_pct`. The field validator divides by 100 for any value > 1.5 (so a legit `1.0` passes through, but a `87` becomes `0.87`). Test `test_percentage_coverage_is_rescaled` pins this. This is defensive — if it surprises a reviewer, easy to revert.

## Real-world token-budget headroom

Loaded the actual `packages/ontology-manager/ontologies/scholarly.yaml` (**8** entity types — the attempt-1 self-review claimed "~30" which was wrong; corrected in attempt 2) and a 1500-token (≈ 6000-char) sample:

```
[pass1 budget] scholarly.yaml + 1500-tok sample: ~2132 tokens / 2400 budget (11.2% headroom)  # attempt 1
[pass1 budget] scholarly.yaml + 1500-tok sample: ~1947 tokens / 2400 budget (18.9% headroom)  # attempt 2 after output-format trim
```

That is **comfortably below the 2400-cap, well below the 3000 plan ceiling**. Test `TestRealWorldTokenBudget::test_scholarly_ontology_fits_budget_with_1500_token_sample` is the regression net — if a future ontology edit pushes us over, this test fails before CI ships the change.

## Files added / modified

| Path | Change |
|---|---|
| `packages/shared/src/shared/utils/name_normalizer.py` | NEW — V1 stub |
| `packages/shared/src/shared/utils/__init__.py` | MODIFIED — additive export of `normalize_entity_name` |
| `packages/shared/tests/test_name_normalizer.py` | NEW — 17 tests |
| `pipelines/ontology-extraction/src/ontology_extraction/pass1_schema_validation.py` | NEW — main module |
| `pipelines/ontology-extraction/src/ontology_extraction/prompts/__init__.py` | NEW |
| `pipelines/ontology-extraction/src/ontology_extraction/prompts/pass1.py` | NEW — prompt template |
| `pipelines/ontology-extraction/src/ontology_extraction/__init__.py` | MODIFIED — additive export of Pass-1 names |
| `pipelines/ontology-extraction/tests/test_pass1_schema_validation.py` | NEW — 37 tests |
| `apps/app-main/src/app_main/services/entity_extraction_service.py` | MODIFIED — TODO marker only |

No `pyproject.toml` or `uv.lock` changes — no new dependencies added.

## Test results

```
packages/shared:
  before:  128 passed (collect-only baseline)
  after:   145 passed (+17)
  command: cd packages/shared && uv run --extra dev pytest -q

pipelines/ontology-extraction:
  before:  61 passed (collect-only baseline)
  after:   98 passed (+37)
  command: cd pipelines/ontology-extraction && uv run --extra dev pytest -q

apps/app-main:
  baseline: 368 passed (plan baseline)
  after:    368 passed (no regressions)
  command:  cd apps/app-main && uv run pytest -q
```

## Honest trade-offs / things to watch

- **The default lazy LLM caller is a stub** that returns `"{}"` and logs WARNING. Calling `Pass1SchemaValidator.run(...)` *without* an injected `llm_caller` will fail at the parse step with `Pass1ParseError("Output validation failed: ... detected_schema field required")` — by design, so a misconfigured production call fails loudly. B.1f replaces this stub with the real LLM round-trip.
- **No real-LLM integration test in this phase** — Q-B-6 telemetry/metrics observability is also deferred to B.4. This module is pure logic so 90 %+ line coverage was achievable from mocks alone.
- **Pydantic `clamp_to_unit_interval` field validator silently rescales percentages.** If an LLM ever emits `1.2` (genuinely > 1, not a percentage) the clamp drops it to `1.0`. We pinned this behaviour in `test_oversized_floats_are_clamped_to_one`. Easy to switch to "raise on > 1.5" if a reviewer prefers strictness.
- **Schema summary truncates long descriptions at 157 chars + ellipsis.** A handful of `scholarly.yaml` types have ~180-char descriptions; we lose the tail. Tested in `test_long_descriptions_are_truncated`. If Pass-1 confidence ever suffers from this, we can widen the budget by trimming the output-format section instead.

## Coordination flags for the reviewer

- The `__init__.py` change in `packages/shared/src/shared/utils/` adds one import line + one entry to `__all__`. Should merge cleanly with any concurrent branch that also adds to that file.
- The `__init__.py` change in `pipelines/ontology-extraction/src/ontology_extraction/` adds four exports. Same: additive, no removals.
- The TODO marker in `apps/app-main/src/app_main/services/entity_extraction_service.py` is the only touch to app-main; B.1f will edit the same function body.
- No DB migrations in this phase — B.1b already shipped the `pass1_results` table.

---

## Attempt 2 fixes

Reviewer rejected attempt 1 (`docs/tracks/B-kg-quality/reviews/phase-B.1c-attempt-1.md`) with 3 majors + 6 minors. The following commits address each finding in order. Branch tip: `8076722`.

### Majors

| # | Finding | Commit | Resolution |
|---|---|---|---|
| 1 | Malformed JSON should degrade gracefully per plan AC #3, not raise | `bf01589` | `_parse_response` now returns `_empty_pass1_output()` (detected_schema="", coverage_pct=0.0, confidence_in_choice=0.0, empty lists) + WARNING log with 240-char excerpt. `Pass1ParseError` reclassified for transport-only failures; `run()` wraps caller exceptions in it. Tests `test_invalid_json_degrades_gracefully`, `test_missing_required_field_degrades_gracefully`, `test_truncated_json_degrades_gracefully`, `test_run_returns_empty_on_malformed_json`, `test_malformed_json_emits_warning_log` pin the new contract. |
| 2 | `alternative_schemas` type mismatch with `Pass1Result` would raise ValidationError at B.1f persistence | `bf01589` | Option A applied: `Pass1Output.alternative_schemas` now `List[Dict[str, Any]]` matching `Pass1Result`. Each entry `{"name": str, "confidence": float}`. Bare strings auto-lift to `{"name": s}` for back-compat (`test_alternative_schemas_legacy_string_format_lifted_to_dict`). End-to-end Pydantic round-trip pinned in `test_dump_feeds_pass1_result_construction_directly`. |
| 3 | 100-type ontology + 1500-token sample blows the 2400-token budget (~4595 tokens, 53% over) | `9a2e3fc` + `bf01589` | `COMPACT_SUMMARY_THRESHOLD = 30`: above-threshold ontologies render names-only (no descriptions). Output-format section trimmed from ~465 to ~280 tokens. `TestStressTokenBudget::test_100_type_ontology_fits_under_budget` and `test_150_type_ontology_still_fits` are the regression net. Worked numbers: synth-100 → 2192 tokens (8.7% headroom); synth-150 → 2367 tokens (1.4% headroom); real scholarly → 1947 tokens (18.9% headroom). |

### Bonus (LLMExtractor LLMManager→ModelManager rename)

Picked Option B (TODO marker) — see commit `339907d`. The full fix requires switching from `manager.generate(...)` (doesn't exist on ModelManager) to `get_model_from_config(model).complete(...)` + a `shared.models.Model` lookup, which is genuinely a DI wiring change that belongs in B.1f. The TODO references this review file so the next reader sees the diagnosis. No behaviour change vs attempt 1 — `try/except ImportError` still silently returns an empty extraction.

### Minors

| # | Finding | Commit | Resolution |
|---|---|---|---|
| 1 | Coverage at 89%, target ≥ 90% | `8076722` | 100% on `pass1_schema_validation.py` AND 100% on `prompts/pass1.py`. New tests: `test_pass1_output_constructor_validation_error_degrades`, `test_pass1_output_constructor_generic_exception_degrades`, `test_none_passed_to_unit_clamp_validator`, `test_empty_description_renders_name_only`. |
| 2 | Candidate-schema list missing `base`, `policy_themes`, `social_profiles` | `9a2e3fc` | Added — `CANDIDATE_SCHEMA_NAMES` is now the full 11-ontology surface. Pinned by `test_candidate_schemas_list_is_complete`. |
| 3 | No salvage when JSON is wrapped in arbitrary prose | `bf01589` | `_salvage_json_object` regex-extracts the first `{...}` block after fence-stripping. Tests `test_prose_around_json_is_salvaged`, `test_prose_then_fenced_json`, `test_salvage_helper_returns_input_when_no_braces`. |
| 4 | Dead `manager = ModelManager()` instantiation in default LLM caller | `bf01589` | Removed; replaced with import-only check (`import llm_manager.manager  # noqa: F401`) so the dep surfaces if absent without allocating throwaway state. |
| 5 | System prompt leaked outside `prompts/` module | `9a2e3fc` | Moved to `prompts/pass1.PASS1_SYSTEM_PROMPT`. The validator imports it from there. `test_run_with_sync_caller` asserts the value matches. |
| 6 | "~30 entity types" claim for scholarly.yaml is wrong (actually 8) | self-review edit (this commit) | Corrected at line 46. Both attempt-1 and attempt-2 budget figures recorded for diff. |

### Verification (attempt 2)

```
pipelines/ontology-extraction: 124 passed (was 98)
  Coverage on pass1_schema_validation.py: 100% (was 89%, target ≥ 90%)
  Coverage on prompts/pass1.py:           100%
packages/shared:               145 passed (unchanged)
apps/app-main:                 368 passed (no regression)

Token-budget worked examples (attempt 2):
  scholarly.yaml + 1500-tok sample:  ~1947 tokens / 2400 budget (18.9% headroom)
  synth 100 types + 1500-tok sample: ~2192 tokens / 2400 budget (8.7% headroom)
  synth 150 types + 1500-tok sample: ~2367 tokens / 2400 budget (1.4% headroom)
```

Public imports re-verified: `from ontology_extraction import Pass1SchemaValidator, Pass1Output, TokenBudgetExceeded, Pass1ParseError` continues to succeed.
