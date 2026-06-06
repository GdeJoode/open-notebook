# Review — Track B Phase B.1c attempt 2

**Branch**: `track/b-pass1-module` (tip `e513f52`)
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-06

## Summary

All three majors and all six minors from attempt 1 are addressed cleanly. Test count is up
from 98 to 124 in `pipelines/ontology-extraction`, coverage is **100%** on both
`pass1_schema_validation.py` and `prompts/pass1.py` (was 89% / 86%). Worked token-budget
numbers in the self-review under-report actual headroom — synth-100 measures **1991 tok**
(self-review said 2192) and synth-150 measures **2066 tok** (self-review said 2367), both
well below the 2400 cap. Graceful malformed-JSON degradation, `alternative_schemas` dict
shape with back-compat lift, and the `_salvage_json_object` regex helper all behave per
contract. No regressions in `packages/shared` (145 pass) or `apps/app-main` (368 pass).

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | `Pass1Output` Pydantic-validates against LLM output schema | OK | All six fields present, validator clamps + coerces defensively, extras ignored. |
| 2 | Mocked `{coverage_pct: 87, ...}` → `coverage_pct = 87.0` (interpreted as rescale) | OK | `test_percentage_coverage_is_rescaled` pins `87 → 0.87`. Plan AC's literal `87.0` is internally inconsistent with the `le=1.0` field constraint; implementer's rescale-to-0.87 interpretation is the right call and was already accepted in attempt 1. |
| 3 | Invalid JSON → `Pass1Output(coverage_pct=0.0, confidence_in_choice=0.0)` + WARNING | OK | `_parse_response` returns `_empty_pass1_output()` for invalid JSON, empty string, whitespace, non-object, truncated JSON, ValidationError, and generic constructor exceptions. WARNING log includes parse reason + 240-char excerpt + ellipsis. `Pass1ParseError` is now reserved for transport-level failures only (well-documented contract distinction). |
| 4 | Token budget ≤ 3000 tokens for ≤ 100 entity types + 5-chunk sample | OK | Verified: synth-100 + 1500-tok sample = **1991 tokens** (cap 2400, plan ceiling 3000), synth-150 + 1500-tok sample = **2066 tokens**. Both stress tests are pinned (`TestStressTokenBudget`). Real-world `scholarly.yaml` (8 types) = **1947 tokens**. |
| 5 | ≥ 90% line coverage of `pass1_schema_validation.py` | OK | **100%** on `pass1_schema_validation.py` AND `prompts/pass1.py` (was 89% / 86% in attempt 1). |

## Test status

```
packages/shared:                     145 passed in 4.02s   (unchanged from attempt 1)
pipelines/ontology-extraction:        124 passed in ~161s  (+26 vs attempt 1's 98)
  Coverage on pass1_schema_validation.py: 100% (138 stmts, 0 missed)
  Coverage on prompts/pass1.py:           100% (48 stmts,  0 missed)
apps/app-main:                       368 passed in ~381s   (no regressions)
```

## Major-finding verifications

### Major 1 — graceful malformed-JSON (verified)

`pass1_schema_validation.py:432-479`:

- `_parse_response` returns `_empty_pass1_output()` on: empty string (446), whitespace (446),
  `JSONDecodeError` (459), non-object top-level (466), `ValidationError` (472), and any other
  constructor exception (479). No path raises from a content-parse failure.
- `Pass1ParseError` is only raised from `run()` when the LLM caller itself throws (transport),
  pinned by `test_run_wraps_llm_caller_exception_in_pass1_parse_error` +
  `test_run_wraps_async_caller_exception`.
- WARNING log via `_log_malformed` (319-332) includes `reason=...` and `excerpt='...'` with
  truncation to 240 chars + `...` suffix. Verified by manual log inspection AND
  `test_malformed_json_emits_warning_log`.
- `_empty_pass1_output()` is centralised (302-316), so every degraded return is bit-identical
  from the caller's perspective.

Plan AC #3 is now met as written.

### Major 2 — `alternative_schemas` dict shape (verified)

`pass1_schema_validation.py:166-176, 222-254`:

- `Pass1Output.alternative_schemas: List[Dict[str, Any]]` matches
  `Pass1Result.alternative_schemas` exactly (verified by `test_dump_feeds_pass1_result_construction_directly`
  which actually constructs a `Pass1Result` from `Pass1Output.model_dump()` and asserts no `ValidationError`).
- Back-compat lift: `normalise_alternative_schemas` validator promotes bare strings to
  `{"name": s}` (line 250-251). Pinned by `test_alternative_schemas_legacy_string_format_lifted_to_dict`.
- Edge cases: dict entries missing `name` are dropped (line 246-248), non-dict/non-string
  entries are silently dropped (line 252-254), and the list is truncated to 3 entries
  (line 254). All three branches covered by dedicated tests.
- Docstring example in `Pass1Output` (lines 112-128) shows the correct persistence pattern
  with the dict-shaped dump feeding `Pass1Result(...)` directly. No more docstring lie.

Persistence path B.1f will rely on `**pass1.model_dump()` and now succeeds.

### Major 3 — 100-type token budget (verified)

`prompts/pass1.py:37, 114-203` and `pass1_schema_validation.py:343-352`:

- `COMPACT_SUMMARY_THRESHOLD = 30` is a module-level constant.
- For ontologies with > 30 types, `build_schema_summary` emits names-only in a single
  brace-wrapped comma-separated group plus a "Types compressed" note (lines 181-192).
- Output-format section trimmed: verified at **241 tokens** (claim ~280; attempt 1 was ~465).
- Worked numbers (this reviewer's measurements):
  - `scholarly.yaml` (8 types) + 1500-tok sample = **1947 tokens** (matches claim, 18.9% headroom)
  - synth-100 + 1500-tok sample = **1991 tokens** (claim 2192 — claim was conservative; 17% headroom on target, 33.6% on plan ceiling)
  - synth-150 + 1500-tok sample = **2066 tokens** (claim 2367 — also conservative; 13.9% headroom on target)
- Stress tests pinned: `test_100_type_ontology_fits_under_budget`,
  `test_150_type_ontology_still_fits`. Boundary test confirms compression kicks in at
  > 30 (`test_compression_kicks_in_above_threshold`) and exactly 30 stays verbose
  (`test_at_threshold_uses_verbose`).

Plan AC #4 is now met with substantial headroom.

## Bonus (LLMExtractor TODO) — verified with one caveat

`pipelines/ontology-extraction/src/ontology_extraction/extractors/llm_extractor.py:35-48`:

- TODO marker is present, well-formed, and references the attempt-1 review file.
- Explains: `LLMManager` doesn't exist (real class is `ModelManager`), `manager.generate(...)`
  doesn't exist either, correct path is `ModelManager().get_model_from_config(model).complete(...)`,
  and the fix is a DI wiring change owned by B.1f.
- Behaviour unchanged from `main` (still `ImportError` → empty `ExtractionResult`), so no
  new bug is shipped.

**Caveat (minor, not blocking)**: The plan §B.1f text does NOT explicitly call out the
`LLMManager → ModelManager` rename. Plan §B.1d touches `llm_extractor.py` to refactor the
prompt-build, but the rename itself isn't scoped in either phase. Best-effort scoping was
clearly the implementer's intent (the TODO is grep-able for `B.1f`), but the next
implementer reading the plan won't see the dep unless they grep the TODOs first. Filed as
**minor — recommend a one-line note in the plan §B.1f** ("includes the
`LLMManager → ModelManager` rename in `llm_extractor.py` — see TODO marker"). Not blocking.

## Minor-finding verifications

| # | Finding | Status |
|---|---|---|
| 1 | Coverage ≥ 90% on `pass1_schema_validation.py` + `prompts/pass1.py` | **100% / 100%** (was 89% / 86%). Three coverage-gap tests added: `test_pass1_output_constructor_validation_error_degrades`, `test_pass1_output_constructor_generic_exception_degrades`, `test_none_passed_to_unit_clamp_validator`, `test_empty_description_renders_name_only`. |
| 2 | Candidate-schema list includes `base`, `policy_themes`, `social_profiles` | All 11 ontologies on disk match `CANDIDATE_SCHEMA_NAMES`. Pinned by `test_candidate_schemas_list_is_complete`. |
| 3 | Prose-around-JSON salvage works | `_salvage_json_object` (`re.search(r"\{.*\}", text, re.DOTALL)`) salvages prose-wrapped JSON. Both `test_prose_around_json_is_salvaged` and `test_prose_then_fenced_json` pass. |
| 4 | Dead `manager = ModelManager()` instantiation removed | Replaced with `import llm_manager.manager  # noqa: F401`. Dependency-surface check without dead allocation. |
| 5 | System prompt moved to `prompts/pass1.PASS1_SYSTEM_PROMPT` | Verified. The validator imports it from there; `test_run_with_sync_caller` asserts the system prompt seen by the LLM caller equals the constant. |
| 6 | Self-review says "8 entity types" for scholarly.yaml | Verified: `scholarly.yaml` has 8 entity types (ScholarlyArticle, Thesis, Periodical, ConferenceEvent, Researcher, ResearchOrganization, EducationalOrganization, ResearchProject). Self-review corrected at line 46 with both attempt-1 and attempt-2 numbers recorded. |

## Edge-case probes (this reviewer)

1. **Legacy string list back-compat**: `Pass1Output(alternative_schemas=["scholarly", "policy"])` →
   `[{"name": "scholarly"}, {"name": "policy"}]`. Lift works in the constructor path
   (not just the parser path). Good.
2. **Out-of-range confidence at boundary**: `confidence_in_choice=1.5` clamps to 1.0
   (because threshold is `if fv > 1.5`, not `>=`). `1.2` and `1.4` also clamp to 1.0.
   `1.6` rescales to `0.016`. Boundary behaviour is consistent with the docstring claim
   that "1.5 is the threshold so a legit 1.0 passes through unchanged".
3. **Multiple JSON blocks in prose** (e.g. `"Here is one: {…FIRST…} and another: {…SECOND…}"`):
   The greedy `re.DOTALL` regex captures everything from the first `{` to the last `}`,
   including the `} and another: {` interlude. `json.loads` then fails with `Extra data`,
   and the graceful malformed-JSON path returns the empty sentinel. **Behaviour is
   defensible** — there's no signal to know which of two blocks is the intended one — but
   worth being aware of. Tested manually; result: `detected_schema == ""` (the safe degradation).
4. **Compact-summary boundary**: at exactly 30 types → verbose form (because threshold is
   `> 30`, not `>=`). Confirmed by `test_at_threshold_uses_verbose`. Above 30 → compact.

## Quality gates run

```
packages/shared:                  145 passed in 4.02s
pipelines/ontology-extraction:   124 passed in ~161s
  pass1_schema_validation.py:    100% (138/138)
  prompts/pass1.py:              100% (48/48)
apps/app-main:                   368 passed in ~381s
```

All green. No skipped tests in the Pass-1 module.

## Issues found

### Blockers

None.

### Major

None. All three from attempt 1 are addressed.

### Minor (optional, not blocking)

1. **Plan §B.1f doesn't enumerate the `LLMManager → ModelManager` fix in `llm_extractor.py`.**
   The TODO marker references the rename and the attempt-1 review, but a fresh implementer
   reading `plan.md` won't see the dependency. Recommend a one-line addition to plan §B.1f's
   "Files to modify" section. Not blocking — `grep -rn "TODO(B.1f)"` already surfaces this
   from the codebase.

2. **Greedy regex in `_salvage_json_object` over-captures across multiple JSON blocks.**
   The current `re.search(r"\{.*\}", text, re.DOTALL)` matches from the first `{` to the
   last `}` in the prose. For LLM responses containing two JSON blocks separated by prose
   (rare but plausible: "Here's option A: {...} or B: {...}"), the salvage produces a
   non-parseable string and the graceful-degradation path kicks in. **This is safe** (no
   crash; empty sentinel + WARNING), but a per-object loop with `json.loads` validation
   could in principle pick the parseable block. Defer — defensible as-is.

3. **`Pass1Result` and `Pass1Output` both have `coverage_pct` with `ge=0, le=1`**, but
   `Pass1Output.clamp_to_unit_interval` rescales > 1.5 inputs while the persistence model
   does not (B.1b). If a future caller bypasses Pass-1's parser and feeds raw LLM output to
   `Pass1Result` directly, the percentage-style scalar would be rejected. Not in scope for
   B.1c, but B.1f should funnel everything through `Pass1Output` before persistence
   (which it will, per the docstring example) — worth a note in the B.1f acceptance criteria.

## Notes / kudos

- Self-review's worked token-budget numbers are **conservative** vs. actual: synth-100
  measured 1991 (vs claimed 2192), synth-150 measured 2066 (vs claimed 2367). Whatever
  the discrepancy source (perhaps the implementer included a system-prompt header in the
  estimation), this is the right direction to be wrong in.
- The `_log_malformed` excerpt format (`reason=... | excerpt='...{ellipsis}'`) is grep-able
  for the orchestrator (B.1e) to count parse failures by reason. Nice touch.
- `_empty_pass1_output()` centralisation means every malformed-path return is identical;
  a downstream test that asserts `result == _empty_pass1_output()` (used three times in the
  test suite) works because every field defaults to a "this is the empty sentinel" value.
- The `Pass1ParseError` is now genuinely useful (transport-only) rather than the
  catch-all-content-failure it was in attempt 1. This is a cleaner contract for B.1e.
- The dict-shaped `alternative_schemas` end-to-end test (`test_dump_feeds_pass1_result_construction_directly`)
  catches the exact Major-2 regression at the contract boundary. Good design.
- Three explicit branch-coverage tests (`test_pass1_output_constructor_validation_error_degrades`,
  `test_pass1_output_constructor_generic_exception_degrades`, `test_none_passed_to_unit_clamp_validator`)
  show the implementer didn't game coverage with noisy "import path is reachable" tests.
- Loguru→stdlib bridging in `test_malformed_json_emits_warning_log` is a known sharp edge
  that's now solved in a reusable pattern.

## Decision rationale

Zero blockers, zero majors. Three minor recommendations none of which would impede B.1d or
B.1e. The token-budget headroom on the 150-type stress case (13.9%) gives downstream phases
real room to add a few hundred more tokens to the prompt (e.g. for a worked example) without
re-engineering compression. The graceful-degradation contract is now what the plan said it
should be, the persistence shape is what B.1b's `Pass1Result` expects, and `Pass1ParseError`
means something specific.

This is a clean foundation for B.1d (Pass 2), B.1e (multi-schema orchestrator), and B.1f
(EntityExtractionService wiring).

## Next steps

**APPROVED** — ready for human approval / merge.

If picked up for merge, the implementer may want to:

1. Address minor 1 (one-line plan §B.1f update). Optional; the TODO marker is sufficient.
2. File minor 2 (`_salvage_json_object` multi-object handling) as a B.1e nice-to-have.
3. Note minor 3 (`Pass1Output → Pass1Result` is the only sanctioned persistence path) in the
   B.1f plan body.

None of these block this PR.
