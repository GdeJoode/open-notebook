# Review — Track B Phase B.1c attempt 1

**Branch**: `track/b-pass1-module`
**Decision**: REVISIONS_NEEDED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-05

## Summary

Solid execution overall — 145 + 98 + 368 tests green, clean module structure, defensive parsing, good test coverage. However, three contract-level issues block approval: (1) acceptance criterion #3 was inverted from "fallback default" to "raise on malformed JSON" without explicit reviewer sign-off, (2) the documented persistence pattern in the `Pass1Output` docstring will raise `ValidationError` due to the `alternative_schemas` type split, and (3) the 100-entity-type ontology budget asserted by AC #4 is ~53% over the plan ceiling under the current schema-summary format. These will trip B.1e and B.1f directly if left unaddressed.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | `Pass1Output` Pydantic-validates against the LLM output schema | ✅ | All six fields present; `test_well_formed_response` + many others pin this. |
| 2 | Mocked `{coverage_pct: 87, ...}` → `coverage_pct = 87.0` | ⚠️ | Plan AC literally says `87.0` but pydantic `le=1.0` would reject. Implementer reinterpreted as "rescale percentage → `0.87`" (`test_percentage_coverage_is_rescaled`). This is the right call given the plan's internal inconsistency, but should be flagged in the plan if accepted. |
| 3 | Invalid JSON → `Pass1Output(coverage_pct=0.0, confidence_in_choice=0.0)` + WARNING | ❌ | Implementer raises `Pass1ParseError` instead. See major #1. |
| 4 | Token budget ≤ 3000 tokens for ≤ 100 entity types + 5-chunk sample | ❌ | A 100-type ontology + 1500-tok sample produces a ~4595-token prompt; `TokenBudgetExceeded` fires. See major #3. |
| 5 | ≥ 90 % line coverage of `pass1_schema_validation.py` | ⚠️ | Measured at 89 % (`pass1_schema_validation.py`) / 86 % (`prompts/pass1.py`). Just under target; trivially fixable. |

## Test status

```
packages/shared:                     145 passed in 2.98s   (+17 new)
pipelines/ontology-extraction:        98 passed in 4.69s   (+37 new)
apps/app-main:                       368 passed in 89.56s  (0 regressions)

Coverage (pass1 modules only):
  pass1_schema_validation.py          89 %  (target ≥ 90 %)
  prompts/pass1.py                    86 %
```

Real-world budget print confirmed: scholarly.yaml (8 entity types — not 30 as self-review claims) + 1500-tok sample → ~2132 tokens / 2400 budget (11.2 % headroom on target, 29 % on plan ceiling).

## Issues found

### 🔴 Blockers (must fix)

(none — nothing is broken for the test set claimed; the issues below are contract divergences and documentation lies, not functional bugs)

### 🟡 Major (must fix)

1. **AC #3 contract inverted without reviewer sign-off** — `pass1_schema_validation.py:323-340`, plan §B.1c AC #3
   - Issue: Plan acceptance criterion #3 specifies `run_pass1` returns `Pass1Output(coverage_pct=0.0, confidence_in_choice=0.0)` on malformed JSON. Implementer changed to "raise `Pass1ParseError`" without explicit deviation approval. Downstream B.1e orchestrator will write retry-or-fallback logic based on the contract; the current behavior changes the integration story.
   - Recommendation: Either (a) add a soft-fallback overload (e.g., `run(..., on_error="raise" | "default")` with `"raise"` as default per implementer's preferred contract), or (b) update the plan AC #3 to record the deviation and rationale before merge. The self-review's "trade-offs" section discusses the choice but does not acknowledge the plan contradicts it.

2. **`Pass1Output` docstring example will raise `ValidationError`** — `pass1_schema_validation.py:88-107`
   - Issue: The class docstring shows `Pass1Result(..., **pass1.model_dump())` as the persistence pattern. This will fail at runtime because `Pass1Output.alternative_schemas: List[str]` does not validate into `Pass1Result.alternative_schemas: List[Dict[str, Any]]`. Verified by direct test (input `["general", "policy"]` → `2 validation errors for Pass1Result: Input should be a valid dictionary`).
   - Recommendation: Update the docstring to show the actual persistence pattern B.1f will use (e.g., `alternative_schemas=[{"name": s} for s in pass1.alternative_schemas]`), or move the example to a `Pass1OutputBuilder.to_pass1_result(...)` helper. The "Field shapes match … exactly" claim in the docstring is currently false. Self-review's "design decisions worth flagging" section mentions the discrepancy verbally but the source-of-truth code example wasn't fixed.

3. **AC #4 budget target unachievable at the 100-entity-type upper bound** — `prompts/pass1.py:101-149`
   - Issue: Plan AC #4 requires the assembled prompt to be ≤ 3000 tokens for "an ontology of ≤ 100 entity types and a 5-chunk sample". Synthesized test (100 types × ~100-char descriptions + 1500-token sample) produces ~4595 tokens — 53 % over the plan ceiling and 91 % over the implementer's 2400 target. `TokenBudgetExceeded` will fire for large ontologies. Scholarly.yaml has only 8 entity types; the real-world test does not stress the AC #4 boundary.
   - Recommendation: Either (a) add an entity-type cap (e.g., truncate `pairs` to top-N most-common-type or alphabetical), or (b) emit `name`-only when total prompt size would exceed budget (description-elided fallback), or (c) update plan AC #4 to lower the cap to ~30 types (which is what the implementer's design actually supports). Add a 100-type regression test either way.

### 🔵 Minor (optional)

1. **Coverage gap of 1 % on `pass1_schema_validation.py`** — covers 113 stmts, 12 missed (89 %). Missing lines: 161, 164-165 (validator None / TypeError branches), 189, 199 (list validator None / non-list branches), 295 (lazy-default fall-through), 352-368 (entire `_default_llm_caller` body). The default-caller branch is intentionally not exercised; the validator-None branches are trivial to cover with two parametrized tests.

2. **Hardcoded candidate-schema list in prompt** — `prompts/pass1.py:69-72` lists `scholarly, general, policy, government, regiodeal, deals, instruments, schema_core` but the repo also ships `base.yaml`, `policy_themes.yaml`, `social_profiles.yaml`. The LLM will never suggest these as `alternative_schemas`. B.1e (multi-schema orchestrator) will inherit the blind spot. Recommendation: pass the candidate list in as a function argument, populated from a directory scan at call time.

3. **Prose-around-JSON not handled** — `_strip_code_fence` only unwraps ``` ``` ``` fences. A response like `Here is the JSON: {…} Hope that helps.` raises `Pass1ParseError("Invalid JSON")`. The plan AC #3 envisions this as malformed and the implementer's behavior is consistent with that, but real LLMs do emit prose-wrapped JSON frequently — a regex like `re.search(r'\{.*\}', text, re.DOTALL)` would salvage it. Defer or address; not blocking.

4. **Dead `manager = ModelManager()` instantiation in `_default_llm_caller`** — `pass1_schema_validation.py:357`. The variable is constructed and never used; the inner closure ignores `manager` and returns `"{}"` directly. Either drop the instantiation entirely, or call through `manager.generate(...)` — but the latter belongs in B.1f. Cleanup-only.

5. **System prompt leaks out of the prompt-template module** — `pass1_schema_validation.py:286-289`. The system-role string ("You are a meticulous ontology curator…") lives inside `run()` rather than in `prompts/pass1.py`. Move it next to the user prompt for testability symmetry. Minor.

6. **Self-review claims scholarly.yaml has "~30 entity types"** — actually 8. The token-budget headroom number is correct; the inflation in the prose is misleading. Worth correcting in the self-review.

## Notes / kudos

- The implementer correctly diagnosed `LLMManager` → `ModelManager` as a real bug (verified by `grep`: `llm_manager.manager` exports `ModelManager`, the import in `llm_extractor.py:36` would always `ImportError` and silently return empty `ExtractionResult`). The choice to route around via DI rather than fix in-place is defensible.
- Token-budget guard fires **before** the LLM call (`test_budget_guard_fires_when_exceeded` confirms `called is False`). Costs are protected.
- Parser robustness is good: empty / whitespace / non-object / null-as-list / non-string-items / truncated-JSON / extra-fields / percentage-style scalars all covered with explicit tests.
- Sync + async LLM caller support (via `hasattr(raw_response, "__await__")`) is elegant and avoids `inspect.iscoroutine`.
- Conservative diacritic-preserving name normalizer matches Q-B-4 exactly; `"Café" → "café"` verified.
- Single import-point promise for `normalize_entity_name` honored: `shared.utils.normalize_entity_name is shared.utils.name_normalizer.normalize_entity_name` (same callable, not a wrapper).
- The TODO marker in `entity_extraction_service.py:84-88` is grep-able (`grep -rn "TODO(B.1f)"` finds it) and explains the wiring intent (sample chunks → run validator → persist via `Pass1ResultRepository.record`). Good.

## Opinion: should the `LLMManager → ModelManager` bug be fixed in this PR?

**Yes — minimal fix in this PR, full integration in B.1f.** Here is the case:

- **For fixing**: B.1a set the precedent by fixing the entity-persistence-drift bug as a bonus. The roadmap RETRO endorses surfacing-then-fixing latent bugs. The rename itself is 2 lines (`LLMManager` → `ModelManager`, possibly `manager.generate(...)` → whatever the actual method is). Leaving `LLMExtractor` broken means `app-main`'s entity-extraction path silently returns empty `ExtractionResult`s today, and B.1f will inherit a confusing surface area.
- **Against fixing**: Strict scope discipline. B.1c's job is the Pass-1 module, not a `LLMExtractor` repair. The PR diff stays small.

**My recommendation**: include a 5-line follow-up commit in this PR that (a) renames the import, (b) updates the method call to whatever `ModelManager`'s public API actually exposes (the implementer would need to verify with `grep`), and (c) adds one regression test in `pipelines/ontology-extraction/tests/test_llm_extractor.py` proving the import doesn't `ImportError`. This is the same energy-efficient fix pattern B.1a used. Alternatively: if the implementer prefers strict scope, **add a `TODO(B.1f)` marker inside `llm_extractor.py:36` referencing this finding so B.1f can't miss it**. Either is acceptable; doing nothing is not — the bug is now known.

## Decision rationale

Three majors block approval — all are contract-level mismatches between the plan and the implementation that will propagate to B.1e/B.1f. None are functional bugs in the code as it stands (tests pass, no regressions), but each will mislead a downstream agent who reads the docstring or trusts the plan AC. Fixing them is mostly documentation + one or two test additions:

- Major #1: either expose a fallback mode or update plan AC #3.
- Major #2: fix the misleading docstring (1-line code change, 1-paragraph doc change).
- Major #3: add a 100-type regression test + either tighten the schema summary or update plan AC #4.

The minors are genuinely optional and can ship as follow-up.

## Next steps

Implementer should:

1. Reconcile AC #3 contract (raise vs fallback). Recommended: add an optional `on_parse_error: Literal["raise", "default"] = "raise"` parameter so both contracts are supported and explicit. Update the plan if the default changes.
2. Fix the `Pass1Output` docstring's persistence example so it actually works, OR add a `to_pass1_result_fields()` helper that does the `[str] → [{"name": str}]` lift.
3. Address AC #4 — either lower the implementer's hard ceiling in code (with a 100-type regression test), or update the plan to acknowledge ≤30 types is the true cap. Self-review should not claim "11.2 % headroom" against the 2400 target without noting that this was measured on an 8-type ontology, not the AC #4 100-type bound.
4. (Optional, but recommended) Apply the 2-line `LLMManager → ModelManager` rename + 1 regression test in `llm_extractor.py`, OR drop a `TODO(B.1f)` marker there.
5. Re-submit for review.
