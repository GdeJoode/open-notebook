# Review — Track B Phase B.1d attempt 1

**Branch**: `track/b-pass2-module`
**Decision**: **APPROVED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-06

## Summary

Phase B.1d delivers a focused, well-tested Pass-2 typed-extraction module that establishes the B4 confidence-everywhere invariant at extraction time. All four plan acceptance criteria are met with strong, deterministic test coverage. The module faithfully mirrors B.1c's conventions (DI seam, lazy default with WARNING canary, token-budget guard, graceful malformed-JSON degradation, Pass2ParseError reserved for transport). No blockers, no majors — only a small handful of nits and design notes worth surfacing.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | `accepted_extensions=[]` → `ExtractionResult` for 3-chunk fixture, equivalent to current `LLMExtractor.extract` | PASS (with interpretation note) | `TestRunPass2::test_three_chunks_with_no_extensions` exercises the 3-chunk fixture and asserts shape + counts + metadata. Implementer chose "shape equivalence via mock" rather than a direct snapshot comparison against `LLMExtractor.extract`. Justified because `LLMExtractor.extract` is currently broken (the documented B.1c-r2 `LLMManager` TODO). Pragmatic and defensible. |
| 2 | Extension injection: prompt body contains type + parent; mock LLM returning the extension label produces `ExtractedEntity` with that label and `confidence=0.85` | PASS | `TestRunPass2::test_extension_injected_and_used` asserts `"EarlyCareerResearcher"` and `"parent: Researcher"` are in the user prompt, mock LLM returns the entity, assertion on label + confidence. |
| 3 | Confidence present and non-null on every entity AND every relation | PASS | `TestRunPass2::test_confidence_present_on_every_element` walks 3 entities and 2 relations from two chunks, asserting `confidence is not None`, `isinstance(float)`, `0.0 <= confidence <= 1.0`. Defensive default in `_clamp_confidence(None) → 0.0` ensures the field invariant holds even on partial LLM output. |
| 4 | ≥ 90% line coverage | PASS | `pass2_typed_extraction.py` 96% (5 missed = lazy `_default_llm_caller`), `prompts/pass2.py` 100%, combined 98%. |

## Test status (independently verified)

- `pipelines/ontology-extraction`: 187 passed (124 baseline + 63 new)
- `packages/shared`: 145 passed
- `apps/app-main`: 368 passed (no regressions)

## Minors (5, none blocking)

1. `_default_llm_caller()` resolves BEFORE the empty-chunks short-circuit (`pass2_typed_extraction.py:403`). Caller doing `await run_pass2([], ontology)` with no DI hits the lazy import + WARNING despite no work to do. Move the `if not chunks` short-circuit above caller resolution.
2. Prompt rule says "missing confidence will be dropped" but parser defaults to 0.0 (`prompts/pass2.py:204` vs `_clamp_confidence`). Asymmetry is intentional but prompt and parser should tell the same story.
3. No regression net for AC #1 against `LLMExtractor.extract` (the snapshot test the plan called for). Implementer's mock equivalence is defensible since `LLMExtractor` is broken. Snapshot ownership moves to B.1f.
4. No orphan-relation policy test. Parser silently includes relations whose source/target text isn't in entities list. One-line test would pin behaviour for B.1e.
5. Self-review "No app-main file touched" is technically correct (only the B.1c-added TODO is in app-main); reword as "no app-main change in B.1d commits" for clarity.

## Kudos

- Per-element confidence assertion test is exactly what the plan asked for — the load-bearing B4 invariant guard.
- 100-type stress test is real safety net.
- Telemetry policy implemented as structured logs with consistent event names; ready for B.4 lift.
- Test file well-organized into named test classes.
- Legacy `subject/predicate/object` back-compat path is a thoughtful transition guard.

## Next steps

APPROVED — ready for merge.
