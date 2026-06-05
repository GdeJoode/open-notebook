# Review — Track A Phase A.1b attempt 2

**Branch**: `track/a-mineru-dispatcher`
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-03

## Summary

Both issues from the attempt-1 review are correctly addressed. The API
schemas now reject invalid `parser_engine` values at the FastAPI/Pydantic
boundary before the service layer is reached (verified — the test asserts
the service mock was *not* awaited). The `parser_engine_used` metadata
now reflects the engine that actually ran (mineru / docling / whisperx)
rather than the user setting, with explicit assignment in each dispatch
branch and a robust transcription/document discriminator for the
IngestionWorkflow path. Test suite expands from 299 → 306 with the
expected +7 (5 settings-router + 2 extractor); 101/101 shared tests
unchanged. Scope is tight (5 files, +229/-5 lines); no incidental
changes outside the review's brief.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| Blocker fix | `SettingsResponse.parser_engine` and `SettingsUpdate.parser_engine` typed as `Optional[Literal[...]]` | ✅ | Both fields at `schemas.py:303` and `schemas.py:335` are `Optional[Literal["simple", "docling", "mineru", "auto"]]`. `Literal` is already imported (`schemas.py:7`). Matches `ContentSettings.parser_engine` in `packages/shared/src/shared/models/settings.py:31-32`. |
| Blocker fix | 422 on invalid value | ✅ | `test_put_rejects_invalid_parser_engine` asserts `resp.status_code == 422`. |
| Blocker fix | Validation fires **before** persistence | ✅ | Same test asserts `svc.update.assert_not_awaited()`. FastAPI rejects the request body during model parsing, so the service layer (and the `UPSERT MERGE`) never runs. |
| Blocker fix | 200 on all 4 valid values | ✅ | `test_put_accepts_each_valid_parser_engine` is parametrized over `["simple","docling","mineru","auto"]` and asserts 200 + echo of the value. All 4 cases pass. |
| Major fix | `engine_used` set per dispatch branch | ✅ | MinerU branch → `"mineru"` (`source_extractor.py:192`); docling HTTP branch → `"docling"` (`source_extractor.py:208`); IngestionWorkflow branch post-execution sets `"whisperx"` if `result.transcription is not None and result.document is None`, else `"docling"` (`source_extractor.py:248-251`). |
| Major fix | `ExtractionResult.metadata["parser_engine_used"]` now uses `engine_used` | ✅ | `source_extractor.py:293` reads `engine_used`, not `resolved_engine`. |
| Major fix | Tests pin `simple`+`.pdf` → `"docling"` | ✅ | `test_simple_setting_records_docling_engine_used` passes. |
| Major fix | Tests pin `mineru`+`.mp3` → `"whisperx"` | ✅ | `test_audio_records_whisperx_engine_used` passes. The audio extension bypasses the dispatcher entirely (extension is not docling-parseable, so `resolved_engine = "docling"` is set as a placeholder); the IngestionWorkflow post-execution branch promotes to `"whisperx"` based on result shape. |
| Side effects | Existing tests still pass | ✅ | `test_records_parser_engine_used_in_metadata` (docling default), `test_routes_to_mineru_when_parser_engine_mineru` (mineru → "mineru"), `test_falls_back_to_docling_for_unsupported_mineru_ext` (fallback → "docling"), `test_auto_setting_resolves_to_docling_in_a1b` (auto → "docling") all continue to assert correctly under the new semantics — none of them depended on the old `resolved_engine == user_setting` identity. |
| Side effects | Test count tallies | ✅ | 299 + 7 (5 + 2) = 306. Confirmed. |
| Side effects | No accidental changes | ✅ | `git diff --stat 093337b..63af72a` touches exactly: `schemas.py`, `source_extractor.py`, two test files, and `status.md`. No frontend/shared changes. |
| Style | Type hints, docstrings intact | ✅ | `engine_used: str` annotation added; docstring updated to spell out the engine-that-ran semantic; comments explain *why*, not *what*. |

## Test status

```
$ uv run --project apps/app-main pytest apps/app-main/tests/ -x --tb=line -q
306 passed, 3 warnings in 76.72s

$ uv run --project apps/app-main pytest apps/app-main/tests/test_settings_router.py::TestParserEngineValidation -v
5 passed:
  test_put_rejects_invalid_parser_engine
  test_put_accepts_each_valid_parser_engine[simple]
  test_put_accepts_each_valid_parser_engine[docling]
  test_put_accepts_each_valid_parser_engine[mineru]
  test_put_accepts_each_valid_parser_engine[auto]

$ uv run --project apps/app-main pytest \
    apps/app-main/tests/test_source_processing_service.py::TestProcessFile::test_simple_setting_records_docling_engine_used \
    apps/app-main/tests/test_source_processing_service.py::TestProcessFile::test_audio_records_whisperx_engine_used -v
2 passed

$ uv run --project packages/shared pytest packages/shared/tests/ -q
101 passed
```

The pre-existing `tests/test_domain.py` collection error
(`ModuleNotFoundError: open_notebook`) was not re-checked here — it
remains a pre-existing condition unaffected by this branch.

## Issues found

### 🔴 Blockers (must fix)

None.

### 🟡 Major (must fix)

None.

### 🔵 Minor (optional)

1. **WhisperX/Docling discriminator silently defaults to "docling" when
   both `result.transcription` and `result.document` are present** —
   `apps/app-main/src/app_main/services/source_extractor.py:248-251`
   - The condition `result.transcription is not None and result.document is None`
     means: only-transcription → `"whisperx"`; everything else → `"docling"`.
     The "both populated" and "neither populated" cases both fall to
     `"docling"`. `IngestionResult` (in
     `pipelines/ingestion/src/ingestion/models/source.py:220-221`) allows
     both to be set independently; in practice the workflow only ever
     populates one, but the contract doesn't forbid both. A future
     hybrid pipeline (audio + transcribed-PDF, audio-with-OCR) would
     silently be reported as `"docling"` here.
   - Not blocking: today's workflow.process() implementation makes the
     both-populated case unreachable, and "neither populated" is
     guarded by `if not result.success: raise` earlier in the flow.
     But the assumption is implicit, not enforced. If A.1c plans to
     consume this field for confidence-based fallback, consider raising
     or logging if neither field is populated (defensive programming).
2. **Inert minor #1 from attempt 1 not addressed (mineru-mock import
   patching)** — `apps/app-main/tests/test_source_processing_service.py:351-365`
   - The brittle `sys.modules` patch pattern persists. Attempt 1's
     review explicitly called this out as MINOR (not blocking), and
     the implementer's status.md notes it as deferred. No change here;
     just noting it's still on the books for A.2 or follow-up cleanup.
3. **The IngestionWorkflow post-execution `engine_used` assignment runs
   even when the workflow ultimately raised (result.success is False)** —
   `apps/app-main/src/app_main/services/source_extractor.py:244-251`
   - The branch sets `engine_used` from `result.transcription` /
     `result.document` immediately after the workflow returns, *then*
     line 253 raises `RuntimeError` if `result.success` is False. On
     a failed run, the `engine_used` value is computed but never used
     (the raise short-circuits the `return ExtractionResult(...)`).
     This is harmless but the dead assignment is slightly awkward.
     Could be deferred until after the success check. Not a bug.

## Decision rationale

Both attempt-1 findings are correctly and completely addressed:

- **Blocker fixed (verified twice)**: The schema typing is exactly what
  was requested. The new test directly proves boundary-rejection
  (`svc.update.assert_not_awaited()`) rather than relying on the 422
  status alone — this is precisely the right assertion shape because
  the original concern was that invalid values could reach the
  persistence layer before being rejected on response construction.
  Pydantic now rejects in the request-body parsing phase, before any
  service method is even called.

- **Major fixed (verified)**: `engine_used` is unambiguously set in
  every dispatch branch. The MinerU and docling-service branches set
  it directly to constants; the IngestionWorkflow branch infers it
  from the result shape using a clean predicate. The two new tests
  exactly cover the cases the attempt-1 review identified as wrong
  (`simple`+`.pdf` and `mineru`+`.mp3`). Importantly, the existing
  mineru/docling tests *also* still pass under the new semantic — the
  refactor is backward-compatible because the old user-setting values
  happened to coincide with what-actually-ran in the cases those tests
  exercised.

The minors above are genuinely minor — defensive-programming
observations and a long-standing brittle test pattern that the
implementer has documented. None blocks merge.

The change is also minimally invasive: 5 files, +229/-5 lines, all
within the review's brief. No accidental changes elsewhere.

## Implementer's claim verification

| Implementer's claim | Verified? | Evidence |
|---|---|---|
| 306 tests pass (was 299, +7) | ✅ | `pytest -q` → `306 passed`. |
| 101 tests in packages/shared pass | ✅ | `pytest -q` → `101 passed`. |
| Schema typing fixed (both fields) | ✅ | `rg parser_engine schemas.py` shows `Optional[Literal[...]]` on both `SettingsResponse:303` and `SettingsUpdate:335`. |
| Invalid → 422 + service never called | ✅ | `test_put_rejects_invalid_parser_engine` asserts both. |
| All 4 valid values → 200 (parametrized) | ✅ | `test_put_accepts_each_valid_parser_engine` parametrized over `["simple","docling","mineru","auto"]`. |
| `engine_used` per dispatch branch | ✅ | Lines 192 (mineru), 208 (docling service), 248-251 (IngestionWorkflow). |
| WhisperX discriminator | ✅ | `result.transcription is not None and result.document is None` at line 248. |
| Tests cover `simple`+`.pdf` → "docling" | ✅ | `test_simple_setting_records_docling_engine_used` passes. |
| Tests cover `mineru`+`.mp3` → "whisperx" | ✅ | `test_audio_records_whisperx_engine_used` passes. |
| Operator warning (minor #2 from attempt 1) added to status.md | ✅ | status.md caveat #6 added (verified in diff). |
| No regressions to existing tests | ✅ | Full 306-test run is green; the four pre-existing parser_engine_used asserts (lines 339, 366, 396, 419) still hold under the new semantic. |

## Next steps

APPROVED. The implementer should:

1. Push the branch (already pushed per status.md note #5).
2. Open a PR against `main` with the attempt-1 + attempt-2 commits.
3. Optional: address minor #1 (defensive guard on neither-populated)
   as part of A.1c when the confidence-fallback consumes this field —
   that's the natural moment to harden the contract.

Recommendation to orchestrator: **proceed to PR creation.**
