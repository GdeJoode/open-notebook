# Review — Track A Phase A.1c attempt 1

**Branch**: `track/a-mineru-fallback`
**Decision**: REVISIONS_NEEDED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-03

## Summary

Phase A.1c lands the confidence scorer, the auto-fallback orchestrator,
and the wiring through `SourceExtractor` / `SourceProcessor`. The unit
tests for the new modules are thorough and acceptance criteria #1, #2,
#3, #4, #5, #6, #7, #9 pass on the mock layer. Two real bugs block
merge: (a) the `source` table in SurrealDB is **SCHEMAFULL** with no
`metadata` field defined in any migration, so the new
`Source.metadata` writes will either fail or be silently dropped in
production — the implementer's "schemaless tables handle it" claim is
factually wrong; (b) `_run_auto_fallback` uses `value or
DEFAULT_THRESHOLD` to resolve the threshold, so a user-set
`docling_min_confidence=0.0` falls back to the 0.95 default (Python
treats `0.0` as falsy). Both are easy to fix.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | `score_docling_extraction` returns `DoclingConfidenceScore`, weights sum to 1.0, overall in [0,1] for any input | ✅ | Asserted in `test_weights_sum_to_one`, `test_score_is_within_unit_interval_for_empty_result`. |
| 2 | Perfect-doc fixture scores ≥ 0.95 | ✅ | `test_score_for_perfect_document_is_high` asserts ≥ 0.95. |
| 3 | Scanned image-only fixture scores ≤ 0.7 | ✅ | `test_score_for_scanned_document_is_low` asserts ≤ 0.7 (observed ~0.28). |
| 4 | `parser_engine="auto"` + score 0.97 ⇒ MinerU not called, `parser_engine_used="docling"` | ✅ | `test_auto_high_confidence_keeps_docling` in `test_source_processing_service.py`. |
| 5 | `parser_engine="auto"` + score 0.60 ⇒ MinerU called, `extraction_fallback_triggered=True` | ✅ | `test_auto_low_confidence_falls_back_to_mineru`. |
| 6 | Docling raising ⇒ auto-mode swallows + calls MinerU + WARNING log | ✅ | `test_docling_exception_triggers_mineru_with_warning`. |
| 7 | `processing_overrides.docling_min_confidence=0.99` raises the bar for that one call | ⚠️ Partial | `test_auto_respects_docling_min_confidence_override` covers 0.99. Bound test for `0.0` is missing AND fails functionally due to the falsy-zero bug below. |
| 8 | `Source.metadata` after auto-extraction contains 4 keys (verified via integration test against `TestClient`) | ❌ | Only mock-level test (`test_lifts_extraction_metadata_to_source_metadata`) was written; no `TestClient` integration test exists. Implementer notes this as "deferred to A.3 Playwright" — acceptable as a documented deferral, but the schemaful-source blocker (see below) means the test would have failed if written. |
| 9 | 50-page fixture scored in < 50 ms | ✅ | `test_score_is_fast_for_50_page_document` measures ~0.1-0.5 ms. |

## Test status

```
apps/app-main: 341 passed, 3 warnings in 68.64s
packages/shared: 105 passed in 2.43s  (claim said 101; actual is 105 — bonus, not a problem)
frontend tsc --noEmit: exit 0 (clean)
A.1b regression placeholder `test_auto_setting_resolves_to_docling_in_a1b`: not present in test code (removed correctly)
```

## Issues found

### 🔴 Blockers (must fix)

1. **`Source.metadata` writes will be rejected or silently dropped by SurrealDB — the `source` table is SCHEMAFULL with no `metadata` field defined in any migration.** — `migrations/1.surrealql:2` defines `DEFINE TABLE IF NOT EXISTS source SCHEMAFULL;`; latest migration is `42.surrealql` and no `DEFINE FIELD … metadata ON TABLE source …` exists. Comparable schemaful tables that need free-form metadata (`source_folder`, `pipeline_cache`) explicitly declare `DEFINE FIELD IF NOT EXISTS metadata ON TABLE … FLEXIBLE TYPE option<object>;` (see `migrations/26.surrealql:31`). The implementer claim in `status.md` lines 685-688 and `source.py` lines 129-131 — "schemaless tables (no migration needed)" — is factually wrong. SurrealDB SCHEMAFULL semantics reject writes to undefined fields; the existing test only exercises a mocked `source_repo.update`, so the bug is invisible to the test suite. In production, either every `_update_source` call with provenance keys will raise (breaking the entire source processing flow) or the `metadata` column will be silently dropped (no UI badge will ever render). All four "metadata" claims in the status doc + the model docstring rest on the wrong assumption.
   - File refs: `migrations/1.surrealql:2`, `packages/shared/src/shared/models/source.py:129-131`, `docs/tracks/A-mineru/status.md:687-688`, `apps/app-main/src/app_main/services/source_processor.py:159-176`.
   - Verification path: a real DB integration test (TestClient + live SurrealDB) would have caught it — acceptance criterion #8 explicitly required this and was skipped.

2. **`docling_min_confidence=0.0` silently reverts to the 0.95 default due to falsy-zero in `_run_auto_fallback`.** — `source_extractor.py:359-362`:
   ```python
   threshold = float(
       getattr(content_settings, "docling_min_confidence", None)
       or DEFAULT_THRESHOLD
   )
   ```
   Python evaluates `0.0 or 0.95` as `0.95` because `0.0` is falsy. A user who explicitly sets `docling_min_confidence=0.0` (or sends `processing_overrides.docling_min_confidence=0.0` to disable the fallback) silently gets the 0.95 default instead — the opposite of what they asked for. The API schema permits `0.0` (`Field(None, ge=0.0, le=1.0)`), so this is a reachable code path. Fix: use a `None` check, e.g. `value if value is not None else DEFAULT_THRESHOLD`. A regression test asserting `threshold=0.0` is respected should be added.
   - File ref: `apps/app-main/src/app_main/services/source_extractor.py:359-362`.
   - Note: the same falsy pattern exists for `int(getattr(document, "page_count", 0) or 0) or _infer_page_count(document)` in `confidence.py:101` — there it is intentional (page_count=0 should fall back to inferred). Not a bug.

### 🟡 Major (must fix)

1. **No integration test against `TestClient` for the auto-fallback metadata round-trip — acceptance criterion #8 explicitly required this and would have surfaced Blocker #1.** — Plan section "Phase A.1c — Tests required" lists "Integration: extend `test_source_extractor.py` with one auto-fallback flow using mocked docling + mineru clients" AND the acceptance criterion #8 reads "verified via integration test against `TestClient`". The implementer wrote only mock-level tests on `_update_source` and acknowledges this as deferred ("HTTP-level integration deferred to A.3 Playwright"). The deferral may be acceptable for the HTTP round-trip, but a real-DB test against the SurrealDB-backed `SourceRepository.update()` would have caught the SCHEMAFULL bug. At minimum, the implementer should either (a) add an integration test that hits a real SurrealDB instance (compose-up or testcontainer) or (b) document that this verification is owed in A.3 and explain why it's safe to ship without it (which they cannot, because of Blocker #1).
   - File ref: `apps/app-main/tests/test_source_processing_service.py:951-991` (mock-only).

2. **`status.md` repeats the wrong claim that "SurrealDB's permissive schemaless tables" make a migration unnecessary.** — `docs/tracks/A-mineru/status.md` lines around 687-688 and the docstring at `packages/shared/src/shared/models/source.py:127-131` both rely on the same incorrect assumption. Once Blocker #1 is fixed (by adding a migration), the docs must be corrected too.

### 🔵 Minor (optional)

1. **`shared` test count claim is slightly off.** — The prompt notes the implementer claim "101 packages/shared tests pass (was 97, +4)". Actual is 105 (probably reflects an extra batch landing). Not a problem — just an update.

2. **`test_docling_exception_triggers_mineru_with_warning` is over-defensive.** — `apps/app-main/tests/test_auto_fallback.py:200-204` ends the warning assertion with `or True`, which makes the assertion vacuous. Loguru-to-caplog interop is documented as flaky elsewhere; a more focused fix would be to capture via `loguru.logger.add(caplog.handler, format=…)` or to assert the side effect (MinerU was called) rather than the log message. Not a blocker but reduces the test's value.

3. **`_run_auto_fallback` builds a new `_DoclingAdapter` class on each call.** — `source_extractor.py:367-385`. Building a class inside an async method on every file is wasteful (small) but more importantly creates a closure over `content_settings`, `emit_key`, `log_stream` from the enclosing scope. Functionally correct but a small dependency-injection refactor (move `_DoclingAdapter` to module scope, take what it needs as constructor args) would be tidier.

4. **`confidence.py:101` page-count fallback uses `or` chain** — `max(1, int(getattr(document, "page_count", 0) or 0) or _infer_page_count(document))`. Works correctly but `_infer_page_count(document)` is only called when `page_count` is 0 (or missing) — fine; the comment above said it's intentional, just noting that the falsy-zero pattern is used here and in `_run_auto_fallback` (Blocker #2), and the inconsistent treatment is something to be careful about in the future.

5. **`auto_fallback.py:163` returns `signals={}` for the "docling failed" sentinel.** — Downstream UI tooltip code in Phase A.2 will need a defensive guard for empty `signals` dicts. Worth documenting on the badge component contract.

6. **`SettingsResponse` and `SettingsUpdate` are duplicate schemas with identical fields.** — Pre-existing, not introduced by A.1c, but the duplication is awkward. Not a A.1c issue.

7. **`test_docling_confidence.py:227-232` has a conditional assertion `if score_default.overall < 0.999`** — this makes the test weaker than it needs to be. If the perfect doc happens to score exactly 1.0, the strict-threshold flip is silently skipped. Better: build the fixture so the perfect-doc score is provably < 1.0 (e.g. add one image), then assert unconditionally.

## Decision rationale

Two real blockers found.

- **Blocker #1** is the more serious of the two: an entire feature
  (the Source.metadata bag) was written assuming the destination table
  is schemaless. It isn't. The unit tests all use mocks so they pass
  trivially, but the moment this PR ships and a user uploads a PDF
  with `parser_engine="auto"`, either the source-update path will
  raise a SurrealDB error and break processing for every source, or
  every metadata write will be silently dropped and the entire UI
  badge contract (Phase A.2) will be unimplementable. A small migration
  (one `DEFINE FIELD IF NOT EXISTS metadata ON TABLE source FLEXIBLE
  TYPE option<object>;` in a new `43.surrealql` + matching down
  migration) is the fix.
- **Blocker #2** is a one-line bug (`or DEFAULT_THRESHOLD` →
  `if value is None else …`) but a real one — silently overriding a
  user-set 0.0 with 0.95 is exactly the kind of "the system lied to
  me" bug we want to catch before merge.

Beyond the blockers, the major item is the absence of any real-DB
integration test — which is what would have caught Blocker #1 — and
the minors are quality nits that don't move the needle.

The 9 acceptance criteria are mostly met at the mock-test layer:
- ✅ #1, #2, #3, #4, #5, #6, #9 cleanly pass.
- ⚠️ #7 passes at 0.99 but the 0.0 lower-bound case is functionally
  broken (Blocker #2).
- ❌ #8 was not verified via TestClient — implementer documented this
  as deferred; deferral was acknowledged in the prompt's caveats so
  it's not by itself a blocker, but the deferral is what hides
  Blocker #1.

## Next steps

Implementer should:
1. Add a SurrealDB migration `migrations/43.surrealql` (and matching
   `43_down.surrealql`) defining a flexible `metadata` field on the
   `source` table: `DEFINE FIELD IF NOT EXISTS metadata ON TABLE
   source FLEXIBLE TYPE option<object>;`. Verify by running the
   migration locally, uploading a PDF in auto mode, and reading back
   `source.metadata` from the DB.
2. Fix the falsy-zero bug in `source_extractor.py:359-362` and add a
   parameterised regression test that calls `_run_auto_fallback` (or
   `extract_with_auto_fallback` directly) with `docling_min_confidence=0.0`
   and asserts that the chosen threshold is 0.0, not 0.95.
3. Update the model docstring in `source.py:127-131` and the relevant
   passages in `docs/tracks/A-mineru/status.md` to remove the
   "schemaless tables handle it" claim — the source table is
   SCHEMAFULL and a migration is required.
4. Address any minors that are cheap to fix; the rest can be filed
   as follow-ups.

Resubmit for review (attempt 2).
