# Review — Track A Phase A.1c attempt 2

**Branch**: `track/a-mineru-fallback`
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-03

## Summary

Attempt 2 fixes both blockers and both majors from attempt 1 with
surgical, well-scoped changes: migration #43 (up + down) declares the
`Source.metadata` field with the same `FLEXIBLE TYPE option<object>`
pattern that migration 26 uses for `source_folder.metadata` and
`pipeline_cache.metadata`; the falsy-zero bug in `_run_auto_fallback`
is replaced with an explicit `None` check and gains regression tests
at both orchestrator (`test_auto_fallback.py`) and extractor
(`test_source_processing_service.py`) layers; a new
`test_metadata_persistence_integration.py` exercises the real
`SourceRepository.update()` body (with `execute_query` patched at the
boundary) and statically validates the migration; the model docstring
and status doc no longer repeat the "schemaless tables handle it"
claim. All test-count targets met: app-main 349 (+8), shared 105
(unchanged), surrealdb-service 45 (unchanged). No regressions, no
scope creep.

## Acceptance criteria check (delta vs attempt 1)

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | `score_docling_extraction` returns `DoclingConfidenceScore`, weights sum to 1.0, overall in [0,1] | ✅ | Unchanged from attempt 1. |
| 2 | Perfect-doc fixture scores ≥ 0.95 | ✅ | Unchanged. |
| 3 | Scanned image-only fixture scores ≤ 0.7 | ✅ | Unchanged. |
| 4 | `parser_engine="auto"` + score 0.97 ⇒ MinerU not called | ✅ | Unchanged. |
| 5 | `parser_engine="auto"` + score 0.60 ⇒ MinerU called | ✅ | Unchanged. |
| 6 | Docling raising ⇒ auto-mode swallows + calls MinerU + WARNING log | ✅ | Unchanged. |
| 7 | `docling_min_confidence` override respected — including the 0.0 boundary | ✅ | Fixed. `test_threshold_zero_keeps_docling_no_fallback` (`test_auto_fallback.py:345`) at the orchestrator layer + `test_auto_respects_threshold_zero_never_falls_back` (`test_source_processing_service.py:598`) at the extractor layer assert the threshold reaches the orchestrator and MinerU is not called on a low-confidence fixture. |
| 8 | `Source.metadata` after auto-extraction contains 4 keys, persisted | ✅ at repo + migration layers; live SurrealDB round-trip still deferred to A.3 | Migration #43 declares the field; `test_metadata_persistence_integration.py` drives the real `SourceRepository.update()` body (patching only `execute_query`) and asserts the four provenance keys reach the SurrealQL UPDATE payload unchanged. Live `INSERT → SELECT` against a real SurrealDB is documented as owed to Phase A.3 Playwright (no testcontainer/in-memory DB infrastructure in the workspace; `surrealdb-python` doesn't accept `mem://`). Honest gap, not a silent skip. |
| 9 | 50-page fixture scored in < 50 ms | ✅ | Unchanged. |

## Test status

```
uv run --project apps/app-main pytest apps/app-main/tests/
  → 349 passed, 3 warnings in 81.37s    (was 341 in attempt 1; delta +8)

uv run --project packages/shared pytest packages/shared/tests/
  → 105 passed in 4.21s                  (unchanged)

uv run --project packages/surrealdb-service pytest packages/surrealdb-service/tests/
  → 45 passed in 5.06s                   (unchanged; migration 43 discovered correctly)

Targeted regression tests:
  test_threshold_zero_keeps_docling_no_fallback            → PASSED
  test_auto_respects_threshold_zero_never_falls_back       → PASSED
  test_update_passes_metadata_through_to_execute_query     → PASSED
  test_update_with_empty_metadata_still_round_trips        → PASSED
  test_construction_then_model_dump_preserves_signals      → PASSED
  test_round_trip_through_model_dump_and_back              → PASSED
  test_migration_43_exists_with_correct_declaration        → PASSED
  test_migration_43_down_removes_field                     → PASSED

Affected-files regression sweep:
  test_auto_fallback.py + test_source_processing_service.py + test_docling_confidence.py
  → 78 passed in 111.52s
```

## Verification of attempt-1 blockers + majors

### 🔴 Blocker #1 — SCHEMAFULL Source table missing metadata declaration

**Status**: Fixed.

- `migrations/43.surrealql:16` declares
  `DEFINE FIELD IF NOT EXISTS metadata ON TABLE source FLEXIBLE TYPE option<object>;` —
  exactly mirrors the established pattern in
  `migrations/26.surrealql:31` (`source_folder.metadata`) and
  `migrations/26.surrealql:67` (`pipeline_cache.metadata`).
- `IF NOT EXISTS` clause is present → idempotent re-runs are safe.
- `migrations/43_down.surrealql:7` is a genuine reverse:
  `REMOVE FIELD IF EXISTS metadata ON TABLE source;`. The down
  migration is also idempotent.
- Discovery: `packages/surrealdb-service` migration-discovery tests
  pick up #43 without modification (17/17 pass).
- The integration test (`test_migration_43_exists_with_correct_declaration`)
  parses the real SurrealQL file and asserts all of: `DEFINE FIELD`,
  `ON TABLE source`, `FLEXIBLE`, `option<object>`, `IF NOT EXISTS`.
  This is a real static trip-wire against accidental removal.

### 🔴 Blocker #2 — Falsy-zero threshold bug

**Status**: Fixed.

- `apps/app-main/src/app_main/services/source_extractor.py:359-366`
  now reads:
  ```python
  raw_threshold = getattr(content_settings, "docling_min_confidence", None)
  threshold = float(
      raw_threshold if raw_threshold is not None else DEFAULT_THRESHOLD
  )
  ```
  The fix is correct: `0.0 is not None` is `True`, so `raw_threshold=0.0`
  flows through as `threshold=0.0` instead of being silently demoted
  to `DEFAULT_THRESHOLD=0.95`.
- An explanatory comment above the change documents *why* the
  `... or DEFAULT_THRESHOLD` pattern was wrong — future-readers won't
  re-introduce the bug.
- Two regression tests added at different layers:
  - **Orchestrator layer** (`test_auto_fallback.py:345`):
    `test_threshold_zero_keeps_docling_no_fallback` calls
    `extract_with_auto_fallback` directly with `threshold=0.0` on a
    low-quality fixture; asserts the chosen engine is `"docling"`,
    `score.threshold == 0.0`, `score.decision == "accept"`, and that
    `mineru.process` was never called.
  - **Extractor layer** (`test_source_processing_service.py:598`):
    `test_auto_respects_threshold_zero_never_falls_back` exercises the
    full `_process_file` path with `docling_min_confidence=0.0` set
    via `ContentSettings`; asserts MinerU was not called and the
    metadata says `parser_engine_used="docling"`,
    `extraction_fallback_triggered=False`.

### 🟡 Major #1 — Integration test for the auto-fallback metadata round-trip

**Status**: Substantively addressed within infrastructure constraints.

The new `apps/app-main/tests/test_metadata_persistence_integration.py`
contains 6 tests structured into 3 classes that genuinely exercise the
boundary (not "elaborate mocks"):

1. `TestSourceRepositoryMetadataPersistence.test_update_passes_metadata_through_to_execute_query` —
   runs the real `SourceRepository.update()` method body (which is
   `BaseRepository.update` at
   `packages/surrealdb-service/src/surrealdb_service/repositories/base.py:167`).
   Only `execute_query` is patched, at the lowest possible boundary
   (the function actually called by `BaseRepository`). Assertions
   verify the SurrealQL query that went out (`"UPDATE source:test1
   MERGE"`), that the metadata key is in `params["data"]`, that all
   4 provenance keys survive (`parser_engine_used`,
   `extraction_confidence`, `extraction_confidence_signals`,
   `extraction_fallback_triggered`), and that the returned Source
   model has the metadata. This test would catch any future field
   filtering in the repository layer.

2. `test_update_with_empty_metadata_still_round_trips` — asserts the
   empty-dict case persists `{}`, not `null` or omitted.

3. `TestSourceMetadataModelRoundTrip` (2 tests) — Pydantic
   `model_dump()` and reconstruction preserve the nested signals
   dict losslessly. Catches accidental `extra='forbid'` or schema
   tightening.

4. `TestMigration43SourceMetadataField` (2 tests) — parses the real
   `migrations/43.surrealql` (and `43_down.surrealql`) text from disk
   and asserts all expected clauses are present. This is a real
   static-validation trip-wire, not a mock.

What it explicitly does *not* cover: a live SurrealDB round-trip
(`migration → INSERT → SELECT`). This gap is honestly documented in
the test module docstring and in `status.md` "Testing gap" section
(lines 939-952), with the rationale (`surrealdb-python` doesn't
support `mem://`; no testcontainer/docker-compose plumbing in the
workspace) and the deferral (Phase A.3 Playwright will drive a real
PDF through the API against a real SurrealDB). Given infrastructure
constraints, this is the strongest test feasible. The combination of
"static migration check + real repository-layer code path" closes
the regression-prevention gap that allowed attempt 1's blocker.

### 🟡 Major #2 — Status doc cleanup

**Status**: Fixed.

- `docs/tracks/A-mineru/status.md:684-688` corrects the original
  attempt-1 claim and links to migration #43.
- `docs/tracks/A-mineru/status.md:692-700` "Modified" line for
  `source.py` is corrected.
- `docs/tracks/A-mineru/status.md:848-860` (Caveat #4) replaces the
  "no migration needed" assertion with the explicit
  `SCHEMAFULL`/migration-43 explanation.
- `docs/tracks/A-mineru/status.md:882-976` is a new
  "Phase A.1c — REVISIONS (attempt 2)" section that itemises the
  four fixes + the testing gap.
- `packages/shared/src/shared/models/source.py:123-135` model
  docstring is corrected.
- `packages/shared/src/shared/models/source.py:154-161`
  `ensure_metadata_dict` docstring references migration #43.

No misleading "schemaless" wording remains in code or docs.

## Issues found

### 🔴 Blockers (must fix)

None.

### 🟡 Major (must fix)

None.

### 🔵 Minor (optional)

These were noted in the attempt-1 review and remain unaddressed —
they are explicitly minor and don't block approval. Listing for
the record / follow-up filing:

1. **`test_docling_exception_triggers_mineru_with_warning` warning
   assertion is still over-defensive** —
   `apps/app-main/tests/test_auto_fallback.py:200-204` retains the
   `or True` pattern. Not regressed in attempt 2 (predates this
   review cycle). Reasonable to file as a follow-up.

2. **`_run_auto_fallback` still builds `_DoclingAdapter` as an inner
   class on every call** —
   `apps/app-main/src/app_main/services/source_extractor.py:371-389`.
   The attempt-2 commit message and status doc explicitly note this
   was a deliberate choice (closure over `content_settings`,
   `emit_key`, `log_stream`). Functionally fine; refactor candidate
   if the pattern recurs.

3. **`auto_fallback.py:163` returns `signals={}` for the
   "docling failed" sentinel** — Phase A.2 UI badge code will need a
   defensive guard. Documented as a contract item for the badge
   component; not a A.1c bug.

4. **Live SurrealDB integration test infrastructure** — the
   pragmatic decision in attempt 2 (static migration validation +
   repository-layer round-trip) is reasonable for this PR, but
   landing `pytest-docker` or `testcontainers-python` in the
   workspace would make future migrations + persistence changes
   cheaper to verify. Worth tracking as a backlog item, not a
   blocker for A.1c.

## Decision rationale

Both attempt-1 blockers (SCHEMAFULL migration miss, falsy-zero
threshold) are fully fixed with the minimum-viable changes plus
defensive regressions at multiple layers. The major-items
(integration test, status doc cleanup) are addressed substantively:
the integration tests exercise the real repository code path at the
right boundary, the static-migration test is a real trip-wire, the
gap that remains (live SurrealDB round-trip) is honestly documented
and owed to A.3. Status doc and model docstring no longer carry the
"schemaless" misclaim.

The diff is tightly scoped to the fix areas — no scope creep, no
opportunistic refactors. Test counts hit the announced targets
exactly: app-main 349 (+8 = 1 orchestrator threshold-zero, 1
extractor threshold-zero, 6 integration), shared 105 (unchanged),
surrealdb-service 45 (unchanged with migration discovered).

No regressions in the affected-files sweep (78/78 pass). Code style
is consistent with the rest of the codebase. Migration syntax is
valid SurrealQL and idempotent.

Attempt 2 of 3 lands cleanly. Approve.

## Next steps

APPROVED — ready for human approval / merge.

Recommended follow-ups (track in backlog, not blocking):

1. File the four minors above as low-priority issues.
2. Phase A.3 Playwright must include the live SurrealDB round-trip
   (real PDF → API → DB → SELECT → assert metadata) that this PR
   defers. The plan already commits to this.
3. Consider adding `pytest-docker` or `testcontainers-python` to
   workspace dev deps so future schema changes can be regression-
   tested end-to-end without waiting for Playwright.
