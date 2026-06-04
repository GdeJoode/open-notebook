# Review — Track A Phase A.1b attempt 1

**Branch**: `track/a-mineru-dispatcher`
**Decision**: REVISIONS_NEEDED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-03

## Summary

The rename + dispatcher + migration are implemented competently and tests pass
(299/299 app-main, 101/101 shared, TS clean). However the API boundary for
`PUT /api/settings` accepts arbitrary strings for `parser_engine` because
`SettingsResponse`/`SettingsUpdate` typed it as `Optional[str]` instead of the
literal set used in `ContentSettings` and `ReprocessRequest`. That lets invalid
values be persisted into SurrealDB before being rejected on response
construction — a real correctness bug, not a style nit. Two smaller issues
(stale `parser_engine_used` metadata claim for `simple`/audio inputs and a
brittle mock pattern) also need attention.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | `PUT /api/settings` round-trip with `parser_engine` | ⚠️ Partial | Round-trip works for valid values, but the schema does not enforce the literal at the boundary — see Blocker #1. |
| 2 | `parser_engine="docling"` keeps existing behaviour | ✅ | `test_records_parser_engine_used_in_metadata` confirms the docling code path still runs and `metadata["parser_engine_used"] == "docling"`. Bit-identical snapshot regression is deferred to integration time (acceptable for A.1b). |
| 3 | `parser_engine="mineru"` calls `MineruHttpClient.process` once + metadata persisted | ⚠️ Partial | `MineruHttpClient.process` is called exactly once (verified). `metadata["parser_engine_used"] == "mineru"` is recorded on `ExtractionResult.metadata`, but it is NOT persisted onto the `Source` record by `SourceProcessor._update_source` — `Source.metadata` field doesn't exist yet (deferred to A.1c per Q-A-3). The plan's wording "`source.metadata["parser_engine_used"] == "mineru"` persisted" is therefore over-claimed in the status; the runtime in-memory check is good enough for A.1b but the status.md should be honest about it. |
| 4 | `mineru` + unsupported ext → fall back to docling + INFO log | ✅ | `test_falls_back_to_docling_for_unsupported_mineru_ext` exercises this. INFO message text matches plan. |
| 5 | Per-source `ReprocessRequest.parser_engine` override | ✅ | Field is correctly typed `Optional[Literal["simple","docling","mineru","auto"]]`; flows through `processing_overrides` → `ContentSettings(...)` re-validation in `SourceProcessor`. |
| 6 | `select_parser_engine` ≥90% coverage | ✅ | 97% line coverage (32 cases). |
| — | Migration idempotent (Q-A-6) | ✅ shape-checked only | Each UPDATE is gated on the destination field being NONE, so re-running is a no-op. Not live-tested against a SurrealDB instance — acknowledged caveat. |
| — | Migration covers all 3 mappings (auto→docling, docling→docling, simple→simple) | ✅ | All three branches present in `migrations/42.surrealql`. NONE/missing case is also handled. |
| — | Field rename complete across py/ts code | ✅ | `rg "default_content_processing_engine_doc"` outside docs/migrations finds only doc comments referencing the rename for human readers. No live callers remain. |

## Test status

```
$ uv run --project apps/app-main pytest apps/app-main/tests/ -x --tb=line -q
299 passed, 3 warnings in 71.57s

$ uv run --project packages/shared pytest packages/shared/tests/
101 passed in 2.45s

$ cd frontend && npx tsc --noEmit
(clean)

$ pytest --cov=app_main.services.parsing.engine_dispatcher
97% (34 stmts, 1 missed) — 32 tests
```

The pre-existing `tests/test_domain.py` collection error
(`ModuleNotFoundError: No module named 'open_notebook'`) reproduces on `main`
unchanged. The implementer's caveat is accurate; not introduced by this branch.

## Issues found

### 🔴 Blockers (must fix)

1. **`SettingsResponse.parser_engine` and `SettingsUpdate.parser_engine` typed as
   `Optional[str]` instead of `Optional[Literal["simple","docling","mineru","auto"]]`** —
   `apps/app-main/src/app_main/api/schemas.py:300,329`
   - Issue: At the API boundary, the field accepts any string. Reproduction:
     ```
     SettingsUpdate(parser_engine='evil_value')  # accepted
     SettingsResponse(parser_engine='evil_value') # accepted
     ContentSettings(parser_engine='evil_value')  # raises ValidationError
     ```
     Because `ContentSettingsRepository.update` (in
     `packages/surrealdb-service/.../repositories/base.py:403-427`) writes the
     raw dict via `UPSERT ... MERGE $data` before re-hydrating via
     `self.model_class(**result[0])`, an invalid value will be **persisted to
     SurrealDB** before validation fires. The subsequent ValidationError on
     response construction then leaves the DB in a broken state (the singleton
     row holds a value that fails the domain model).
   - Impact: the plan explicitly required `parser_engine: 'docling' | 'mineru' |
     'auto'` typing (and the implemented domain enum extends to `'simple'`).
     The `ReprocessRequest.parser_engine` field uses `Literal[...]` correctly;
     `SettingsUpdate` and `SettingsResponse` were missed. This breaks
     acceptance criterion 1 ("round-trip") in the strict reading: invalid input
     can corrupt the singleton.
   - Reference: `mineru_supported_extensions` is typed as `Optional[List[str]]`
     which is acceptable (no enum), but consider whether the API should
     validate non-empty / extension format. (Treat as MINOR follow-up if you
     prefer.)

### 🟡 Major (must fix)

1. **`parser_engine_used` metadata mislabels the engine that actually ran for
   `simple` and for audio/video** —
   `apps/app-main/src/app_main/services/source_extractor.py:163-272`
   - Issue: When `parser_engine == "simple"`, `select_parser_engine` returns
     `"simple"`, but the code then runs the standard `IngestionWorkflow` (or
     `DoclingHttpClient` if `USE_DOCLING_SERVICE`) because there is no separate
     "simple" implementation. The metadata stored is
     `parser_engine_used = "simple"` even though Docling actually ran. The
     implementer documents this caveat in their status.md notes #1 ("simple
     engine is preserved as a value but currently routes through Docling
     because there is no separate simple-extraction implementation"), but the
     metadata field is intended to be consumed downstream (UI badge in A.2,
     auto-fallback in A.1c). Recording an engine name that did not actually
     run is misleading at best.
   - Audio/video files take the `else` branch at line 157-161 and get
     `resolved_engine = "docling"`, which IngestionWorkflow then routes through
     WhisperX. Recording `"docling"` for an audio file is similarly wrong.
   - Recommendation: distinguish "user setting" from "engine that ran". Either
     (a) record `parser_engine_used` only after the engine actually dispatches
     (i.e. set to `"mineru"` only after MinerU runs; set to `"docling"` only
     after IngestionWorkflow/DoclingHttpClient runs; otherwise omit/null), or
     (b) leave it as a record of the *setting* and add a separate
     `effective_pipeline` field for what really executed. The current shape
     will confuse A.1c's confidence-fallback logic when it inspects
     `parser_engine_used` to decide whether to retry with MinerU.

### 🔵 Minor (optional)

1. **Mock patching strategy in MinerU routing tests is brittle** —
   `apps/app-main/tests/test_source_processing_service.py:351-365`
   - The two new MinerU tests patch `sys.modules["app_main.services.mineru_http_client"]`
     with a `MagicMock`. This works because `_process_file` does a lazy
     `from app_main.services.mineru_http_client import MineruHttpClient`, but
     it has two failure modes: (a) if the real module was already imported in
     the same test session by another test, the lazy import hits the real
     class first; (b) the test must rely on the import statement being lazy
     (the test will silently break if `MineruHttpClient` is ever hoisted to
     module-level). Consider `patch("app_main.services.source_extractor.MineruHttpClient",
     create=True)` once you move the import (or accept the brittleness; it's
     working today).
2. **`SettingsForm` Zod schema accepts `mineru` / `auto` but the dropdown
   only renders `docling` / `simple`** —
   `frontend/src/app/(dashboard)/settings/components/SettingsForm.tsx:23,200-204`
   - If a user sets `parser_engine="mineru"` via API (the only way in A.1b),
     visits Settings, and saves the form without changing anything, the Radix
     `<Select>` rendering an unknown value silently submits whatever the form
     state holds. This is a real footgun until A.2 wires the 4-way dropdown.
     The implementer's caveat #3 acknowledges this scope choice, but the
     practical effect is that A.1b's only consumer of `parser_engine = mineru`
     is the per-source `ReprocessRequest` override — the global Settings field
     is essentially read-only until A.2. Worth a sentence in the status.md
     warning operators not to set MinerU globally via API yet.
3. **`mineru_supported_extensions` is absent from the `SettingsForm` Zod
   schema** — `frontend/src/app/(dashboard)/settings/components/SettingsForm.tsx:19-65`
   - Field exists in the type but not in the form. Acceptable since no UI
     for it lands until A.2; flagged so it is not forgotten.
4. **Down migration is lossy mineru → docling** — `migrations/42_down.surrealql:7-9`
   - Acknowledged in the file's leading comment. Users who downgrade after
     selecting MinerU silently revert to Docling. Acceptable for V1, but the
     status.md should call this out so operators don't downgrade casually.
5. **`tests/test_domain.py` change is inert** — `tests/test_domain.py:251`
   - The test file fails at import time on `main` (and HEAD) due to
     `ModuleNotFoundError: No module named 'open_notebook'`. The implementer
     dutifully renamed the assertion, but it will never execute. Not a
     regression; clean-up follow-up at most.
6. **Migration not live-tested** — `migrations/42.surrealql`
   - Status.md caveat #4 is honest about this. SurrealQL is shape-checked
     but never executed against a real instance. Recommend at minimum a
     local `docker compose up` + manual seed + `surreal migrate` run before
     merge, to confirm the syntax against the SurrealDB version pinned in
     the project.

## Decision rationale

One blocker (API boundary validation can persist invalid values to the
singleton row) and one major (misleading `parser_engine_used` metadata) push
this to REVISIONS_NEEDED. The blocker is a small fix (tighten two Literals)
but its impact is real: any client that posts `{"parser_engine": "anything"}`
writes garbage into the source-of-truth record before the response is
rejected. The major (`parser_engine_used` semantics) needs a quick decision
between recording-the-setting vs. recording-the-engine-that-ran, before A.1c
builds on top of this metadata.

Everything else — dispatcher purity, coverage, field-rename completeness,
migration shape, frontend types — is solid.

## Implementer's claim verification

| Implementer's claim | Verified? | Evidence |
|---|---|---|
| 299 tests pass (was 263, +36) | ✅ | `pytest -q` → `299 passed`. 32 dispatcher + 4 extractor = +36. |
| 101 tests in packages/shared pass | ✅ | `pytest -q` → `101 passed`. |
| TypeScript clean | ✅ | `npx tsc --noEmit` → no output. |
| No references to old field in code | ✅ | `rg "default_content_processing_engine_doc"` outside docs/migrations returns only doc-comments noting the rename. |
| Migration idempotent | ✅ on inspection | Each UPDATE WHERE-gates on the destination field being NONE. Re-running is a no-op. Not stack-tested. |
| Migration covers all 3 mappings | ✅ | `auto`/`docling`/NONE → `docling`; `simple` → `simple`. |
| `simple` engine routes through Docling, pre-existing behaviour | ✅ | Verified against `git show main:.../source_extractor.py` — main also routed `simple` through Docling unconditionally; no behavioural drift. |
| `auto` pinned to docling in A.1b | ✅ | `test_auto_setting_resolves_to_docling_in_a1b` exists and asserts `metadata["parser_engine_used"] == "docling"` when `parser_engine="auto"`. |
| SettingsForm UI options unchanged (Docling + Simple) | ✅ | Lines 200-204 still render only those two options; Zod schema accepts all four. |
| Migration shape-checked but not live-tested | ✅ | Plausible based on the SurrealQL itself; no fixture or integration test exists. |
| Pre-existing `tests/test_domain.py` collection failure | ✅ | Reproduces on `main` with the same `ModuleNotFoundError`. Not introduced by this branch. |

## Next steps

Implementer should address:

- **Blocker #1**: Change `SettingsResponse.parser_engine` and
  `SettingsUpdate.parser_engine` to
  `Optional[Literal["simple","docling","mineru","auto"]]` in
  `apps/app-main/src/app_main/api/schemas.py`. Add a test in
  `apps/app-main/tests/test_settings_router.py` that posts an invalid value
  and asserts a 422 response. Confirm no existing test relied on the loose
  typing.
- **Major #1**: Decide on `parser_engine_used` semantics. If it should reflect
  *what actually ran*, branch its assignment so it is only set when the
  dispatcher's resolved engine actually executed (e.g. set to "mineru" only
  inside the `if use_mineru:` branch; set to "docling" only in the
  IngestionWorkflow/DoclingHttpClient branches; never set "simple" until a
  real simple-extraction path exists). Add a test asserting that for
  `parser_engine="simple"` + a `.pdf` file the metadata says `"docling"` (or
  whatever the chosen convention is). If the field is intended to mirror the
  user setting, rename it to `parser_engine_setting` and add a separate
  `effective_engine` field that reflects what ran.
- Minors are optional but #2 (operator warning in status.md about setting
  MinerU globally via API in A.1b) is cheap and prevents a real footgun.

Re-submit for review after the blocker + major are addressed.
