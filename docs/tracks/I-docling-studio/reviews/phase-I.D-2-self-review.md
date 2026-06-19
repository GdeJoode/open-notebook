# Phase I.D-2 — Full Docling conversion config — Self-review

Branch: `track/i-d2-docling-config`
Commits: `ac8239d` (ingestion forwarding) → `cdc8d88` (backend round-trip) → `3cbef2f` (frontend)

## Scope delivered

Five docling conversion fields, controllable per-run, end-to-end:

| Field (UI / override key)                | DoclingConfig field          | Default (AC3) |
|------------------------------------------|------------------------------|---------------|
| `docling_do_code_enrichment`             | `do_code_enrichment`         | `True`        |
| `docling_do_formula_enrichment`          | `do_formula_extraction`      | `True`        |
| `docling_do_picture_classification`      | `do_picture_classification`  | `None` → VLM toggle |
| `docling_generate_page_images`           | `generate_page_images`       | `False`       |
| `docling_image_scale`                    | `images_scale`               | `2.0`         |

Thread: PipelineConfigPanel → `DoclingPipelineConfig` → `POST /sources/{id}/reprocess`
(`ReprocessRequest`) → `processing_overrides` → `source_processor` merge into
`ContentSettings` → `build_ingestion_config` → `DoclingConfig` → `to_docling_options()`
→ `PdfPipelineOptions`.

## Stale plan path (flagged)

The plan (`I.D-2`) names `packages/shared/.../pipeline_config.py` and "shared model add fields".
That file does not exist. The real chain is `pipelines/ingestion/.../config.py` (`DoclingConfig`
already had all 5 fields) + `packages/shared/.../models/settings.py` (`ContentSettings`) +
`apps/app-main/.../services/ingestion/config_builder.py`. Wired accordingly.

## Acceptance criteria

**AC1 — all 5 fields round-trip via the API (request → DoclingConfig).** MET.
- `ReprocessRequest` accepts all 5 (`image_scale` pre-existed; 4 added).
- `source_processor.process_source` merges `processing_overrides` into `ContentSettings`
  via `model_dump()` + `update()` (existing mechanism); the 4 new fields now exist on
  `ContentSettings` so they survive the round-trip rather than being dropped on
  re-validation.
- `build_ingestion_config` maps each onto `DoclingConfig`.
- Tests: `test_id2_override_fields_map_through`, `test_id2_override_payload_round_trip`
  (simulates the exact merge the processor does).

**AC2 — backend passes them to the docling converter options.** MET, and required a fix.
- `to_docling_options()` previously forwarded `generate_page_images`, `images_scale`,
  `do_picture_classification`, `do_picture_description` — but **silently dropped**
  `do_code_enrichment` and `do_formula_extraction`. Wired both (commit `ac8239d`).
  Verified the real docling field names against the installed package: docling expects
  `do_formula_enrichment` (not `_extraction`) and `do_code_enrichment` — both present in
  `PdfPipelineOptions.model_fields`.
- Test: `test_id2_forwarded_to_docling_options` builds the real `PdfPipelineOptions` and
  asserts all 5 values land (passes; docling is installed in the venv).

**AC3 — UI defaults = today's effective values; no behaviour change.** MET.
- Enrichment on, page images off, scale 2.0: defaults unchanged from existing `DoclingConfig`.
- Picture classification: the subtle one. Today it is bound to `use_vlm`. The new field
  defaults to `None` on both `ContentSettings` and `ReprocessRequest`, and is left
  **undefined** in `DEFAULT_PIPELINE_CONFIG`. `config_builder` falls back to `use_vlm` when
  the value is `None`, so an unconfigured source behaves exactly as before. The
  `reprocess_source` override filter (`if v is not None`) drops a `None` classification, so
  it never overrides the coupling unless the user explicitly toggles it.
- Tests: `test_id2_defaults_preserve_current_behaviour`,
  `test_id2_classification_follows_vlm_when_unset` (both VLM directions),
  `test_id2_classification_decoupled_from_vlm`, and the frontend
  `sources.id2.test.ts` asserting `docling_do_picture_classification` stays `undefined`.

## Mental inversion — how could this be wrong?

- **"Classification override accidentally fires by default."** Guarded three ways: field
  defaults to `None`; `DEFAULT_PIPELINE_CONFIG` omits it; `reprocess_source` strips
  `None`-valued overrides before they reach the merge. Covered by tests on both the
  follows-VLM and decoupled paths.
- **"`do_formula_extraction` vs docling's `do_formula_enrichment` mismatch silently no-ops."**
  Checked the installed docling `PdfPipelineOptions.model_fields` directly; the forwarding
  uses `do_formula_enrichment=self.do_formula_extraction`. The forwarding test asserts the
  resulting option value, so a renamed/dropped kwarg would fail the test rather than pass
  silently.
- **"New ContentSettings fields break persistence / existing rows."** All `Optional` with
  defaults; `RecordModel` is `extra="allow"`; the surrealdb settings repo uses
  `ContentSettings` directly (no hardcoded field list) and `SettingsUpdate` is
  `extra="allow"`. Old rows missing the fields hydrate to defaults. `config_builder` reads
  via `getattr(..., None)` so even a `ContentSettings` constructed without them works.
- **"Frontend sends keys the backend rejects."** `ReprocessRequest` has no `extra` config →
  pydantic default `extra="ignore"`; the panel already sent `privacy` (ignored) before this
  phase, so no regression.
- **"Image-scale slider lets through out-of-range values."** Slider clamped 1–4 (step 0.5);
  `ReprocessRequest.docling_image_scale` now carries `ge=1.0, le=4.0`. (Note: the global
  `ContentSettings.docling_image_scale` keeps no range constraint — unchanged, out of scope.)

## Tests run (actual)

- Backend (WSL venv): `pytest tests/test_source_processing_service.py tests/test_config_router.py`
  → **53 passed**. The 9 `TestBuildIngestionConfig` cases include 6 new I.D-2 tests; the
  forwarding test exercises real docling.
- `ruff check` on the 3 source files I authored substantively
  (`settings.py`, `config_builder.py`, `schemas.py`) → clean. Pre-existing ruff
  import-sort/unused-import findings in `config.py`, `sources_processing.py`, and the test
  module are on lines I did not touch and were left as-is (surgical).
- Frontend: `tsc --noEmit` → exit 0; `npm run lint` → exit 0 (warnings only, all
  pre-existing incl. the unused `DEFAULT_PIPELINE_CONFIG` import in the panel which predates
  this phase); `npm test` → **18 passed** (4 new).

## Not fully verifiable here

- **No live ingestion run.** The forwarding test constructs `PdfPipelineOptions` but does not
  run docling against a real PDF — verifying that docling itself honours the options is
  outside a unit test's reach (and needs GPU/models). E2E ingest with page-images on / code-
  enrichment off would confirm the visible effect.
- **UI rendering** is asserted only via tsc + lint + a defaults unit test. The vitest harness
  is node-env / pure-logic by design (no jsdom/testing-library); component/DOM behaviour is
  the E2E suite's job. No Playwright spec was added for this panel change.
