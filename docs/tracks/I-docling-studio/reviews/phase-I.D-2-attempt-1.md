# Phase I.D-2 — adversarial review (attempt 1)

> Branch: `track/i-d2-docling-config`, diff `main...HEAD`
> Reviewer: adversarial-reviewer agent
> Plan: `docs/tracks/I-docling-studio/plan.md` §I.D-2 (3 ACs)
> Date: 2026-06-19

## VERDICT: REVISIONS_NEEDED → resolved

AC1 (round-trip) and AC2 (forwarding) were correct. AC3 (no behavior change) failed: code +
formula enrichment flipped OFF→ON for every existing user. Resolved in revisions.

## Findings

### 1. BLOCKER — code + formula enrichment silently flip OFF→ON for all existing users (AC3)
On `main`, `to_docling_options()` never forwarded `do_code_enrichment` / `do_formula_enrichment`,
so docling used its native defaults (both **False**) — effective behavior OFF. After the branch:
`DoclingConfig` defaulted both True AND now forwards them, so a default ingestion produced
`PdfPipelineOptions.do_code_enrichment=True`, `do_formula_enrichment=True`. Verified by direct exec.
Three default vectors all pointed ON: `DoclingConfig` dataclass defaults (config.py:116,119),
`ContentSettings` defaults (settings.py:108,111), `ReprocessRequest` `Field(True)`
(sources_processing.py:61-62), and the frontend `DEFAULT_PIPELINE_CONFIG` (sources.ts:56-57).

### 2. Major — AC3 test was theater
`test_id2_defaults_preserve_current_behaviour` asserted the DoclingConfig dataclass fields were
True — which was always true on main too, so it could never catch the flip (which lives at the
`to_docling_options()` forwarding boundary).

## High-risk spot assessment
1. AC3 defaults preserve behavior — FAIL (code/formula OFF→ON). Classification PASS (None→use_vlm,
   identical both VLM directions); generate_page_images PASS (False→False); images_scale PASS (2.0).
2. AC2 formula field-name — PASS. Real docling field is `do_formula_enrichment`; forwarding maps
   `do_formula_enrichment=self.do_formula_extraction` correctly; `do_code_enrichment` real + forwarded.
3. AC1 round-trip names — PASS. Names match at every hop, for both reprocess AND upload paths.
4. images_scale 1–4 — PASS (minor: old 0.5x option now unreachable; cosmetic).
5. settings mirror/persistence — PASS (ContentSettingsRepository is generic).
6. tests real — MIXED: AC1/AC2 tests genuine; AC3 test theater (Finding 2).

## AC scorecard (pre-revision)
- AC1 (round-trip): MET. AC2 (forwarding): MET. AC3 (no behavior change): FAILED.

---

## Attempt 1 — revisions

Effective default for code + formula enrichment forced back to OFF at every layer; AC3 test moved
to the real `to_docling_options()` boundary. Backend `-k id2`: **8 passed**; frontend tsc clean +
`sources.id2.test.ts` 4/4.

| # | Severity | Resolution |
|---|---|---|
| 1 | BLOCKER | Defaults set to OFF on all four vectors: `DoclingConfig.do_code_enrichment`/`do_formula_extraction` = False (config.py); `ContentSettings.docling_do_code_enrichment`/`_formula_enrichment` = False (settings.py); `ReprocessRequest` fields default `None` (sources_processing.py); `DEFAULT_PIPELINE_CONFIG` code/formula = false (sources.ts). Override paths (explicit true) still reach docling as ON. Preserves pre-I.D-2 behavior (docling-native OFF). |
| 2 | Major | Replaced the dataclass-field AC3 test with `test_id2_defaults_reach_docling_options_off` (default `to_docling_options()` → both False — fails on the flip) + `test_id2_overrides_reach_docling_options_on` (explicit opt-in → both True). Updated the frontend defaults test to assert false. |

### Note
This keeps code/formula enrichment OFF by default per AC3. If richer-by-default conversion is
desired, that is a deliberate, separately-approved behavior change — not made here.
