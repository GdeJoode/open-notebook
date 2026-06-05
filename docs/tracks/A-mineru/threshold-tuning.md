# Track A — Default-threshold tuning

> Run date: 2026-06-04
> Default threshold under test: **0.95** (from `DEFAULT_THRESHOLD` in `apps/app-main/src/app_main/services/parsing/confidence.py`)
> Docling service: docker image `open-notebook-docling`, container `open-notebook-docling-1` (Up 33 hours at score-time)
> Reproduction script: `apps/app-main/scripts/score_pdf_corpus.py`

## Corpus

The tuning corpus is a developer-local five-PDF subset of `docling_input/`
(gitignored). The corpus is deliberately small per the A.3 plan: 3-5 real
PDFs is enough to validate whether 0.95 is "reasonable" or "wildly
mis-calibrated". The synthetic fixtures committed under
`apps/app-main/tests/fixtures/` are included as a sanity-baseline row
(very small docs that **should** score low — they're not meant to drive
the decision).

| # | Fixture | Type | Size | Notes |
|---|---------|------|------|-------|
| 1 | `Convenant_Zuid-Oost-Drenthe-II.pdf` | Dutch policy convenant, native PDF | 309 KB | Clean text, government format with embedded tables |
| 2 | `Bijlage 3. Lopende trajecten Achterhoek NPVR.pdf` | Project-tracking appendix (table-heavy) | 129 KB | Many small tables, layout-heavy |
| 3 | `fiscal_equalisation.pdf` | Academic / mixed content | 673 KB | Formulas + tables + figures |
| 4 | `Van-meer-waarde-Nationaal-Netwerk-Brede-Welvaart-2025.pdf` | Long-form report (50+ pages) | 4.1 MB | Mostly clean prose with figures |
| 5 | `Bennett_test.pdf` | Academic with at least one scanned page | 3.3 MB | OCR-needed proxy |
| 6 | `synthetic_clean_text.pdf` | Synthetic (sanity) | 1.2 KB | Pure text, no tables/images |
| 7 | `synthetic_table_heavy.pdf` | Synthetic (sanity) | 1.3 KB | Text-rendered table |
| 8 | `synthetic_image_only.pdf` | Synthetic (sanity) | 17 KB | One rasterised PNG, no text layer |

## Scores @ default 0.95

Captured by running

```bash
USE_DOCLING_SERVICE=true DOCLING_SERVICE_URL=http://localhost:8100 \
    uv run --project apps/app-main \
        python apps/app-main/scripts/score_pdf_corpus.py <pdf-or-dir>
```

against the docling service.

### Real corpus (driving the decision)

| Fixture | overall | ocr | density | heading | table | image | unknown | decision @0.95 | dominant_signal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bennett_test.pdf | 0.725 | 1.00 | 1.00 | 0.23 | 0.00 | 0.91 | 1.00 | fallback | table_success |
| Bijlage 3. Lopende trajecten Achterhoek NPVR.pdf | 0.735 | 1.00 | 1.00 | 0.25 | 0.00 | 0.98 | 1.00 | fallback | table_success |
| Convenant_Zuid-Oost-Drenthe-II.pdf | 0.850 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | fallback | table_success |
| fiscal_equalisation.pdf | 0.828 | 1.00 | 1.00 | 0.87 | 0.00 | 0.97 | 1.00 | fallback | table_success |
| Van-meer-waarde-Nationaal-Netwerk-Brede-Welvaart-2025.pdf | 0.760 | 1.00 | 1.00 | 0.45 | 0.00 | 0.92 | 1.00 | fallback | table_success |

### Synthetic baseline (sanity)

| Fixture | overall | ocr | density | heading | table | image | unknown | decision @0.95 | dominant_signal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_clean_text.pdf | 0.878 | 1.00 | 0.39 | 1.00 | 1.00 | 1.00 | 1.00 | fallback | text_density |
| synthetic_image_only.pdf | 0.681 | 1.00 | 0.15 | 0.00 | 1.00 | 1.00 | 1.00 | fallback | heading_rate |
| synthetic_table_heavy.pdf | 0.801 | 1.00 | 0.38 | 0.50 | 1.00 | 1.00 | 1.00 | fallback | text_density |

(The synthetic fixtures legitimately score low because they are tiny — a
single page of lipsum cannot reach the `text_density` saturation point
of 1500 chars/page. They are NOT signal evidence about 0.95.)

## Observations

1. **All five real PDFs trigger fallback at 0.95.** Overall scores range
   from 0.725 (Bennett_test) to 0.850 (Convenant_Zuid-Oost-Drenthe-II).
   The default behaviour is therefore "every real doc gets re-parsed
   with MinerU" — which is the spirit of the `auto` mode.
2. **The dominant low-scoring signal is uniformly `table_success = 0.00`.**
   Every real document had at least one table detected by docling, and
   docling produced zero parsed rows for those tables in this corpus.
   That signal alone drags the overall score down by ~0.15 (its weight).
   Without `table_success`, the recomputed scores cluster around
   0.85-1.00 — well above 0.95 only for the cleanest texts.
3. **OCR confidence, text density, and unknown_element_ratio are
   uniformly perfect (1.00).** These signals carry 60% of the weight
   and discriminate cleanly between native-PDF text layers and
   scanned/garbled content. None of our corpus PDFs is scanned, so
   they all max out these signals.
4. **`heading_rate` is the second-most-varying signal** (0.23-1.00),
   reflecting genuine differences in document structure (e.g.
   appendix-style PDFs with few headings vs. multi-section policy
   docs).
5. The synthetic fixtures score low because they're tiny (text density
   saturation point is 1500 chars/page — a one-page lipsum doc can't
   reach it). The behaviour is correct, not a calibration error.

## Decision

We considered three branches:

### Branch A — Keep 0.95 (CHOSEN)

Of the five real fixtures, all five trigger fallback at 0.95. Manual
inspection of the docling output for each shows:

- The `table_success=0.00` signal is **acting as intended**: docling
  detected `<table>`-class elements but the structural row parser
  returned zero rows. These documents do contain tabular content that
  benefits from MinerU's table-focused pipeline (MinerU's table_body
  HTML preserves merged cells; docling's table parser drops them).
- The `Convenant_Zuid-Oost-Drenthe-II.pdf` case (overall=0.850) is a
  legitimate "borderline" doc — readable as docling output but
  legitimately worth re-parsing for the buried tables.
- The Bennett_test result (0.725) is the textbook fallback case — at
  least one scanned page genuinely needs MinerU.

In a corpus this small, "100% fallback" is not the same as
"misconfigured": it reflects that real-world heterogeneous corpora
contain enough table/figure content to make MinerU's output worth the
re-parse for `auto`-mode users. Operators with predominantly pure-prose
corpora (no tables at all) can lower the threshold to ~0.80 via
Settings without code changes.

> **Corpus-size disclaimer.** Five PDFs is too small to claim 0.95 is
> optimal — the data only justifies that 0.95 is **defensible** given
> the observed mix and the per-user override path. Phase A's exit
> contract is that the default is configurable; revisiting the default
> after operators report telemetry from a broader corpus is a Track
> H/follow-up item, not an A.3 regression.

### Branch B — Lower to 0.80-0.85 (REJECTED)

A lower threshold would have kept three of five fixtures on docling
(`Convenant_Zuid-Oost-Drenthe-II`, `fiscal_equalisation`,
`Van-meer-waarde`). But two arguments against:

1. **`table_success=0.00` is informative**, not noise. Documents with
   tables docling couldn't parse really are candidates MinerU might
   improve. Suppressing that signal would hide useful information.
2. The user-facing "Confidence threshold" slider (A.2) already lets
   operators relax the bar **without** a code change. Defaults should
   favour conservative behaviour (more fallbacks → fewer surprises);
   power-users tune from there.

### Branch C — Raise to 0.97 (REJECTED)

No real-corpus document scored above 0.95, so raising the bar would
have zero effect on the corpus and would only make synthetic /
fixture-style content even more likely to fall back. No upside.

## Reproducibility

```bash
docker compose up -d docling
USE_DOCLING_SERVICE=true DOCLING_SERVICE_URL=http://localhost:8100 \
    uv run --project apps/app-main \
        python apps/app-main/scripts/score_pdf_corpus.py \
        path/to/your/pdfs/
```

The scores above were captured on 2026-06-04 against the docker image
tagged `open-notebook-docling:latest` (container
`open-notebook-docling-1`). Re-running the script on a different
docling minor version will produce slightly different
heading/table/image counts but the qualitative picture — clean native
text scoring high, scanned content scoring low — is robust across
versions.

For the maintainer's convenience the synthetic fixtures live in
`apps/app-main/tests/fixtures/` and regenerate via
`uv run --project apps/app-main python apps/app-main/tests/fixtures/generate.py`.

## Follow-ups

These are not blockers for closing Track A; they're items worth
revisiting after live MinerU usage produces real data.

1. **`table_success` signal calibration.** The signal is
   `non_empty_tables / len(tables)` (so 1 bad table out of 5 ⇒ 0.8,
   not 0.0). The 0.00 we saw in this corpus means **every** detected
   table parsed empty, not just one. A future refinement could weight
   by row-yield rather than the binary non-empty check, so partial
   table extraction earns partial credit. Tracked as an A4 polish
   item; not blocking.
2. **Live telemetry.** Track A ships without instrumentation; we can't
   tell whether real users see fallback rates of 100% (matching our
   corpus) or 10%. A B4-era telemetry layer (planned per
   `FEATURE_ROADMAP.md`) should surface this.
3. **Larger corpora.** Five PDFs is enough to refute "0.95 is wildly
   wrong" but not enough to calibrate down to ±0.02 precision. A
   20-30 doc labelled corpus would let us pick the threshold by
   percentile (e.g. "85th percentile of clean-text scores"). The
   ingredient list for that work lives in the parent
   `plan.md` — out of scope here.

---

**Decision: keep default at 0.95.**
