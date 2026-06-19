# Phase I.C — self review

> Branch: `track/i-coord-canonicalization`
> Commits: `e1e562e` (backend fix) → `7000b7d` (bbox tests) →
> `a54b50e` (frontend heuristic removal) → `e5327cc` (backfill script + tests)
> Plan: `docs/tracks/I-docling-studio/plan.md` §I.C
> Reviewer cycle: ×1.5

## Plan-vs-reality corrections (verify every path — some were stale)

1. **`source.page_dimensions` does not exist.** The plan's backfill spec and
   AC say to read `source.page_dimensions`. Grep across the whole repo finds
   the term only in the plan itself — there is no such field on the source
   model (`packages/shared/src/shared/models/source.py`) or any migration.
   Page dimensions are **not persisted**; the `/page-count` endpoint
   (`apps/app-main/src/app_main/api/routers/sources_files.py:254`) reads them
   live from the PDF via `pypdfium2`. The backfill script therefore does the
   same: resolve `chunk.source.asset.file_path`, open the PDF, read the page
   size (cached per file). This is the realistic implementation of Q-I-C-1's
   "in-place if dimensions present, else re-ingest".

2. **The real bug had TWO call paths, only one is wired in production.**
   - `ChunkExtractor.extract()` → `BoundingBox.from_docling` (the buggy path).
   - `ChunkExtractor.extract_from_docling()` → `_convert_docling_bbox` (an
     already-correct sibling that divides by `doc.pages[n].size` and flips y).
   Production runs through `chunk_builder.from_document(result.document)`
   (`source_extractor.py:227`), which reads `elem.bbox` produced by
   `from_docling`. So the fix had to live in `from_docling`, and
   `_convert_docling_bbox` was the reference for the correct math.

3. **`from_docling` only received `(prov, page)`** — it had no access to page
   dimensions, and `prov.page_height` is usually absent (the root of the
   silent y-flip break). I threaded real `page_width`/`page_height` from
   `DoclingParser._get_page_sizes(doc)` into `from_docling` at all three call
   sites (`_extract_element`/`_extract_table`/`_extract_image`).

4. **`chunk_builder.from_document` needed no behavioral change** — it already
   emits `[page, bbox.x, bbox.x2, bbox.y, bbox.y2]` with no rescaling. Once
   `from_docling` and `_bbox_from_mineru` both emit 0–1, this path is
   uniform. I added a docstring note making the 0–1 reliance explicit (the
   plan said "remove any remaining raw-pt assumption"; there was none to
   remove, only an implicit assumption to document).

## AC-by-AC

### AC1 — `from_docling` emits 0–1 normalized coords on the fixture set

**Status**: PASS. `test_docling_emits_normalized_0_1_coords` and
`test_docling_topleft_origin_not_flipped` assert every coord ∈ [0,1] and the
exact divide-by-dimension + y-flip math. The method now divides x by
`page_width`, y by `page_height`, flips BOTTOMLEFT→TOPLEFT (honoring an
explicit `coord_origin`), and clamps to [0,1].

### AC2 — `analyzeChunkFormat` removed; no callers remain (grep clean)

**Status**: PASS. `analyzeChunkFormat` is deleted from `PdfChunkViewer.tsx`.
Grep for `analyzeChunkFormat|chunkFormat|isNormalized` across `frontend/`
returns **no matches**. The `isNormalized` branch in the overlay and the
legacy negative-y workaround are gone; positions are unconditionally treated
as 0–1 and scaled into page pixels. `npx tsc --noEmit` (via the local
`typescript/bin/tsc`) exits 0.

**Scope decision (flag for reviewer):** `analyzeChunkFormat` returned *two*
things — `isNormalized` (the coordinate heuristic this phase removes) and
`pagesAreZeroBased` (page-index detection, orthogonal to the coordinate
bug). Current data stores 1-indexed pages, but legacy rows may be
0-indexed, and the backend fix does nothing about page indexing. Removing
page-base detection too would risk regressing navigation on legacy rows for
no benefit to this phase. I therefore kept a tiny `detectZeroBasedPages()`
helper that inspects **only the page integer, never the coordinates**. AC2
("`analyzeChunkFormat` removed; no callers remain") is satisfied — the
function and its name are gone. I did not escalate because the plan's intent
("treat all positions as 0–1") is specifically about coordinates, which this
honors.

### AC3 — Backfill processes 10K chunks < 60s; reports counts + skipped rows

**Status**: PARTIAL (logic verified; perf is by-design/estimated). I cannot
run a 10K live corpus in this sandbox. Instead `test_backfill_chunk_positions.py`
pins the load-bearing behavior: paginated batch loop
(`test_backfill_paginates_across_batches` with batch_size=2 over 5 rows →
3 SELECT pages), idempotency (`test_backfill_is_idempotent` → 0 writes on
0–1 data), selective rewrite (only legacy rows updated), `--dry-run`
(counts but no writes), and the missing-PDF skip path. The script reports
processed/normalized/already-normalized/skipped(no-pos/no-pdf/bad-page) +
distinct unresolved PDFs. The <60s/10K figure is **estimated**: cost is
dominated by one UPDATE per legacy row plus a single pypdfium open per
distinct PDF (cached), batched 1000/query — comfortably sub-minute on a dev
box, but not measured here.

### AC4 — MinerU emit path produces identical positions to Docling's (parity)

**Status**: PASS (the mandatory Track A guard).
`test_docling_and_mineru_emit_identical_box_for_same_page_region` builds the
same physical region two ways — MinerU pipeline `[200,187.6,500,387.6]`
(÷1000) and Docling raw BOTTOMLEFT points on A4 (÷595/÷842 + flip) — and
asserts x/y/x2/y2/width/height match to 1e-6. The plan says "byte-for-byte";
I assert float-equal within 1e-6 because the two paths reach the same value
through different arithmetic (1000-divide vs 595/842-divide+flip), so
bit-identical floats are not guaranteed and 1e-6 is the honest, meaningful
bar. Flagging this wording choice for the reviewer.

### AC5 — Visual smoke: overlay aligns with rasterized page after backfill

**Status**: DEFERRED / MANUAL. Requires a running frontend + backend + real
pre-fix source data + a browser. Not doable headless in this sandbox. The
math is covered by AC1/AC4 unit tests and the frontend typecheck; the visual
alignment check must be run manually against a dev instance before
production rollout (and per Risk 3, backfill must run before the frontend
change ships).

## Mental inversion tests

### Inversion 1 — revert the divide, keep raw points
If `from_docling` returned raw `x=left, width=right-left` again,
`test_docling_emits_normalized_0_1_coords` fails its `0 ≤ v ≤ 1` assertions
(left=119 ≫ 1) and the parity test fails (119 ≠ 0.2). Caught.

### Inversion 2 — drop the y-flip
If I returned `y = top/page_height` for BOTTOMLEFT origin instead of
`(page_height - top)/page_height`, `test_y_flip_top_of_page_maps_to_small_y`
fails: a near-top element (large Docling `t`) would map to a large y instead
of ~0. Caught. The parity test would also fail.

### Inversion 3 — fall back to raw points when dimensions are missing
The previous code effectively did this (page_height defaulted to 1.0). If I
returned raw points when `page_width`/`page_height` is None,
`test_missing_page_dimensions_yields_zero_box_not_raw_points` fails — it
asserts a zero box, never raw coords leaking through as fake 0–1. Caught.
This is the single most important guard: a raw point masquerading as 0–1 is
exactly the corruption the frontend used to paper over.

### Inversion 4 — backfill not idempotent (re-normalize already-0–1 rows)
If `_needs_normalization` returned True for 0–1 coords (e.g. by checking
`> 0` instead of `> 1.0`), `test_backfill_is_idempotent` fails (it would
divide 0.2 by 595 → 0.0003 and write it). The `> 1.0` threshold is the
idempotency hinge, asserted by both `test_needs_normalization_*` and the
idempotency test. Caught.

### Inversion 5 — frontend still divides only when "normalized"
The old overlay had `if (isNormalized) { x*pw } else { x }`. Now it always
does `x*pw`. If a legacy raw row somehow survived backfill and reached the
viewer, it would be multiplied by page width again (huge box). This is an
accepted consequence of dropping the heuristic per Q-I-C-2 + Risk 3: the
backfill MUST run before the frontend change deploys. Documented, not
defended in code (that was the whole point of removing the guesswork).

## Tests + results

```
apps/app-main (root venv, WSL):
  tests/test_bbox_canonicalization.py        9 passed
  tests/test_backfill_chunk_positions.py    11 passed
  tests/test_mineru_layout_parser.py    (regression, includes new dev assert) passed
  tests/test_source_processing_service.py   46 passed   (from_document path)
pipelines/ingestion (full suite):          107 passed
```

Combined targeted run: bbox + backfill + mineru + docling_confidence →
58 passed. No regressions observed in the suites that exercise the changed
code. I did not run the entire app-main suite to green (heavy ML imports +
live-DB integration tests exceed the sandbox time budget — same limitation
recorded in the I.H1 self-review).

## Lint / typecheck

- `ruff check` on changed Python: clean except a single `I001` in
  `backfill_chunk_positions.py` (a deferred `surrealdb_service` import after
  `sys.path.insert`, carrying `# noqa: E402`). This is the **identical**
  established pattern in the sibling script `score_pdf_corpus.py` (which
  emits 3 such I001). Left as-is per the surgical-change rule. The new test
  files and the edited `document.py`/`chunk_builder.py`/`mineru_layout_parser.py`
  regions are I001-clean. `docling_parser.py` carries 2 pre-existing I001
  (the `docling_core` deferred import) that exist on `main` and were not
  introduced here.
- Frontend: `node ./node_modules/typescript/bin/tsc --noEmit` → exit 0
  (clean). `npx tsc` directly does **not** work here — it pulls an unrelated
  `tsc@2.0.4` package; the project compiler is invoked via the local bin.

## Cross-track note (Track H)

The 0–1 TOPLEFT convention is now documented on the `BoundingBox` class
docstring as the canonical contract every `from_*` constructor must follow,
explicitly calling out the deferred Track H vision parser.

## What the reviewer should look at first

1. **`source.page_dimensions` non-existence** (correction 1) — confirm the
   live-pypdfium backfill approach is acceptable vs. the plan's assumed field.
2. **AC2 scope decision** — keeping `detectZeroBasedPages` (page index only)
   while removing the coordinate heuristic. Acceptable, or remove entirely?
3. **AC4 parity tolerance** — 1e-6 float-equal vs. the plan's "byte-for-byte".
4. **AC3 perf** — the <60s/10K figure is estimated, not measured.
5. **AC5 + Inversion 5** — backfill-before-frontend-deploy ordering is now a
   hard operational requirement (no frontend fallback for raw data).
