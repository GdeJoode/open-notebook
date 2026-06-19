# Phase I.D-4 — Result tabs (Markdown / Images / Structure) — Self-review

Branch: `track/i-d4-result-tabs`

## Scope

Added a tab strip to the inspect-workspace right pane —
**Properties / Markdown / Images / Structure / Config** — and the three new
result viewers. Strictly scoped to I.D-4; I.D-1 (LayersBar), I.D-2
(PipelineConfigPanel fields), and I.D-3 (ChunkActionsToolbar) logic is
untouched.

## Files

Created:
- `frontend/src/components/source/inspect/MarkdownViewer.tsx`
- `frontend/src/components/source/inspect/ImageGallery.tsx`
- `frontend/src/components/source/inspect/StructureViewer.tsx` (placeholder)
- `frontend/src/components/source/inspect/ConfigTab.tsx` (reprocess panel split out of PropertiesPanel)
- `frontend/src/lib/utils/image-pagination.ts` + `.test.ts`
- `frontend/e2e/track-i/result-tabs.spec.ts`
- `apps/app-main/tests/test_source_images.py`

Modified:
- `frontend/src/components/source/inspect/DocumentInspectWorkspace.tsx` — tab strip in right pane
- `frontend/src/components/source/inspect/PropertiesPanel.tsx` — now the read-only Properties tab (header + Pipeline section removed; Pipeline moved to ConfigTab)
- `frontend/src/lib/api/sources.ts` — `getImages()`
- `frontend/package.json` / `package-lock.json` — `rehype-sanitize@^6.0.0`
- `apps/app-main/src/app_main/api/routers/sources_files.py` — `GET /{id}/images` + shared `_resolve_output_dir` helper

### Stale plan paths

The plan lists files-to-modify as only `DocumentInspectWorkspace.tsx`. In
practice I also touched `PropertiesPanel.tsx` (to extract the reprocess panel
into the new Config tab so Properties is purely metadata) and added
`ConfigTab.tsx`. The plan's tab list explicitly includes a separate **Config**
tab, so this split is implied by the plan even though those files were not
enumerated. The plan also assumes `react-markdown` / `remark-gfm` /
`rehype-sanitize` all needed adding; the first two were already installed, so
only `rehype-sanitize` was added (as the recon note anticipated).

## AC-by-AC

### AC1 — Each tab renders without console error

- Tab strip uses the existing shadcn `Tabs` primitive. Radix unmounts inactive
  content, so each tab's component mounts only when selected.
- `MarkdownViewer`, `ImageGallery`, `StructureViewer`, `ConfigTab`, and the
  Properties content all handle their empty/loading/error states, so none can
  throw on missing data (no markdown → empty state; no images → empty state;
  image-list error → error state with retry).
- E2E test `all five tabs render without console errors (AC1)` switches through
  every tab with a `console.error` + `pageerror` watchdog asserting `[]`.
- Verifiability here: spec validated by `--list` only (headless Playwright
  cannot run in this environment). The assertion logic is in place; an actual
  run is needed to fully confirm. `tsc --noEmit` is clean and `npm run lint`
  reports only warnings (see below).

### AC2 — Markdown XSS-safe (sanitized)

- `MarkdownViewer` uses `react-markdown` + `remark-gfm` + `rehype-sanitize`.
- Two layers: react-markdown v10 does **not** render raw embedded HTML unless
  `rehype-raw` is added — which I deliberately did **not** add — so raw
  `<script>`/`<img onerror>` never become live DOM. `rehype-sanitize` is layered
  on as explicit defense-in-depth per the AC, so even a future change in
  react-markdown's default would be caught.
- E2E test feeds a markdown fixture containing `<script>window.__xss=true` and
  an `onerror` `<img>`, then asserts `window.__xss !== true` and that no
  `<script>` exists under `.prose`.
- Verifiability: assertion logic in place; needs a real headless run to execute.

### AC3 — Images lazy-load (≤6 in flight)

Bounded by **two** mechanisms, documented in `ImageGallery` and
`image-pagination.ts`:
1. **Pagination** — only one page of `IMAGE_PAGE_SIZE` (=6) images is mounted at
   a time, so at most 6 `<img>` requests can ever start together. This is the
   hard upper bound and is unit-tested (`pageSlice(...).length <= 6` for every
   page of a 25-item list).
2. **Native lazy-loading** — each `<img>` carries `loading="lazy"`, deferring
   any that are off-screen within the page.
- E2E test returns 13 image filenames and asserts `figure img` count ≤ 6, every
  mounted img has `loading="lazy"`, and the Next-page control is present.
- Unit tests (`image-pagination.test.ts`) ran green: **6 passed**.

## Image-list source (the key recon question)

There is **no** per-chunk image filename in persisted data. Picture chunks store
only `metadata.classification` (`pipelines/ingestion/.../chunking/extractor.py`),
not a filename. Images are written to disk by the exporter as
`{output_dir}/output/extracted_info/images/image_NNN_pageN.png` plus
`*_description.txt` sidecars.

Decision: rather than render a permanently-empty Images tab, I added a
**minimal, clearly-correct listing endpoint** `GET /sources/{id}/images` that
enumerates that exact directory — mirroring the directory-resolution and
path-safety logic already in the sibling `GET /sources/{id}/images/{filename}`
serve endpoint (I factored the shared resolution into `_resolve_output_dir`). It
filters to image extensions and excludes the `.txt` sidecars, returns a sorted
list, and returns `{"images": []}` (not 404) when there is no output dir / no
images so the gallery shows a clean empty state. This is exactly the "trivial
addition mirroring the existing serve endpoint" the brief permits. Backend tests
(`test_source_images.py`, 4 cases) ran green.

If the reviewer prefers zero backend surface in this sub-feature, the frontend
already degrades to a clean empty state, so the endpoint could be dropped and
the Images tab would simply always show "No extracted images" — but that would
make the tab effectively dead, which seemed worse than a 30-line mirror endpoint.

## StructureViewer placeholder

Intentional placeholder per the plan — it renders a "Structure graph coming in a
later phase (I.F)" empty state with an icon. I.F replaces it with
`<StructureGraphView>` (Sigma.js). No tree/graph logic was built here.

## Mental inversion — how could this be wrong?

- **Tab content height collapse**: `Tabs` is `flex h-full flex-col`, `TabsList`
  is in a `flex-shrink-0` header, and each `TabsContent` is `flex-1 min-h-0`, so
  the active tab fills remaining height and its own `overflow-y-auto` scrolls.
  Verified the class chain by reading `tabs.tsx`; not visually confirmed (no
  headless render here) — a manual smoke is the residual risk.
- **Right pane too narrow for 5 tabs**: triggers use `px-2 text-xs` and the list
  is `overflow-x-auto`, so they scroll rather than overflow at the 10% min pane
  width. Residual: cosmetic only.
- **`source.full_text` null during async processing**: typed `string | null`;
  MarkdownViewer renders an empty state for null/empty. Safe.
- **Image `src` before API base resolves**: `apiUrl` is fetched once; until it
  resolves each figure shows a spinner instead of a broken `<img>`. No request
  fires early.
- **Regression to I.D-1/I.D-3**: PropertiesPanel still renders the chunk
  metadata + `ChunkActionsToolbar`; LayersBar and the middle-pane viewer are
  untouched. The only behavioral change is that the reprocess panel now lives in
  the Config tab instead of inline under Properties.

## Test / check results

- `node ./node_modules/typescript/bin/tsc --noEmit` — clean (no output).
- `npm run lint` — only pre-existing-style warnings; my files contribute one
  `@next/next/no-img-element` warning on `ImageGallery`'s `<img>`, consistent
  with the codebase's existing choice (PdfChunkViewer, ExtractionTab) to use
  raw `<img>` for backend-served files. No errors.
- `vitest run` — 24 passed (3 files), including the 6 new image-pagination tests.
- `python -m pytest apps/app-main/tests/test_source_images.py` — 4 passed.
- `playwright test result-tabs.spec.ts --list` — 3 tests listed (parses).

## Not fully verifiable in this environment

- Headless Playwright cannot run, so the three E2E assertions (AC1 no-console-
  error, AC2 sanitization at runtime, AC3 lazy/≤6 at runtime) are validated only
  by `--list` + the underlying logic + unit tests. A real E2E run + a manual
  visual smoke of the tab strip remain.
