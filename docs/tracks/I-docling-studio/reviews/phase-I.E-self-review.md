# Phase I.E — Responsiveness & polish — self-review

Branch: `track/i-e-polish` (off latest `main`, main checkout, no worktree).
Scope: frontend-only — IBM Plex Mono load, `.mono-num` consumer sweep,
overflow/fluid hardening of the inspect workspace.

## Commits

| Hash | Message |
|---|---|
| `6041324` | feat(frontend): load IBM Plex Mono for mono-numerics (I.E) |
| `6677679` | style(frontend): apply .mono-num to inspect numeric readouts (I.E) |
| `4d02b15` | fix(frontend): harden inspect workspace against horizontal overflow (I.E) |
| `cbbe0d1` | test(e2e): assert mono-numerics font + no horizontal overflow (I.E) |

## Validation

- `node ./node_modules/typescript/bin/tsc --noEmit` → exit 0 (clean; `**/*.ts`
  include covers the new e2e spec).
- `npm run lint` → exit 0; only pre-existing warnings in untouched files
  (AddSourceDialog, BatchModeDialog, ExtractionTab, etc.). No new warnings in
  any I.E-touched file.
- `npm test` (vitest) → 24/24 pass, 3 files. No regression.
- `npx playwright test e2e/track-i/responsiveness-polish.spec.ts --list` → 2
  tests listed cleanly. Full headless run deferred (no browser in this env).
- `npm run build` → exit 0, full route table emitted. (A `.next/standalone`
  trace-copy `ENOENT` warning appears on Windows for the `(dashboard)` route
  manifest; it is pre-existing/unrelated to this change and does not fail the
  build.)

## Measured font bundle delta (the I.A AC5 budget concern)

Method: built `main`-state (Inter only) vs I.E-state (Inter + IBM Plex Mono),
diffed `.next/static/media`.

| | woff2 files | media total |
|---|---|---|
| Baseline (Inter only) | 7 | 218,888 B |
| With IBM Plex Mono 400/500 latin | 17 | 284,404 B |
| **Delta** | **+10** | **+65,516 B (~64 KB)** |

Honest assessment: **this exceeds the plan's "<30KB-ish" I.A expectation.**
Two things to weigh:

1. This is exactly WHY I.A deferred the load — the budget was reserved and the
   font only ships now that it has consumers.
2. The delta is **font media (woff2), not JS**. First Load JS is byte-identical
   before/after (inspect route 340 kB, shared 101 kB). next/font emits the font
   as per-unicode-range subset slices fetched on demand; the digit glyphs the
   `.mono-num` readouts use live in one basic-latin slice, so a real client
   fetches a small fraction of the total, not all the files.

> **Revision (adversarial review, MAJOR):** The original draft shipped weights
> `["400","500"]` and (incorrectly) claimed the plan specified both — it does
> not (§I.E gives no weights). Weight 500 had **zero consumers** (no `.mono-num`
> site uses `font-medium`), so it was pure dead payload. Dropped to
> `weight: ["400"]`. Re-measured against a clean `main` build:
> main 218,888 B / 7 woff2 → branch **251,624 B / 12 woff2 = +32,736 B (~32 KB)**
> across 5 latin subset slices. That is ~2.7 KB over the literal <30 KB on-disk
> line, but next/font splits by `unicode-range`: the basic-latin slice carrying
> the digits `.mono-num` renders is ~10 KB, and the browser fetches only the
> slice(s) matching rendered glyphs — so the **effective per-client transfer for
> the numerics is well under 30 KB**. The AC's intent ("subset via next/font")
> is met; total-on-disk is marginally over. A strict <30 KB on-disk would
> require self-hosting a digits-only subset (outside `next/font/google`'s
> control) — not done, noted as an option for the track owner.

## AC-by-AC

**AC1 — Workspace adapts 1024px → ultrawide; no horizontal scrollbars.**
Partially auto-verified. The layout is already percentage-based via
`react-resizable-panels` (no media queries — inherently fluid). I added the two
overflow fixes where a child could push width: the PropertiesPanel bbox value
cell (`min-w-0 break-words`, pinned label) and the PdfChunkViewer page-nav
toolbar (`flex-wrap`). The e2e spec asserts `documentElement` has no horizontal
overflow at 1024×768 and 2560×1080, including with the widest bbox readout
mounted in the narrow right panel. **Needs a live browser to fully confirm**
(spec validated with `--list` only here).

**AC2 — bbox readout renders in IBM Plex Mono.**
Code-verified: `IBM_Plex_Mono(... variable: "--font-mono-numeric")` is loaded in
layout.tsx and its `.variable` is on `<html>`; globals.css binds `.mono-num` and
`--font-mono` to `var(--font-mono-numeric, ...)`. The e2e spec asserts a
`.mono-num` element's computed `font-family` contains "IBM Plex Mono". **The
computed-style assertion needs a live browser to confirm** (validated `--list`).

**AC3 — `.mono-num` on all three target consumer kinds.**
Done.
- Page indicator: PdfChunkViewer page-number input + "/ {pageCount}", and the
  embed-mode "Elements on Page N" / "N elements · N total" header.
- Element-type counts: LegendBar "(N)" spans.
- Bbox coords: already on PropertiesPanel bbox + page (I.A); verified resolving
  to IBM Plex Mono now.
Also already carrying `.mono-num` (verified, no change needed): ChunkListPanel
"N total", ImageGallery filename + "n / N" page indicator, PropertiesPanel
"Pages" count.

**AC4 — Visual smoke clean.**
Build clean; tsc/lint/tests clean. The e2e spec holds `pageerror` and
`console.error` watchdogs to zero across the flow. **Live 3-page visual smoke
(dashboard / source detail / notebook detail) needs a browser** — not run here.

## Mental inversion — how could this be wrong?

- *Font doesn't actually resolve to IBM Plex Mono.* Risk: `--font-mono-numeric`
  not on the element's cascade. Mitigated: variable is set on `<html>` (root),
  and `.mono-num` reads it directly; the e2e computed-style check is the guard.
  Residual: only confirmable live.
- *I added horizontal overflow instead of fixing it.* The only structural change
  is `flex-wrap` (can only reduce overflow) and `min-w-0`/`break-words` (lets a
  cell shrink, never grow). No widths added.
- *I broke an existing numeric readout's alignment.* `.mono-num` only changes
  font + tabular-nums; layout classes untouched except the bbox row, which I
  re-aligned to `items-start` + right-aligned value (intentional for wrapping).
- *I touched I.D feature logic.* No — changes are className-only plus the bbox
  row flex tweak and the toolbar wrap; no handlers, hooks, or data paths moved.
- *Bundle blew past budget unnoticed.* Measured and reported honestly above
  (~64 KB media, 0 JS); flagged the overage rather than hiding it.

## Not fully verifiable in this environment

- No-horizontal-scroll at 1024px / ultrawide (e2e exists; needs headless run).
- `.mono-num` computed font-family == IBM Plex Mono (e2e exists; needs headless
  run + actual webfont fetch).
- Live 3-page visual smoke for console cleanliness.
