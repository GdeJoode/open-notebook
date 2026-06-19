# Phase I.B — self review

> Branch: `track/i-inspect-workspace`
> Commits: `5b859b1` (store + deps + vitest) → `3cc0ee2` (PdfChunkViewer mode) →
> `b11b292` (workspace + panels + route + button) → `2324f9d` (e2e spec)
> Plan: `docs/tracks/I-docling-studio/plan.md` §I.B
> Reviewer cycle: ×1.5

## Plan-vs-reality corrections (paths / deps verified)

1. **`react-resizable-panels` resolved to `^4.11.2`, not the plan's `^3`.** The
   plan (and the dep mitigation in §6 Risk 6) assumed v3 with the classic
   `PanelGroup` / `PanelResizeHandle` API. The current latest stable is **v4**,
   whose public API is `Group` / `Panel` / `Separator` and whose layout type is
   a **map keyed by panel id** (`Layout = { [id: string]: number }`), not the v3
   size **array**. I used v4 and adapted accordingly: `defaultLayout` /
   `onLayoutChanged` are keyed by `inspect-left` / `inspect-middle` /
   `inspect-right`. Q-I-B-1 confirmed the *library*, not a major version; I read
   this as "use the current stable" rather than pin a year-old major. Flagging
   for the reviewer in case a v3 pin was intended.

2. **The dashboard route group does NOT wrap pages in `AppShell`.**
   `(dashboard)/layout.tsx` only does auth + model-status gating; each page
   renders its own shell (e.g. `notebooks/[id]/page.tsx` wraps `<AppShell>`;
   the existing `sources/[id]/page.tsx` uses a bare `h-screen` flex). The plan
   says "new route wrapping `<AppShell>`" — I did exactly that in
   `inspect/page.tsx`, matching the notebook-detail precedent.

3. **No JS unit-test runner existed.** The frontend had only Playwright. The
   plan requires a store reducer unit test, so I added `vitest` (devDep) with a
   `node` environment + a tiny in-memory `localStorage` shim
   (`vitest.setup.ts`) so the Zustand `persist` middleware works without pulling
   in jsdom. Added `test` / `test:watch` scripts. This is new test
   infrastructure; flagging it explicitly.

4. **Virtualization dep choice.** Per the recon note I added
   `@tanstack/react-virtual@^3.14.3` (lightweight, ~no transitive weight) for
   the >1K-chunk left pane rather than hand-rolling windowing.

## AC-by-AC

### AC1 — `/sources/{id}/inspect` renders 3 panels separated by resize handles

**Status**: PASS (verified by build + e2e collection + code). The route exists
at `frontend/src/app/(dashboard)/sources/[id]/inspect/page.tsx` and `next build`
emits it (`/sources/[id]/inspect`, 22.8 kB). `DocumentInspectWorkspace` renders
a `Group` with three `Panel`s (chunk list / PDF / properties) and two
`Separator`s between them. The e2e test `renders 3 panes + handles with correct
ARIA` asserts three `role="region"` panes and exactly two `role="separator"`
handles. Full browser execution is deferred (see "Environment limits").

### AC2 — Drag handle resizes panels; layout reflows without overflow

**Status**: PASS (logic + e2e authored; browser run deferred). The `Separator`
drives live resize natively; `onLayoutChanged` writes the new sizes to the
store. The e2e test `drag resizes + persists across reload` drags the first
separator ~20% of the group width to the left, polls that the left region's
measured width dropped by >10px, and asserts `document.body.scrollWidth -
clientWidth <= 1` (no horizontal overflow). Panels use `min-h-0` /
`overflow-hidden` wrappers so content scrolls inside its pane rather than
pushing the group wider.

### AC3 — Resized sizes persist across navigation and reload (Zustand persist)

**Status**: PASS for the persist contract (unit-tested); reload restore
authored in e2e. The store persists **only** `panelSizes` (via `partialize`)
under `document-workspace-storage`. The unit test
`persists only panelSizes to storage` asserts `panelSizes` is serialized and
`activePage`/`activeChunkId` are NOT (those are navigation-scoped by design).
The e2e test reads the persisted `left` value after the drag, reloads, and
asserts it restores (`toBeCloseTo`). Note: across **navigation** the store is a
module singleton so the layout is already in memory; across **reload** it
rehydrates from localStorage — both paths covered.

### AC4 — Keyboard nav: Tab between panels; Arrow keys move

**Status**: PASS (two mechanisms). (a) The `Separator` is natively focusable
(`tabIndex=0`) and v4 handles Arrow-key resize on it; the e2e test
`keyboard resize keeps focus inside the workspace` focuses a separator, presses
ArrowLeft ×8, polls that the persisted left size dropped, and asserts
`document.activeElement` is still a `role="separator"` (focus did not escape the
workspace). (b) `ChunkListPanel` is a single-tab-stop `role="listbox"` with
ArrowUp/Down/Home/End moving selection and scrolling the active row into view —
this is the "Arrow keys reorder/move selection" half of the AC. Tab order is:
chunk-list listbox → separator → (PDF pane controls) → separator → properties.

### AC5 — ARIA: each panel `role="region"` + `aria-label`; handles expose orientation

**Status**: PASS. Each of the three pane wrappers is a `role="region"` with an
`aria-label` ("Chunk list" / "PDF preview" / "Properties"). v4's `Separator`
auto-emits the full WAI-ARIA separator contract: `role="separator"`,
`aria-orientation`, `aria-valuemin/max/now`, `aria-controls`. The e2e test
asserts `aria-orientation="vertical"` on both separators — v4 inverts the
group orientation for the divider, so a horizontal `Group` yields vertically-
oriented separators (the divider line is vertical). I also added a
human-readable `aria-label` to each separator ("Resize chunk list" / "Resize
properties panel").

> Correction (adversarial review attempt 1, BLOCKER #1): an earlier draft of
> this section and the e2e spec asserted `aria-orientation="horizontal"`, which
> is wrong for v4 — the library emits `"vertical"` here, which is what AC5
> ("handles expose orientation") actually requires. The implementation was
> always correct; the test assertion and this note were fixed to `"vertical"`.

### AC6 — No regression in the existing Chunks tab

**Status**: PASS. `PdfChunkViewer`'s new props are all optional with `mode`
defaulting to `"embed"`; the Chunks-tab call site
(`SourceDetailContent.tsx`) was not changed in a behavior-affecting way — it
still renders `<PdfChunkViewer sourceId chunks />` with no mode, so the
two-pane embed layout is byte-identical. The only Chunks-tab edit is an added
"Open Inspect" `<Button>` above the viewer (entry point to the new route) and a
flex wrapper to seat it; the viewer container keeps `min-h-0` full height.
`tsc --noEmit` is clean and `next build` succeeds, so no type/build regression.

## Mental inversion tests

### Inversion 1 — break the persist partialize
If `partialize` also serialized `activePage`/`activeChunkId`, the unit test
`persists only panelSizes to storage` fails (`parsed.state.activePage` would be
defined). Caught. This guards against navigation state leaking across reloads.

### Inversion 2 — drop the clamp
If `setPanelSizes` stored raw values, `clamps panel sizes below MIN_PANEL_PCT`
and `…above MAX_PANEL_PCT` fail (a 2% or 95% pane would persist and could render
a degenerate, unrecoverable layout on reload). Caught by 3 clamp tests + NaN
test.

### Inversion 3 — make PdfChunkViewer fullscreen the default
If `mode` defaulted to `"fullscreen"`, the Chunks tab would lose its internal
chunk-list pane (AC6 regression). The default is `"embed"` and the embed render
path is untouched; the fullscreen branch is gated behind `!fullscreen`. The
existing I.A design-tokens spec still renders source detail clean.

### Inversion 4 — wrong layout key shape (v3 array vs v4 map)
If I'd passed `defaultLayout` as an array (v3 habit), v4 would ignore it and the
panes would fall back to `defaultSize`; `onLayoutChanged` would also yield
`undefined` for the id lookups and the store would never update — the e2e
persist assertion (`persistedLeft < 22`) would fail. The id-keyed map is the
load-bearing detail; covered by the drag/persist e2e.

### Inversion 5 — separator not keyboard-focusable
If the separator lost `tabIndex` (e.g. via a `disabled` Group), the AC4 test's
`expect(sep).toBeFocused()` fails and the focus-stays-inside assertion fails.
Native v4 focusability is relied upon, not re-implemented.

## Tests + results

```
frontend (vitest, node env + localStorage shim):
  src/lib/stores/document-workspace-store.test.ts   9 passed

frontend typecheck:  node ./node_modules/typescript/bin/tsc --noEmit  → exit 0
frontend lint:       npm run lint  → 0 errors (warnings only, all pre-existing;
                     0 warnings in any new I.B file)
frontend build:      npm run build → ✓ 18/18 pages; /sources/[id]/inspect emitted
e2e collection:      npx playwright test --list e2e/track-i/inspect-workspace.spec.ts
                     → 3 tests collected
```

## Environment limits (honest)

- **E2E not executed in browser.** No dev server + Chromium available headless
  here. The spec is authored and validated by collection only; full run is
  deferred to a local `npm run e2e` (same handling as the I.A design-tokens
  spec). The drag/keyboard/persist assertions are my best-effort encoding of
  the ACs but have not been observed green against a live render.
- **`next build` standalone-trace warning.** The build prints
  `⚠ Failed to copy traced files for …(dashboard)\page.js [ENOENT …
  page_client-reference-manifest.js]`. This concerns the dashboard **root**
  page manifest during standalone trace copying on Windows — a pre-existing
  Next 15 + Windows quirk, unrelated to the inspect route (which compiled and
  emitted cleanly). Page generation itself succeeded (18/18). I did not
  "fix" it (out of scope, not introduced by this phase).

## New dependencies (with sizes)

| Dep | Version | Why | Size note |
|---|---|---|---|
| `react-resizable-panels` | `^4.11.2` | ARIA + keyboard resize handles (Q-I-B-1) | ~14KB gzipped per plan §6 (v4; not independently re-measured) |
| `@tanstack/react-virtual` | `^3.14.3` | virtualized >1K-chunk left pane | lightweight, headless (no styles); ~3–4KB gzipped |
| `vitest` (devDep) | `^4.1.9` | required store unit test; no JS runner existed | dev-only, not in client bundle |

## What the reviewer should look at first

1. **v4 vs v3** (correction 1) — confirm using the current stable major
   (id-keyed `Layout`) is acceptable vs. a literal `^3` pin.
2. **New vitest infra** (correction 3) — node env + in-memory localStorage
   shim instead of jsdom. Acceptable, or prefer jsdom?
3. **AC4 split mechanism** — separator-keyboard-resize (the "move" half) +
   listbox-arrow-selection (the "move selection" half). Both present; confirm
   that satisfies "Arrow keys to reorder selection".
4. **E2E deferral** — none of the three tests have been run in a browser here;
   they need a local dev-server run before merge.
5. **`PropertiesPanel` reprocess** reuses `sourcesApi.reprocess` with default
   pipeline config — confirm that's the intended right-pane "Pipeline config"
   surface (plan says "active chunk metadata + Pipeline config").
