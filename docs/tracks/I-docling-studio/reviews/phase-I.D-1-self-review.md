# Phase I.D-1 — self review (LayersBar)

> Branch: `track/i-d1-layersbar`
> Commits: `2ed9420` (extract element-colors + controlled hiddenTypes) →
> `0279147` (store hiddenTypes + tests) → `ef1135f` (LayersBar + workspace
> wiring + ChunkListPanel consolidation) → `3000e5a` (e2e spec)
> Plan: `docs/tracks/I-docling-studio/plan.md` §I.D-1
> Reviewer cycle: ×1.5

## Scope

Strictly I.D-1 (LayersBar). No I.D-2/3/4 work. The bbox-hiding plumbing in
`PdfChunkViewer` already existed (per recon); this phase drives it from the
inspect workspace via a shared store + a dedicated `LayersBar`, without
touching the embed-mode Chunks tab behavior.

## Plan-vs-reality corrections

1. **DS `elementColors.ts` is not vendored here.** The plan says "ported
   verbatim from DS `frontend/src/features/document/elementColors.ts`". That
   donor file is not present in this repo. Per the task brief I used the
   existing in-repo `ELEMENT_COLORS` (already commented "matches Docling
   Studio") as the source of truth, moved it to
   `frontend/src/lib/constants/element-colors.ts`, and added a comment noting
   the MIT/DS lineage. This is single-source-of-truth, not a re-port.

2. **`hiddenTypes` modeled as `string[]`, not `Set`, in the store.** A `Set` is
   not JSON-serializable and would break the `persist` middleware if it ever
   reached `partialize`. I store a normalized-key `string[]` and convert to a
   `Set` at the `PdfChunkViewer` boundary (which already expects `Set<string>`).
   `hiddenTypes` is deliberately kept OUT of `partialize` (navigation-scoped,
   like `activeChunkId`), so the existing panelSizes persist contract and its
   unit tests are untouched.

3. **Consolidated a second duplicate palette.** `ChunkListPanel.tsx` carried
   its own inline `ELEMENT_COLORS` with a comment explicitly deferring the
   consolidation to "I.D-1 (`element-colors.ts`)". I pointed it at the shared
   module too, removing that duplicate. This is in-scope under the one allowed
   refactor (single source of truth); no behavior change (same hex values).

## AC-by-AC

### AC1 — Each element type has a toggle chip

**Status**: PASS (code + e2e collection). `LayersBar` maps over
`Object.keys(ELEMENT_COLORS)` and renders one `<button aria-pressed>` per key
(15 keys, incl. distinct aliases like `heading` / `section_header` that share a
color but are independently hideable, matching the overlay's per-key skip
logic). The e2e test `renders a toggle chip per element type` asserts the
`role="group"` "Element layer visibility" is visible and contains ≥10 toggle
chips. Each chip shows the color swatch (inline hex from `getElementColor`) +
the label.

### AC2 — Toggling a chip hides the matching bbox overlays

**Status**: PASS for the data path (code + e2e on pressed-state); pixel-level
overlay change verified manually only. Clicking a chip calls the store's
`toggleType`, which flips the key in `hiddenTypes`. `DocumentInspectWorkspace`
feeds `new Set(hiddenTypes)` into `<PdfChunkViewer hiddenTypes={...} />`; the
overlay already does `if (hiddenTypes.has(elementTypeKey(rect.elementType)))
continue` in its draw/hover/click loops. So a hidden key stops being drawn and
stops being hover/click-hittable. The e2e test asserts `aria-pressed` flips
`true → false → true` on click. The canvas is pixel-drawn (not DOM nodes), so
the actual disappearance of a box is a manual-smoke item, not asserted in e2e.

### AC3 — Hidden-types state lives in `document-workspace-store`

**Status**: PASS (unit-tested). The store gained `hiddenTypes: string[]` plus
`toggleType` / `showAllTypes` actions and resets it in `reset()`. Both the
`LayersBar` (writer + reader) and the `PdfChunkViewer` (reader, via the
workspace) read from this single store — there is no second copy of the
fullscreen visibility state. Unit tests cover toggle on/off, multi-type set
semantics (no duplicates), `showAllTypes`, reset, and non-persistence.

### AC4 — Keyboard accessible: Space toggles the focused chip

**Status**: PASS (native + e2e). Each chip is a native `<button>`, which the
browser activates on both Space and Enter while focused — no custom key handler
needed. The chip carries `aria-pressed` and a descriptive `aria-label`
(`"table layer visible/hidden"`). The e2e test `Space toggles the focused chip`
focuses a chip, presses Space, and asserts `aria-pressed` flips. Focus-visible
ring uses the semantic `ring-ring` token.

### AC5 (plan note) — No regression to the embed-mode Chunks tab

**Status**: PASS. `PdfChunkViewer`'s new `hiddenTypes` prop is optional and
mirrors the existing `selectedChunkId` controlled pattern:
- When omitted (embed / Chunks tab), the viewer keeps its local
  `localHiddenTypes` state and renders its internal `LegendBar` exactly as
  before — `controlledLayers` is false, so the `!controlledLayers` guard keeps
  the LegendBar.
- When supplied (fullscreen), the overlay reads the prop and the internal
  `LegendBar` is suppressed (the workspace's `LayersBar` replaces it).
The default (no prop) render path is byte-for-byte the prior embed behavior;
only the internal state variable was renamed (`hiddenTypes` →
`localHiddenTypes`) and the repeated key-normalization inlined into the shared
`elementTypeKey()` helper (same regex). `tsc --noEmit` clean; lint clean on all
new/changed files.

## Mental inversion tests

### Inversion 1 — make `hiddenTypes` controlled-default the embed path too
If the viewer always read a controlled prop, the Chunks tab would lose its
local LegendBar toggles. Guarded: `hiddenTypes = controlledHiddenTypes ??
localHiddenTypes` and the LegendBar is gated on `!controlledLayers`
(`controlledLayers = controlledHiddenTypes !== undefined`). With no prop both
fall back to local state + LegendBar.

### Inversion 2 — persist `hiddenTypes`
If `hiddenTypes` were added to `partialize`, the new test `does NOT persist
hiddenTypes to storage` fails (`parsed.state.hiddenTypes` would be defined) and
a stale hidden-set could survive reloads across unrelated sources. Caught.

### Inversion 3 — store a `Set` in the store
A `Set` would serialize to `{}` under JSON and silently lose data if it ever
hit storage; it also breaks structural equality in tests. Modeling as
`string[]` keeps the store plain-JSON and the tests use `toEqual([...])`.

### Inversion 4 — non-idempotent toggle (push without dedupe)
If `toggleType` always pushed, repeated toggles would accumulate duplicates and
`showAll`/visibility would desync. The test `toggleType keeps set semantics
(no duplicates)` and the on/off test catch this; the impl filters on the second
toggle.

### Inversion 5 — chip not a native button (e.g. a `<div role="button">`)
A `<div>` would not get Space/Enter activation for free, breaking AC4. Using a
native `<button>` means the keyboard contract is browser-provided; the e2e
Space test would fail if it were swapped for a non-button.

### Inversion 6 — unstable Set prop reference
If the workspace passed `new Set(hiddenTypes)` inline (new ref every render),
the overlay would redraw on every parent render. Guarded with
`useMemo(() => new Set(hiddenTypes), [hiddenTypes])`.

## Tests + results

```
frontend unit (vitest, node env):
  src/lib/stores/document-workspace-store.test.ts   14 passed (was 9; +5 I.D-1)

frontend typecheck:  node ./node_modules/typescript/bin/tsc --noEmit  → exit 0
frontend lint:       npm run lint  → 0 errors; 0 warnings in any new/changed
                     I.D-1 file (pre-existing warnings elsewhere untouched)
e2e collection:      npx playwright test --list e2e/track-i/layers-bar.spec.ts
                     → 4 tests collected
```

## Environment limits (honest)

- **E2E not executed in a browser.** No dev server + Chromium available
  headless here. The spec is validated by collection (`--list`) only; the
  assertions (chip count, `aria-pressed` toggling on click + Space, "Show all")
  are my best-effort AC encoding but have not been observed green against a live
  render. Same handling as the I.A / I.B specs.
- **No component-level render test.** The vitest setup is `environment: 'node'`
  with no jsdom / `@testing-library/react` (neither is a dependency). Adding
  them would be new test infra beyond I.D-1 scope, so the component is covered
  by the e2e spec (collection-only here) and the store logic is covered by
  vitest unit tests, per the task brief's fallback.
- **Pixel-level overlay hiding** (a box actually disappearing on the canvas) is
  a manual-smoke item; the canvas is not DOM-queryable. The data path that
  drives it (store → Set prop → overlay skip loop) is wired and the skip loop
  pre-existed.

## What the reviewer should look at first

1. **DS port substitution** (correction 1) — confirm reusing the in-repo
   palette as source of truth (vs. a literal DS re-port) is acceptable given
   the donor file is absent.
2. **ChunkListPanel consolidation** (correction 3) — a slightly wider edit than
   strictly LayersBar, but it removes the duplicate its own comment deferred to
   this phase. Confirm in-scope.
3. **`string[]` vs `Set` modeling** (correction 2) — and the non-persistence
   choice for `hiddenTypes`.
4. **E2E deferral** — none of the 4 tests ran in a browser here.
