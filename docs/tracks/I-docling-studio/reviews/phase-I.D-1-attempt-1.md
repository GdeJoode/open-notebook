# Phase I.D-1 — adversarial review (attempt 1)

> Branch: `track/i-d1-layersbar`, diff `main...HEAD` (5 commits)
> Reviewer: adversarial-reviewer agent
> Plan: `docs/tracks/I-docling-studio/plan.md` §I.D-1
> Date: 2026-06-19

## VERDICT: APPROVED

0 blockers, 0 majors. The two likeliest real failures (key-normalization no-op; embed-mode
regression) were checked hardest and are both clean. All ACs met. tsc clean, lint clean on all
I.D-1 files, 14/14 unit tests pass. E2E deferred (no headless browser — repo-consistent, as I.A/I.B).

## High-risk spot assessment
1. **Embed-mode no-regression** — PASS. `controlledLayers = controlledHiddenTypes !== undefined`;
   `hiddenTypes = controlledHiddenTypes ?? localHiddenTypes`; internal LegendBar gated `!controlledLayers`.
   The two embed callers (`SourceDetailContent.tsx:816`, `PreprocessingTab.tsx:99`) pass no prop →
   unchanged; only `DocumentInspectWorkspace.tsx:183` passes it. Embed change is a local-var rename only.
2. **Toggle→overlay key round-trip** — PASS. LayersBar stores `Object.keys(ELEMENT_COLORS)` (already
   normalized); overlay skip uses `elementTypeKey()` = `toLowerCase().replace(/\s+/g,'_')`. Keys match.
3. **Keyboard a11y** — PASS. Chips are native `<button>` (Space/Enter built-in), `aria-pressed={!isHidden}`,
   individually focusable. E2E focuses + presses Space.
4. **I.B persist contract** — PASS. `partialize` still returns only `{ panelSizes }`; new state excluded
   (explicitly tested). New toggle tests are real (add/remove/idempotent/showAll/reset).
5. **ChunkListPanel scope** — PASS (legitimate). Old dot-color map hexes byte-identical to the shared
   module; the old comment already deferred consolidation here. No visual change.
6. **Color extraction fidelity** — PASS. All 15 keys/hexes/DEFAULT_COLOR/normalization identical to prior inline.
7. **Chip set completeness** — PASS (with Minor 1). 15 chips from ELEMENT_COLORS keys.
8. **Tests real** — PASS. Inverting toggle/aria-pressed/partialize each fails a test.

## AC scorecard
- Each element type has a toggle chip (≥10): MET (15 chips).
- Toggling hides matching overlays: MET (data path wired; pixel disappearance is manual-smoke — canvas).
- State in `document-workspace-store`: MET (unit-tested).
- Keyboard (Space toggles focused chip): MET.
- No embed Chunks-tab regression: MET.

## Minor findings (non-blocking — logged as follow-ups, not fixed in I.D-1)
1. **Chips derive from the color map, not the document's actual element types** (`LayersBar.tsx:29`). A data
   type absent from ELEMENT_COLORS gets DEFAULT_COLOR overlays but no chip (can't be hidden); and chips show
   for types absent in the current doc. AC met literally; a data-driven chip set (intersect with `typeCounts`)
   would be more honest. Candidate for I.D-4/I.E polish.
2. **Alias chips share colors and are independently hideable** (`text`/`paragraph`, `list`/`list_item`,
   `heading`/`section_header`). Documented intent; possible user confusion. Consider grouping aliases.
3. **`getElementColor` dropped the old optional-chaining null-safety** on `element_type` (`ChunkListPanel.tsx:160`).
   Type contract says non-optional `string`, so safe; a malformed null payload would throw instead of gray-fallback.

## Decision
Merged as-is. Minors are genuine but non-blocking; logged for a later polish pass (I.D-4 / I.E).
