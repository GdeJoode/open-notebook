# Phase I.B — adversarial review (attempt 1)

> Branch: `track/i-inspect-workspace`, diff `main...HEAD` (5 commits)
> Reviewer: adversarial-reviewer agent
> Plan: `docs/tracks/I-docling-studio/plan.md` §I.B (6 ACs)
> Date: 2026-06-19

## VERDICT: REVISIONS_NEEDED

The production code is sound — all six ACs are satisfied at the code level, verified
against the installed react-resizable-panels **v4** compiled source. The failure was in
the **E2E test and the self-review narrative**, which asserted the opposite of what the
library emits. tsc clean, 9/9 unit tests pass, lint clean.

## Findings

### 1. BLOCKER — E2E asserts the wrong `aria-orientation`; self-review documents the wrong value
`frontend/e2e/track-i/inspect-workspace.spec.ts:177` (+ AC comments :19, :173); self-review AC5 §.

react-resizable-panels v4 computes a separator's `aria-orientation` as the **inverse** of the
group orientation. The workspace uses `<Group orientation="horizontal">`, so each `Separator`
emits `aria-orientation="vertical"` — which is exactly what AC5 ("handles expose orientation")
requires. The spec asserted `"horizontal"` (would fail on a real run) and the self-review's AC5
section claimed the same and presented it as PASS. The implementation was always correct.

*Fix*: assert `"vertical"` in the spec, update the AC5 comment + self-review. **Resolved** —
see Revisions.

### 2. Major — Full E2E suite never executed; AC2/AC3/AC4 encoded but unobserved
`phase-I.B-self-review.md` admits no browser run (no headless dev-server + Chromium in sandbox).
The reviewer traced drag/persist/keyboard against the v4 compiled source and expects them to pass
once #1 is fixed, but this must be demonstrated on a local run before merge, not asserted.

### 3. Minor — `aria-activedescendant` can briefly reference a non-rendered virtualized row
`ChunkListPanel.tsx:144-146`. Self-heals via `scrollToIndex`; one-frame edge. Follow-up.

### 4. Minor — "Tab between panels" is satisfied via focusable contents, not focusable regions
The `role="region"` wrappers aren't themselves focusable; everything inside is keyboard-reachable
and focus stays in the workspace. Spirit of AC4 met. No fix required.

## High-risk spot assessment
1. **AC6 no Chunks-tab regression** — PASS. `PdfChunkViewer` new props all optional; `mode`
   defaults to `"embed"`; `controlledSelection = selectedChunkId !== undefined` is false for the
   existing caller, preserving the uncontrolled path. No required prop added.
2. **AC3 persist round-trip** — PASS (wiring verified). Store persists only `panelSizes`
   (`partialize`); v4 layout values are percentages so the [10,80] clamp is meaningful;
   `defaultLayout` fed back on mount and honored (key-count == panel-count); zustand persist
   rehydrates synchronously before mount (no race). Reload step still wants the live E2E run.
3. **Store unit tests** — PASS, real. Assert exact clamp boundaries (95→80, 2/8→10, NaN→10) and the
   partialize contract; a min/max swap would fail. 9/9 pass.
4. **AC5 ARIA** — PASS (implementation); test/self-review were wrong (BLOCKER #1).
5. **AC4 keyboard** — PASS at code level (Arrow/Home/End + scrollToIndex; roving via
   activedescendant; native separator keyboard resize). Live run pending.
6. **Virtualization >1K** — PASS. Canonical TanStack Virtual setup (ref + getScrollElement +
   estimateSize 76 + overscan 8 + translateY rows).
7. **v4 vs plan `^3`** — ACCEPTABLE. v4.11.2 current stable; Q-I-B-1 confirmed the library not a
   major; API used correctly; no double-persist (manual zustand, not autoSaveId);
   package.json/lock consistent.
8. **Overflow / min-h-0** — PASS. AppShell main `min-h-0 overflow-hidden`; panes `h-full
   overflow-hidden` with children scrolling internally. Flex min-height trap avoided.

## AC scorecard
- AC1 (route renders 3 panels + handles): MET (code; live render low-risk).
- AC2 (drag resizes, no overflow): MET (code) / live behavior deferred to E2E run.
- AC3 (persist across nav + reload): MET (round-trip wired + unit-tested; reload wants live run).
- AC4 (Tab between panels, Arrow moves): MET (code) / live deferred.
- AC5 (region+aria-label; handles expose orientation): MET — separators emit `aria-orientation="vertical"`.
- AC6 (no Chunks-tab regression): MET (tsc/lint clean; uncontrolled path preserved).

## Tooling observed by reviewer
- `tsc --noEmit` → exit 0; `npm test` → 9/9; `npm run lint` → 0 errors. E2E not executed (headless).

---

## Attempt 1 — revisions

| # | Severity | Resolution |
|---|---|---|
| 1 | BLOCKER | Spec now asserts `aria-orientation="vertical"` (`inspect-workspace.spec.ts:181`); AC5 comments (:19, :173) and the self-review AC5 section corrected with a note explaining v4's inversion. Implementation unchanged (was already correct). |
| 2 | Major | Full E2E execution remains deferred — no headless dev-server + browser in this sandbox (same constraint as the I.A spec). AC2/AC3/AC4 are code-verified (reviewer traced them against the v4 source) + unit-tested where applicable; a local `npm run e2e` run is required before relying on them as observed-green. Documented honestly, not claimed green. |
| 3 | Minor | Deferred (follow-up): `aria-activedescendant` one-frame edge under virtualization (self-heals via scrollToIndex). |
| 4 | Minor | No fix required — "Tab between panels" met via focusable contents; focus stays in workspace. |
