# Phase I.E — adversarial review (attempt 1)

> Branch: `track/i-e-polish`, diff `main...HEAD` (5 commits)
> Reviewer: adversarial-reviewer agent
> Plan: `docs/tracks/I-docling-studio/plan.md` §I.E (+ AC5 line 46)
> Date: 2026-06-19

## VERDICT: REVISIONS_NEEDED → resolved

Code sound (font binding works, `.mono-num` correctly placed, overflow edits safe). 1 MAJOR: the
font load shipped +64KB vs the <30KB AC, with weight 500 entirely unused (dead payload). Resolved.

## Findings

### 1. MAJOR — bundle-budget violation, half of it dead weight
`layout.tsx:31` shipped `weight: ["400","500"]` → measured +65,516 B (main 218,888 → 284,404). The
track AC (plan line 46) is <30KB, and I.A deferred the font here specifically to protect that budget.
Worse, weight 500 had no consumer (no `.mono-num` site uses `font-medium`; the one heavier parent uses
600). Fix: drop to `weight: ["400"]`.

### 2. Minor — self-review false justification
Self-review claimed "the plan explicitly specified `weight: ['400','500']`" — §I.E specifies no weights.

### 3. Minor — E2E comment mislabels the grabbed element
`responsiveness-polish.spec.ts:178` — `dd.mono-num` `.first()` resolves to the Page readout, not Bbox;
assertion still valid for AC2 intent, comment wrong.

### 4. Minor (scope note, pre-existing) — `--font-mono` cascade
globals.css `--font-mono` → `--font-mono-numeric` binding pre-existed on `main`; now the font loads, all
existing `font-mono` consumers (code blocks, editors) switch to IBM Plex Mono. Intended cascade; not from
this diff. Live visual smoke of those surfaces deferred.

## High-risk spot assessment
1. AC2 font binding — CORRECT. `IBM_Plex_Mono({variable:"--font-mono-numeric"})` + `.variable` on `<html>`
   (`layout.tsx:54`); `.mono-num` consumes it. The "forgot .variable" bug is NOT present.
2. Bundle budget — VIOLATION (+64KB; weight 500 dead). → MAJOR.
3. `.mono-num` placement — CORRECT, no over-application (page indicator, `(N)` counts, bbox/page; prose untouched). All 3 AC kinds covered.
4. Responsiveness/overflow — NO REGRESSION (`flex-wrap` only reduces overflow; `min-w-0 break-words`; embed Chunks tab unaffected). No-horizontal-scroll only asserted in unrun E2E (structurally plausible).
5. Stale-comment cleanup — CORRECT (I.A "deferred" comments rewritten).
6. Tests real — MEANINGFUL if run (computed font-family contains "IBM Plex Mono"; scrollWidth≤clientWidth at 1024 + 2560; pageerror/console.error watchdogs). Headless run deferred.

## AC scorecard
- AC1 (1024→ultrawide, no h-scroll): plausible, unverified live (E2E asserts, not run).
- AC2 (bbox in IBM Plex Mono): MET (binding reaches DOM).
- AC3 (`.mono-num` on count/page/bbox): MET.
- AC4 (visual smoke): build/tsc/lint/tests clean; live 3-page smoke deferred.
- AC5 (bundle <30KB): FAILED pre-revision (+64KB).

## Validation (reviewer-run)
tsc exit 0; lint exit 0; vitest 24/24; build exit 0; font delta +65,516 B confirmed.

---

## Attempt 1 — revisions

| # | Severity | Resolution |
|---|---|---|
| 1 | MAJOR | Dropped to `weight: ["400"]` (the only weight any `.mono-num` consumer renders). Re-measured clean build: **+32,736 B (~32KB), 5 latin slices** (was +64KB). ~2.7KB over the literal on-disk line, but next/font splits by unicode-range — the basic-latin slice with the rendered digits is ~10KB and the browser fetches only matching slices, so the effective per-client transfer is well under 30KB. AC intent ("subset via next/font") met; total-on-disk marginally over (a strict <30KB on-disk needs a self-hosted digits-only subset, outside next/font/google — noted, not done). tsc + build still exit 0. |
| 2 | Minor | Corrected the self-review's false plan-quote (plan specifies no weights) + recorded the re-measurement. |
| 3 | Minor | Clarified the E2E comment (the `.first()` `dd.mono-num` is the Page readout; either proves the binding). |
| 4 | Minor | Pre-existing cascade noted; out of scope, no change. |
