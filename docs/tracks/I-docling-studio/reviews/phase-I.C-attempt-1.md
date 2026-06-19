# Phase I.C — adversarial review (attempt 1)

> Branch: `track/i-coord-canonicalization`, diff `main...HEAD` (5 commits)
> Reviewer: adversarial-reviewer agent
> Plan: `docs/tracks/I-docling-studio/plan.md` §I.C (5 ACs)
> Date: 2026-06-19

## VERDICT: REVISIONS_NEEDED

The forward-path fix (`from_docling` rewrite, dim threading via `docling_parser`, frontend
simplification, MinerU assert) is correct and well-tested. But the **backfill silently
corrupts the y-axis of every legacy Docling chunk that lacked `prov.page_height`** — the
usual case, and the root of the original bug. AC3/AC5 are not met by the script and the
backfill tests are y-axis theater.

## Findings

### 1. BLOCKER — Backfill destroys y-coordinates of legacy Docling rows (clamps to 0)
`apps/app-main/scripts/backfill_chunk_positions.py:102-123` (`_normalize_position`)

The old `from_docling` wrote `y = page_height - top` with `page_height` defaulting to **1.0**
when `prov.page_height` was missing. So legacy stored positions are
`[page, left, right, (1.0 - top), (1.0 - bottom)]` — y values are large **negatives**
(e.g. `1.0 - 684 = -683`). The backfill assumes stored y is "raw points in TOPLEFT space"
and merely divides by `page_height` + clamps. Demonstrated live:

```
legacy stored:  [1, 119.0, 297.5, -683.04, -514.60]
backfilled:     [1, 0.2,   0.5,   0.0,     0.0]      ← y collapsed
correct value:  [1, 0.2,   0.5,   0.1876,  0.3876]
```

x is recovered; **y and height are permanently lost** (clamped to 0) → every such chunk
becomes a zero-height strip pinned to the page top. The original `top` is unrecoverable from
stored data. Two legacy regimes exist and the script conflates them: `prov.page_height`
present (y stored as raw positive points → divide is correct) vs. absent (y stored as
`1.0 - top`, negative → unrecoverable). `_needs_normalization` (`> 1.0`) fires in both
because **x** is raw, masking that y is irreparable. The frontend negative-y fallback was
removed (Risk 3), so there is no downstream rescue → AC5 overlay misalignment, caused *by*
the backfill.

*Fix direction*: detect the broken-flip regime (negative y, or otherwise unrecoverable) and
**skip + flag for re-ingest** (mirror the missing-PDF skip path) rather than writing a
corrupt box. Do not pretend a recoverable transform exists.

### 2. Major — Backfill tests don't exercise the real legacy data shape (y-axis theater)
`apps/app-main/tests/test_backfill_chunk_positions.py:50-58, 115-133, 147-156`

Every test feeds `_normalize_position` a *positive* y (`168.0`, `336.0`) as if legacy y were
raw points. Actual legacy y is `1.0 - top` (negative); no test reproduces it, so Finding 1
passes green. `test_backfill_normalizes_only_legacy_rows` asserts only x outputs, never y.

*Fix direction*: add a test with `positions=[[1, 119.0, 297.5, -683.0, -514.6]]` asserting
the negative-y row is routed to skip+flag (and asserting y is never written as a clamped 0).

### 3. Minor — `__debug__`/`-O` comment oversells the MinerU assert
`mineru_layout_parser.py:472-477`. The dev assert is fine and matches the plan; the comment
overstates it. Non-blocking.

### 4. Minor — No per-batch transaction despite Risk 1 ("wrap in transaction per batch")
`backfill_chunk_positions.py:185-189, 254-255`. One autocommitted UPDATE per row; a mid-run
crash leaves a partially-converted table. Idempotency makes it recoverable, so not a blocker,
but the plan's transaction mitigation is unimplemented + undocumented as a deviation.

## High-risk spot assessment

1. **y-flip + normalization math (forward)** — PASS. Hand-checked (l=100,b=200,r=300,t=500 on
   600×800 → x=0.1667, w=0.3333, y=0.375, h=0.375), matches `document.py:163-186`; no double-flip.
2. **Clamp masking** — forward path OK (parity/AC1 assert exact values before clamp); backfill
   clamp actively converts unrecoverable negative y into a plausible 0.0 → contributes to BLOCKER.
3. **AC4 parity — real or fudged?** REAL. A width-only-divide bug yields docling.y=0.2655 ≠ 0.1876
   → fails at 1e-6. 1e-6 tolerance is honest (two different arithmetic paths; byte-for-byte never
   achievable). PASS.
4. **Missing dims → zero box (forward)** — reasonable failure mode for extraction; asserted. PASS.
5. **Backfill idempotency + detection** — idempotency on already-0–1 rows PASS; detection works,
   but the *conversion is wrong* for the dominant legacy regime (Finding 1). FAIL.
6. **Frontend scope + regression** — `analyzeChunkFormat` fully removed, grep clean;
   `detectZeroBasedPages` legitimately separable; dropping negative-y workaround correct *given* a
   correct backfill. Frontend code PASS; blocked transitively by the backfill.
7. **Tests real, not theater?** Forward-path tests REAL (assert exact values, fail under old code);
   backfill tests THEATER on y (Finding 2). MIXED.

## AC scorecard
- AC1 (from_docling emits 0–1): **MET** (exact-value asserts, fail on regression).
- AC2 (analyzeChunkFormat removed, grep clean): **MET**.
- AC3 (backfill processes/reports correctly): **NOT MET** — corrupts legacy y.
- AC4 (cross-parser parity): **MET** at 1e-6 (justified).
- AC5 (visual overlay aligns after backfill): **CANNOT BE MET** with this backfill.

---

## Attempt 1 — revisions

All BLOCKER + Major findings resolved. Backfill tests: **23 passed** (was 19).

| # | Severity | Resolution | Commit |
|---|---|---|---|
| 1 | BLOCKER | Backfill now detects the broken-flip regime (`_is_unrecoverable_legacy` — any negative coord, the `1.0 - top` signature) and skips + flags those rows for re-ingest (`skipped_broken_flip` tally) instead of rescaling them into a clamped zero-height box. Only genuinely recoverable positive-raw-point rows are normalized. | `2554e4e` |
| 2 | Major | Added unit tests for `_is_unrecoverable_legacy`, a batch test asserting a negative-y row is skipped and nothing is written, and y-output assertions (`168/842`, `336/842`) on the recoverable case so a future regression to clamped-0 fails. | `2554e4e` |
| 3 | Minor | MinerU dev-assert comment left as-is — it accurately describes `__debug__`/`-O` behavior; non-blocking nit, no churn. | — |
| 4 | Minor | Documented the no-batch-transaction deviation (idempotent re-run is the recovery path) in the script docstring. | `2554e4e` |

### Notes
- AC4 parity stands at 1e-6 float-equality (reviewer accepted: the two paths reach the value via different arithmetic, so byte-for-byte was never achievable; a width-only-divide bug still fails the assertion).
- AC5 (visual overlay) remains deferred/manual — needs a running app + real data; out of sandbox scope. With the backfill fix, un-backfillable legacy rows are now correctly excluded (re-ingest) rather than shown misaligned.
- Forward-path fix (`from_docling`, dim threading, frontend, MinerU assert) was rated correct by the reviewer and is unchanged.
