# Track I — Docling Studio integration — status

Append-only ledger. One row per phase attempt.

| Phase | Title | PR | Branch | Date | Status |
|---|---|---|---|---|---|
| I.A | design tokens | [PR #34](https://github.com/GdeJoode/open-notebook/pull/34) | `track/i-design-tokens` | 2026-06-16 | in review |
| I.A | design tokens (attempt 2) | [PR #34](https://github.com/GdeJoode/open-notebook/pull/34) | `track/i-design-tokens` | 2026-06-17 | revisions pushed (BLOCKER 1 + Majors 2, 3 + Minors 4, 6) |
| I.H1 | upload guards + per-IP rate limiting | — | `track/i-upload-guards` | 2026-06-19 | adversarial review attempt 1: REVISIONS_NEEDED (2 BLOCKER: JSON-endpoint 500, page-count OOM) |
| I.H1 | upload guards (revisions) | — | `track/i-upload-guards` | 2026-06-19 | all findings resolved; 9/9 tests pass; ready for review (AC5 multi-worker shared-state deferred — in-memory backend) |
| I.A | design tokens | [PR #34](https://github.com/GdeJoode/open-notebook/pull/34) | `track/i-design-tokens` | 2026-06-18 | merged to main (commit 21e0108) |
| I.H1 | upload guards + per-IP rate limiting | — | `track/i-upload-guards` | 2026-06-19 | merged to main (commit c577847, `--no-ff`) |
| I.C | coordinate canonicalization | — | `track/i-coord-canonicalization` | 2026-06-19 | adversarial review attempt 1: REVISIONS_NEEDED (1 BLOCKER: backfill corrupted legacy negative-y rows) |
| I.C | coordinate canonicalization (revisions) | — | `track/i-coord-canonicalization` | 2026-06-19 | BLOCKER+Major resolved (broken-flip rows skip+re-ingest); 23 backfill tests pass; ready for review |
| I.C | coordinate canonicalization | — | `track/i-coord-canonicalization` | 2026-06-19 | merged to main (commit 55183ba, `--no-ff`) — ⚠ run backfill_chunk_positions.py BEFORE frontend ships |
| I.B | inspect workspace (3-pane resizable) | — | `track/i-inspect-workspace` | 2026-06-19 | implemented; 9/9 store unit tests pass; tsc clean; lint clean (new files); next build OK; e2e collects (full run deferred); ready for review |
| I.B | inspect workspace (revisions) | — | `track/i-inspect-workspace` | 2026-06-19 | adversarial review attempt 1: REVISIONS_NEEDED (1 BLOCKER: wrong separator aria-orientation in spec) → fixed; code was correct; ready for review |
