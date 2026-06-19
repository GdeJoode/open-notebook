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
