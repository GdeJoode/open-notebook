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
| I.B | inspect workspace (3-pane resizable) | — | `track/i-inspect-workspace` | 2026-06-19 | merged to main (commit 63e745e, `--no-ff`) — ⚠ live E2E run still pending local; backfill-before-deploy still applies (I.C) |
| I.D-1 | LayersBar (element-type visibility toggles) | — | `track/i-d1-layersbar` | 2026-06-19 | implemented; 14/14 store unit tests pass; tsc clean; lint clean (new files); e2e collects (4 tests, full run deferred); ready for review. Extracted element-colors to shared module (single source of truth); hiddenTypes in store as non-persisted string[]; embed Chunks tab unchanged |
| I.D-1 | LayersBar | — | `track/i-d1-layersbar` | 2026-06-19 | adversarial review: APPROVED (0 blockers/majors, 3 non-blocking minors logged) → merged to main |
| I.D-2 | full Docling conversion config | — | `track/i-d2-docling-config` | 2026-06-19 | implemented (5 toggles end-to-end; found+fixed unforwarded code/formula enrichment) |
| I.D-2 | full Docling conversion config | — | `track/i-d2-docling-config` | 2026-06-19 | adversarial review attempt 1: REVISIONS_NEEDED (1 BLOCKER: code/formula enrichment flipped OFF→ON for existing users, AC3) → fixed (defaults off on all layers); 8 backend + 4 frontend tests pass |
| I.D-2 | full Docling conversion config | — | `track/i-d2-docling-config` | 2026-06-19 | merged to main (commit 4c4e5b0, `--no-ff`) |
| I.D-3 | chunk merge/split | — | `track/i-d3-chunk-mutate` | 2026-06-19 | implemented; ChunkMutator service + merge/split endpoints on sources_crud.py (plan's chunks.py path was stale); 12/12 mutator tests pass (merge concat/union/resequence, split offset/proportional/insert, atomicity rollback ×2); ruff clean; tsc clean; lint clean (new files). Real SurrealQL BEGIN/COMMIT transaction for atomicity. ChunkActionsToolbar wired into PropertiesPanel. chunk_edit table deferred to I.H2 (loguru audit only). Ready for review |
| I.D-3 | chunk merge/split | — | `track/i-d3-chunk-mutate` | 2026-06-19 | adversarial review attempt 1: REVISIONS_NEEDED (2 MAJOR: transaction error-propagation unproven + rollback test theater; multi-box split geometry wrong) → fixed (execute_transaction guard + box-unit split); 17 mutator tests pass |
| I.D-3 | chunk merge/split | — | `track/i-d3-chunk-mutate` | 2026-06-19 | merged to main (commit 3d5cf11, `--no-ff`) |
| I.D-4 | result tabs (Markdown/Images/Structure) | — |  | 2026-06-19 | adversarial review: APPROVED (0 blockers/majors, 3 minors). Markdown rehype-sanitize XSS-safe; ImageGallery 6/page DOM-gated lazy-load; new GET /sources/{id}/images listing; ConfigTab split out; StructureViewer placeholder for I.F. 24 frontend + 4 backend tests |
