# Track K — Entity resolution & deduplication — status

Append-only ledger. One row per phase attempt.

| Phase | Title | PR | Branch | Date | Status |
|---|---|---|---|---|---|
| — | (planning) | — | — | 2026-06-22 | track-planner producing plan.md |

## Basis
- B.8c assessment: V1 normalizer resolves identical surface forms (107 cross-doc) but fragments variants (BZK 8-way, "minister" 23-way). See `../B-kg-quality/reviews/phase-B.8c-resolution-assessment.md`.
- Swap-point: `packages/shared/src/shared/utils/name_normalizer.py::normalize_entity_name` (persistence dedup key + filtering both call it).
- Two layers: K.1-K.2 cheap NL normalization (quick wins) → K.3+ full Q9/M4 vocabulary resolution.
