# Track K — Entity resolution & deduplication — status

Append-only ledger. One row per phase attempt.

| Phase | Title | PR | Branch | Date | Status |
|---|---|---|---|---|---|
| — | (planning) | — | — | 2026-06-22 | track-planner producing plan.md |
| K.1 | NL-aware normalizer + precision guard + measurement harness | (pending) | `track/k1-nl-normalizer` | 2026-06-22 | implemented — all 9 ACs green, ready for review |

## Phase K.1 — NL-aware normalizer + precision guard + measurement harness — 2026-06-22

**Branch**: `track/k1-nl-normalizer` (off main). Commits: NL rules (`nl_normalization.py` + `name_normalizer.py` compose) → harness + fixtures + tests.

**Delivered**
- `packages/shared/src/shared/utils/nl_normalization.py` — `_LEADING_ARTICLES`, `_ROLE_ORG_PREFIXES` (ordered longest-first), `strip_leading_noise` (tail-preserving guard, `_MIN_TAIL_LEN=2`), `canonicalize_spelling` (curated dict: `koninkrijkrelaties`→`koninkrijksrelaties`).
- `packages/shared/src/shared/utils/resolution_metrics.py` — `measure_fragmentation` → `FragmentationReport` (distinct-canonical count + size histogram + merged-cluster member lists), `count_false_merges`, `count_unmerged_must_merge`. DB-free; normalizer is injected so the same code measures baseline vs candidate; `entity_type` folded into the cluster key (homograph guard).
- `name_normalizer.py` — composes V1 → strip_leading_noise → canonicalize_spelling. Public signature unchanged; docstring rewritten (V1-stub framing dropped). `Apple Inc.` doctests/tests still green.
- Fixtures: `tests/fixtures/entity_resolution/{convenant_entities.jsonl (1402 entities, frozen live dump from the 4 measured Convenant sources), must_merge.jsonl (9 pairs), must_not_merge.jsonl (8 adversarial pairs)}`.
- Tests: `test_nl_normalization.py`, `test_resolution_metrics.py` (incl. per-pair parametrized over-merge canary), extended `test_name_normalizer.py` (NL ACs + B.8 hash-contract).

**Measured (AC8) over the frozen 1402-entity Convenant fixture**
- Distinct-canonical: **1360 (V1) → 1314 (K.1)** = **−46**.
- BZK full-form cluster (`binnenlandse zaken en koninkrijksrelaties`, type `other`): **6 surface forms → 1** canonical key (article/role-prefix/spelling variants collapsed). The `organization`-typed slice collapses a further 3→1. Across both type-keys the BZK/Binnenlandse-Zaken family drops from ~14 surface forms to 2 keys (split only by `entity_type`).
- False-merges over `must_not_merge.jsonl`: **0** (AC7). Unmerged over `must_merge.jsonl`: **0** (AC6).

**AC ledger**: AC1✓ AC2✓ AC3✓ AC4✓ AC5✓ AC6✓ (0 unmerged) AC7✓ (0 false-merge) AC8✓ (−46, BZK 6→1) AC9✓ (hash derive-rule unchanged).

**Regression**: `packages/shared/tests/` 264 passed. B.8 `test_entity_persistence_service.py` + `test_entity_repository_roundtrip.py` 15 passed. ruff clean on changed files.

**Notes / tricky pairs**
- The K.1 must_merge corpus deliberately does NOT include the `BZK ↔ Binnenlandse Zaken en Koninkrijksrelaties` (abbreviation↔full-form) pair — that crosses two distinct normalized keys (`bzk` vs `binnenlandse zaken...`) and is K.2's curated-abbreviation job, not K.1's prefix-strip job. Including it here would have falsely failed AC6.
- `Provincie Groningen` ↔ `Groningen` (the city): both normalize to the bare name `groningen`; distinctness is carried by `entity_type` (`organization` vs `location`), which the harness folds into the cluster key. Fixture pair encodes the differing types; documented in `test_nl_normalization.py::TestTailPreservationCanary`.
- `_MIN_TAIL_LEN=2` is the guard that keeps `Gemeente a`-style 1-char tails from being stripped to noise while letting `bzk` (3) through.

## Basis
- B.8c assessment: V1 normalizer resolves identical surface forms (107 cross-doc) but fragments variants (BZK 8-way, "minister" 23-way). See `../B-kg-quality/reviews/phase-B.8c-resolution-assessment.md`.
- Swap-point: `packages/shared/src/shared/utils/name_normalizer.py::normalize_entity_name` (persistence dedup key + filtering both call it).
- Two layers: K.1-K.2 cheap NL normalization (quick wins) → K.3+ full Q9/M4 vocabulary resolution.
