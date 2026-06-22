# Track K — Entity resolution & deduplication — status

Append-only ledger. One row per phase attempt.

| Phase | Title | PR | Branch | Date | Status |
|---|---|---|---|---|---|
| — | (planning) | — | — | 2026-06-22 | track-planner producing plan.md |
| K.1 | NL-aware normalizer + precision guard + measurement harness | (pending) | `track/k1-nl-normalizer` | 2026-06-22 | implemented — all 9 ACs green, ready for review |
| K.1 rev3 | Option A: no cross-type NAME collisions; name-only false-merge gate | (pending) | `track/k1-nl-normalizer` | 2026-06-22 | implemented — 0 name-only false-merges, ready for review |

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

## Phase K.1 rev3 — Option A: no cross-type NAME collisions — 2026-06-22

**Decision** (after 3 review cycles): K.1 normalization must NOT create cross-type
name collisions. Relations resolve endpoints by name ALONE
(`WHERE canonical_name = $name`, no type filter), so a name shared by two
different-typed real entities corrupts the graph regardless of the
`(canonical_name, entity_type)` dedup key. The rev2 over-merge canary checked at
`(name, type)` and so missed name-only collisions; rev3 fixes the criterion.

**The fix**
- `nl_normalization.py::_ROLE_ORG_PREFIXES` reduced to the org-only set:
  **`("het ministerie van ", "ministerie van ")`**. REMOVED: `minister van`,
  `staatssecretaris van`, `de minister van`, `de staatssecretaris van` (person
  leaders — `Minister van BZK` is a PERSON), and `gemeente`, `provincie`
  (municipality leaders — `Gemeente Groningen` the ORG ≠ `Groningen` the city).
  Articles (`de`/`het`/`een`) and `ministerie van` kept (org → org-tail, same
  type, no cross-type collision).
- `resolution_metrics.py::count_false_merges` now compares at the **NAME level**
  (ignores `entity_type`) — the criterion relations actually depend on. Added
  `count_false_merges_with_type` for `(name, type)` fragmentation diagnostics;
  `measure_fragmentation` still keys on `(name, type)` (mirrors persistence
  dedup key).
- `must_not_merge.jsonl`: added/encoded the cross-type pairs that must stay
  distinct at name-only — `Minister van BZK` ↔ `Ministerie van BZK`,
  `Gemeente Groningen` ↔ `Groningen`, `Provincie Groningen` ↔ `Groningen`.
  Gate: **0 name-only collisions** across all of them.
- `must_merge.jsonl`: removed the person-minister↔org pairs that are now
  (correctly) distinct (`Minister van BZK` ↔ `BZK`; `De Minister van … KR` ↔ …).
  Kept org forms + article variants.
- `notebook_merge_service.py`: KEEPS the rev2 `(name, type)` bucket key (correct
  + consistent with persistence). With zero name-only cross-type collisions the
  `name_to_canon` relation-endpoint rewrite is now safe. Regression test
  `test_same_name_different_type_stays_distinct` kept (now the two normalize to
  DIFFERENT strings → trivially distinct); added
  `test_relation_endpoints_do_not_cross_types` confirming the endpoint rewrite
  routes minister vs ministry to distinct endpoints.

**Final strip list**: leading articles `de` / `het` / `een`; org leaders
`het ministerie van` / `ministerie van`. (No person-role or municipality
leaders.)

**Re-measured over the frozen 1402-entity Convenant fixture**
- Distinct-canonical: **1360 (V1) → 1341 (K.1 rev3)** = **−19**. Smaller than
  rev2's −46 BY DESIGN — rev2's extra merges were partly WRONG cross-type
  merges (person/municipality leaders stripped onto org/location names).
- BZK full-form cluster (`binnenlandse zaken en koninkrijksrelaties`, type
  `other`): **4 surface forms → 1** (two bare spelling variants + two
  `Ministerie van …` forms via ministerie-strip + spelling). Person-role forms
  no longer fold in.
- **Name-only false-merges over `must_not_merge.jsonl`: 0.** Unmerged over
  `must_merge.jsonl`: 0.
- `Minister van BZK` → `minister van bzk`; `Ministerie van BZK` → `bzk` —
  confirmed DIFFERENT strings.

**Regression**: `packages/shared/tests/` 273 passed; merge + persistence +
roundtrip suites green (296 total across the validation command). B.8 hash_id
derive-rule unchanged (`TestB8HashContract` green). ruff clean on changed src +
shared test files (pre-existing import-sort finding in
`test_notebook_merge_service.py` left untouched — not introduced here).

## Basis
- B.8c assessment: V1 normalizer resolves identical surface forms (107 cross-doc) but fragments variants (BZK 8-way, "minister" 23-way). See `../B-kg-quality/reviews/phase-B.8c-resolution-assessment.md`.
- Swap-point: `packages/shared/src/shared/utils/name_normalizer.py::normalize_entity_name` (persistence dedup key + filtering both call it).
- Two layers: K.1-K.2 cheap NL normalization (quick wins) → K.3+ full Q9/M4 vocabulary resolution.
