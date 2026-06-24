# Track Q — Status Ledger

> Post-extraction triage & merge pipeline. Plan: `./plan.md` (draft, awaiting human approval).
> REUSE-FIRST: matching (K.5), merge/provenance (B.8/O.1/K.3), review-queue precedent (K.5 CandidateReport) are MERGED. Track Q = triage layer + UI surface.

| Phase | Title | Reuse-vs-new | Status | PR | Notes |
|-------|-------|--------------|--------|----|-------|
| Q.1 | Config loader & validator | NET-NEW (small) | ready-for-review | `track/q-config-impl` | loader reads `config/triage_config.json`; 16 tests pass, ruff + mypy clean; see Q.1 section below |
| Q.2a | Relation merge (edge-dedup + cross-doc provenance + dup backfill) | NET-NEW | not-started | — | **secures the promotion model**; persist currently DUPLICATES edges (70 live dups); prerequisite for Q.2 |
| Q.2 | Signals (effective degree w/ weak-edge promotion, recurrence, affected set) | NET-NEW + partial reuse | not-started | — | RELATED counts when recurring ≥ weak_promotion_min_docs; depends on Q.2a |
| Q.3 | Status assignment + migration 59 + change-log | NET-NEW logic, reuse status/properties | not-started | — | adds `reference` status, `manual_override`, status_change_log |
| Q.4 | Pipeline orchestration (after-extraction hook) | ORCHESTRATION (reuse B1/B2) | not-started | — | hook at entity_extraction_service.py:1155 & :1362 |
| Q.5 | UI surface | REUSE-HEAVY | not-started | — | extends KG view + resolution-hub patterns |

## Open decisions (await operator confirm — see plan §3/§5)
- [ ] R1: triage fires on re-filter path (:1362) too? (rec: yes)
- [ ] R2: manual_override = column (migration 59) vs properties key? (rec: column)
- [ ] R3: status-change log + triage queue as DB tables vs file? queue migration in 59 or 60? (rec: tables)
- [ ] R5: named-programme limitation — queue-catch acceptable for v1, alias rule deferred? (rec: yes)

## Phase Q.1 — Config loader & validator (ready-for-review)

**Branch**: `track/q-config-impl` (worktree-isolated; the `track/q-config` name was already
held by the shared checkout — reviewer/human may rename on merge).
**Commits**: `4c1c318` (loader) → `1eb3565` (tests).

**Files created**:
- `apps/app-main/src/app_main/services/triage/__init__.py` — package marker, re-exports loader API.
- `apps/app-main/src/app_main/services/triage/triage_config.py` — `TriageConfig` (frozen dataclass),
  `TriageConfigError`, `load_triage_config(path=None, *, reload=False)`.
- `apps/app-main/tests/test_triage_config.py` — 16 tests.

**Config convention matched**: frozen `@dataclass`, mirroring the plan's named reuse anchor
`entity_filtering.config.FilteringConfig` (not pydantic) — the triage config is static committed
JSON, not env-driven settings. Lookups (`tier_for`, `predicate_strength`) are O(1) via pre-built
reverse maps; the instance is frozen and cached one-per-resolved-path.

**Acceptance criteria** (all met / tested):
1. ✅ `load_triage_config()` → `TriageConfig`, `version == 1` from committed file.
2. ✅ `tier_for`: BeleidsThema→active, Budget→reference, Person→unsure_review, unmapped→unsure_review.
3. ✅ `predicate_strength`: VERSTERKT→structural, RELATED→weak, unmapped→weak.
4. ✅ `weak_promotion_min_docs == 2`; `core_active_min_degree`/`reference_isolated_max_degree`/`well_connected_threshold` accessors.
5. ✅ Overlapping tier lists, missing key, bad version, malformed JSON all raise `TriageConfigError` (no silent default).
6. ✅ Read-only, cached per path, file never mutated (asserted).

**Validation**: `uv run pytest apps/app-main/tests/test_triage_config.py -q` → 16 passed.
`uv run python -c "from app_main.api.app import create_app"` → OK. ruff clean. mypy clean
(via `--with mypy`; mypy not in the base env).

**Note for reviewer**: this worktree's branch did not contain the Track-Q commits, so the
impl branch was based off the Q-config commit `3b0f6e1` (which carries `config/triage_config.json`
+ the plan). No files outside `apps/app-main/.../triage/`, the test, and this ledger were touched.

## Frozen anchors (reuse, do not change)
- K.5 `candidate_dedup_service.py` (matcher) · K.3 `recanonicalization_service.py` · B.8/O.1 `entity_persistence_service.py::persist_filtered_result`
- migration 39 (entity.status free string, no enum ASSERT; properties flexible; primary_type; idx_entity_status) · migration 58 (relation) · highest migration = 58 → Q uses 59
- P.1 1024-dim mxbai embedding pin
