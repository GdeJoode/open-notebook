# Track Q — Status Ledger

> Post-extraction triage & merge pipeline. Plan: `./plan.md` (draft, awaiting human approval).
> REUSE-FIRST: matching (K.5), merge/provenance (B.8/O.1/K.3), review-queue precedent (K.5 CandidateReport) are MERGED. Track Q = triage layer + UI surface.

| Phase | Title | Reuse-vs-new | Status | PR | Notes |
|-------|-------|--------------|--------|----|-------|
| Q.1 | Config loader & validator | NET-NEW (small) | merged ✅ | `track/q-config-impl` | loader reads `config/triage_config.json`; lookup maps wrapped in MappingProxyType (immutable cache); APPROVED rev1 (immutability blocker fixed + test) |
| Q.2a | Relation merge (edge-dedup + cross-doc provenance + dup backfill) | NET-NEW | merged ✅ | `track/q-relation-merge` | persist UPSERTs per `(in,out,relation_type)` (union `source_documents`, max conf, no-clobber properties); backfill (dry-run + idempotent); O.1 roundtrip green; APPROVED. Live DB shows 0 dup groups (216 active edges) — the 70-dup figure predates O.1/re-extraction; backfill verified via seeded-dup test |
| Q.2 | Signals (effective degree w/ weak-edge promotion, recurrence, affected set) | NET-NEW + partial reuse | merged ✅ | `track/q2-signals` | `TriageSignalsService` + batched `effective_degree_for_entities`; promotion flip asserted (1-2→3+); affected set = new+changed-degree+changed-docs vs pre-batch snapshot; ONE batched DB call; APPROVED |
| Q.3 | Status assignment + migration 59 + change-log | NET-NEW logic, reuse status/properties | merged ✅ | `track/q3-status` | migration 59 (manual_override bool COLUMN + status_change_log TABLE); StatusAssignmentService (tier×degree×override truth table, manual_override ALWAYS wins, 4-path defense); set_status override-conditioned no-op; 49 tests green; APPROVED |
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

---

## Phase Q.2a — implementation summary (ready for review)

**Branch**: `track/q-relation-merge` (off `main`)
**Commits**: `cf9e706` (forward fix + backfill), `76c33b9` (tests)

### What changed
- `apps/app-main/src/app_main/services/entity_persistence_service.py`
  - **Before**: the relation loop ran an UNCONDITIONAL `RELATE $s->relation->$t SET source_documents = [$source_id]` per extracted edge — the same edge from another doc made a DUPLICATE row, so per-edge cross-doc recurrence was never tracked.
  - **After**: a new `_upsert_relation()` helper decides create-vs-union AFTER the O.1 type-safe endpoints resolve (the O.1 resolution path is untouched):
    - existing active edge `(in, out, relation_type)` → `UPDATE ... SET source_documents = array::union(source_documents, [$source_id])`, keep `max(confidence)`, merge `properties` (existing keys retained, new keys overlaid — no clobber; merged in Python because `object::extend`/`object::merge` are absent on SurrealDB v2.x).
    - no existing edge → `RELATE` a fresh edge (the pre-Q.2a behaviour).
  - Idempotent: re-persisting the same `(edge, source_id)` is a no-op union (length unchanged). New `relations_merged` counter + updated persist summary logs + return dict.
- `scripts/backfill_relation_merge.py` (new) — collapses existing duplicate edges: groups active relations by `(in, out, relation_type)`, keeps one survivor, unions provenance/properties/max-confidence, deletes redundant rows. `--dry-run`, idempotent, non-destructive to non-duplicates (mirrors `scripts/backfill_entity_embeddings.py`).

### Tests (all green)
`uv run pytest apps/app-main/tests/test_relation_merge.py packages/surrealdb-service/tests/test_relation_endpoint_resolution_roundtrip.py apps/app-main/tests/test_entity_persistence_service.py -q` → **67 passed**.
- `test_relation_merge.py` (testcontainers): same-edge-two-docs → 1 edge len-2; idempotent re-persist (stays len-2); RELATED ×3 docs → len-3; max-conf + no-clobber property merge; backfill collapses a seeded dup set + idempotent; non-dup untouched; dry-run writes nothing.
- `test_relation_endpoint_resolution_roundtrip.py` (O.1) — **unchanged, still green**: confirms type-safe endpoint resolution + name-only fallback unaffected.
- `test_entity_persistence_service.py` — updated for the upsert write path (edge-existence SELECT before create) + new merge-branch unit test.
- `uv run python -c "from app_main.api.app import create_app"` → OK.

### Live-DB finding (honest note)
Backfill `--dry-run` against the running `open_notebook` DB (port 8000): **216 active relations, 0 duplicate `(in,out,relation_type)` groups**. The plan's "70 live dups" figure predates O.1 / a re-extraction; this DB is already clean, so the backfill is a safe no-op here (exercises its idempotency clause). Collapse correctness is proven by the seeded-dup integration test, not by collapsing the live DB.

### Frozen constraints respected
O.1 type-safe endpoint resolution, migration 39/58 relation schema (no schema change — `source_documents` already exists), B.8 hash_id, K.7a typed endpoints — all untouched. Only the create-vs-union decision after endpoints resolve changed.

---

## Phase Q.2 — implementation summary (ready for review)

**Branch**: `track/q2-signals` (off `main`, which carries Q.1 + Q.2a)
**Commits**: `71a82c4` (repo batched degree helper), `c90aff5` (signals service + tests)

### What changed
- `packages/surrealdb-service/.../repositories/entity.py` — ADD read-only `effective_degree_for_entities(ids, structural_predicates, weak_predicates, weak_promotion_min_docs) -> dict[id,int]`. ONE round-trip for the whole batch (NOT one query per node). Additive, no schema change.
  - **Promotion in SQL**: an edge counts when `relation_type INSIDE $structural` **OR** (`relation_type NOT INSIDE $structural AND array::len(source_documents ?? []) >= $min_docs`). Treating "weak" as "not structural" means an unmapped predicate (config-default weak) promotes exactly like `RELATED`. The single-doc RELATED is excluded by the `>= $min_docs` guard; a multi-doc RELATED passes it.
  - **Scan scope**: `WHERE status = 'active' AND (in INSIDE $ids OR out INSIDE $ids)` — pulls every counting edge touching the batch in one SELECT; the per-id tally is summed in Python (each counting edge adds 1 to **each** of its endpoints in the batch → AC5 both-endpoints; a self-loop adds 1 via set semantics).
- `apps/app-main/src/app_main/services/triage/signals_service.py` (new) — `TriageSignalsService`:
  - `effective_degree(id)` (single, delegates to the batched helper), `degree_bucket(d)` → `"0"|"1-2"|"3+"` keyed on `well_connected_threshold` (3), `doc_count(entity)` distinct `source_documents`.
  - `compute_report(batch_entities, pre_batch_snapshot) -> SignalsReport` — one batched degree call; per-id `EntitySignals(effective_degree, bucket, doc_count, is_new, degree_changed, doc_count_changed)`.
  - `compute_affected_set(...)` — new + changed-effective-degree + changed-doc-count. **Snapshot contract documented**: the caller MUST capture `pre_batch_snapshot` (`{id: {"degree", "doc_count"}}`) BEFORE the merge; capturing it after would make every node look unchanged and break idempotency (plan R4).
  - Predicate lists + threshold sourced from the Q.1 config (`cfg.predicate_strength` / `cfg.weak_promotion_min_docs` / `cfg.well_connected_threshold`); nothing hardcoded.

### Tests (`apps/app-main/tests/test_triage_signals_service.py`)
`uv run --project apps/app-main pytest apps/app-main/tests/test_triage_signals_service.py -q` → **6 passed, 1 skipped**.
- **Promotion flip asserted directly** (AC1): 2 `BIJDRAGT_AAN` + 1 single-doc `RELATED` → degree 2, bucket `1-2`; same edge reaches `source_documents >= 2` → degree 3, bucket `3+`.
- doc-count distinct + idempotent (AC2); affected set = new + changed-degree + changed-docs, unchanged excluded, promotion-threshold-crossing marks both endpoints (AC3); **ONE batched call asserted** via a call-count on the fake repo (AC4); structural + promoted-weak edge counts for both endpoints (AC5).
- Integration `test_integration_promotion_flip_and_batched_query` exercises the REAL SurrealQL promotion filter against a live container; **skips cleanly** here because the `docker` Python SDK is not installed in the app-main test venv (same graceful-skip as the Q.2a roundtrip suite). The unit `FakeEntityRepository` mirrors the exact SQL promotion arithmetic.

### Quality gates
- `ruff check` on all changed files → clean. `mypy signals_service.py` → clean.
- `uv run --project apps/app-main pytest apps/app-main/tests/test_triage_config.py -q` → 17 passed (no Q.1 regression).
- `uv run python -c "from app_main.api.app import create_app"` → OK.

### Environment note (honest)
This worktree's `.venv` resolves `app_main` only under `uv run --project apps/app-main` (the root `open-notebook` project does not depend on `app-main`). The `docker` SDK is absent, so `requires_docker` tests skip — no new dependency was introduced (out of scope for this phase). The real-SQL promotion path is covered by the integration test wherever Docker + the SDK are present.
## Phase Q.3 — implementation summary (ready for review)

**Branch**: `track/q3-status` (off `main` — Q.1 config-loader + Q.2a relation-merge merged)
**Commits**: `5d01902` (migration 59 + repo set_status), `1556b0b` (service + log repo), `2d62391` (tests)

### Operator decisions implemented
- **manual_override = dedicated bool COLUMN** (migration 59), not a `properties` key (R2).
- **status-change log = SurrealDB TABLE** `status_change_log` (R3), not an append-only file.

### What changed
- `migrations/59.surrealql` / `59_down.surrealql` — `DEFINE FIELD IF NOT EXISTS manual_override ON entity TYPE bool DEFAULT false` + `DEFINE TABLE IF NOT EXISTS status_change_log SCHEMAFULL` (fields `entity record<entity>`, `old_status`, `new_status`, `reason`, `batch_id`, `changed_at datetime DEFAULT time::now()`; indexes on `entity` and `batch_id`). Additive + idempotent (`IF [NOT] EXISTS`), mirrors migration 54. Down removes ONLY the two Q.3 objects; does NOT touch the migration-39 `status` field or `idx_entity_status`.
- `packages/shared/src/shared/models/entity.py` — `manual_override: bool = False`; `status` doc note adds `reference`.
- `packages/surrealdb-service/src/surrealdb_service/repositories/entity.py` — persist `manual_override` on create; carry it through verbatim on re-upsert (extraction never clears an operator pin); new `set_status(entity_id, status, *, respect_override=True)` whose UPDATE is conditioned on `manual_override = false OR NONE` in one atomic statement (race-free no-op when pinned).
- `apps/app-main/src/app_main/services/triage/status_assignment_service.py` (new) — pure `assign(entity, tier, effective_degree) -> StatusDecision`; side-effecting `apply(decision, batch_id)` writes + logs ONLY on an actual change of a non-pinned entity.
- `apps/app-main/src/app_main/services/triage/status_change_log.py` (new) — thin `append` / `list_for_entity` / `list_for_batch` over the migration-59 table.

### Status-assignment truth table (config: core_active_min=1, well_connected=3)
| tier | effective_degree | status | queue_flags |
|------|------------------|--------|-------------|
| active | ≥ 1 | `active` | — |
| active | 0 | `active` | `isolated core actor / possible extraction gap` |
| reference | 0–2 | `reference` | — |
| reference | ≥ 3 | `reference` | `candidate active` |
| unsure_review | any | `reference` | `unsure — operator decides` |
| (any, `manual_override == true`) | any | operator's status (unchanged) | — (no write, no log) |

### How manual_override-wins is enforced (defence in depth)
1. **Decision layer**: `assign()` short-circuits when `manual_override` is set — returns the operator's status, no flags, `changes_status == False`.
2. **Apply layer**: `apply()` is gated on `changes_status` — a pinned (or unchanged) decision writes nothing and logs nothing.
3. **DB layer**: `set_status(respect_override=True)` conditions the UPDATE on `manual_override = false OR NONE` in a single statement, so even a stale decision computed before a pin can't overwrite it (returns False → `apply` does NOT log a change that did not happen).
4. **Re-upsert**: extraction's entity upsert preserves the existing `manual_override` verbatim, so a re-extraction never clears a pin.

### Tests (all green)
`uv run pytest apps/app-main/tests/test_status_assignment_service.py apps/app-main/tests/test_migration_59.py -q` → **24 passed (unit)** + **6 passed (testcontainer)**.
- Unit: full tier×degree truth table, manual_override-wins (pure), apply() write+log on real change / suppressed on unchanged / suppressed under override / suppressed when DB-layer override race wins, re-run idempotency (log written once), dict-row support.
- Integration (migration 59): manual_override defaults false; `entity.status` accepts `reference` (no ASSERT rejection); down→forward roundtrip preserves migration-39 status field + index, drops only Q.3 objects, idempotent both ways (restore in `finally`); status_change_log CRUD via the repo (changed_at populated by schema default); set_status override no-op + operator force-path.

### Regressions checked (green)
`test_migrations_roundtrip.py` + `test_entity_repository_roundtrip.py` (21 passed), `test_relation_merge.py` + `test_entity_persistence_service.py` (61 passed), `test_entity_model.py` (12 passed). `from app_main.api.app import create_app` → OK. `ruff check` on all changed files → clean.

### Env note (worktree)
Run with `PYTHONPATH=<wt>/apps/app-main/src:<wt>/packages/shared/src:<wt>/packages/surrealdb-service/src` and `uv run --no-sync --with docker --with testcontainers` — the shared `.venv` had `app_main` editable-linked to a different worktree, so the worktree src is prepended explicitly and `docker`/`testcontainers` injected for the requires_docker suites.

### Out of scope (deferred to Q.4 per plan)
The review queue itself is Q.4 — Q.3 only RETURNS the `queue_flags` in the decision (it does not write a queue table). The after-extraction hook and orchestration are Q.4.
