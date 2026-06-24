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

---

## Phase Q.4 — Pipeline orchestration (after-extraction hook) — ready for review

**Branch**: `track/q4-orchestration` (off `main`)
**Commits**: `40556ae` batch triage/relation helpers + persist returns batch ids → `13fe665` orchestrator + queue + backbone + migration 60 + unit tests → `57f6048` router + DI + hook at both persist sites → `2612d58` integration + router tests.

### Orchestration flow (B1→B6)
`TriagePipeline.run(source_id, persisted_entity_ids, batch_id)` — pure orchestration over FROZEN reuse services:
- **B1 match** — `CandidateDedupService.propose_candidates` (K.5); review-band proposals touching the batch are surfaced as match-conflict queue rows (never auto-merged).
- **B2 merge** — done by the caller's `persist_filtered_result` (B.8/O.1/Q.2a) BEFORE the hook; `RecanonicalizationService` (K.3) injected for collision merges.
- **B3 signals** — `TriageSignalsService.compute_report` (Q.2).
- **B4 status** — `StatusAssignmentService.assign`/`apply` (Q.3), manual_override-wins.
- **B5 queue** — `TriageQueueRepository.upsert` (dedup-by-entity).
- **B6 backbone** — `BackboneCheckService.warnings_for` (weak-edge warnings for affected degree≥3 nodes).

### Affected-set / snapshot choice (R4) — choice (b), proven idempotent
The hook fires AFTER the merge, so the orchestrator cannot observe the pre-merge degree/doc-count. Of the plan's two options it takes **(b): re-triage the batch entities + their active relation neighbours**, with an EMPTY pre-batch snapshot so Q.2 marks the whole superset "new". This is safe because every downstream WRITE is idempotent under re-derivation:
- status — `apply()` writes + logs ONLY on an actual change (pure function of tier×degree×override; unchanged re-run = no-op, no DB write, no log row).
- provenance — counted by Q.2a's per-edge `source_documents` UNION at persist time; triage only READS degree/doc-count, never double-counts.
- queue — `upsert` dedups by entity (one OPEN row per entity, UPDATEd not appended).

The AC3 "unchanged nodes untouched" guarantee is met at the WRITE layer (log/queue stability), which the run-twice integration test asserts. Cost of (b) over (a): unchanged-but-touched nodes are re-EVALUATED (DB reads) but produce zero spurious writes.

**Run-twice proof** (`test_run_twice_is_idempotent`): same batch twice → identical statuses, `status_change_log` row count UNCHANGED, `r2.statuses_changed == 0`, open-queue count UNCHANGED.

### Queue dedup mechanism
`triage_queue` table (migration 60): at most ONE OPEN row per entity. `upsert` SELECTs an existing open row for the entity and UPDATEs it (signals/reason/batch refresh) else CREATEs; only OPEN rows dedup, so a decided row never re-opens. Dedup verified against the live container (`test_triage_queue.py`).

### Migration 60 (triage_queue)
`migrations/60.surrealql` + `60_down.surrealql` — `DEFINE TABLE IF NOT EXISTS triage_queue SCHEMAFULL`; fields `entity record<entity>`, `name`, `type`, `structural_degree`, `doc_count`, `reason`, `decision DEFAULT 'open'`, `batch_id`, `created_at`/`updated_at DEFAULT time::now()`; indexes on `entity` + `decision` + `batch_id`. Additive + idempotent (`IF [NOT] EXISTS`), follows migration 54/59 precedent. Down removes ONLY the `triage_queue` table (cascades its fields/indexes). Migration 59 already merged → this is **60** (the plan's "new 60 if Q.3 already merged" branch).

### Hook placement (R1 — both persist sites, guarded, non-blocking)
`entity_extraction_service._run_triage(source_id, persist_result)` fires after `persist_filtered_result` on BOTH paths — main (~:1155) and re-filter (~:1362) — OUTSIDE the filtering try/except, with its own log-and-continue guard (mirrors the filtering-failure isolation at ~:1141). Pipeline lazily constructed via `dependencies.get_triage_pipeline()` on first persist. `persist_filtered_result` now RETURNS `persisted_entity_ids` (additive) so the hook re-triages exactly the batch. A triage failure logs an error and extraction still reports success (asserted by `test_step_failure_is_non_blocking` + the `_run_triage` guard).

### Operator decision endpoint (override-wins)
`POST /api/triage/queue/{id}/decision` → `EntityRepository.set_manual_override(status, manual_override=True)` (the ONLY writer of `manual_override = true`, unconditional). A subsequent pipeline run leaves the pinned status untouched (DB-layer guard in `set_status`). Router tests + integration confirm.

### Named-programme limitation (R5, documented + asserted)
NOVEX/Regio-Deals reach `active` ONLY when typed `Programma`/`Deal`/`RegioDeal` (config active tier). Typed `Organisatie` the SAME programme falls to unsure_review → `reference` + queue. Asserted by `test_named_programme_typing_limitation`. Queue-catch is the accepted v1 behaviour; an alias rule is deferred.

### Files
- Create: `services/triage/triage_pipeline.py`, `triage_queue.py`, `backbone_check.py`; `api/routers/triage.py`; `migrations/60.surrealql` + `60_down.surrealql`.
- Modify: `entity_extraction_service.py` (hook + `_run_triage`), `entity_persistence_service.py` (returns `persisted_entity_ids`), `EntityRepository` (`triage_rows_for_entities`, `relations_for_entities`, `set_manual_override`), `dependencies.py` (`get_triage_pipeline`/`get_triage_queue_repo`/`get_status_change_log_repo`), `api/app.py` (router register).

### Tests (all green)
`uv run --no-sync --with docker --with testcontainers pytest <Q.4 suite> + test_triage_signals_service.py + test_status_assignment_service.py` → **56 passed** (unit + testcontainer integration; Q.2/Q.3 regression clean). `create_app()` → OK. `ruff check` on all changed files → clean (the 2 pre-existing `ExtractionResult` F821 forward-refs in entity_extraction_service.py are untouched by this diff).

### Env note (workspace .venv)
Restored env was missing `testcontainers`/`docker` (member dev-group deps). Canonical invocation: `uv run --no-sync --with docker --with testcontainers pytest …` (matches the Q.3 note). `--no-sync` prevents the workspace-member desync; `--with` injects the testcontainer deps transiently.

---

## Phase Q.5 — UI surface (triage) — READY FOR REVIEW

Branch: `track/q5-ui` (off `main`, worked in the main tree). Commits:
- `14c72b8` fix(triage): match-conflict upsert preserves status-flag signals (Q.5) — the folded Q.4 review minor.
- `78b18bd` feat(triage): thread optional status filter onto entity listing (Q.5) — additive backend seam.
- `b25cae7` feat(triage): entity-keyed override endpoint + triage API client (Q.5).
- `e49c59e` feat(triage): KG status filter, override toggle, triage-queue view (Q.5) — the UI.

### Q.4 review minor folded (match-conflict clobber)
`_surface_match_conflicts` upserted a queue row with `structural_degree=0, doc_count=0` and would CLOBBER a prior status-flag row's real signals + reason for the same entity. Fix in `triage_queue.py`:
- `upsert` treats an incoming zero degree/doc_count as "unknown" on UPDATE (never overwrites a prior non-zero signal).
- New `merge_reason` flag (`_merge_reason` pure helper, idempotent) APPENDS the incoming reason instead of replacing it; `_surface_match_conflicts` passes `merge_reason=True`.
Idempotency preserved (re-run appends nothing). Tested: `test_match_conflict_does_not_clobber_status_flag_signals` (testcontainer) + `test_merge_reason_appends_and_is_idempotent` (pure).

### Backend status filter (additive)
- `EntityRepository.list_entities` / `count_entities` accept optional `status` (backed by `idx_entity_status`); the list projection now surfaces `status`, `manual_override`, `source_documents`.
- `KnowledgeGraphService.list_entities` / `get_entity` join the Q.2 signals (`structural_degree`, `doc_count`) in ONE batched relation query; degrades gracefully on failure.
- `routers/knowledge_graph.py` exposes `?status=` on GET /entities.
- New additive `POST /api/triage/entities/{id}/override` pins/unpins ANY entity (detail-panel toggle) via the existing `set_manual_override` (override-wins) and closes the entity's open queue row on a pin. The Q.1–Q.4 services are otherwise unchanged (FROZEN respected).

### Frontend
- `lib/api/triage.ts` (mirrors `entity-resolution.ts`): queue, decide, setEntityOverride, backboneWarnings.
- `lib/utils/triage.ts` (pure, unit-tested): statusDisplay, buildDecision, groupByReason, primaryReason, isOpen, totalWeakEdges, isBackboneRisk.
- `lib/hooks/use-triage.ts`: queue/backbone queries + decide/override mutations (invalidate triage + knowledge-graph caches).
- `components/knowledge-graph/StatusBadge.tsx`, `StatusFilter.tsx`, `BackboneWarnings.tsx`.
- `knowledge-graph/triage/page.tsx`: review queue grouped by reason with promote/keep/override decision controls; loading/error/empty states reuse the resolution-hub structure.
- `knowledge-graph/page.tsx`: StatusFilter in the filter row; Status/Degree/Docs columns; triage-queue link; detail-panel triage block (status badge + degree/docs + `manual_override` Switch → override API).
- `knowledge-graph.ts`/`use-knowledge-graph.ts`: optional `status` list param + status/manual_override/structural_degree/doc_count on the Entity type.

### AC status (plan Q.5)
1. KG table renders default/loading/error/empty + filters by status — DONE (table states + StatusFilter → `?status=` request; E2E asserts the filter re-query).
2. Each row/detail shows structural-degree + cross-doc-count — DONE (Degree/Docs columns + detail block; service joins them).
3. Detail panel shows status + manual_override toggle; toggling calls the API + sticks across re-run — DONE (Switch → `POST /override`, override-wins enforced server-side by `set_manual_override` + the `set_status` guard; router tests assert the contract).
4. Triage-queue view lists rows (id/name/type/degree/docs/reason) + decision control; deciding closes the row — DONE (grouped queue + promote/keep/override; E2E asserts decide-closes).
5. Backbone warnings render for degree>=3 weak-edge entities — DONE (`BackboneWarnings` component + `useBackboneWarnings`; unit-tested summary/risk logic).
6. Keyboard-accessible + reuses resolution-hub patterns — DONE (aria-labels on the filter/toggle/decision buttons; queue page mirrors the resolution-hub loading/error/empty + dismissed-set structure).

### Tests
- Frontend: `npm test` → **77 passed (10 files)** incl. 13 new pure-helper tests (`__tests__/triage-utils.test.ts`). `npx tsc --noEmit` → clean (e2e included). `npm run lint` → no new warnings in Q.5 files (pre-existing warnings elsewhere untouched).
- Frontend E2E: `e2e/track-q/triage.spec.ts` — 4 route-mocked scenarios (status filter → `?status=reference`, detail override toggle → `POST /override` with `manual_override=true`, queue decide-closes, empty + error states). Discovered by Playwright; the project has no `webServer` config so these run against a running dev server (manual / CI), matching the track-k convention.
- Backend: `uv run --no-sync pytest test_triage_queue.py test_triage_router.py test_knowledge_graph_service.py test_knowledge_graph_router.py` → **38 passed** (clobber fix + override endpoint + status-filter service/router). `test_entity_repository_roundtrip.py::test_list_entities_status_filter` → passed (live-DB status filter). `test_triage_pipeline.py` → no regression. `create_app()` → OK.

### Manual smoke checklist (not yet executed against the live corpus)
- [ ] KG table → set status filter to `reference` → table narrows; URL/query carries `status=reference`.
- [ ] Open a degree>=3 entity → detail shows degree/docs + status badge; flip override → toast "Pinned"; re-run extraction → status preserved.
- [ ] Triage queue → grouped reasons render; Promote a row → toast + row leaves; entity status flips to active.
