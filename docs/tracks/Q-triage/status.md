# Track Q — Status Ledger

> Post-extraction triage & merge pipeline. Plan: `./plan.md` (draft, awaiting human approval).
> REUSE-FIRST: matching (K.5), merge/provenance (B.8/O.1/K.3), review-queue precedent (K.5 CandidateReport) are MERGED. Track Q = triage layer + UI surface.

| Phase | Title | Reuse-vs-new | Status | PR | Notes |
|-------|-------|--------------|--------|----|-------|
| Q.1 | Config loader & validator | NET-NEW (small) | not-started | — | config file DONE; loader reads `config/triage_config.json` |
| Q.2a | Relation merge (edge-dedup + cross-doc provenance + dup backfill) | NET-NEW | ready-for-review | `track/q-relation-merge` | persist write-path now UPSERTs per `(in,out,relation_type)` (union `source_documents`, max conf, no-clobber properties); backfill collapses dups (dry-run + idempotent); O.1 roundtrip green. Live DB now shows 0 dup groups (216 active edges) — the 70-dup figure predates O.1/re-extraction; backfill verified via seeded-dup test |
| Q.2 | Signals (effective degree w/ weak-edge promotion, recurrence, affected set) | NET-NEW + partial reuse | not-started | — | RELATED counts when recurring ≥ weak_promotion_min_docs; depends on Q.2a |
| Q.3 | Status assignment + migration 59 + change-log | NET-NEW logic, reuse status/properties | not-started | — | adds `reference` status, `manual_override`, status_change_log |
| Q.4 | Pipeline orchestration (after-extraction hook) | ORCHESTRATION (reuse B1/B2) | not-started | — | hook at entity_extraction_service.py:1155 & :1362 |
| Q.5 | UI surface | REUSE-HEAVY | not-started | — | extends KG view + resolution-hub patterns |

## Open decisions (await operator confirm — see plan §3/§5)
- [ ] R1: triage fires on re-filter path (:1362) too? (rec: yes)
- [ ] R2: manual_override = column (migration 59) vs properties key? (rec: column)
- [ ] R3: status-change log + triage queue as DB tables vs file? queue migration in 59 or 60? (rec: tables)
- [ ] R5: named-programme limitation — queue-catch acceptable for v1, alias rule deferred? (rec: yes)

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
