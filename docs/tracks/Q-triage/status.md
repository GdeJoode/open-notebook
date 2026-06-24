# Track Q — Status Ledger

> Post-extraction triage & merge pipeline. Plan: `./plan.md` (draft, awaiting human approval).
> REUSE-FIRST: matching (K.5), merge/provenance (B.8/O.1/K.3), review-queue precedent (K.5 CandidateReport) are MERGED. Track Q = triage layer + UI surface.

| Phase | Title | Reuse-vs-new | Status | PR | Notes |
|-------|-------|--------------|--------|----|-------|
| Q.1 | Config loader & validator | NET-NEW (small) | not-started | — | config file DONE; loader reads `config/triage_config.json` |
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

## Frozen anchors (reuse, do not change)
- K.5 `candidate_dedup_service.py` (matcher) · K.3 `recanonicalization_service.py` · B.8/O.1 `entity_persistence_service.py::persist_filtered_result`
- migration 39 (entity.status free string, no enum ASSERT; properties flexible; primary_type; idx_entity_status) · migration 58 (relation) · highest migration = 58 → Q uses 59
- P.1 1024-dim mxbai embedding pin
