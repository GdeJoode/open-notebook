# Review — Track B Phase B.5a attempt 1

**Branch**: `track/b-orphan-connector` (PR #22)
**Decision**: **REVISIONS_NEEDED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-09

## Summary

Orphan-connector module is well-crafted: 38 focused tests, 98%/100% coverage, robust JSON parsing, token-budget guard fires before LLM call. Three substantive gaps prevent approval — all integration boundary, not functional bugs.

## Acceptance criteria check

All 6 plan-level ACs PASS at module level (38 unit tests).

## Major (3)

### M1: `list_orphans_for_source` ships with zero tests

`packages/surrealdb-service/.../entity.py:596-677`. The new repo method is the ONLY production seam orphan-connector depends on. No test for empty source_id, entity-SELECT exception, per-entity edge-probe exception, no-orphans path. The N+1 query pattern (1 + N round-trips) is acknowledged in self-review but uncovered.

### M2: Workflow Stage 14 has no test coverage

`workflow.py:565-610`. The new Stage 14 (orphan-connect after dedup) is invoked by `FilteringWorkflow.process()` but never tested at the workflow level. Both `orphan_cfg.enabled=False` skip and the `OrphanTokenBudgetExceeded` recovery branch are dead code in the test suite.

### M3: Orphan-confirmed relations bypass ontology validation

`workflow.py:566`. Stage 14 runs AFTER Stage 11 (ontology constraint filter). LLM-invented relation_types ("KNOWS_SECRETLY", "ATE_LUNCH_WITH") are appended to `filtered_relations` without ontology check. Plan §583 ambiguous on whether validation applies. Decision must be explicit before B.1f wires this in.

## Minors (7)

1. `enabled=True` + missing DI silently skips — add WARNING log
2. Single oversize chunk aborts entire `confirm_connections` batch
3. Naming deviation: `OrphanConnectorConfig.enabled` vs plan's `orphan_connect_enabled`
4. `nc=-1` log artifact for non-list chunks (use "n/a")
5. Cross-source orphan semantics undocumented (entity in A+B with relation from B → not orphan when queried for A — defensible but undocumented)
6. Self-relation handling not pinned by test
7. Self-review "77 passed" surrealdb-service includes 25 skipped

## Kudos

- JSON parse pipeline (code-fence unwrap → braced-object salvage → loads → type checks)
- Token-budget guard envelope matches what LLM actually receives
- Loose chunk shape (text/body, id/chunk_id, entities as str/dict/ExtractedEntity)
- `seen_pairs` dedup keyed on (orphan_norm, partner_norm) handles multi-chunk co-occurrence
- Sync/async caller via `hasattr(raw, "__await__")` — clean DI surface

## Next steps

1. Fake-repo tests for `list_orphans_for_source` (5 paths)
2. 2-3 workflow-integration tests for Stage 14 (happy + disabled + budget recovery)
3. Decide ontology-bypass: reorder Stage 14 before validation OR document why bypass is intentional + pin with test
4. Address Minor #1 (warning on enabled+missing-DI) for B.1f safety
