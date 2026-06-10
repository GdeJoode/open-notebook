# Review — Track B Phase B.5a attempt 2

**Branch**: `track/b-orphan-connector` (PR #22)
**Decision**: **APPROVED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-10
**Attempt 1**: REVISIONS_NEEDED (0 blockers + 3 majors + 7 minors)

## Summary

Attempt 2 closes all 3 majors with behaviour-pinning tests + a defensible decision on ontology bypass. 6 new orphan-query tests + 5 new Stage 14 tests + ontology-bypass pin. 39/39 orphan-connector, 494/495 entity-filtering (1 pre-existing unrelated), 437/437 app-main.

## Major resolutions

**M1**: `list_orphans_for_source` test coverage — 6 tests pin empty-source-id, entity-SELECT raises, edge-probe raises, no entities, all orphans, mixed.

**M2**: Stage 14 workflow coverage — 5 tests: disabled-skips, enabled-happy, budget-exceeded-recovers, missing-DI-warning, ontology-bypass-pin.

**M3**: Ontology-bypass — kept current behaviour (Stage 14 after Stage 11). Documented in workflow.py + orphan_connector.run() + self-review. Pin test builds constrained ontology, mock LLM returns out-of-ontology relation, asserts survival. Reordering before Stage 11 fails the pin.

## Minor resolutions

5/7 resolved; 2 deferred per prompt.

## New minors (non-blocking, B.5b follow-ups)

1. WARNING fires on default config — `enabled=True` default + missing DI = WARNING spam until B.1f wires DI
2. Generic exception passthrough — only `OrphanTokenBudgetExceeded` caught; LLM crashes propagate
3. Query-substring stub dispatch fragility — refactor could mis-route

## Test status (reproduced)

- surrealdb-service: 6 orphan-query + 58 non-docker
- entity-filtering: 5 Stage 14 + 39 orphan-connector + 494 full
- app-main: 437 passed

## Kudos

- Bypass pin test exemplary: real constrained ontology + WHOLE pipeline cross-stage assertion
- Loguru → stdlib bridge for caplog documented and reusable
- Cross-source semantics in docstring captures subtle invariant
- Self-review traces every minor back to test/doc change
- N+1 query defended explicitly ("orphan set is by definition small")

## Next steps

APPROVED — ready for merge.
