# Review — Track B Phase B.4 attempt 1

**Branch**: `track/b-confidence-telemetry` (HEAD `9e35198`)
**Decision**: **APPROVED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-08

## Summary

Phase B.4 lands the always-on telemetry sink (migration 47 + `shared.services.metrics`), wires the two call sites (`extraction.complete` + `extraction.auto_fallback`), surfaces confidence on the KG table and relation rows via `ConfidenceBar`, and persists the `ConfidenceFilter` choice in localStorage. Backend well-tested (unit + testcontainers roundtrip incl. migration-47 idempotency); exactly-once + payload shape pinned. Frontend clean (TS + lint); Playwright spec is structurally correct.

## Acceptance criteria check

| # | Criterion | Status |
|---|---|---|
| 1 | KG page shows confidence bar on every entity tile | PASS |
| 2 | Filter slider hides entities below threshold; localStorage persists | PASS |
| 3 | `run_extraction` writes exactly one `extraction.complete` row | PASS — happy path, zero-chunks, paused-review all pinned |
| 4 | Auto-fallback writes one `extraction.auto_fallback` per source | PASS — both engine branches + both-fail-no-metric tested |
| 5 | Migration 47 idempotent (B.0 harness verifies) | PASS — `test_migration_47_is_idempotent` |
| 6 | Playwright spec verifies bar + filter UI | PASS (code) — env-failure on shared port `8502` traced to stale dev server, not spec |

## Test status (independently verified)

- `packages/shared`: 154 passed
- `packages/surrealdb-service`: 77 non-docker + 25 docker passed
- `apps/app-main`: 408 passed (no regressions)
- frontend tsc + lint: clean
- Playwright B.4 spec: 4/4 per implementer; reviewer hit stale-bundle env issue (not code)

## Minors (6, non-blocking)

1. **Duplicate `ConfidenceBar` in `ResolutionLogTab.tsx`** — pre-dates this PR; uses yellow vs amber. Visual inconsistency between Resolution Log tab and Entities tab.
2. **`extraction.auto_fallback` payload includes extra `threshold` key** beyond `{confidence, decision, engine_used}` in plan. Safe (FLEXIBLE), but document or drop.
3. Plan §531 mentions "source-detail KG tab" that doesn't exist — strike from plan.
4. `get_all_entities_and_relations` SELECT omits confidence — Sigma graph view can't tint by confidence. Implementer flagged.
5. `ConfidenceFilter` only emits `onChange` when persisted value > 0 — undocumented prop contract.
6. `metrics_pool_bound_to_live` fixture uses module-level monkeypatch — works but stale-risk if test ordering changes.

## Kudos

- Telemetry hook covers zero-chunks early-return + paused-review skip (negative-space tests)
- `test_record_metric_handles_missing_persistence_module` exercises ImportError branch
- Migration 47 file is doc-rich (FLEXIBLE rationale, composite index)
- `_avg_entity_confidence` module-scope helper, testable in isolation
- `ConfidenceBar` defensive `clamp01` handles undefined/NaN
- Both-engines-fail path emits no metric and spec pins it (correct: would muddy dashboards)

## Decision rationale

0 blockers · 0 majors · 6 minors. Track A's RETRO #5 closure unblocked. Approved.

## Next steps

1. **Before merge**: re-run Playwright against fresh dev server from this branch (env caveat).
2. Optional follow-up: consolidate the duplicate `ConfidenceBar` + plan-housekeeping for §531.
