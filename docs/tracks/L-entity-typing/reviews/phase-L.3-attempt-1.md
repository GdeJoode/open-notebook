# Review — Track L Phase L.3 attempt 1

**Branch**: `track/l3-enum-types`
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-22

## Summary

L.3 adds `programme` and `technology` as canonical entity types via a pure-code change (enum addition + L.1 bridge repoint + activated L.1/L.2 test assertions). The no-migration claim is genuinely correct: migration 50 already left `entity.entity_type` as bare `TYPE string` with no ASSERT, and no migration 51-57 re-added an entity_type constraint, so the planned migration 58 would have been a literal no-op. The L.1/L.2 test edits are legitimate activations (interim `creative_work` -> `programme`; removal of `other` re-pin expectations), not weakening. Suite green (99 passed), B.8 dedup contract intact.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | `programme`/`technology` in `_ALLOWED_ENTITY_TYPES`; RegioDeal -> `programme` (not other/creative_work) | ✅ | enum verified at runtime; `test_regiodeal_bridge_lands_on_programme_not_other` asserts != other AND != creative_work |
| 2 | `{label:"Technologie"}` -> `technology` | ✅ | `test_technologie_residual_alias_lands_on_technology` |
| 3 | programme/technology write succeeds against live schema | ✅ (re-interpreted) | Live schema has no ASSERT (migration 50) so any string persists; AC3 intent met. Two persist-path tests assert upserted entity carries the real canonical |
| 4 | Migration 58 idempotent up/down/up | ✅ (justified deviation) | NO migration written — would be a no-op since live constraint already relaxed. Documented in status.md + plan Risk (a) anticipated this |
| 5 | Decision L-D1 recorded (extend-enum vs ontology-type-as-entity_type) | ✅ | status.md documents extend-enum choice + B.8 rationale |

## Test status

```
uv run pytest apps/app-main/tests/test_entity_persistence_service.py \
  packages/ontology-manager/tests/test_canonical_bridge.py \
  packages/shared/tests/test_entity_type_aliases.py \
  packages/surrealdb-service/tests/test_entity_repository_roundtrip.py -q
99 passed in 8.14s
```

## Verification performed

1. **No-migration correct** — `migrations/50.surrealql:29` = `DEFINE FIELD OVERWRITE entity_type ON entity TYPE string;` (no ASSERT). Grepped migrations 51-57: the only ASSERT (`migrations/51.surrealql:34`) is on the LLM-pipeline-stage enum, not `entity.entity_type`; `migrations/56.surrealql:42` defines `entity_type` on `alias_overlay` (different table, DEFAULT '', no ASSERT). Migrations 54/55 reference entity_type only in comments confirming additive-only. **No later migration re-added an entity_type constraint** — no-migration is genuinely correct.
2. **programme/technology valid additions** — runtime check confirms both in `_ALLOWED_ENTITY_TYPES`; guard logic unchanged (only the membership set grew). Persist tests confirm a programme/technology entity persists with that exact entity_type via the real `persist_filtered_result` path.
3. **Bridge repoint correct** — `Deal`/`GovernmentService` -> `programme` (was interim `creative_work`); `TechArticle`/`Technology`/`SoftwareApplication` -> `technology`. `general.yaml:133` confirms `Technology` declares `schema_org_type: schema:TechArticle`, matching the mapping. Runtime check: every value in `_CANONICAL_BY_SCHEMA_ORG` is in `_ALLOWED_ENTITY_TYPES` (NONE missing). No Dutch literals in bridge keys; `TestLanguageAgnostic` grep-guard green.
4. **Test updates are activations not weakening** — every changed assertion flips an interim/pending expectation to its activated value (`creative_work`->`programme`; `other` re-pin expectation removed). No assertions deleted/loosened; new explicit assertion classes added (`TestProgrammeAndTechnology`, `TestL3ProgrammeTechnologyActivation`). Residual-resolver assertions unchanged (always emitted the real targets).
5. **B.8 intact** — entity.py untouched (`hash_id` derive-rule unchanged); every emitted entity_type in the extended enum.
6. **Scope** — diff is 6 files, no migration/.surql added; ruff clean on both changed source files; `create_app` imports OK; L-D1 documented in status.md.

## Issues found

### 🔴 Blockers (must fix)

None.

### 🟡 Major (must fix)

None.

### 🔵 Minor (optional)

1. **AC3/AC4 wording vs implementation** — `docs/tracks/L-entity-typing/plan.md:148-150,159-160` still prescribe `migrations/58.surrealql` + a roundtrip harness as falsifiable AC. The implementation correctly deviated to no-migration, and status.md documents why, but the plan AC text was not reconciled. Follow-up: annotate plan L.3 AC3/AC4 as superseded-by-no-migration so future readers don't expect migration 58. Does not block — the deviation is sound and the live-schema reality (no ASSERT) makes the migration a no-op, exactly as plan Risk (a) warned.

## Decision rationale

The two load-bearing claims both hold: (1) no-migration is genuinely correct — verified migrations 51-57 contain no entity_type ASSERT on the `entity` table, so migration 58 would be a no-op against the already-relaxed live schema; and (2) the L.1/L.2 test edits are activations, not weakening — each flips a documented interim/pending value to its real canonical and the suite adds (not removes) coverage. B.8 is intact (entity.py untouched, all types in enum), suite is green (99/99), bridge values all valid, no Dutch literals, ruff clean, create_app imports. Zero blockers, zero majors. One minor doc-reconciliation follow-up.

## Next steps

APPROVED — ready for human approval / merge. File the minor plan-AC reconciliation as a follow-up note.
