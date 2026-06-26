# Track S — status

## Phase S.2 — Self-healing repair migration (Backend) — READY FOR REVIEW

**Branch**: `track/s2-drift-repair-migration` (off `main` @ `2a6c80e`, S.1 merged).
**Commits**: `891315a` (migration), `ce88605` (tests).

### Delivered
- `migrations/65.surrealql` — idempotent forward-guard. For every strict
  (non-`option<>`, no-`VALUE`) field WITH a safe default from the S.1 inventory,
  `UPDATE <table> SET f = f ?? <default>`. Covers 21 tables incl. `entity`
  (mirror of 61) and `source` (mirror of 64) as idempotent overlaps.
- `migrations/65_down.surrealql` — comment-only documented no-op (mirrors
  61_down / 64_down; re-NONE-ing would re-introduce the drift).
- `packages/surrealdb-service/tests/test_migration_65_strict_drift_guard.py` —
  8 `@requires_docker` tests, all green.

### Deliberately excluded (no safe coalesce default)
Content/text bodies (`chunk.text`, `source_embedding.content/embedding`,
`doc_node.self_ref`, `preprocessing_result.naive_summary`), required FKs/ids
(`*.source`, `*.source_id`, `*.entity`, `*.name`), type discriminators
(`entity.entity_type`, `relation.relation_type`, `claim.claim_type/statement`,
`job.job_type`, `chunk.element_type/order`, `metrics.event_type`), and edge
endpoints (`in`/`out` on relation/derived_from/next_node/parent_of/reference/
refers_to). These are the S.4 prevention rule's domain, not a coalesce target.

### Test evidence (per acceptance criterion)
- **AC1 synthetic-drift repro** — `test_{entity,source,transformation}_drift_fails_pre_passes_post`
  forge a NONE strict field (DEFINE FIELD OVERWRITE option<> → CREATE NONE →
  re-DEFINE strict without backfill), assert `UPDATE` raises pre-65 and lands
  post-65. All 3 PASS.
- **AC2 idempotent** — `test_migration_65_idempotent_double_apply` (double-apply
  no error) + `test_migration_65_clean_row_unchanged` (set values survive). PASS.
- **AC3 full chain** — `test_migration_65_discovered_and_applied`: the session
  fixture's strict transactional runner applies 1→65 clean and records v65. PASS.
  Existing `test_migrations_roundtrip.py` (19 tests) still green — no regression.
- **AC4 no-coalesce** — `test_no_safe_default_fields_not_coalesced`: static
  block-scoped grep of the on-disk body; no forbidden field is coalesced. PASS.
- **AC5 down** — `test_migration_65_down_is_noop`: cleaned SQL empty, no error. PASS.

Run: `8 passed` (S.2 suite), `19 passed` (existing roundtrip suite).

### Honest assessment of marginal value
On **current staging** migration 65 repairs **0 rows** — S.1 proved no live
drift, and 61/64 already healed the only two historically-drifted tables. Its
value is (a) the consolidated invariant in one place, (b) self-healing on any
UNSCANNED or freshly-restored environment (dev/prod), and (c) a validated,
reusable mechanism. The `entity`/`source` blocks are pure idempotent overlap
with 61/64. This is insurance + consolidation, not a live fix; the real
recurrence-stopper is the S.4 prevention rule (pair every new strict field with
a coalescing UPDATE in the SAME migration).

### Not touched
Live staging untouched (S.3 is the gated live-apply checkpoint).
