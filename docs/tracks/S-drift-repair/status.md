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

---

## Phase S.2b — drift-only WHERE generalization (live-debug fix, attempt 2) — READY FOR REVIEW

**Branch**: `track/s2b-drift-where-generalize` (off `main`).
**Commits**: `e0d85df` (migration), `154863d` (test).

### Why (adversarial review REVISIONS_NEEDED)
Migration 65 runs at app STARTUP. A SurrealDB SCHEMAFULL `UPDATE` re-validates
AND REWRITES every matched row. The original blanket `UPDATE <table> SET f = f ??
d` matched ALL rows on ~19 tables, so on payload-bearing tables (multi-MB rows)
it rewrote every row and overwhelmed the WS connection — observed live as a
startup crash on `extraction_result` (entities/relations multi-MB payloads).
Attempt 1 patched only `extraction_result` with a drift-only WHERE; the reviewer
correctly flagged this under-generalizes (`pass1_results`, `job`, `metrics`,
`dead_letter` payloads could crash the same way on an unscanned env, and the
migration ships explicitly as insurance for unscanned envs).

### Delivered
- `migrations/65.surrealql` — drift-only WHERE on ALL 19 UPDATE statements. Each
  statement's WHERE mirrors its SET field list exactly: one `<f> = NONE` disjunct
  per coalesced field, OR-joined, `RETURN NONE` retained. On a healthy DB every
  statement matches 0 rows → zero row rewrites anywhere → true no-op + startup-
  safe; only genuinely-drifted rows are touched. SET lists UNCHANGED (S.1/S.2
  validated). Excluded no-safe-default fields untouched. One-line header comment
  added: "Each UPDATE's WHERE mirrors its SET — drift-only ... Keep WHERE and SET
  field lists in lockstep."
- WHERE==SET verified programmatically for all 19 blocks — **zero mismatches**
  (entity 17, source 1, relation 8, chunk 6, claim 3, dead_letter 3, doc_node 1,
  entity_suggestion 7, episode_profile 3, extraction_result 6, job 5, metrics 2,
  model_route 3, pass1_results 6, preprocessing_result 1, triage_queue 6,
  transformation 1, status_change_log 1, speaker_profile 2).
- New test `test_extraction_result_drift_repaired_clean_untouched`: forges
  `extraction_result.entity_count = NONE`, asserts UPDATE fails pre-65 / lands
  post-65 (drift repaired), AND asserts a clean row (entity_count=7) is untouched
  — pins the WHERE no-op that keeps startup safe on large rows.
- Sharpened `test_migration_65_clean_row_unchanged` docstring to frame entity as
  the generalized-WHERE no-op on a multi-field block.

### Test evidence (per acceptance criterion)
- **AC1 (WHERE mirrors SET ×19)** — programmatic check above: all 19 match, no
  reconciliation needed.
- **AC2 (existing drift tests still pass)** — `test_{entity,source,transformation}_drift_fails_pre_passes_post`
  all PASS: a drifted row matches its WHERE → repaired.
- **AC3 (new extraction_result test)** — `test_extraction_result_drift_repaired_clean_untouched`
  PASS: fail-pre / pass-post + clean-row-untouched.
- **AC4 (idempotent, full 1→65 chain on fresh container)** — session fixture's
  strict transactional runner builds a fresh container and applies 1→65 clean;
  `test_migration_65_discovered_and_applied` + `test_migration_65_idempotent_double_apply`
  PASS.
- **AC5 (down no-op)** — `test_migration_65_down_is_noop` PASS.

Run: `9 passed in 7.07s` (full S.2 suite incl. new test) against a real
SurrealDB container.

### Not touched
Live staging untouched (already at v65; the file change is version-gated and only
affects future/unscanned envs). `.serena/project.yml` left as-is (pre-existing
working-tree change, not part of this phase).
