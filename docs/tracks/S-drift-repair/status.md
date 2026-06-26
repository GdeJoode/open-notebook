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

---

## Phase S.4 — Prevention rule + lockstep guard + RETRO — READY FOR REVIEW

**Branch**: `track/s4-prevention` (off `main` @ `7950ba4`, S.2/S.2b merged).
**Commits**: `1b14658` (prevention rule docs), `00b6e25` (lockstep test).

### Delivered
- **Prevention rule** — new `migrations/README.md`, placed next to the `.surql`
  files where a migration author actually looks. States the rule crisply ("when
  you add a strict non-`option<>` field to an EXISTING SCHEMAFULL table, backfill
  in the SAME migration with `UPDATE <t> SET f = f ?? <d> WHERE f = NONE RETURN
  NONE`"), the two-line why, a correct example, the do/don't (prefer `option<>`;
  never blanket-UPDATE a payload-bearing table), and cross-links to migrations
  61/64/65 and `docs/tracks/S-drift-repair/`. Cross-referenced from
  `docs/development/contributing.md` Schema Management.
- **Lockstep static test** — `packages/surrealdb-service/tests/test_migration_65_where_mirrors_set.py`.
  Pure-file parser (no DB/Docker): for each `UPDATE` block in `65.surrealql`,
  asserts `set(WHERE '= NONE' disjuncts) == set(SET 'f = f ?? default' fields)`,
  and pins the expected 19-block count. Enforces the drift-only-WHERE invariant
  the S.2b reviewer flagged so a future "added a SET field, forgot the WHERE
  disjunct" regression fails CI. **2 passed**; sanity-checked by removing one
  disjunct → red (`entity: SET-only=['weight']`) → restored → green.
- **Memory note** — `strict-field-drift-migrations.md` already carries the blanket-
  UPDATE-crashes-startup → drift-only-WHERE lesson (added in S.2b); no edit needed.

### Test evidence
- `uv run --project packages/surrealdb-service pytest packages/surrealdb-service/tests/ -q -k "migration"`
  → **68 passed, 109 deselected** (incl. the new 2 lockstep tests and the 9
  `@requires_docker` migration-65 tests against a real container). No regressions.

---

## RETROSPECTIVE — Track S (CLOSED)

**The drift class.** A SurrealDB `SCHEMAFULL` table re-validates the WHOLE record
on any `UPDATE`. A strict (non-`option<>`) field added to an existing table by a
later migration with a `DEFAULT` stays `NONE` on pre-existing rows (DEFAULT applies
only to new rows). The strict type then silently blocks ALL future writes to those
rows (`Found NONE for field X, expected a <type>`) — not just writes to that field.
It surfaced one table at a time during live writes: `entity` (mig 61, Track Q),
`source` (mig 64, R.0e).

**S.1 found ZERO live drift.** The read-only inventory against staging proved that
every strict, non-`option<>`, no-VALUE field on all 31 active tables already reads
non-NONE: the reactive 61/64 had already healed the only two historically-drifted
tables, and nothing else had accrued NONE. So the planned "self-healing repair
migration" had **0 rows to repair on staging** — it ships purely as a forward-guard
for unscanned/freshly-restored environments.

**The irony — the forward-guard nearly caused a startup crash.** Migration 65 runs
at app startup. The first cut was a blanket `UPDATE <t> SET f = f ?? d` per table.
But a SCHEMAFULL `UPDATE` re-validates AND **REWRITES** every matched row, and the
blanket form matches ALL rows — so on payload-bearing tables with multi-MB rows
(`extraction_result` entities/relations, `pass1_results`, `job`/`metrics`/
`dead_letter` payloads) it rewrote the whole table even when clean and overwhelmed
the WS connection. Observed live as a startup crash on `extraction_result`. The
guard meant to prevent the drift class nearly bricked startup itself.

**Resolution — drift-only WHERE.** S.2b put a `WHERE <f> = NONE OR ... RETURN NONE`
on all 19 `UPDATE` statements, each WHERE mirroring its SET field list exactly. On
a healthy DB every statement matches 0 rows → zero rewrites anywhere → true no-op +
startup-safe; only genuinely-drifted rows are touched. The lockstep static test
(S.4) now pins WHERE==SET so the mirror can't silently drift apart.

**The real fix is the prevention rule, not the migration.** Migration 65 repairs 0
rows on current staging; its value is consolidation + insurance for unscanned envs.
The actual recurrence-stopper is the S.4 rule documented in `migrations/README.md`:
pair every new strict field with a coalescing `UPDATE ... WHERE = NONE` in the SAME
migration (and prefer `option<>` when NONE is valid; never blanket-rewrite a
payload-bearing table). That moves the fix from reactive (one table at a time, in
production) to preventive (at authoring time, in review).

**Phases**: S.1 inventory (zero live drift) · S.2 forward-guard migration 65 ·
S.2b drift-only-WHERE generalization (startup-safe) · S.3 live verify (staging at
v65, writable) · S.4 prevention rule + lockstep guard + this retro.

**Status: Track S CLOSED.**
