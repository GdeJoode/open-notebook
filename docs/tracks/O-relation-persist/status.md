# Track O — status

## O.2a — non-destructive relation-table remediation migration (2026-06-25)
**State**: COMPLETE + container-proven (review attempt 2 applied). Ready for
re-review. Live apply is O.2b (gated).

**Branch**: `track/o2-relation-remediation` (off `main`).

### Review attempt 2 — blockers/major addressed
- **Blocker 1 (58 destroys edges before 62 runs)**: rewrote `migrations/58.surrealql`
  from `REMOVE TABLE IF EXISTS relation` to the SAME non-destructive
  DELETE-junk + `DEFINE TABLE OVERWRITE` approach 62 uses. Reconciled
  `58_down.surrealql` to a documented no-op (its old down was itself a
  destructive `REMOVE TABLE`). Rationale for editing an applied migration:
  envs at version >= 58 have it recorded in `_sbl_migrations` and will NOT
  re-run it (version gate), so the edit cannot change their state — for them 62
  re-asserts the correct definition anyway; only envs still at <= 57 execute 58,
  and those are exactly the population we must protect. Documented in the 58
  header.
- **Blocker 2 (AC4 test didn't exercise the failing path)**: added
  `test_runner_sequence_58_then_62_preserves_edges` — seeds N=7 real edges at a
  simulated v57 healthy state and drives the ACTUAL `AsyncMigrationRunner` over
  the on-disk 58 AND 62 bodies. **Verified it FAILS against the original
  destructive 58 (7 seeded -> 0 survived) and PASSES against the non-destructive
  58 (7 -> 7, in/out intact).**
- **Major 3 (runner swallows non-first-statement errors)**: the runner now
  applies migration bodies through `execute_transaction` (per-statement status
  inspection via `_check_transaction_response`) instead of `execute_query`
  (which returns only statement 1 and swallows later errors). A failed
  non-first statement now raises and the version is NOT bumped. Added
  `test_runner_surfaces_failing_second_statement`. Side effect: this surfaced a
  latent always-failing statement in `migrations/7.surrealql`
  (`remove table speaker_profile` on a fresh DB) that was silently swallowed
  before — fixed to `remove table if exists`. Full 62-migration chain applies
  clean under strict checking; only that one historical migration had a latent
  silent failure. Updated the 4 affected runner unit tests in `test_migrations.py`
  to patch `execute_transaction`.
- **Minor 4**: AC2 plan text says "INFO FOR TABLE ... TYPE RELATION" but on
  2.6.5 the kind is only in `INFO FOR DB` (see finding below); the test reads
  `INFO FOR DB`. Noted so the plan text can be reconciled separately.

### What shipped
- `migrations/62.surrealql` — self-healing remediation:
  1. `DELETE relation WHERE in = NONE OR out = NONE` (drops only null-endpoint
     legacy junk; no-op on healthy edges).
  2. `DEFINE TABLE OVERWRITE relation SCHEMAFULL TYPE RELATION FROM entity TO entity`.
  3. Re-assert migration-39 fields/indexes with `OVERWRITE`.
- `migrations/62_down.surrealql` — documented no-op (mirrors migration 61 down).
- `packages/surrealdb-service/tests/test_migration_62_relation_remediation.py`
  — 4 `@requires_docker` tests, one per acceptance criterion.

### Load-bearing empirical finding (SurrealDB 2.6.5)
**`DEFINE TABLE OVERWRITE ... TYPE RELATION` PRESERVES edge records on a healthy
TYPE RELATION table.** AC4 test seeds N=5 real edges, applies 62, and asserts
exactly 5 survive with in/out intact — passes. OVERWRITE rewrites the table
*definition*, not its records, so the chosen strategy is non-destructive on
healthy environments (unlike migration 58's `REMOVE TABLE`). No conditional
INFO-gated fallback was needed.

Secondary finding: on 2.6.5 `INFO FOR TABLE relation` does NOT expose the table
*kind*; the `TYPE RELATION` clause lives in `INFO FOR DB` under
`tables.relation` (e.g. `DEFINE TABLE relation TYPE RELATION IN entity OUT entity
SCHEMAFULL`). The test reads the kind from `INFO FOR DB` (authoritative); the AC
wording "INFO FOR TABLE output contains TYPE RELATION" is a false negative on
this version.

### Per-criterion evidence (all PASS)
- AC1 discovered+applied: `test_migration_62_discovered_and_applied`.
- AC2/AC3 drift→convert→RELATE lands: `test_drifted_relation_converted`.
- AC4 safety invariant (N edges preserved, 62 alone): `test_healthy_edges_preserved`.
- AC4 end-to-end (N edges preserved across 58->62 runner sequence):
  `test_runner_sequence_58_then_62_preserves_edges`.
- AC5 idempotent: `test_migration_62_idempotent`.
- AC6 down present + sane: `migrations/62_down.surrealql`.
- Runner no-silent-swallow: `test_runner_surfaces_failing_second_statement`.
- AC7 no regressions: roundtrip 14/14, migration unit 17/17, entity-persistence +
  relation-merge 61/61.

### Tests (attempt 2)
- `packages/surrealdb-service/tests/test_migration_62_relation_remediation.py` — 6 passed.
- `packages/surrealdb-service/tests/test_migrations_roundtrip.py` — 14 passed.
- both files together (ordering safety) — 20 passed.
- `packages/surrealdb-service/tests/test_migrations.py` (runner units) — 17 passed.
- `apps/app-main/tests/test_entity_persistence_service.py` + `test_relation_merge.py` — 61 passed.

### Commits
- `783a03a feat(migrations): non-destructive relation-table remediation (migration 62)`
- `5f812d1 test(migrations): container proof for migration 62 relation remediation`
- attempt-2 commits appended (58 non-destructive, runner strict seam, 7 fix,
  sequence + swallow tests) — see branch log.

### Note for O.2b
Both migration 58 (now non-destructive) and 62 are safe on healthy DBs. On the
live staging DB the drifted table has 3 null-endpoint legacy rows — 58/62 delete
those and convert the kind, then the O.1 persist replay lands the edges. No real
edge data is at risk on any environment.

---

## O.1 — relation persistence: type + name endpoint resolution (2026-06-23)
**State**: code fix COMPLETE + tested; live re-verification BLOCKED on a live-DB
schema-drift remediation (see `escalations.md` → O.1). Ready for review.

**Branch**: `track/o-relation-persist`

### Diagnosis (instrumented, measured on live data)
Live replay of the stored extraction for `source:052dtl7jrwu1czlpnui4`
(150 relations) through the real endpoint-resolution:
- OLD (alias-typed, no fallback): **44 resolved / 106 skipped**.
- Per missing endpoint: **95 TYPE-only misses** (entity exists under a DIFFERENT
  type than the alias-only relation side computed — e.g. relation `concept`/
  `programme` vs entity bridge type) + **38 genuine NAME misses** (surface form
  never persisted as an entity).
- Conclusion: BOTH causes, dominated by type.

### Fix
1. Type the relation endpoint the SAME way the entity was — through the L.1
   bridge (`_resolve_entity_type`), not the alias-only `_normalize_entity_type`.
   Carried `source_type`/`target_type` (per-edge homograph disambiguator) wins,
   else the bridge-resolved batch map; both bridge-resolved so the typed lookup
   matches the persisted row.
2. `_resolve_endpoint_id`: typed lookup first (K.7a homograph safety), name-only
   `LIMIT 1` fallback on a typed miss so an edge is never silently dropped.
3. RELATE binds the resolved id back to a record link
   (`LET $s = type::thing($sid)`), since `SELECT VALUE id` yields a string and
   RELATE rejects a bare-string in/out.

NEW (bridge-typed + name-only fallback) on the same doc: **113 resolved / 37
skipped** — all 37 residual are genuine name misses (out of O.1 scope).

### Re-verification (BLOCKED)
In-process replay of the fixed persist against live staging reports
`relations_created: 124`, but `SELECT count() FROM relation` stays at 3: the
live `relation` table is a pre-migration-39 NORMAL table (not
`TYPE RELATION`), so SurrealDB rejects every `RELATE`
(`Found record ... which is not a relation, but expected a NORMAL`). This is a
live-DB schema-drift blocker independent of the O.1 code; remediating it touches
the frozen `relation` table → escalated for sign-off (see `escalations.md`).

The code itself is proven by 6 `@requires_docker` roundtrip tests on a freshly
migrated container (the table is correctly `TYPE RELATION` there) — including a
bridge-only-type endpoint creating its edge and a typed-miss name-only fallback
creating its edge.

### Tests
- `apps/app-main/tests/test_entity_persistence_service.py` — 52 passed.
- `apps/app-main/tests/test_notebook_merge_service.py` — passed (61 together).
- `packages/surrealdb-service/tests/test_relation_endpoint_resolution_roundtrip.py`
  — 6 passed (2 new O.1 cases).
- `from app_main.api.app import create_app` — import OK.
- ruff: all changed files clean.

### Commits
- `diag(relation): instrument endpoint resolution to measure skip causes (O.1)`
- `fix(relation): bridge-type endpoints + name-only fallback so edges persist (O.1)`
- `fix(relation): RELATE via record-link + carried-type precedence + log level (O.1)`
- `test(relation): cover bridge-only-type endpoint + name-only fallback (O.1)`

### Files
- `apps/app-main/src/app_main/services/entity_persistence_service.py` (fix)
- `apps/app-main/tests/test_entity_persistence_service.py` (tests)
- `packages/surrealdb-service/tests/test_relation_endpoint_resolution_roundtrip.py` (tests)
- `scripts/diag_relation_persist.py` (read-only diagnostic)
