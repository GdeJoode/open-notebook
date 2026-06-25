# Track O — sprint plan (O.2 completion)

> Resumes the O.1 escalation (`escalations.md`). O.1 code is merged + container-proven;
> the live `relation` table is a pre-migration-39 `NORMAL` table, so `RELATE` is
> rejected categorically. O.2 closes the blocker via escalation **Option A**: an
> idempotent, self-healing remediation migration, then a gated live apply.

**Workflow**: track methodology — `implementer` → `adversarial-reviewer` (≤3 attempts →
`escalation-handler`). Run in the **main tree** (`uv run pytest`), no worktree (shared `.venv`).
Live writes are **authorized with a checkpoint**: container-prove first, show dry-run +
backup, single confirm at the live-apply boundary.

---

## Phase O.2a — Self-healing relation-table remediation migration (Backend)

**Why**: convert any environment whose `relation` table drifted to `NORMAL` back to
`TYPE RELATION FROM entity TO entity` (migration 39's definition) **without destroying
real edges on already-healthy environments**.

**Deliverables**
- `migrations/62.surrealql` — idempotent remediation.
- `migrations/62_down.surrealql` — documented no-op (data/schema repair has no faithful inverse; mirror migration 61's down convention).
- `apps/app-main/tests/test_migration_62_relation_remediation.py` (or `packages/surrealdb-service/tests/`) — `@requires_docker` test covering the four invariants below.

**Top risk (load-bearing)**: SurrealDB v2 cannot necessarily convert a populated/NORMAL
table to `TYPE RELATION` in place. If the migration must `REMOVE TABLE` + redefine, a naive
version would silently destroy edges on a healthy DB when applied elsewhere (other devs / prod).
The migration MUST be non-destructive on healthy tables. Research `DEFINE TABLE OVERWRITE ...
TYPE RELATION` semantics (Context7 + WebSearch) before writing.

**Acceptance criteria** (all yes/no)
1. Runner auto-discovers version 62; `62.surrealql` applies clean on a fresh migrated container.
2. Drifted case: a `NORMAL` `relation` table (old pre-39 field shape + null-endpoint junk rows) → after 62, the table kind is `TYPE RELATION`. NOTE: on SurrealDB 2.6.5 the table kind is exposed via `INFO FOR DB` (under `tables.relation`), **not** `INFO FOR TABLE` (which omits the kind) — the test must read the authoritative `INFO FOR DB` source.
3. After 62 on the drifted DB, a `RELATE $a->relation->$b` between two entities succeeds (`count() FROM relation` increases by 1).
4. **Safety invariant**: a healthy `TYPE RELATION` table holding N real edges → after 62, still exactly N edges, each edge's `in`/`out` intact (no loss).
5. Idempotent: applying 62 twice yields the same end state with no error.
6. `62_down.surrealql` present + sane.
7. The 6 existing O.1 `@requires_docker` relation-roundtrip tests stay green.

**Mandatory commands**: `uv run --project apps/app-main pytest -k "migration_62 or relation_roundtrip"` (+ targeted surrealdb-service suite).

**Evidence**: test output showing invariants 2/3/4/5; `INFO FOR TABLE relation` before/after.

**Branch**: `track/o2-relation-remediation`. **Depends on**: none (O.1 merged).

---

## Phase O.2b — Live staging apply + relation re-verify (Integration · LIVE CHECKPOINT)

**Why**: prove the end-to-end claim on the real corpus — relations actually persist after
the schema is repaired.

**Deliverables**
- A backup artifact: `INFO FOR TABLE relation` + `SELECT * FROM relation` (the 3 rows) captured to a file before any mutation.
- A short runbook/evidence doc under `docs/tracks/O-relation-persist/` (before/after counts).

**Procedure (orchestrator-run, not blind-delegated)**
1. Snapshot pre-state + write backup. Show you the dry-run plan and backup. **← single confirm.**
2. Apply migration 62 to `SURREAL_DATABASE=staging` via the app migration runner.
3. Replay the stored O.1 persist for `source:052dtl7jrwu1czlpnui4`.

**Acceptance criteria**
1. Backup of pre-state written before any mutation.
2. Post-migration `relation` on staging is `TYPE RELATION`.
3. Replay creates ≥113 edges (`count() FROM relation`: 3 → ≥113), matching the in-process O.1 measurement.
4. Entity data unchanged: `count() FROM entity` identical before/after (B.8 invariant).

**Evidence**: before count (3), after count, sample of 3 created edges, entity-count parity.

**Branch**: `track/o2-live-verify`. **Depends on**: O.2a (APPROVED + merged).
