# Track S — systematic strict-field drift repair

> **Planned 2026-06-26, to run AFTER Track R.0 closes.** Eliminates the systematic
> SCHEMAFULL strict-field drift across the staging DB that has surfaced one table at a
> time during O/P/R live writes.

## Problem (the recurring pattern)
SurrealDB SCHEMAFULL tables re-validate the WHOLE record on any UPDATE. When a strict
(non-`option<>`) field is added by a later migration with a `DEFAULT`, that default only
applies to NEWLY created rows — rows that predate the migration keep `NONE`. The strict
type then rejects the row on the first UPDATE (`Found NONE for field X, expected a <type>`),
silently blocking ALL writes to that row (app + scripts).

Confirmed instances so far (each fixed reactively, one table at a time):
- `entity` → migration 61 (Track Q) — many strict fields coalesced.
- `relation` → migration 62 (Track O) — table-KIND drift (NORMAL→RELATION), related but distinct.
- `source.private` → migration 64 (R.0e) — blocked the source aggregate embedding.

Memory: `strict-field-drift-migrations.md` (find/fix/test recipe + likely-affected tables:
`notebook`, `note`, `model`, settings singletons, `chunk`, `source_embedding`, …).

**Goal**: one idempotent, self-healing sweep that repairs EVERY drifted strict field on
EVERY table, so no future live write hits this class of error again — and so any drifted
environment self-heals on migrate.

**Workflow**: track methodology — `implementer` → `adversarial-reviewer` (≤3 → escalation).
Main tree, `uv run pytest`, no worktree. Live apply = gated checkpoint. Strict transactional
runner (O.2a) in effect.

---

## Phase S.1 — Drift inventory (Discovery, read-only)
**Why**: know the full surface before writing the repair; reactive one-table-at-a-time fixing is the bug.
**Deliverables**: a generated inventory (committed under `docs/tracks/S-drift-repair/`) that, for
every table, lists each non-`option<>` strict field (type + DEFAULT) AND, measured against the LIVE
`staging` DB (read-only), which of those are `NONE` on existing rows (count per field).
**How**: parse `migrations/*.surrealql` for all `DEFINE FIELD ... ON <table>` (skip `option<...>`,
skip computed `VALUE` fields); cross-check each against `INFO FOR TABLE`; probe NONE counts per row set.
**Acceptance**
1. Inventory covers ALL user tables (not just entity/source); each entry: table, field, strict type, default, live-NONE-count.
2. Distinguishes genuinely NONE-rejecting strict fields from `option<>`/computed (which are fine).
3. Identifies which tables currently have writable-blocking drift on live staging.
**Branch**: `track/s1-drift-inventory`. **Depends on**: none.

## Phase S.2 — Self-healing repair migration (Backend)
**Why**: fix every drifted strict field in one idempotent migration; supersede the reactive per-table fixes.
**Deliverables**: a migration (next free version) that, per affected table, `UPDATE <table> SET f = f ?? <default>, ...`
for every strict field from the S.1 inventory (mirror migrations 61/64). Down = documented no-op.
**Acceptance**
1. For every table in the inventory, a `@requires_docker` test reproduces the drift (NONE strict field) and asserts a subsequent `UPDATE <table> SET <something>` FAILS pre-migration and PASSES post-migration.
2. Idempotent: double-apply no-op; clean rows unchanged.
3. Full migration chain applies clean under the strict transactional runner on a fresh container.
4. Covers entity + source too (idempotent overlap with 61/64 — re-running their coalesce is a no-op).
5. Down present + sane.
**Branch**: `track/s2-drift-repair-migration`. **Depends on**: S.1.

## Phase S.3 — Live apply + writability verification (Integration · LIVE CHECKPOINT)
**Why**: prove staging is fully writable; no table left with NONE-strict drift.
**Procedure**: snapshot affected row counts → apply the migration to staging (strict runner) → verify.
**Acceptance**
1. Migration applied to staging; version advanced.
2. For each previously-drifted table, an UPDATE on an existing row now succeeds (probe; read-only-then-one-write per table).
3. Re-running the S.1 inventory against staging reports ZERO NONE-rejecting strict fields on existing rows.
4. No row counts changed; only NONE→default coalesced (no data loss).
**Branch**: `track/s3-drift-live-verify`. **Depends on**: S.2.

## Phase S.4 — Prevention note + RETRO (docs)
**Why**: stop the class from recurring.
**Deliverables**: a short ARCHITECTURE/CONTRIBUTING note: "when adding a strict field to an existing
SCHEMAFULL table, ALWAYS pair the `DEFINE FIELD` with a coalescing `UPDATE <table> SET f = f ?? default`
in the SAME migration" (the root-cause prevention); update `strict-field-drift-migrations.md`; RETRO.
**Acceptance**: prevention rule documented where migration authors will see it; Track S status CLOSED.
**Depends on**: S.3.
