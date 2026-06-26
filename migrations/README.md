# Database migrations

SurrealQL migrations applied in version order by the strict transactional runner
(`packages/surrealdb-service`). Each `N.surrealql` has a matching `N_down.surrealql`.

## RULE: backfill when you add a strict field to an existing SCHEMAFULL table

> **When you add a strict (non-`option<>`) field to an EXISTING `SCHEMAFULL` table,
> you MUST, in the SAME migration, backfill existing rows:**
>
> ```surrealql
> DEFINE FIELD <field> ON <table> TYPE <strict_type> DEFAULT <default>;
> UPDATE <table> SET <field> = <field> ?? <default> WHERE <field> = NONE RETURN NONE;
> ```

**Why (two lines).** A SurrealDB `SCHEMAFULL` table re-validates the WHOLE record on
any `UPDATE`. A `DEFAULT` applies only to rows created *after* the migration, so
pre-existing rows keep `NONE` in the new strict field — and the strict type then
**silently blocks ALL future writes to those rows** (`Found NONE for field X, expected
a <type>`), not just writes to that field. The app's own updates and every pipeline
write to the affected rows fail until the field is backfilled.

### Do / don't

- **Prefer `option<>`** if `NONE` is genuinely a valid value — then no backfill is
  needed and you must NOT coalesce it (coalescing would destroy the `NONE` signal,
  e.g. `source.embedding = NONE` means "aggregate not yet computed").
- **Use a drift-only `WHERE <field> = NONE`** on the backfill `UPDATE`, and keep the
  `WHERE` disjuncts in **lockstep** with the `SET` field list (one `<f> = NONE` per
  coalesced `f`). A blanket `UPDATE <table> SET f = f ?? d` (no `WHERE`) matches every
  row and a `SCHEMAFULL` `UPDATE` re-validates **and REWRITES** each matched row — on a
  payload-bearing table with multi-MB rows (`extraction_result`, `job`, `metrics`,
  `dead_letter`, `pass1_results`) that blanket rewrite overwhelms the WS connection and
  **can crash startup**. With the drift-only `WHERE`, a healthy DB matches 0 rows →
  zero rewrites → true no-op; only genuinely-drifted rows are touched.
- **Never invent a value for a required field with no honest default** (content/text
  bodies, FKs/ids, type discriminators, edge `in`/`out`). If such a field is genuinely
  required, fix it at insert time, not with a guessed coalesce.

### Correct example

A later migration adds a strict `weight` to the existing `entity` table:

```surrealql
DEFINE FIELD weight ON entity TYPE float DEFAULT 0.0;
-- Backfill pre-existing rows in the SAME migration (drift-only WHERE):
UPDATE entity SET weight = weight ?? 0.0 WHERE weight = NONE RETURN NONE;
```

## Reference fixes & full story

This drift class bit `entity`, `source`, and several other tables one at a time as
they surfaced during live writes. The reference fixes:

- [`61.surrealql`](./61.surrealql) — reactive backfill of `entity` strict fields.
- [`64.surrealql`](./64.surrealql) — reactive backfill of `source.private`.
- [`65.surrealql`](./65.surrealql) — consolidated self-healing forward-guard: a
  drift-only-`WHERE` coalesce over every safe-default strict field, `WHERE` mirroring
  `SET` on all 19 statements. This is the canonical example of the rule above; a
  lockstep static test
  (`packages/surrealdb-service/tests/test_migration_65_where_mirrors_set.py`) enforces
  that every block's `WHERE` matches its `SET`.

Full investigation, inventory, and retrospective:
[`docs/tracks/S-drift-repair/`](../docs/tracks/S-drift-repair/).
