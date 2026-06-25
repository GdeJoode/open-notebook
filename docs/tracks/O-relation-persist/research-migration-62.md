# O.2a research — SurrealDB v2.6.5 NORMAL→RELATION conversion

Pre-implementation research (orchestrator, Context7 + WebSearch) to de-risk the
load-bearing unknown for migration 62. The container test is the final authority;
this is the design basis the reviewer should check against.

## Environment
- SurrealDB **2.6.5** (image `surrealdb/surrealdb:v2`), python client `surrealdb>=1.0.4`.
- Migration runner: `packages/surrealdb-service/.../migrations.py` — auto-discovers
  `<int>.surrealql`, applies once per env, tracks in `_sbl_migrations`. Next free version = **62**.

## Findings
1. **Why migration 39 didn't convert the live table.** Advisory
   [GHSA-27vq-hv74-7cqp](https://github.com/surrealdb/surrealdb/security/advisories/GHSA-27vq-hv74-7cqp):
   `DEFINE TABLE OVERWRITE` *silently failed* to overwrite a `TYPE RELATION` table
   **before 2.1.4**. Migration 39 used a plain `DEFINE TABLE relation TYPE RELATION`
   (no OVERWRITE); on an env where `relation` pre-existed as NORMAL, the redefine did
   not take effect → the table stayed NORMAL → `RELATE` rejected ("expected a NORMAL").
   On 2.6.5 (≥2.1.4) OVERWRITE works, so it is the correct conversion tool now.
2. **OVERWRITE does not guarantee record preservation.** SurrealDB docs explicitly say
   "if you need to preserve data when changing a table definition, back it up to a
   temporary table"; issue #5602 shows OVERWRITE can break field types. → The healthy-DB
   **edge-preservation invariant (O.2a AC#4) must be proven by the container test**, not assumed.
3. **`ALTER TABLE` cannot change TYPE** (NORMAL↔RELATION). Only DEFINE TABLE (OVERWRITE) can.

## Recommended strategy (validate empirically; adjust if the test disproves it)
A single `.surrealql` can't branch on table kind. Make it a no-op on healthy envs by
construction:
```surql
-- 1. drop ONLY null-endpoint junk (the drifted NORMAL rows). No-op on healthy edges,
--    whose in/out are set.
DELETE relation WHERE in = NONE OR out = NONE;
-- 2. re-assert the canonical edge definition (migration 39), now with OVERWRITE so it
--    actually converts a drifted NORMAL table on 2.6.5; identical no-op on a healthy one.
DEFINE TABLE OVERWRITE relation SCHEMAFULL TYPE RELATION FROM entity TO entity;
DEFINE FIELD ... ON relation ...;   -- mirror migration 39 fields/indexes (OVERWRITE)
```
- **Healthy env**: real edges have non-null in/out → DELETE matches nothing; OVERWRITE
  re-asserts the same def → edges preserved (must verify OVERWRITE doesn't wipe).
- **Drifted env (staging)**: 3 null-endpoint junk rows deleted → table empty → OVERWRITE
  converts NORMAL→RELATION cleanly.

**If the test shows OVERWRITE wipes edges on 2.6.5**: fall back to making the conversion
conditional in `migrations.py` (read `INFO FOR TABLE relation`, only redefine when kind is
NORMAL), since healthy envs are already correct and need no touch.

## Container test must reproduce
- Force-drift: define `relation` as NORMAL with the pre-39 field shape + insert null-endpoint rows.
- Apply 62 → assert `INFO FOR TABLE relation` contains `TYPE RELATION`; junk gone; a `RELATE` lands.
- Healthy path: migrated container with N real edges → apply 62 → still N edges, in/out intact.
- Idempotency: apply 62 twice → no error, same end state.
