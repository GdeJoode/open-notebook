# Review — Track Q Phase Q.2a attempt 1

**Branch**: `track/q-relation-merge`
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-24

## Summary

Q.2a replaces the unconditional `RELATE` in `entity_persistence_service.py` with an UPSERT (`_upsert_relation`: existing-edge lookup → union, else create) plus an idempotent dedup backfill. The two load-bearing properties hold: O.1 endpoint resolution is untouched (new edges still created via the same `type::thing` record-link RELATE after `_resolve_endpoint_id` resolves), and the union is genuinely idempotent, non-clobbering, and direction/type-correct. All 6 docker-backed integration tests and the O.1 roundtrip suite are green.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | Same relation from 2 source_ids → ONE edge, `source_documents` len 2 | ✅ | `test_same_edge_two_docs_yields_one_edge_len2` (real container) |
| 2 | Re-persist same (edge, source_id) → no-op union, len stays 2 | ✅ | `test_re_persist_same_edge_same_doc_is_noop` |
| 3 | Backfill collapses seeded dups, re-run idempotent | ✅ | `test_backfill_collapses_seeded_dups_and_is_idempotent` + dry-run test |
| 4 | No non-duplicate deleted; endpoints + relation_type preserved; O.1 unaffected | ✅ | unique `VERSTERKT` edge survives; O.1 roundtrip suite 6/6 green |
| 5 | Weak RELATED across 3 docs → `source_documents` len 3 | ✅ | `test_related_edge_three_docs_len3` |

## Test status

```
apps/app-main: tests/test_relation_merge.py tests/test_entity_persistence_service.py → 61 passed in 9.12s
  test_relation_merge.py (6 requires_docker, all RAN not skipped) → 6 passed in 7.83s
packages/surrealdb-service: tests/test_relation_endpoint_resolution_roundtrip.py → 6 passed in 8.07s
create_app import OK
ruff: All checks passed!
```

## Load-bearing verification

**1. O.1 NOT regressed.** `_resolve_endpoint_id` (entity_persistence_service.py:224) and the `type::thing`/`LET $s..RELATE` binding are NOT in the diff — unchanged. `_upsert_relation` is the only new decision, invoked AFTER endpoints resolve (line 543-544 → 660). A no-existing-edge lookup still RELATEs a fresh edge (`_upsert_relation` line 345-364, returns True → `relations_created`). The existing-edge lookup (line 305-314) binds `str(src_id)`/`str(tgt_id)` — the SAME resolved ids the create path uses (line 357) — not raw names. AC1 proves it: the 2nd persist's lookup matched the just-created edge.

**2. Idempotent union + no-clobber.** Lookup matches `in = type::thing($sid) AND out = type::thing($tid) AND relation_type = $rel_type AND status = 'active'` — resolved record-link endpoints, directional (`in`/`out` distinct), type-scoped. `array::union(source_documents, [$source_id])` is a no-op on repeat (AC2). Confidence keeps `max` (line 322-327). Properties merge Python-side: existing as base, new keys overlay, existing retained (line 320-321) — `test_merge_keeps_max_confidence_and_merges_properties` asserts `sig_a` retained + `sig_b` added. Status filter is safe: migration 58 line 35 defines `status DEFAULT "active"`, so RELATE-created edges (which don't set status) get `active` and ARE matched on re-persist.

**3. Backfill** groups by `(in,out,relation_type)` (line 91-94), collapses only `len > 1` groups (line 95) — unique edges never touched (AC4). Union provenance + max-conf + no-clobber merge (line 108-120), dry-run writes nothing (line 159-170). Idempotent: post-collapse each group is singular.

## Issues found

### 🔴 Blockers
None.

### 🟡 Major
None.

### 🔵 Minor (optional follow-up)

1. **Non-deterministic survivor row identity in backfill** — `scripts/backfill_relation_merge.py:105`. `_load_active_relations` SELECTs without `ORDER BY`, so `edges[0]` (the survivor) is not a deterministic row across runs. The DATA outcome is identical (provenance unioned, idempotent after first run), so this is cosmetic — but if a deterministic survivor id is wanted (e.g. for audit logs), add `ORDER BY id` to the load query.
2. **Directionality not directly asserted in tests** — `apps/app-main/tests/test_relation_merge.py`. The lookup is directional by construction (`in`/`out`), but no test asserts `(A->B)` stays distinct from `(B->A)`. Construction is correct; a one-line assertion would lock the guarantee against future edits.
3. **Archived-edge edge case** — `entity_persistence_service.py:308`. The lookup filters `status = 'active'`, so re-persisting an edge whose only prior row was archived/merged creates a NEW active edge rather than reviving the archived one. This is arguably correct (archived edges should not accumulate provenance) and produces no duplicate among ACTIVE edges, but is undocumented behavior.

## Decision rationale

0 blockers + 0 majors → APPROVED. Both load-bearing properties verified: O.1 not regressed (endpoint resolution untouched in the diff; new edges still created; roundtrip suite green) AND the union is idempotent + non-clobbering + direction/type-correct (proven by 6 real-container tests + a focused unit test). Scope clean: no migration/schema/hash_id/EntityRepository changes. The three minors are cosmetic/hardening and do not block merge.

## Next steps

APPROVED — ready for human approval / merge. Minors can be filed as optional follow-up (most cheaply: a directionality assertion and an `ORDER BY id` on the backfill load).
