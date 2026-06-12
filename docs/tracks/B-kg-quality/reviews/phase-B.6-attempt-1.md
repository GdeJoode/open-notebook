# Review — Track B Phase B.6 attempt 1

**Branch**: `track/b-notebook-merge` (HEAD `65e09d0`) (PR #25)
**Decision**: **REVISIONS_NEEDED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-12

## Summary

Implementation well-structured; API + dialog wiring clean; 6 unit tests + Playwright pass. **BUT AC #2 (idempotency) structurally broken in production** — merge service relies on `updated_at` mismatch to count writes, but `EntityRepository.upsert_entity` unconditionally runs `UPDATE … SET updated_at = time::now()` on every call. Unit test "passes" only because mock fakes static post-write `updated_at` that does NOT match production semantics. Second concern: missing guard against `target_id ∈ source_ids` at API boundary.

## Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| 1 | Synchronous merge endpoint | PASS |
| 2 | Idempotent re-run → 0 entities, 0 relations | **FAIL in production** (relations idempotent; entities counter inflates) |
| 3 | OECD union type_tags | PASS |
| 4 | Relations dedup max-confidence | PASS |
| 5 | UI dialog multi-select + dry-run | PASS |
| 6 | Playwright happy path | PASS |

## Blockers (2)

### B1: Idempotency check uses `updated_at` mismatch (broken in production)

`notebook_merge_service.py:367-385` decides `entities_changed += 1` by comparing `existing_before.updated_at` vs `existing_after.updated_at`. Production `upsert_entity` (entity.py:226-240) always executes `SET …, updated_at = time::now()`. Result: second merge counts EVERY contributed entity as `entities_merged += 1`.

Unit test passes only because mock writes SAME `"2026-06-12T00:00:00Z"` to before+after. Production diverges: `time::now()` never equals previous.

**Impact**: AC #2 fails in production. Dialog promises "Re-running is a no-op"; operator sees inflated counts on re-run.

**Fix recommendation**: change idempotency signal from timestamp to semantic content (compare merged-fields vs existing row pre-write; skip-counter when set-equal). Don't mutate the timestamp-based mock.

### B2: No guard against `target_id ∈ source_ids` at API boundary

`notebooks.py:209-244`. HTTP caller can POST `{source_ids: ["notebook:n1"], target_id: "notebook:n1"}` → service re-merges entities into themselves → phantom writes + counter inflation (compounds B1).

UI dialog excludes target from picker; API has no equivalent server-side check.

**Fix**: reject at router/service entrypoint with 422.

## Major (2)

### M3: `_find_target_entity` has NO notebook-scope filter; docstring claims it does

`notebook_merge_service.py:619-645`. Docstring says "Look up entity in the target notebook". SQL is `SELECT … FROM entity WHERE canonical_name = $... AND entity_type = $... LIMIT 1` — no notebook filter. `notebook_id` parameter unused.

Functionally tolerable (entity rows globally keyed by `(canonical_name, entity_type)` per migration 39) but documented contract WRONG. Compounds B1 — reasoning about idempotency requires correct understanding of probe.

**Fix**: rename `_find_canonical_entity` + update docstring, OR add `WHERE source_documents ANYINSIDE $target_source_ids` filter.

### M4: Test mock diverges silently from `upsert_entity` production contract

`test_notebook_merge_service.py:192-204` — mock `_upsert` sets `target_after_write[k] = {**before, "updated_at": "2026-06-12T00:00:00Z"}`. Violates production contract that `upsert_entity` ALWAYS advances `updated_at`. Faithful mock would write strictly-greater timestamp → `test_idempotent_re_run` would fail and surface B1.

**Impact**: false assurance of idempotency. B.7's full E2E validation would be first place bug surfaces.

**Fix**: align mock to advance timestamp; rewrite idempotency test to invariant on entity row semantic content, not `updated_at`.

## Minors (5)

1. `_relation_exists` + `_create_relation` = 2 round-trips per relation (extra cost)
2. `_find_target_entity` runs twice per existing entity (extra round-trip)
3. API accepts archived notebooks as source/target without validation
4. `MergeNotebookDialog` has no unit/component tests (Playwright happy-path only)
5. Conflict UX deferred to future-B/Track M4 (self-flagged, acceptable)

## Tests (independently verified)

- `apps/app-main`: 500 passed
- `packages/shared`: 154 passed
- 6 notebook-merge service tests pass (but B1 + M4 mean false assurance)
- Frontend tsc + lint clean
- Playwright 1/1 pass on port 8503

## Kudos

- Clean separation: aggregation → conflict detection → writes (3 phases with docstrings explaining WHY)
- `normalize_entity_name` REUSED from shared.utils (single import-point, honors Q-B-4/Q-B-7)
- Q-B-5 `type_match_required=True` default — both branches pin-tested
- **Relation idempotency CORRECTLY implemented** via existence probe + IF-block (entity path should mirror this)
- B.4 telemetry call wired correctly
- Dialog UX: dry-run before commit, amber conflict rows, Switch+Label htmlFor, Radix focus trap
- `_tags_are_disjoint` correctly skips conflict check for single-contributor cases

## Decision rationale

Two blockers + two majors preclude approval. User explicitly requested HIGH quality bar; this is LAST feature phase. Letting broken AC #2 through means B.7's integration validation operates on false assurance. Tests passing necessary but not sufficient when mock diverges from production in the exact dimension under test.

## Next steps

1. Fix B1 — change idempotency signal away from `updated_at` to semantic content comparison
2. Fix B2 — reject `target_id ∈ source_ids` at router/service entrypoint with 422
3. Fix M3 — rename `_find_target_entity` and update docstring (or add notebook filter)
4. Fix M4 — align mock `_upsert` to advance timestamp; rewrite idempotency test to invariant on semantic content
5. Re-submit
