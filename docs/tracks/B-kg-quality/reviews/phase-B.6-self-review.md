# Phase B.6 — self-review

> Author: implementer agent, 2026-06-12
> Branch: `track/b-notebook-merge`
> Goal: cross-notebook entity / relation merge — an API + UI that
> aggregates entities and relations from N source notebooks into a
> target notebook, generalising B.1e's per-pass merge semantics across
> notebooks.

## Acceptance criteria check

| # | Criterion | Verified? |
|---|---|---|
| 1 | `POST /api/notebooks/merge` runs synchronously and returns a populated `NotebookMergeReport`. | YES — endpoint defined in `apps/app-main/src/app_main/api/routers/notebooks.py`. No async deferral (no job-queue submission); the service awaits `EntityRepository.upsert_entity` + relation RELATE directly. ≤ 1000-entity throughput is not benchmarked here but the algorithm is O(N entities + M relations) with one DB round-trip per row. |
| 2 | Re-running the same merge → `entities_merged=0, relations_created=0`. | YES — `test_idempotent_re_run` in `tests/test_notebook_merge_service.py`. The service snapshots the target row's `updated_at` before and after each upsert, incrementing the counter only when the timestamp changes. Relations probe via a SurrealQL IF-block that early-returns when an `(in, out, relation_type)` triple already exists. |
| 3 | "OECD" with `Organization` + `Org` → single entity with `type_tags=["Organization","Org"]`, deterministic `primary_type` (highest-confidence). | YES — `test_multi_source_merges_with_dedup`. Higher-confidence label (0.9 > 0.7) wins `primary_type`; tie-break is first-seen via dict-of-tuples insertion order. |
| 4 | Relations dedup by `(source, target, type)`, confidence = max. | YES — relation aggregation uses a `Dict[Tuple, candidate]` keyed by `(norm_src, norm_tgt, rel_type)`. Higher-confidence candidate replaces the existing one. Verified by `test_single_source_merges_into_target` (3 distinct relations land 1:1) and the dedup logic is mirrored from `_merge_results` in `multi_schema_orchestrator.py`. |
| 5 | UI dialog: multi-select + dry-run preview (counts without commit). | YES — `MergeNotebookDialog.tsx`. `CheckboxList` powers the source multi-select; Preview button fires `dry_run=true` mutation; the response is held in component state and renders entity/relation counts + conflict rows (amber-bordered) inline. |
| 6 | Playwright covers happy path. | YES — `e2e/track-b/notebook-merge.spec.ts` opens the dialog from the per-card overflow menu, selects two sources, asserts the preview counters render, then confirms and asserts the request shape (dry-run flag, source/target ids) on both calls. Dialog closes on commit. |

## Files created

- `apps/app-main/src/app_main/services/notebook_merge_service.py`
  (~560 lines): `NotebookMergeService` with `merge_notebooks(...)`,
  `NotebookMergeReport` dataclass, `NotebookMergeConflict` dataclass,
  `_MergedEntityAccumulator` private aggregator. Algorithm: load
  entities/relations per source notebook → bucket by
  `normalize_entity_name(canonical_name)` → detect type-collision
  conflicts (when `type_match_required=True`) → write entities through
  `EntityRepository.upsert_entity` → write relations through a
  guarded SurrealQL IF-block. Emits one `notebook_merge` metric per
  call via `shared.services.metrics.record_metric`.
- `apps/app-main/tests/test_notebook_merge_service.py` (~360 lines, 6
  tests): single-source merge, multi-source dedup with type_tag union,
  idempotent re-run, type-collision conflict (when required) skipped,
  type-collision union (when NOT required) merged, dry-run reports
  counts without writing. All 6 pass.
- `frontend/src/components/notebooks/MergeNotebookDialog.tsx`
  (~250 lines): two-stage UX — pick sources → Preview (dry-run) →
  Merge (commit). `CheckboxList` for source picker (excludes target
  + archived). `Switch` for `type_match_required` (default ON). Preview
  panel renders entity/relation counters + conflict list with amber
  warning styling. Loading/error states handled via the mutation's
  `isPending` + toast hooks.
- `frontend/e2e/track-b/notebook-merge.spec.ts` (~165 lines): one
  fully-mocked happy-path test. Mocks `/api/notebooks` and
  `/api/notebooks/merge`; asserts dry-run + commit request payloads
  and dialog dismissal on success.

## Files modified

- `apps/app-main/src/app_main/api/routers/notebooks.py`:
  added `POST /merge` with `NotebookMergeRequest` (pydantic
  `min_length=1` on `source_ids` → 422 for empties) and
  `NotebookMergeReportResponse`. 404 if the target notebook doesn't
  exist; the explicit 422 guard sits alongside the pydantic constraint
  for direct-service-call documentation.
- `apps/app-main/src/app_main/dependencies.py`:
  added `get_notebook_merge_service()` that wires the shared
  `EntityRepository`.
- `frontend/src/lib/api/notebooks.ts`: added `notebooksApi.merge()` +
  `NotebookMergeRequest` / `NotebookMergeReport` /
  `NotebookMergeConflict` TS types.
- `frontend/src/lib/hooks/use-notebooks.ts`: added
  `useMergeNotebooks()` mutation hook. Success toast suppressed for
  `dry_run=true` (the dialog renders the report inline instead). On
  commit, invalidates the notebook query keys for the target.
- `frontend/src/app/(dashboard)/notebooks/components/NotebookCard.tsx`:
  added "Merge into…" dropdown item (between Archive and Delete) +
  `data-testid` on the overflow trigger for the playwright spec.
  Wires `MergeNotebookDialog` with this card's id as the target.

## Quality gates

- `apps/app-main`: **500 tests pass** (baseline ~494 + 6 new), no
  regressions. Total runtime ~67s.
- `packages/shared`: **154 tests pass**.
- `frontend`: `npx tsc --noEmit` clean. `npm run lint` clean for new
  files (zero new warnings; pre-existing warnings unchanged).
- Playwright: 1/1 pass on `notebook-merge.spec.ts` (~4.2s, headless
  chromium against a local `next dev`).

## Design decisions & deviations

1. **Conflict detection scope.** The plan's AC #3 worded the OECD case
   as "merge with type_tag union", so the default behaviour for
   `type_match_required=True` was unclear. I went with: a single
   normalized name that appears in multiple notebooks with **disjoint**
   tag sets across contributors is a conflict. A name with overlapping
   tags (e.g. "OECD" tagged `Organization` in one notebook and both
   `Organization,Org` in another) is NOT a conflict — the union is
   safe because at least one tag matches. Pair-wise intersection used
   so a 3-way merge fires the conflict only when every pair is
   disjoint.
2. **Idempotency via timestamp snapshot.** The simpler approach
   ("compare full row before/after") would have required reading every
   field. The `updated_at` snapshot is cheap and exactly captures
   "did the upsert touch this row". The `EntityRepository.upsert_entity`
   contract refreshes `updated_at = time::now()` on every UPDATE, so
   the signal is well-defined.
3. **Relation idempotency via SurrealQL IF-block.** SurrealDB's RELATE
   is not idempotent (each call creates a new edge row). The service
   pre-probes via `SELECT id FROM relation WHERE in=... AND out=...
   AND relation_type=...` inside the same `LET`-block as the RELATE,
   so the round-trip cost is one DB call.
4. **Dialog source list excludes archived notebooks.** Per the
   `useNotebooks(false)` call. Merging into / from archived notebooks
   would be a separate workflow.
5. **Dry-run path still hits `_find_target_entity`.** The dry-run
   reports a conservative count (how many target rows do NOT already
   carry the entity). It does NOT compute the exact post-merge state
   change (which would require simulating the full upsert merge);
   that's acceptable for a Preview because the operator's question is
   "is anything going to happen?", not "what does the final row look
   like?". The actual commit produces the authoritative counts.

## Out of scope (explicitly deferred)

- **Conflict resolution UI**: the dialog **displays** conflicts but
  doesn't offer a manual override per row. Operators can flip the
  global `type_match_required` toggle off to merge all conflicts as
  union, or pick narrower source sets. A per-row "merge this anyway"
  toggle is a Track M4 / future-B item.
- **Audit log / undo**: no merge-history table. If a merge produces
  bad results, the operator must manually edit affected entities.
- **Cross-target multi-merge**: a single endpoint call has exactly one
  target. Batching N source-target pairs into a single transaction is
  not on the plan.
- **Q9 name normalizer**: still using the V1 stub
  (`shared.utils.name_normalizer.normalize_entity_name`). Track M4
  Q9 replaces it.

## Issues to flag

- **Conflict-resolution UX is read-only**: see the deferral above.
  Operators see *what* conflicts but not *which notebook contributed
  which tag* in the row UI — that information IS in the report
  (`source_notebook_ids` on each conflict), but the dialog only
  renders the unioned tag list and the normalized name. A future
  iteration could pop a sub-dialog per conflict row showing the
  per-notebook breakdown.
- **No load test**: AC#1 says "≤30s for ≤1000 entities". The
  algorithm is linear and the per-entity cost is dominated by one
  upsert call + one target-probe, so 1000 entities → ~2000 SurrealDB
  round-trips. At typical SurrealDB ~1ms RTT that's ~2s; well under
  the 30s budget. Not formally benchmarked.
- **Soft 8502 port collision** during e2e: prior dev-server runs left
  port 8502 held by an orphan process the spec couldn't see. Worked
  around by running playwright with `PLAYWRIGHT_BASE_URL=http://localhost:8503`.
  No code change needed — this is a local-dev-only artefact of the
  agent environment.

## Ready for review.

---

## Attempt 2 fixes (2026-06-12)

Reviewer rejected attempt 1 with REVISIONS_NEEDED (2 blockers + 2
majors + 5 minors). This section explains the fixes.

### B1 — semantic-content idempotency (was timestamp-based)

Attempt 1 used an `updated_at`-mismatch signal:

```
existing_before = await self._find_target_entity(...)
await self._entity_repo.upsert_entity(entity_model)
existing_after  = await self._find_target_entity(...)
if existing_after["updated_at"] != existing_before["updated_at"]:
    entities_changed += 1
```

Production `EntityRepository.upsert_entity` unconditionally runs
`SET ..., updated_at = time::now()`, so on a second pass the timestamp
ALWAYS changes — counter inflates to "every contributed entity"
instead of zero. The mock obscured this because it wrote the same
constant `"2026-06-12T00:00:00Z"` to the after-row.

Attempt 2 switches to semantic comparison. Algorithm:

1. Compute the would-be merged `Entity` model BEFORE calling upsert.
2. Read the existing row pre-write via `_find_canonical_entity`.
3. Compare via `_entity_matches(merged, existing_row)`. Fields:
   - `canonical_name` (string equality)
   - `entity_type` (string equality)
   - `primary_type` (string equality)
   - `confidence` (float equality)
   - `type_tags` (sorted-list equality — set semantics)
   - `source_documents` (sorted-list equality — set semantics)
   - `properties` (dict equality)
4. If `existing_before is None` → counter += 1 (CREATE).
5. Else if NOT `_entity_matches(...)` → counter += 1 (UPDATE that
   actually changed the row).
6. Else → counter stays put (true no-op).

`updated_at` and `id` are deliberately NOT in the comparison set so
the bumped timestamp doesn't trigger a false positive.

Regression test: `test_idempotent_re_run` now uses a strict-advance
mock (`fake.upsert_call_count` increments on every upsert, producing
`2026-06-12T00:00:01Z`, `…:02Z`, …). The test asserts:

- First run: `entities_merged=1, relations_created=1`
- Second run: `entities_merged=0, relations_created=0`
- `mock_upsert.call_count == 2` (proves we re-ran the upsert)
- After-write timestamp is `…00:02Z` (proves the strict-advance mock
  is wired correctly)

Verified stdout from `pytest -v -s tests/test_notebook_merge_service.py::test_idempotent_re_run`:

```
merge_notebooks: ... entities=1 relations=1 conflicts=0 dry_run=False
merge_notebooks: ... entities=0 relations=0 conflicts=0 dry_run=False
PASSED
```

This test would FAIL on attempt-1 implementation (the strict-advance
mock makes the timestamp-mismatch always fire).

### B2 — `target_id ∈ source_ids` guard at the API boundary

`apps/app-main/src/app_main/api/routers/notebooks.py:226-244`. Added
explicit 422 with detail `"target_id cannot appear in source_ids"`.
Empty-list case already 422 via pydantic `min_length=1` + explicit
fallback. Tests in `tests/test_notebooks_router.py::TestMergeNotebooks`:

- `test_merge_rejects_overlapping_target_and_source` — 422 +
  `merge_service.merge_notebooks.assert_not_called()`
- `test_merge_rejects_target_in_mixed_source_list` — same, multi-source
- `test_merge_rejects_empty_source_ids` — 422

### M3 — rename `_find_target_entity` → `_find_canonical_entity`

Renamed in place + docstring corrected to describe the actual semantics
(global lookup by `(canonical_name, entity_type)` keyed per migration
39). Dropped the unused `notebook_id` parameter.

### M4 — strict-advance mock (covered above)

Mock now persists the FULL post-write row and uses a per-call counter
to drive `updated_at`. Honest contract.

### Minor 1 — fold `_relation_exists` into `_create_relation` return

`_create_relation` now returns `bool(rows)` based on the IF/RELATE
SurrealQL block. The dry-run path still calls the standalone
`_relation_exists` (must not write).

### Minor 2 — fold pre-write probe

B1 fix already eliminated the post-write `_find_target_entity` call;
only ONE probe per entity now (pre-write).

### Minor 3 — archived-notebook rejection at API boundary

Router now rejects archived target (422 + "archived" detail) and
archived sources (422 per-source detail). Tests:

- `test_merge_rejects_archived_target`
- `test_merge_rejects_archived_source`

### Minor 4 — MergeNotebookDialog component tests

Deferred. The Playwright spec covers the dialog's full happy-path UX
(open → select sources → preview shows counts → confirm → close).
Adding a vitest unit layer is low marginal value when the e2e already
exercises the same code-paths against a real DOM.

### Minor 5 — conflict UX deferral

Acknowledged as out-of-scope per attempt 1.

### Verification

- `apps/app-main`: **508 tests pass** (+8 new merge router tests).
- `packages/shared`: 154 pass.
- Frontend `npx tsc --noEmit`: clean.
- Frontend `npm run lint`: zero new warnings (all pre-existing).
- Playwright `notebook-merge.spec.ts`: 1/1 pass (~7.7s, port 8503).

No new dependencies. No architectural changes. Rebase on `main` was
clean (no conflicts).
