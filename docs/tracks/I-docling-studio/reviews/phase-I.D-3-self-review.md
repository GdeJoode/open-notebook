# Phase I.D-3 — Chunk merge/split — Self-review

Branch: `track/i-d3-chunk-mutate`
Commits: `8e5480e` (backend), `c78c2a6` (frontend)

## Scope

Third of four I.D sub-features: merge/split mutations on a source's chunk
sequence, exposed at `/sources/{id}/chunks/{chunk_id}/merge` and `/split`,
with a toolbar in the inspect workspace. Strictly scoped to I.D-3 — no
touch to I.D-1/2/4.

## Stale plan paths (flagged)

The plan (`plan.md` → I.D-3) names files that do not match the codebase:

- Plan: new `apps/.../api/routers/chunks.py` with `/chunks/{id}/merge|split`.
  Reality: chunk endpoints live in `sources_crud.py` under
  `/sources/{source_id}/chunks/...`. Endpoints added there, mirroring the
  existing `update_chunk`/`delete_chunk`/`create_chunk` style and the
  `{chunk_id:path}` converter (chunk ids contain colons).
- Plan AC4 references appending to a `chunk_edit` table — explicitly
  deferred to I.H2 per the plan body; for I.D-3 we log via loguru only.

## AC-by-AC

### AC1 — Merge (text concat + positions union + remove + resequence)
Met. `ChunkMutator.merge` concatenates `keep.text + "\n\n" + drop.text`
(separator documented as `MERGE_SEPARATOR`), unions positions by
**concatenating** the two position lists (the faithful "union": the merged
chunk's overlay covers every line-box of both inputs — documented in the
service docstring and the `_split_positions`/merge comments). The absorbed
chunk is `DELETE`d and every later chunk's `order` is decremented by 1 to
compact the gap. Direction is documented: by default merge with the **next**
chunk by order; an explicit adjacent `target_chunk_id` may be supplied. The
lower-`order` chunk always survives.
Test: `test_merge_concats_text_unions_positions_resequences`,
`test_merge_with_explicit_adjacent_target`.

### AC2 — Split (text split at offset + proportional positions + order insert)
Met. `ChunkMutator.split` cuts text at `cursor_offset` (original keeps
`[:offset]`, new chunk gets `[offset:]`). Positions are apportioned
proportionally by the text fraction `offset/len`: each bbox is cut
vertically at `y1 + (y2-y1)*fraction` so the first half takes the top band
and the second the bottom band — monotonic and union-conserving. The new
chunk is inserted at `order+1` after shifting all later chunks up by 1.
Test: `test_split_at_offset_creates_two_chunks`,
`test_split_positions_proportional_single_box`,
`test_split_positions_conserves_union`.

### AC3 — Atomicity (Q-I-D3-1)
Met, with a **real** SurrealQL `BEGIN TRANSACTION ... COMMIT TRANSACTION`
block. The exact guarantee:

- Both ops build a single multi-statement transaction string and send it
  through the canonical `execute_query` seam (which forwards it unchanged
  to `AsyncSurreal.query`). SurrealDB executes the block server-side and
  rolls the whole block back if any statement errors; the SDK then raises,
  which `execute_query` re-raises as `RuntimeError`. So a mid-op failure
  leaves no partial write — this is genuine BEGIN/COMMIT, not a
  validate-then-write emulation.
- One seam caveat I verified against surrealdb 1.0.6: `query()` returns
  only the **first** statement's result for a multi-statement query, so I
  do **not** read mutated rows from the transaction's return value —
  callers re-fetch via a separate `SELECT` after the commit succeeds. This
  is purely a read-back detail; it does not weaken the atomicity of the
  write block.
- Additionally, all *validation* (chunk exists, belongs to the source,
  adjacency, offset bounds) runs in Python before the transaction is built,
  so the common failure modes never reach the database.

The atomicity tests inject a failure inside the transaction (`fail_on`)
and assert the store is byte-for-byte identical to the pre-op snapshot:
`test_merge_rollback_leaves_state_intact`,
`test_split_rollback_leaves_state_intact`.

### AC4 — Audit trail
Met (as scoped). Each op logs a before/after summary via loguru
(`chunk merge: ...` / `chunk split: ...` with text lengths and position
counts). The durable `chunk_edit` table is **deliberately not** built here
— deferred to I.H2 per the plan.

## Mental inversion (how could this be wrong?)

- **"Union" interpretation.** I read "positions = union of bboxes" as
  list-concatenation (cover all line-boxes), not a single enclosing
  rectangle. A single enclosing rect would over-cover whitespace between
  the two chunks; concatenation is the faithful coverage and matches how
  the overlay renders one rect per position entry. Documented explicitly.
- **Split position math is an approximation.** Without per-line geometry we
  can't split a multi-line bbox at the true line boundary. The vertical
  proportional cut is monotonic and conserves the union, but for a chunk
  whose text/line distribution is uneven the cut won't land exactly on a
  line. Acceptable for an operator tool; documented in `_split_positions`.
- **Order resequencing races.** Merge decrements `order` for all later
  chunks; split increments. Both happen inside the same transaction as the
  delete/create, so concurrent readers see either the pre- or post-state,
  never a half-applied sequence. Concurrent *writers* to the same source
  are out of scope (no row locking beyond the transaction); this matches
  the rest of the chunk CRUD surface.
- **`target_chunk_id` non-adjacent / cross-source.** Rejected with a
  `ChunkMutationError` → HTTP 400 before any write
  (`test_merge_non_adjacent_target_rejected`). Cross-source chunks are
  filtered out because we only load chunks `WHERE source = $source`.
- **Offset at the boundary.** `cursor_offset` of 0 or `len` would create an
  empty chunk; rejected as out-of-range
  (`test_split_offset_out_of_range_rejected`).

## Tests run (real output)

Backend (WSL, project venv):
```
tests/test_chunk_mutator.py ............  12 passed in 3.70s
tests/test_chunk_mutator.py tests/test_schemas.py tests/test_reextract_router.py
  35 passed, 3 warnings in 59.34s   (warnings are pre-existing SWIG deprecations)
```
Ruff: `All checks passed!` on all five changed Python files.

Frontend:
```
tsc --noEmit  → exit 0
npm run lint  → no errors; only pre-existing warnings in unrelated files
               (none in ChunkActionsToolbar / use-sources / api/sources)
```

## Not fully verifiable in this environment

- **Live transaction rollback against a real SurrealDB.** The BEGIN/COMMIT
  block is asserted against a FakeDB that models all-or-nothing apply; the
  actual server-side rollback semantics of SurrealDB's transaction block
  are relied upon (documented, standard SurrealQL) but not exercised here.
- **End-to-end UI flow** (click Merge/Split → re-fetch → overlay updates).
  No Playwright spec added in I.D-3 (the plan lists only the mutator test);
  the toolbar is type-checked and lint-clean but not E2E-tested.
