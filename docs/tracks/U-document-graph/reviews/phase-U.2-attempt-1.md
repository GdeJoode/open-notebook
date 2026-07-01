# Review — Track U Phase U.2 attempt 1

**Branch**: `track/u2-mentions`
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-27

## Summary
The `mentions` (source->entity) projection is a faithful, idempotent derived view of
`entity.source_documents` reusing R.2 `entity_weight` and R.6 `normalize_entities_for_signal`
verbatim. Migration 66 is non-destructive (guarded null-endpoint DELETE + DEFINE TABLE OVERWRITE)
and its preservation test genuinely fails under a blanket DELETE (verified by adversarial patch).
All 50 reviewed tests pass against a live SurrealDB container; canonical data is provably untouched.

## Acceptance criteria check
| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | mentions edges from array; count = U.1 estimate (67) | PASS | Projection = 1:1 from source_documents; container seed yields the expected df>=2 edges, singleton dropped. Unit + DB tests. |
| 2 | each edge carries a weight (R.2 verbatim) | PASS | `entity_weight` imported verbatim from kg_source_scorer; rd_w>bw_w; all weights>0. |
| 3 | idempotent, no dup edges | PASS | clear+RELATE; 2nd run clears exactly what 1st created; pair-set deduped (test_regenerate_is_idempotent). |
| 4 | singleton/generic noise handled | PASS | df==1 spoke dropped; generics down-weighted ~6x not removed. |
| 5 | canonical entity/source rows untouched | PASS | Snapshot (incl. updated_at) byte-identical pre/post regenerate; loader is read-only active-only. |
| 6 | document->entity->document traversal returns K4; papers isolated | PASS | Traversal reaches all 4 convenanten; entity-less paper never reached; endpoint shape matches U.4. |

## Independent verification beyond the supplied tests
- Migration 66 guarded-DELETE: patched `DELETE mentions WHERE in=NONE OR out=NONE` -> blanket
  `DELETE mentions;` and re-ran `test_healthy_mentions_edges_preserved` -> FAILED
  (`edge loss: 5 -> 0`). Restored original (guarded) -> PASS. The test genuinely catches the
  destructive mistake. Full 1->66 chain applies clean (`test_migrations_applied`). Down is a
  documented no-op (sane for a schema repair).
- `entity_weight` / `normalize_entities_for_signal` are the SAME functions R.2 uses (imported,
  not re-implemented). `_representative_entity_ids` keys in the SAME `concept::name|slot` space
  as the normalizer (`test_no_unmapped_concepts` asserts unmapped==0 -> every edge anchors a real entity).
- `clear_mentions` is `DELETE mentions;` only — never touches relation/cites/entity/source.
- `relate_mention` interpolates record-id literals only in the RELATE arrow slots (SurrealDB #4232),
  validated via `ensure_record_id`; all metadata bound as params. `load_mentions_edges` parameterized.
- New endpoints unauthenticated — consistent with every pre-existing knowledge_graph.py endpoint (no new gap).
- No TODO/FIXME/print/emoji in new code.

## Test status
```
packages/shared  test_mentions_projection.py ............ 20 passed
packages/shared  (full regression) ......................490 passed
surrealdb-service migration_66 + roundtrip ..............23 passed (incl. 1->66 chain, healthy-edge preservation)
app-main         test_mentions_regenerate_db.py ......... 7 passed (@requires_docker)
app-main         KG router/service regression ...........45 passed
```

## Issues found
### Blockers (must fix)
None.

### Major (must fix)
None.

### Minor (optional, follow-up)
1. **relate_mention docstring/behaviour mismatch** — `entity.py` `relate_mention`: docstring says
   "False on a bad id / transport error" but the code `raise`s on a transport error (only bad-id
   returns False). The caller (`regenerate`) wraps the call in try/except and counts it as `failed`,
   so behaviour is correct; only the docstring is stale.
2. **emitted_concepts approximation** — `regenerate()` derives `emitted_concepts` from the post-cut
   edge set (`len({e.entity_id})`) rather than the projection stats; under a non-zero `min_weight`
   this under-counts concepts that were weight-filtered out. Telemetry-only; not an acceptance field.
3. **Double active-entity load** — `regenerate()` calls `load_active_entity_source_map()` twice
   (projection + active count). Both reads, harmless; a single load could be threaded through.

## Decision rationale
0 blockers + 0 majors. Every acceptance criterion verified against a live container, not mocks.
The one load-bearing safety claim (migration 66 preserves healthy edges) was independently
falsification-tested. The three minors are telemetry/doc nits that do not affect correctness,
safety, or the U.4 contract. APPROVED.

## Next steps
Ready for human approval / merge. The live regenerate remains correctly gated (table still 0 rows
on staging). Minors can be filed as follow-up. U.4 can consume `GET /knowledge-graph/document-graph`
as-is (`{edges:[{id,source,target,weight,concept_name,concept_type,document_frequency}], count}`).
