# Review — Track PL Phase PL.4 attempt 1

**Branch**: `track/pl4-pipeline-definition` (`5d72fd6..4288aa1`)
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-30

## Summary

PL.4 is a clean, behavior-identical consolidation of the PL.1–PL.3 source
auto-chain into one declarative `SOURCE_PIPELINE` + `advance_source` driver. The
scattered handler enqueues moved verbatim into `source_pipeline.py`; handlers are
now thin. `processing_stage` is exposed on the source read API. I independently
drove the `advance_source` dispatch table (incl. double-advance, parked/terminal,
chunk-guard scoping) and ran the full requested suite + `@requires_docker` DB
behavioral tests + the migrations roundtrip. No blockers, no majors.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | Chain driven by `advance_source`; handlers hold no ad-hoc next-stage enqueues; unit tests on the stage-transition table | ✅ | `source_pipeline.py` holds the table; `test_source_pipeline.py` (13) pins each transition + gates. Only `submit_command_job` left in `handlers.py` is the NOTE auto-link (`handlers.py:469`). |
| 2 | `processing_stage` returned by the source read endpoint(s); a test asserts it | ✅ | `sources_crud.py:310,363`; `test_sources_crud.py::test_processing_stage_returned` (graphed) + `::test_processing_stage_defaults_ingested` (None→ingested). |
| 3 | End-to-end ingest produces the same result as PL.3 (no regression); suites green | ✅ | Handler diff vs PL.3 is a 1:1 relocation; DB test `test_auto_extract_materializes_source_graph_and_completes` passes; 1385 passed / 3 pre-existing docling failures (identical on `main`). |

## Test status

```
test_source_pipeline.py ......................... 13 passed (21.58s)
-k "pipeline|processing_stage|autoextract|insights|entity_extract|sources_crud"
                                                  87 passed, 1385 deselected (40.70s)
  (incl. @requires_docker: gate_db, graph_db, processing_stage_db, notebook_auto_insights_db — testcontainers spun up, NOT skipped)
app-main -m "not requires_docker"                 1385 passed, 2 skipped, 3 failed
  (the 3 = TestBuildIngestionConfig docling-not-installed — VERIFIED identical on main; env-only, not a regression)
migrations_roundtrip (all 72 + down/forward)      19 passed (8.13s)
my own advance_source adversarial suite           7 passed (double-advance, parked-on-repeat, guard scoping)
```

## Issues found

### Blockers (must fix)
None.

### Major (must fix)
None.

### Minor (optional / follow-up)

1. **`advance_source` is not enqueue-deduped (documented, accepted).** `advance_source`
   reads `processing_stage` and dispatches the successor, but for the enqueue-based
   stages (EMBED/EXTRACT) it does NOT itself advance the stage value — that happens
   in the next handler. So a double-call on `embedded` enqueues `run_entities` +
   `run_summaries` TWICE (I confirmed this with a direct test). This is the explicitly
   documented design (module docstring lines 41–45: "relies on job idempotency") and
   is **behavior-identical to PL.3**, where each handler also re-fired on every
   invocation with no dedup. Not a defect for this refactor; noting only because the
   word "idempotent" in the brief could be read as enqueue-dedup, which it is not.
   If true enqueue-dedup is ever wanted, that's a NEW behavior (out of PL.4 scope).

## Decision rationale

Verified per-criterion against the #1 bar (behavior-identical to PL.3):

1. **Behavior-identity.** `git diff pl3..pl4 handlers.py` is a pure relocation: the
   embed→extract enqueue, the toggle-gated insights chain (`_maybe_chain_insights`),
   the inline mentions refresh (`_refresh_source_mentions`), and the `embedded_chunks>0`
   guard all moved into `source_pipeline.py` unchanged. Same enqueue conditions
   (chunk-count guard, auto_insights toggle, schema gate). The DB end-to-end test
   that materializes the graph and reaches `complete` passes through the new path.

2. **`advance_source` dispatch + idempotency.** My own adversarial tests confirm:
   double-advance from `embedded` re-enqueues (relies on job idempotency, = PL.3);
   double-advance from `extracted` re-runs the idempotent clear-then-relate refresh
   and re-writes graphed/complete safely; `awaiting_schema_review`/`failed`/`complete`
   NEVER advance even on repeated calls (`_TERMINAL_OR_PARKED` guard at line 234,
   checked BEFORE stage lookup); the `embedded_chunks<=0` guard gates ONLY the EMBED
   fan-out (line 252–256: `stage_name is StageName.EMBED`), so `ingested→embed` and
   `extracted→graph` are correctly unaffected. The schema gate stays in the extract
   handler (reraises `SchemaReviewPendingError`); `advance_source` is only on the
   success path — correct, no double-handling.

3. **Consolidation complete.** Grep confirms the only `submit_command_job` in
   `handlers.py` is the NOTE auto-link (`handlers.py:469`, a different pipeline). All
   source next-stage dispatch goes through `advance_source` / `_best_effort_enqueue`.

4. **API + `refresh_source` invariant.** `SourceResponse.processing_stage` added
   (`schemas.py:631`), returned by `get_source` + `update_source`, tested.
   `MentionsProjectionService.refresh_source` is **byte-identical to `main`**
   (`git diff main..pl4` on that file is empty) — the full-projection-then-scoped-write
   invariant is untouched; only the call site moved into `_run_graph_inline`.
   `get_processing_stage` uses the proven `SELECT VALUE … FROM $id` pattern (mirrors
   `get_aggregate_embedding` at source.py:450) with best-effort None fallback.

5. **The 2 folded minors.** (a) `test_migration_72_backfills_none_and_row_stays_writable`
   mirrors migration-71's S.4 test (NONE `notebook.auto_insights` repaired AND row
   stays writable — passed under docker). (b) `complete`-with-zero-edges semantics are
   documented in the `source_pipeline.py` module docstring (lines 34–39).

6. **No regression.** Full app-main suite green except the 3 `TestBuildIngestionConfig`
   docling-import failures, which I verified fail identically on `main` (environment:
   docling not pip-installed) — the documented 3-docling baseline. The note auto-link
   chain (`_handle_embed_single_item`) is untouched.

## Next steps

APPROVED — ready for human approval / merge. The single minor (enqueue-dedup naming)
is a documentation/scope note only; no code change required for PL.4.
