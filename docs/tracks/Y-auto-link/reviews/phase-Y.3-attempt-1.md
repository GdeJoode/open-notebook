# Review — Track Y Phase Y.3 attempt 1

**Branch**: `track/y3-autolink-job`
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-29

## Summary

Y.3 wires the background auto-link job (`NOTE_AUTO_LINK`) chained off a successful
note embed, folds the two Y.2 minors, and closes Track Y (ARCHITECTURE §12,
FEATURE_ROADMAP, RETRO). The load-bearing error-isolation property holds under an
independently-driven partial-failure test; the trigger point is correct against
the real EmbeddingService; the job is idempotent; the shared-DB ranking caveat
reproduces identically on `main` (genuinely pre-existing, not Y.3). No blockers,
no majors.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | New embedded note → job auto-links; on-demand still works; both idempotent | ✅ | `test_job_links_a_new_embedded_note`, `test_job_is_idempotent_on_rerun` (DB) + Y.2 suites green |
| 2 | Job failure isolated (note created; logged; no 500/corruption) | ✅ | `test_job_failure_leaves_note_and_edges_intact` + `test_autolink_enqueue_failure_does_not_fail_embed` + my partial-failure heal test |
| 3 | ARCHITECTURE §12 + RETRO; Track Y CLOSED; roadmap; suites green | ✅ | §12 accurate, §13 preserved; RETRO honest; additive roadmap |

## Independent verification

1. **Error isolation (load-bearing).** Both seams confirmed.
   - Enqueue seam (`handlers.py:410-428`): wrapped in try/except, logs, returns
     the embed result on a queue failure — `test_autolink_enqueue_failure_does_not_fail_embed` PASS.
   - Job seam: auto-link is a separate downstream job writing only `related_note`
     edges; `handle_note_auto_link` raises on hard failure (worker marks FAILED),
     never touches note CRUD. DB test PASS. I additionally drove a PARTIAL failure
     (1st relate succeeds, 2nd raises): note + embedding byte-intact (`dim==3`),
     at most a partial edge, and an idempotent re-run converged to the full set
     `{n1,n2}` with zero duplicate rows. The worst case is "missing edges, self-healed
     on re-run" — never note corruption / blocked embedding / permanent half-write.
2. **Trigger point.** `handlers.py:410` gates on `item_type=="note" and result.embeddings_created > 0`.
   Verified against the REAL `EmbeddingResult` (`pipelines/embeddings/.../service.py:25,338,355`):
   a no-content note yields `embeddings_created=0` → no chain; source embeds → no chain.
   Tests `test_no_autolink_when_note_produced_no_embedding`, `test_no_autolink_for_source_embed` PASS.
3. **Idempotency.** `test_job_is_idempotent_on_rerun` PASS (identical targets + count);
   my partial-heal test PASS (no dup rows after re-run).
4. **Y.2 minors.** `_ensure_embedding` tuple refactor behavior-preserving — the
   not_found / no-embedding / embedded paths reproduce the original `status`
   resolution without the double-`get`; 20 Y.2 service/router tests PASS.
   `invalid_id`-is-MCP-only doc on `NoteAutoLinkResponse` is accurate (the HTTP
   route 422s a bad id before the service; Y.2 `..._rejected_422_before_service` green).
5. **No regression + shared-DB caveat.** `test_note_similarity_roundtrip` passes in
   ISOLATION on both branch and main. Co-running it after `test_mcp_auto_link_note`
   FAILS IDENTICALLY on `main` (`cosine ranking wrong: got []` — accumulated MCP-test
   notes push the seeded notes out of top-k). Confirmed pre-existing; Y.3 touches
   neither file.
6. **Closure integrity.** ARCHITECTURE §12 accurate (3-layer flow, embed split,
   two-half trigger, isolation, sync model, data reality, note↔source extension);
   "Further reading" preserved as §13, no clobber. RETRO honest. Track Y CLOSED.
   FEATURE_ROADMAP additive.

## Test status (re-run, this environment, Docker available)

```
test_handle_embed_note_autolink.py + test_handle_note_auto_link.py + test_handlers.py  → 11 passed
test_handle_note_auto_link_db.py (docker)                                              → 3 passed
test_note_auto_link_service.py + test_notes_auto_link_router.py (Y.2 regression)        → 20 passed
combined app-main auto-link suite                                                       → 34 passed
surrealdb test_mcp_auto_link_note.py (docker)                                           → 5 passed
job-queue package                                                                       → 38 passed
adversarial partial-failure + idempotent-heal (mine, docker)                            → 1 passed
main co-run reproduces the shared-DB ranking failure                                    → confirmed pre-existing
```

## Issues found

### 🔴 Blockers — none
### 🟡 Major — none
### 🔵 Minor (optional, follow-up)

1. **Partial-write window is undocumented in the test file.** `apps/app-main/tests/test_handle_note_auto_link_db.py:119` only exercises the all-or-nothing case (relate raises on the FIRST pair → 0 edges). The real isolation guarantee is "a partial write may land but is self-healed by the idempotent re-run", which I had to prove separately. Consider adding a partial-failure-then-rerun test so the self-healing property is regression-guarded in-tree. (Behavior is correct; coverage of the partial case is the gap.)

## Decision rationale

0 blockers, 0 majors. The load-bearing isolation property survived adversarial
partial-failure probing; the trigger condition is correct against production
code, not just mocks; the job is idempotent and self-healing; the only failing
test is a pre-existing shared-DB ordering artifact that reproduces on `main`.
Docs are accurate and additive. → APPROVED.

## Next steps

Ready for human approval / merge. The one minor (partial-failure regression test)
can be filed as a Y follow-up; it does not block.
