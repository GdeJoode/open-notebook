# Review — Track Y Phase Y.2 attempt 2

**Branch**: `track/y2-autolink-ondemand`
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-29

## Summary

The auto-link orchestrator, HTTP endpoint, and embedding-free MCP tool meet all
Y.2 acceptance criteria. All 25 listed tests pass, and an independent
reviewer-authored fresh-container test (precise-cosine notes, three reruns)
confirmed the load-bearing invariants directly: threshold is inclusive at the
boundary, below-threshold/self are never linked, and reruns produce the identical
edge set with no row growth. No blockers or majors.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | edges only ≥ min_similarity, top-k capped, no self, idempotent | ✅ | Independently verified on fresh container (below) |
| 2 | endpoint + MCP both drive service/repo, return summary; no-embedding → embedded-first (endpoint) or clean needs_embedding (MCP), no 500 | ✅ | Endpoint awaits real `EmbeddingService.embed_note`; MCP returns `needs_embedding` |
| 3 | tests for service (threshold/top-k/idempotency/self) + endpoint + MCP | ✅ | 12 + 8 + 5 = 25, all pass |

## Test status

```
app-main:   test_note_auto_link_service.py + test_notes_auto_link_router.py → 20 passed (24.5s)
surrealdb:  test_mcp_auto_link_note.py (@requires_docker)                   → 5 passed (8.9s)
Y.1 repo:   test_note_similarity_roundtrip + test_migration_68_*            → 15 passed
adjacent:   test_notes_router.py + test_note_service.py                     → 16 passed
reviewer:   test_zz_adversarial_y2_verify.py (precise cosine, 3 reruns)     → 1 passed (then removed)
```

## Independent verification (per the brief)

1. **Idempotency on a real DB — PASS.** My own test seeded q + hi(cos1.0) +
   on(cos0.8 exact) + lo(cos0.6) on a 4-dim axis (isolating them in the shared
   DB), ran `auto_link_note` three times, and asserted the `out`-edge set AND the
   `count()` are identical across all three runs. No duplicate `(in,out)` rows,
   no growth. `relate_note`'s clear-before-relate (`notebook.py:299-317`) holds.
2. **Threshold + top-k + self-exclusion — PASS.** With `min_similarity=0.8`:
   cos1.0 and cos0.8-boundary were linked (boundary inclusive — confirmed; the
   service gate is `float(score) < eff_min` → below, so `>=` links,
   `note_auto_link_service.py:220`), cos0.6 was NOT linked, self was NOT linked.
   `test_auto_link_note_top_k_caps_links` independently shows `k=2` caps to 2.
3. **Route validation = 422 not 500 — PASS.** `_validate_note_id`
   (`notes.py:30-47`) runs `_validate_record_id` + `note:` prefix before the
   service; `k`/`min_similarity` bounded by `Query(ge/le)` (`notes.py:171-182`).
   I confirmed the validator regex (`base.py:22`) rejects
   `note:x; REMOVE TABLE note; --` and `note:x OR true`, accepts only `table:id`.
   The repo `relate_note` re-validates before interpolation
   (`notebook.py:272-280`), so a bad id can't reach the RELATE write.
4. **No-embedding path, no 500 — PASS.** Endpoint: `_ensure_embedding`
   (`note_auto_link_service.py:122-157`) awaits the real
   `EmbeddingService.embed_note` (`pipelines/embeddings/.../service.py:327`),
   guards it in try/except, and treats `embeddings_created < 1` as
   `needs_embedding` (200). MCP: `SELECT VALUE embedding` → `needs_embedding`
   when empty (`server.py:394-404`). Verified by tests; no 500 path.
5. **MCP stays embedding-free — PASS.** `server.py` imports only repo
   primitives + connection helpers (no torch/ollama/embedding). The
   surrealdb-service `pyproject.toml` has zero embedding deps. The embed-then-link
   convenience is app-main-only (`get_note_auto_link_service`,
   `dependencies.py:418-429`).
6. **Summary correctness + canonical untouched — PASS.** `created +
   below_threshold + skipped == candidates_considered` balanced in every run of
   my test and the suite. My note-row snapshot (title/content) before/after three
   runs was unchanged. `to_dict` always populates the required schema fields on
   every status, so `needs_embedding`/`not_found` can't raise a response-model
   error. Tests assert invariants, not brittle positional ranks (confirmed by
   reading `test_mcp_auto_link_note.py`).

## Issues found

### 🔴 Blockers (must fix)

None.

### 🟡 Major (must fix)

None.

### 🔵 Minor (optional)

1. **Double `note_repo.get` / double existence fetch** —
   `notes.py:201` (route 404 check) + `note_auto_link_service.py:133` and again
   at `:194`. The note row is fetched up to three times on the not-found/needs-
   embedding path. Harmless and gives clean 404-vs-needs_embedding separation,
   but a single fetch could be threaded through. Follow-up only.
2. **`embedded` schema field documented on response but not in the `status`
   docstring enum** — `schemas.py` `status` description lists
   `linked | needs_embedding | not_found` (correct); the MCP tool can also return
   `invalid_id` (`server.py:372`). The HTTP schema never returns `invalid_id`
   (it 422s first), so the two surfaces legitimately differ — note only.

## Decision rationale

0 blockers + 0 majors → APPROVED. All three acceptance criteria are met and the
two load-bearing properties called out in the brief (idempotency on a real DB,
route-layer 422 validation) were independently reproduced on a fresh container by
the reviewer, not merely trusted from the suite. The MCP layering is genuinely
embedding-free (verified at the dependency and import level), and the injection
payload from the Y.1 review is rejected here too and cannot reach the write.

## Next steps

Ready for human approval / merge. The two minors can be filed as Y.3 follow-ups
(neither blocks). Y.3 (background job) can build on this orchestrator unchanged.
