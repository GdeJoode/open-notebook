# Review — Track X Phase X.1 attempt 1

**Branch**: `track/x1-retrieval-provenance` (`a2a8c24..e0897fd`)
**Decision**: REVISIONS_NEEDED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-29

## Summary

The source_embedding hydration is correct and the staging probe genuinely proves the
load-bearing argmax claim (cosine top-1 == fn::vector_search collapsed math::max, 1e-9) for
chunk-bearing source hits. All required suites are green. However, the hit->chunk mapping is
NOT exhaustive: `fn::vector_search`/`fn::text_search` also emit `source_insight` hits whose
`parent_id` is a `source:` id, and `_hit_source_id` treats those as source hits — attaching a
`source_embedding` chunk's page that did NOT produce the hit's score. This is exactly the
"hydrated page can mismatch the winning chunk" failure mode (a wrong-page citation for X.2).
It is latent only because the staging DB currently has zero insights, but the embedding
pipeline does populate `source_insight.embedding`, and `fn::vector_search` searches them.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | Each hit carries provenance keys; None/absent handled | ✅ | Stable 5-key set + `source` seeded on every hit; merge never clobbers a real value with None. |
| 2 | Existing callers/tests unchanged (backward-compat) | ✅ | `hydrate` kwarg defaults True; additive keys; all callers delegate unchanged. Suites green. |
| 3 | Unit shape tests + read-only staging probe on a known source | ✅ | 17 mocked + 2 staging probes pass; real page/section on `source:dndibxmjveoxk7tfqfsl`. |

Acceptance criteria as literally written are met. The blocker below is a correctness gap in
the load-bearing mapping that the criteria under-specify (they say "the underlying chunk",
silently assuming every hit is chunk-derived) — but it is the precise failure this review was
asked to hammer, and it would surface as a wrong-page citation in X.2.

## Test status (independently re-run)

```
test_search_provenance.py ............................ 17 passed
surrealdb-service full (not requires_docker) ......... 92 passed, 2 skipped
pipelines/retrieval test_service.py ................. 26 passed
app-main test_search_service.py + test_search_router . 19 passed
app-main test_search_rerank_router.py ............... 6 passed
staging probe (SURREAL_DATABASE=staging, read-only) .. 2 passed
  - test_vector_hit_carries_real_chunk_page_and_section PASSED
  - test_hydrated_page_matches_the_chunk_the_hit_came_from PASSED (cosine top-1 == max, 1e-9)
```

Note: the staging probe self-skips unless `SURREAL_DATABASE` is exactly `staging`. The repo
`.env` has CRLF line endings, so naive `source .env` yields `staging\r` and silently skips
both probes (observed). Had to export the var clean to actually run them — worth a guard note.

## Issues found

### 🔴 Blockers (must fix)

1. **`source_insight` hits get a wrong-chunk page (mapping mismatch)** —
   `repositories/search.py:25-37` (`_hit_source_id`) + `:114-122` (merge).
   - Issue: `fn::vector_search` (`migrations/4.surrealql:91-104,129-132`) and `fn::text_search`
     (`:34-40`) emit `source_insight` hits with `id = source_insight:...` and `parent_id =
     source:Y`. `_hit_source_id` returns `parent_id` (it starts with `source:`), so the hit is
     hydrated from `_best_chunk_per_source`, which only queries `source_embedding`. The insight
     hit is therefore stamped with the source's top *source_embedding* chunk's `chunk_id` /
     `physical_page` — a different row that did not produce the insight hit's score. Reproduced:
     an `{id: source_insight:X, parent_id: source:Y, similarity: 0.95}` hit comes back with
     `chunk_id=chunk:emb_top, physical_page=12` (the embedding-leg winner, not the insight).
   - The staging argmax proof (`test_search_provenance_staging.py:115-124`) reconstructs fn's
     max over `source_embedding WHERE chunk IS NOT NONE` only — it excludes `source_insight`, so
     it cannot catch this. Latent on staging solely because `source_insight` has 0 rows there
     (verified read-only); `pipelines/embeddings/service.py:451-472` (`embed_insight`) does
     populate `source_insight.embedding` in the normal pipeline.
   - Impact: X.2 would emit a citation whose page belongs to a chunk that was never the matching
     unit — the exact wrong-page citation this track must prevent. The implementer was aware
     insights map to a source (`search.py:29` names `source_insight`) but did not guard the page.
   - Recommendation: state the issue, not the fix — but the hit `id` prefix (`source_insight:`)
     is available to distinguish these from chunk-bearing `source:` hits.

### 🟡 Major (must fix)

1. **Mapping not proven for the insight path; test is non-exhaustive** —
   `test_search_provenance_staging.py:98-142`.
   - Issue: the only argmax proof excludes `source_insight` from the reconstructed collapse, so
     the suite asserts correctness for a strictly narrower hit population than `fn::vector_search`
     actually returns. There is no test (mocked or staging) exercising an insight hit. Given the
     blocker above, the test set does not cover the path that breaks.
   - Recommendation: add coverage for the `source_insight`-hit case (mocked is sufficient and
     does not need staging insights).

### 🔵 Minor (optional)

1. **Text-only `section_path`/`element_type` is from an arbitrary chunk** — `search.py:155-168`.
   The text branch attaches the source's first-by-`order` chunk's structural fields. `physical_page`
   is correctly left None (criterion 2 honesty satisfied — no guessed page), but `section_path`
   is still a structural attribute of a chunk unrelated to the BM25 match. Documented as a
   "best-effort source-level hint", so acceptable, but it is a milder form of the same
   attach-from-an-unrelated-row pattern; X.2 should not treat text-hit `section_path` as precise.
2. **Unconditional extra round-trip on every search** — `search.py:225-226,273-274` and the
   surrealdb-service search router / MCP server now hydrate by default, adding one batched SELECT
   per search call. Single batched query, acceptable; note for perf-sensitive callers.
3. **Staging probe skip is brittle to CRLF env** — `test_search_provenance_staging.py:32`. An
   exact-string `!= "staging"` guard silently skips when the value is `staging\r` (this repo's
   `.env` is CRLF). A `.strip()` would avoid a false "passed" via skip.

## Decision rationale

One reproduced blocker (a hydrated page that can mismatch the winning unit for `source_insight`
hits) forces REVISIONS_NEEDED per the decision matrix and the explicit instruction that "a
hydrated page that can mismatch the winning chunk is a BLOCKER." The source_embedding path —
the primary path the `ask` graph uses — is correct and well-proven; the gap is the insight
hit class, which is real (pipeline populates insight embeddings; fn searches them) though
currently dormant on staging. Criteria 3, 4, 5 are fully satisfied; criterion 1 is satisfied
for chunk-derived hits but the mapping is wrong for insight-derived hits.

## Next steps

Implementer: address the blocker (insight hits must not be stamped with an unrelated
`source_embedding` chunk's page/chunk_id) and add the missing insight-hit test, then re-submit.
The source_embedding hydration, fn:: untouched-ness, backward-compat, and graceful degradation
all hold and need no rework.
