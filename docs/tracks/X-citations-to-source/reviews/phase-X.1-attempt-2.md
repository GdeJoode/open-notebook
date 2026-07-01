# Review — Track X Phase X.1 attempt 2

**Branch**: `track/x1-retrieval-provenance` (new commit `4a51352`)
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-29

## Summary

The attempt-1 blocker (a `source_insight` hit inheriting an unrelated `source_embedding`
chunk's page) is closed. Chunk-level provenance is now attached only to hits whose **own**
`id` is a `source:` record (`_hit_is_chunk_backed`), while the source-level anchor still comes
from `parent_id` (`_hit_parent_source`). Insight hits, note hits, and the entire text-only
path get all chunk keys `None` and never reach `_best_chunk_per_source`. The source_embedding
argmax path is byte-unchanged and its staging proof (cosine top-1 == fn `math::max`, 1e-9)
still passes. No blockers or majors remain.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | Each hit carries provenance keys; None/absent handled; **mapping correct** | ✅ | Chunk keys attached only for chunk-backed (`source:` own-id) hits; insight/note/text → None. Stable 5-key set + `source` always seeded. |
| 2 | Existing callers/tests unchanged (backward-compat) | ✅ | Signatures unchanged; additive keys; full suites green (96 + 26 + 25). |
| 3 | Unit shape tests + read-only staging probe | ✅ | 21 mocked (was 17; +insight/text-only coverage) + 2 staging probes pass. |
| 4 | Graceful degradation (no crash on lookup failure / notes / missing chunk) | ✅ | `test_lookup_failure_degrades_gracefully`; try/except at `search.py:148`. |
| 5 | `fn::` untouched (no migration) | ✅ | No surrealql/migration files in the diff. |
| 6 | Tests non-vacuous (prove the mapping, not just key-present) | ✅ | Staging argmax proof intact; insight-alongside-embedding test proves the fixed same-source case. |

## Verification of the fix (per coordinator items)

1. **Blocker closed** — `_hit_is_chunk_backed` (`search.py:42-60`) decides chunk provenance
   from the hit's own `id` prefix; only `source:` qualifies. The merge loop
   (`search.py:152-160`) gates on it and looks up `best_by_source.get(hit["id"])`, so a
   `source_insight:` hit is never looked up and keeps all chunk keys `None` (`source` set from
   `parent_id`). Independently reproduced the original attempt-1 case
   (`{id: source_insight:X, parent_id: source:Y}`, embedding present, lookup returning page 12
   for `source:Y`): result `chunk_id=None, physical_page=None, source=source:Y`, lookup
   **not called**. `test_insight_alongside_embedding_hit_for_same_source` proves the same-source
   case (embedding hit → page 7, insight → page-less, lookup receives only `["source:s1"]`) —
   non-vacuous.
2. **Text-only attaches no chunk keys** — `hydrate_provenance` early-returns when
   `embedding is None` after seeding `source` (`search.py:133-134`); the previous arbitrary
   first-chunk `section_path`/`element_type` branch is gone. `test_text_only_attaches_no_chunk_keys`
   asserts every chunk key `None` and the lookup is not called.
3. **source_embedding path unchanged + still correct** — `_best_chunk_per_source` SELECT is
   byte-identical to attempt 1 (same `vector::similarity::cosine` / `ORDER BY _sim DESC` /
   first-per-source). Staging probe `test_hydrated_page_matches_the_chunk_the_hit_came_from`
   still passes (1e-9).
4. **No regression / fn untouched** — full non-docker surrealdb-service suite 96 passed,
   retrieval 26 passed, staging probe 2 passed, app-main search+rerank 25 passed; no migration
   in the diff; no residual `_hit_source_id` references.

## Test status (independently re-run)

```
test_search_provenance.py ............................ 21 passed
surrealdb-service full (not requires_docker) ......... 96 passed, 2 skipped
pipelines/retrieval test_service.py ................. 26 passed
app-main search_service + search_router + rerank ..... 25 passed
staging probe (SURREAL_DATABASE=staging, read-only) .. 2 passed
  - test_hydrated_page_matches_the_chunk_the_hit_came_from PASSED (cosine top-1 == max, 1e-9)
independent blocker reproduction ..................... insight hit -> all chunk keys None, lookup NOT called
```

## Issues found

### 🔴 Blockers
None.

### 🟡 Major
None.

### 🔵 Minor (optional, non-blocking — carry to X.2/X.3 or follow-up)

1. **Unconditional extra round-trip on every vector/hybrid search** — `search.py:144-147` and
   the surrealdb-service search router / MCP server hydrate by default (one batched SELECT per
   search). Single batched query; acceptable. (Carried from attempt 1.)
2. **Staging probe skip is brittle to CRLF env** — `test_search_provenance_staging.py:32`
   exact-string `!= "staging"` skips silently on `staging\r` (this repo's `.env` is CRLF).
   A `.strip()` would prevent a false "passed via skip". Cosmetic. (Carried from attempt 1.)
3. **X.2 note** — only chunk-backed (`source:`) vector/hybrid hits carry a `chunk_id`. Insight
   and text hits intentionally carry none, so the X.3 membership guard must scope its check to
   hits that actually have a `chunk_id` (the status doc already flags this).

## Decision rationale

Zero blockers, zero majors → APPROVED per the decision matrix. The one reproduced blocker from
attempt 1 is closed and covered by a non-vacuous regression test; the correct source_embedding
path and its staging proof are untouched; backward-compat, graceful degradation, and the
fn::-untouched constraint all hold. Remaining items are minors.

## Next steps

Ready for human approval / merge. Proceed to X.2 (cited answers) consuming the surfaced
`chunk_id`/`physical_page`/`section_path` keys; scope the X.3 membership guard to hits bearing
a `chunk_id`.
