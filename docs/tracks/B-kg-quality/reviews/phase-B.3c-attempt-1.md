# Review — Track B Phase B.3c attempt 1

**Branch**: `track/b-soft-nudge` (PR #21)
**Decision**: **REVISIONS_NEEDED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-09

## Summary

5 ACs met end-to-end; 2 banner components render right states; 5 endpoints wired with reasonable shapes; sentinel-resume documented + tested at JSON schema + TTL boundaries. 19/19 new unit tests pass, 436 app-main full suite, clean tsc, no new lint warnings, 5 Playwright specs parse.

**BUT** the resume-sentinel is only filtered at JSON schema endpoint + TTL exporter. It is NOT filtered at the extraction pipeline boundary — leaks into Pass-2 LLM prompt as a real extension type.

## Blockers (1)

### B1: Resume sentinel leaks into Pass-2 LLM prompt

Files:
- `apps/app-main/src/app_main/services/entity_extraction_service.py:343-353` (builds `accepted_by_schema` from full list)
- `pipelines/ontology-extraction/src/ontology_extraction/multi_schema_orchestrator.py:557`
- `pipelines/ontology-extraction/src/ontology_extraction/prompts/pass2.py:135-173` (`_format_accepted_extensions` — no sentinel filter)

When user clicks `[Resume]`, `resume_extraction` appends `{type_name: "_resumed_without_extensions", is_resume_sentinel: true, ...}` to `accepted_extensions`. Review-gate predicate clears (list non-empty). But `_run_multi_schema` then iterates the same list to build `accepted_by_schema`. The sentinel has no `schema_name` → broadcast into every applicable schema's list. `run_pass2` passes it to `build_pass2_prompt`. `_format_accepted_extensions` (no `is_resume_sentinel` check) renders:

```
## Accepted Extension Types
- **_resumed_without_extensions** (no parent)
```

LLM is now instructed to treat `_resumed_without_extensions` as a first-class entity type. The sentinel persists in `accepted_extensions` FOREVER → every Resume permanently pollutes every future Pass-2 prompt for that notebook.

**Evidence**: `grep -rn "is_resume_sentinel" --include="*.py"` → only 2 filter sites, both in `schemas.py` (JSON browse + TTL). No filtering in extraction pipeline.

**Fix**: filter at every consumer of `accepted_extensions`. Narrow blast radius = filter at `_run_multi_schema:343-353`. Defence-in-depth = also at `_format_accepted_extensions`. Add regression test `test_run_multi_schema_filters_resume_sentinel`.

## Major (1)

### M1: Module comment + self-review claim a frontend filter that does not exist

`schemas.py:759` comment: "the frontend already filters them by checking `type_name.startswith('_')`". `grep` confirms NO such filter exists in `SchemaBrowser.tsx`. Misleading documentation compounds B1.

Fix: either implement claimed frontend filter (defence-in-depth) OR remove the comment + correct self-review §2.

## Minors (5)

1. `useResumeExtraction` forward-references `PAUSED_EXTRACTION_QUERY_KEY` (runtime-safe but awkward)
2. Close-X button wired to `mark_read` (persists) but `aria-label="Hide banner"` (transient)
3. 30s polling vs AC #1's "within 5s" — documented trade-off
4. `paused_count` derived from job-list length, not dedicated count query — fine at scale
5. Sentinel `created_at` uses `isoformat()` (+00:00) vs Z-suffix — frontend handles both

## Tests (independently verified)

- `apps/app-main/tests/test_schemas_soft_nudge.py`: 19 passed
- `apps/app-main` full: 436 passed
- frontend tsc clean, lint clean, 5 specs parse

## Kudos

- `_make_app` helper + `dependency_overrides` pattern clean
- Sentinel-filtering tests at JSON + TTL boundaries lock those filters
- Soft-import dance for B.3b — both branches merge in either order
- Idempotent endpoint design across all 5 mutations
- A11y: `Alert` role + Switch htmlFor pairing
- Defensive `_project_event` projection tolerates dict-shaped or object-shaped rows

## Decision rationale

The implementation is otherwise clean. But the resume button — centrepiece of AC #4 — introduces a real correctness regression on the next-extraction path. The user clicks Resume expecting "proceed with current schema"; what actually happens is "proceed with current schema PLUS a fake type for the LLM to consider". Bug is contained (no crash) but WILL show in production as weird LLM extractions after Resume — exactly the foot-gun review steps catch.

## Next steps

1. Fix B1 — filter sentinels before any extraction-pipeline consumer sees them. Add `test_run_multi_schema_filters_resume_sentinel`
2. Fix M1 — implement claimed frontend filter OR remove misleading comment + correct self-review
3. Optional: address minors
