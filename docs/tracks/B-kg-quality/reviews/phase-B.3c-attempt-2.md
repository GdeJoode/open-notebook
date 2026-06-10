# Review — Track B Phase B.3c attempt 2

**Branch**: `track/b-soft-nudge` (PR #21, HEAD `04f5982`)
**Decision**: **APPROVED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-10
**Attempt 1**: REVISIONS_NEEDED — 1 blocker (sentinel leak) + 1 major + 5 minors

## Summary

Attempt-1 blocker (resume sentinel leaking into Pass-2 LLM prompt) closed with belt-and-braces filtering at 2 backend seams + defence-in-depth UI filter + explicit type-contract field. Blocker carries real regression test pinning the boundary. Prompt builder carries 2 unit tests covering mixed-list and all-sentinel branches. M1 (false-claim comment) rectified. All 5 minors landed. Quality gates: 437 app-main + 19 soft-nudge + 65 pass2 + 243 ontology-extraction tests green; tsc + lint clean.

## Acceptance criteria (all PASS)

All 5 ACs met. AC #4 (resume proceeds) — sentinel now filtered before LLM prompt.

## Blocker B1 — RESOLVED

Trace confirmed by file inspection:

1. `resume_extraction` (schemas.py:1014-1034) appends sentinel with Z-suffix `created_at`
2. `EntityExtractionService._run_multi_schema` (entity_extraction_service.py:370-382) calls `_is_resume_sentinel(ext)` (helper L41-57) before routing — sentinel never reaches per-schema bucket
3. `_format_accepted_extensions` (pass2.py:135-187) filters at L158 + omits entire section when all-sentinel (L159-160)
4. `SchemaBrowser` (frontend L96-100) drops `is_resume_sentinel === true` AND `type_name.startsWith('_')`
5. Type contract carries `is_resume_sentinel?: boolean`

5 filter sites + helper + type contract.

**Regression test** (test_entity_extraction_service.py:864-953) constructs NotebookSchema with real `X` + sentinel; asserts:
- Real `X` survives under `"scholarly"` bucket
- NO bucket carries `is_resume_sentinel=True` (inverted across every bucket — robust against future broadcast regressions)

Mental inversion: removing the filter at L373 would broadcast sentinel everywhere → test fails. Pin correct.

**Pass-2 unit tests** (test_pass2.py:237-291): mixed list + sentinel-only both pinned; both would fail under inversion.

## Major M1 — RESOLVED

Module-level comment at schemas.py:749-776 enumerates all 5 filter sites — authoritative reference. Self-review §2/§3 corrected. Frontend filter implemented, not merely documented.

## Minors — All resolved

- M1 forward-ref: PAUSED_EXTRACTION_QUERY_KEY at L147, before useResumeExtraction at L159 ✓
- M2 aria-label="Mark as read" matches mutation ✓
- M3 30s polling documented as deliberate trade-off ✓
- M4 paused_count via list length — deferred with tracking ✓
- M5 Z-suffix on created_at ✓

## Edge-case probes

1. Multiple resume sentinels → each filtered independently ✓
2. String-shaped `is_resume_sentinel: "true"` → backend service/pass2 catch via `bool(...)`/truthy; TTL+JSON sites use strict `is True` (would slip). Sentinel-write centralised at resume_extraction using Python `True` — safe operationally. Inconsistency = minor follow-up.
3. Pre-existing `_underscore_intended` type → frontend filter would hide. Belt-and-braces. Minor risk documented.

## New minors (non-blocking, follow-up)

1. **Inconsistent sentinel-predicate idiom** — schemas.py:214/636 use `is True`/`is not True`; entity_extraction_service.py uses `bool(...)`; pass2.py uses truthy `not ext.get(...)`. All agree operationally. Recommend collapsing all backend sites onto `_is_resume_sentinel` helper.
2. **No UI-rendering test pins SchemaBrowser filter** — backend gates prevent sentinels reaching frontend in normal flow; defence-in-depth filter at L96-100 lacks render test.
3. **`startsWith('_')` heuristic** — belt-and-braces but risky for future legitimate `_internal_*` type.

## Test status (independently verified)

- `test_run_multi_schema_filters_resume_sentinel` → 1 passed
- `test_schemas_soft_nudge.py` → 19 passed
- `apps/app-main` full → 437 passed
- `test_pass2.py` → 65 passed (includes 2 new sentinel tests)
- `ontology-extraction` full → 243 passed
- tsc + lint clean

## Kudos

- `_is_resume_sentinel` helper with docstring naming WHY consumers must filter — single source of truth
- Regression test asserts both positive (X survives) AND inverted negative (NO sentinel in ANY bucket) — robust against future broadcast regressions
- All-sentinel section-omission guard prevents empty header reaching LLM
- Module-level comment enumerates all 5 filter sites — authoritative reference
- Self-review §"Attempt 2" carries grep evidence — audit cheap

## Decision rationale

Attempt-1 blocker was a real correctness regression: every Resume click would have permanently polluted every future Pass-2 prompt with `_resumed_without_extensions` as phantom entity type. Fix lands filter at narrowest seam (service-layer broadcast loop), backs with defence-in-depth at prompt builder, extends with surgical regression tests. Frontend UI filter + type contract close M1 documentation gap. Minors all addressed. Quality gates green.

Edge-case predicate inconsistency is only residual smell — sentinel-write centralised so all predicates agree operationally. Filed as minor.

## Next steps

APPROVED — ready for merge.
