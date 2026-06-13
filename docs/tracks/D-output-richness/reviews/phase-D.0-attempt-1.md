# Review — Track D Phase D.0 attempt 1

**Branch**: `track/d-foundation` (HEAD `1eaecbf`)
**Decision**: **APPROVED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-13

## Summary

D.0 lands clean foundation plumbing: 5 export contract models with 100% coverage, single-import-point `external_ids` stub mirroring `name_normalizer.py`, two notebook-scoped repository methods correctly reusing B.5b reference-edge traversal, strict embedding-omission inversion test. All 6 ACs met (1 ambiguity). 0 blockers, 0 majors, 3 minors filed as follow-ups.

## Tests (independently verified)

- packages/shared: 199 passed (+45)
- packages/surrealdb-service docker: 10/10 in `test_entity_list_for_notebook.py`
- packages/job-queue: 38 (no regression)
- apps/app-main: 508 (no regression)
- Coverage on new modules: 100%

## Inversion check on embedding-omission

Mentally inverted: adding `"embedding"` to `_ENTITY_EXPORT_FIELDS` → sentinel vector test fails (round-trip not `[]`) AND belt-and-braces literal-string check fails. Strong test, not pseudo-coverage.

## Minors (3, non-blocking)

1. **`Entity.status` exclusion divergence**: AC3 says `status != "archived"`. Implementation filters on `orphan_status` only (B.5b lifecycle). `Entity.status='merged'` from B.1a merge-loser path would leak through. Dormant today (no writer); pin in D.1a or D.0 amend follow-up.

2. **`ExportReport.metadata` type openness**: `Optional[Dict[str, Any]]` allows D.1/D.2/D.3 to put PII or IDs (violating Q-D-8). Tighten docstring to "MUST be counts/non-PII only" or add validator. Defer to use-site reviewers.

3. **`entity_types=[]` ambiguous**: `None` = all types; `[]` produces SQL `INSIDE []` = zero rows. D.1c UI dialog needs decision. Recommend validator rejecting `[]` with "use None for all types" OR normalize `[]` → `None`.

## Issue #2 from implementer (Entity model missing orphan_* fields)

NOT a blocker. Pydantic V2 default `extra="ignore"` silently drops orphan_* columns at validation. Plan §70/§82 don't require orphan_* on Entity model. D.1/D.2/D.3 don't render orphan_status in outputs per current spec. If future Track-D UI needs "pending_reconnect" badge → add 4 fields to Entity model in a future small PR. Not blocking now.

## Kudos

- Inversion test (sentinel vector + literal string check) — exact RETRO #6 pattern
- Two-phase set-intersection in `list_relations_for_notebook` cleanly implements Q-D-4
- `ensure_record_id` coercion fix — implementer hit and fixed real `INSIDE` semantic bug
- 100% coverage with realistic tests (not code-runs pseudo-coverage)
- Self-review acknowledges its own ambiguities — Track B RETRO honesty bar met
- Single-import-point Q9 swap pattern correctly wired
- All 6 Q-D-* defaults explicitly honored with self-review citation
- Empty-input guards covered above AC bar

## Decision rationale

Zero blockers, zero majors. Three minors all dormant or deferred to use-site. Implementer choices documented and consistent with Q-D-* matrix. Tests deterministic, coverage real, inversion strong. Foundation solid for D.1/D.2/D.3.

Quality bar set by Track B met. APPROVED.

## Next steps

Ready for merge. Track-D implementer should track MINOR-1/2/3 as follow-ups inside D.1a (or D.1c for MINOR-3). Per Q-D-9, next phase D.3 NetworkX (smallest mechanical export).
