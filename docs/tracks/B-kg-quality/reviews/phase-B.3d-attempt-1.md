# Review — Track B Phase B.3d attempt 1

**Branch**: `track/b-reextract-prompt` (HEAD `8fbbb36`, branched from pre-B.3c-merge)
**PR**: #24
**Decision**: **REVISIONS_NEEDED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-10

## Summary

B.3d implementation correct on own merits — 27 backend tests, sound service-level dedup, banner UI accessible. **Fatal problem**: B.3c merged onto main in a shape the implementer did not anticipate. Main now ships a DIFFERENT `GET /events` endpoint inside `notebook_events.py`. Rebasing produces real conflicts; the wire contracts differ between B.3c (on main) and B.3d (branch).

## Acceptance criteria

5/6 PASS; AC#5 (per-source "Schema changed" badge) deferred — banner covers primary need.

## Blockers (3, all overlap-driven)

### B1: Duplicate `/events` + `/mark_read` endpoints with INCOMPATIBLE contracts

- **main** (B.3c): `/events?type=str,str&unread=bool&limit=int` returns `NotebookEventView`
- **branch** (B.3d): `/events?type=str&unread_only=bool` returns `_NotebookEventView`
- **main mark_read**: full event_id, returns `{event_id, success}`
- **branch mark_read**: event_id suffix, returns `{ok: true}`

**Fix**: Drop branch's `/events` + `/mark_read` handlers. Re-point banner hooks at main's `notebook_events.py` endpoint. Re-write `useSchemaChangedEvents` to delegate to existing `useNotebookEvents(id, {types: ['schema_changed'], unread: true})`.

### B2: Rebase requires substantial manual conflict resolution + frontend reconciliation

Same root cause. Branch's 2-dot diff DELETES B.3c files (banner, soft-nudge spec, notebook_events.py). Naive merge destroys B.3c. 3-way rebase required preserving:
- All B.3c surface (banner mounts, dismiss endpoint, paused-extraction endpoints, full hook + API)
- B.3d's reextract candidates/jobs endpoints
- Both `schema-soft-nudge.spec.ts` AND `schema-reextract.spec.ts` pass post-rebase

### B3: `page.tsx` integration conflict — 3 banners must coexist

Post-rebase must render SchemaSoftNudge + ExtractionPausedBanner + ReextractPromptBanner together.

## Majors (2)

### M4: `PAUSED_FOR_REVIEW` NOT in `_INFLIGHT_STATUSES`

`reextract_service.py:69` — `_INFLIGHT_STATUSES = {"queued", "processing", "retrying"}`. A source whose latest job is `PAUSED_FOR_REVIEW` will get a new job enqueued → TWO competing jobs (paused awaiting `/extraction/resume` + freshly queued). Worker picks new one; paused becomes orphan state.

**Fix**: Add `paused_for_review` to `_INFLIGHT_STATUSES` (user must resume first) OR actively cancel paused job before enqueueing. Pin test.

### M5: No annotation on V1 "all sources" candidate test

`test_returns_all_notebook_sources` asserts V1 shape but no docstring explains WHY V1 ignores op type. Future B.3d-r1 narrowing would silently over-narrow.

## Minors (5, optional)

1. `enqueue_reextract` silently lenient on cross-notebook source_ids
2. Banner returns null during first poll (up to 30s) — documented intentional
3. `useReextractCandidates` enabled condition reasonable
4. `mark_read` URL `event_id_suffix` prefix-strip is one-shot (`notebook_event:notebook_event:abc` slips through)
5. AC#5 badge deferred — acceptable

## Tests verified

- `test_reextract_service.py`: 13 passed (env hang on `llm_manager` blocked full router tests; static-verified 14 router tests structurally correct)
- `schema-reextract.spec.ts`: 4 hermetic specs

## Kudos

- Self-review preempted the overlap concern — accurate hazard-flagging
- `ReextractService.enqueue_reextract_jobs` textbook async batch op — per-element try/except + structural protocol + no shared mutable state
- Service test "completed/failed jobs do NOT dedup" — exact behavioral pin
- Default-select-every-candidate dialog UX matches user intent
- Resilient degradation: candidates returns [] on repo error; mark_read failures logged-not-raised

## Next steps

1. Rebase on current main
2. Reconcile schemas.py (keep B.3c blocks + B.3d reextract endpoints, drop B.3d /events handlers)
3. Re-point banner hooks at main's `useNotebookEvents` / `useMarkEventRead`
4. Update Playwright mock to `{event_id, success}` shape
5. Address Major #4: PAUSED_FOR_REVIEW dedup semantics + service-level test
6. Add docstring annotation per Major #5
7. Re-run full suites (backend + frontend + both Playwright specs)
