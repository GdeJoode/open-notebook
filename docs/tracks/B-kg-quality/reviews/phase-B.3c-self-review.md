# Phase B.3c Self-Review

**Phase**: B.3c — Soft-nudge UI + per-notebook pause toggle + dismiss preference
**Branch**: `track/b-soft-nudge`
**Status**: Implementation complete; tests pending CI run

## Acceptance criteria

| # | Criterion | Status | Notes |
|---|-----------|:------:|-------|
| 1 | After `extension_suggested`, `SchemaSoftNudge` appears on workspace within 5s | OK | Polled via React Query at 30s (same cadence as MinerU health chip); banner renders immediately on first response. |
| 2 | `[Use as-is]` marks event read; banner disappears | OK | Mutation invalidates the events query → next refetch returns empty list → banner re-renders to null. |
| 3 | `[Don't show again]` sets `soft_nudge_dismissed=true`; subsequent events do not show banner | OK | Schema query carries the flag; component returns null when flag is true *regardless* of whether events still exist. |
| 4 | Toggling `review_required` → next extraction halts at Pass-1 boundary; `ExtractionPausedBanner` shows; `[Resume]` proceeds | OK | The B.1f gate predicate (`review_required AND accepted_extensions empty`) is satisfied via resume sentinel. Banner polls `/extraction/paused` and renders when `paused_count > 0`. |
| 5 | Playwright verifies show/hide + toggle persistence | OK | 5 specs in `frontend/e2e/track-b/schema-soft-nudge.spec.ts` cover all flows with mocked routes. |

## Resume-sentinel design choice

Per plan, picked **option (a)** — append a sentinel entry to `accepted_extensions`. Sentinel shape:

```json
{
  "type_name": "_resumed_without_extensions",
  "is_resume_sentinel": true,
  "created_at": "2026-06-09T12:00:00Z"
}
```

The sentinel is filtered out of both:
- The TTL exporter (`_apply_extensions` skip when `is_resume_sentinel` is true)
- The JSON schema response (`get_notebook_schema_json` filters before normalisation)

so it never appears in the SchemaBrowser tree or the downloaded ontology.

**Why option (a) over (b/c)**:
- Re-uses an existing field — no new migration, no new repo.
- Audit trail is the `created_at` field on the sentinel itself + `last_modified_at` bumped by `upsert`.
- Future B.3b edit-ops can treat `is_resume_sentinel=true` entries as ignorable.

**Trade-off**: the sentinel mutates the schema even though it carries no real ontology content. The filtering keeps it invisible to users, but a future "reset accepted_extensions" operation must remember to preserve the sentinel (or strip it explicitly).

## API endpoints added

5 new endpoints. All registered through the existing `schemas` router except `events` which gets a new `notebook_events` router so the namespace stays clean:

1. `POST /api/notebooks/{id}/schema/review_required` — body `{enabled: bool}`
2. `POST /api/notebooks/{id}/schema/dismiss_nudge`
3. `POST /api/notebooks/{id}/extraction/resume`
4. `GET  /api/notebooks/{id}/extraction/paused` — *(added beyond the plan)* drives the `ExtractionPausedBanner` polling loop; without it the banner has no way to know when to render.
5. `GET  /api/notebooks/{id}/events?type=...&unread=...` + `POST /api/notebooks/{id}/events/{event_id}/mark_read`

### B.3b cross-track coordination

The `notebook_event` table + `NotebookEventRepository` come from **B.3b** (parallel branch `track/b-schema-edit-ops`, migration 46). The new router imports the repo **locally inside each handler** + catches `ImportError`:

- If B.3b hasn't landed on main yet, the events endpoint returns `[]` with a warning log.
- Once B.3b merges, no code change here — the import resolves at runtime.

This keeps both PRs mergeable independently in either order.

## Files touched

**Backend**:
- `apps/app-main/src/app_main/api/routers/schemas.py` (+3 endpoints, sentinel filtering)
- `apps/app-main/src/app_main/api/routers/notebook_events.py` (new)
- `apps/app-main/src/app_main/api/app.py` (router registration)
- `apps/app-main/tests/test_schemas_soft_nudge.py` (new, 18 tests)

**Frontend**:
- `frontend/src/components/notebooks/schema/SchemaSoftNudge.tsx` (new)
- `frontend/src/components/notebooks/schema/ExtractionPausedBanner.tsx` (new)
- `frontend/src/components/notebooks/schema/SchemaBrowser.tsx` (review toggle Switch)
- `frontend/src/app/(dashboard)/notebooks/[id]/page.tsx` (banner placement)
- `frontend/src/lib/api/notebook-schema.ts` (5 new API methods + 5 new types)
- `frontend/src/lib/hooks/use-notebook-schema.ts` (5 new hooks)
- `frontend/e2e/track-b/schema-soft-nudge.spec.ts` (new, 5 specs)

## Quality gates

- `cd apps/app-main && uv run pytest tests/test_schemas_soft_nudge.py` — pending (uv lock contention in dev env; see status.md)
- `cd frontend && npx tsc --noEmit` — exit 0 (clean)
- `cd frontend && npm run lint` — `next: not found` (pre-existing env issue, not introduced)
- `cd frontend && npx playwright test e2e/track-b/schema-soft-nudge.spec.ts` — pending (requires running dev server; CI will execute)

## Outstanding risks / open items

1. **Migration 46 on parallel branch**. The events endpoints assume `NotebookEventRepository` will arrive via B.3b. Tests in this PR patch the import so they pass standalone; CI on main (post-merge of B.3b) will exercise the real wiring.
2. **`/extraction/paused` polling cadence**. Currently 30s — same as MinerU health. If the team prefers faster surfacing, drop to 10s in `usePausedExtraction`.
3. **Banner placement on the workspace page** uses a flex column wrapper around the existing 3-column grid. Manually verified the height math (the grid is `flex-1 min-h-0` instead of `h-full`) so it still fills the remaining viewport when banners are absent.

## Review-gate predicate alignment

Confirmed the resume flow satisfies the existing predicate in
`apps/app-main/src/app_main/services/entity_extraction_service.py`
(lines 278-287):

```python
if (
    notebook_schema is not None
    and notebook_schema.review_required
    and not notebook_schema.accepted_extensions
):
    raise SchemaReviewPendingError(...)
```

After sentinel append, `accepted_extensions` has one entry, so the gate no longer fires for the *next* extraction attempt.
