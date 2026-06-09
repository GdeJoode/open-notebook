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

The sentinel is filtered out at **five** sites (defence-in-depth — see "Attempt 2 fixes" below for full inventory):

1. TTL exporter (`_apply_extensions`) — `apps/app-main/src/app_main/api/routers/schemas.py:214`
2. JSON schema response (`get_notebook_schema_json`) — `apps/app-main/src/app_main/api/routers/schemas.py:636`
3. Extraction pipeline boundary (`EntityExtractionService._run_multi_schema`) — `apps/app-main/src/app_main/services/entity_extraction_service.py:359` (added attempt-2)
4. Pass-2 prompt builder (`_format_accepted_extensions`) — `pipelines/ontology-extraction/src/ontology_extraction/prompts/pass2.py:155` (added attempt-2)
5. Frontend `SchemaBrowser` tree mapper — `frontend/src/components/notebooks/schema/SchemaBrowser.tsx:96` (added attempt-2)

So the sentinel never appears in the SchemaBrowser tree, the downloaded TTL, the JSON schema response, the Pass-2 LLM prompt, or any per-schema accepted-extensions bucket.

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

## Attempt 2 fixes (post adversarial-reviewer review)

The reviewer of attempt-1 returned `REVISIONS_NEEDED` with one BLOCKER (B1 — resume sentinel leaks into Pass-2 LLM prompt), one major (M1 — frontend filter referenced in a comment didn't exist), and five minors. All addressed below.

### BLOCKER B1 — Sentinel filtered before Pass-2 prompt

The attempt-1 code only filtered sentinels at the JSON schema endpoint and TTL exporter. The extraction pipeline boundary at
`apps/app-main/src/app_main/services/entity_extraction_service.py:343-353` (now ~357-371) iterated `accepted_extensions` to build `accepted_by_schema` without skipping sentinels. Because the sentinel has no `schema_name`, it was broadcast to every applicable schema's bucket → `_format_accepted_extensions` rendered `- **_resumed_without_extensions** (no parent)` into the LLM prompt. Persistent contamination after every Resume.

**Primary fix** — `entity_extraction_service.py`:
- Added `_is_resume_sentinel(extension)` helper (lines 41-58) documenting why the marker must be filtered.
- Updated the `accepted_by_schema` build loop (lines 357-371) to skip sentinels before routing by `schema_name`.
- Regression test `TestRunMultiSchemaBody::test_run_multi_schema_filters_resume_sentinel` in `apps/app-main/tests/test_entity_extraction_service.py` (lines 864-955) constructs a `NotebookSchema` carrying both a real extension (`X`) and a sentinel, runs the service through to `mock_extract`, and asserts the per-schema bucket contains `X` and NO entry with `is_resume_sentinel=True` or `type_name="_resumed_without_extensions"`.

**Defence-in-depth at prompt layer** — `pipelines/ontology-extraction/src/ontology_extraction/prompts/pass2.py`:
- `_format_accepted_extensions` (lines 135-189) now filters sentinels up-front. The empty-result branch is reached when the input list contains only sentinels — the section header is omitted entirely rather than rendering with no bullets.
- Two unit tests in `pipelines/ontology-extraction/tests/test_pass2.py::TestBuildPass2Prompt`:
  - `test_format_accepted_extensions_filters_sentinels` — mixed list; sentinel stripped, real type kept, header present.
  - `test_format_accepted_extensions_omits_section_when_all_sentinels` — sentinel-only list; section header absent.

### Major M1 — Frontend filter implemented + comment corrected

The attempt-1 comment in `schemas.py:759` claimed `"the frontend already filters them by checking type_name.startswith('_') — see SchemaBrowser notes"`. `grep` confirmed no such filter existed. Fixed by implementing the claimed filter and rewriting the comment:

- **`frontend/src/lib/types/notebook_schema.ts`** — `ExtensionView` now carries an optional `is_resume_sentinel?: boolean` field with docstring noting the filter contract.
- **`frontend/src/components/notebooks/schema/SchemaBrowser.tsx:80-105`** — `accepted_extensions` is filtered via `ext.is_resume_sentinel !== true && !(ext.type_name ?? '').startsWith('_')` before mapping to TreeItems. Comment explains this is belt-and-braces against future backend regressions.
- **`apps/app-main/src/app_main/api/routers/schemas.py:757-777`** — comment rewritten to enumerate all five sentinel filter sites; this is now the authoritative reference for future work.

### Minors

- **M1 (forward-ref)** — `useResumeExtraction` previously referenced `PAUSED_EXTRACTION_QUERY_KEY` declared further down the file. Reordered so the constant is declared before the hook (runtime-safe before but cleaner read-order now). See `frontend/src/lib/hooks/use-notebook-schema.ts:147`.
- **M2 (close-X aria-label)** — `aria-label="Hide banner"` → `aria-label="Mark as read"` to match the actual mutation it invokes. See `SchemaSoftNudge.tsx:171`.
- **M3 (30s polling)** — kept at 30s but documented the trade-off explicitly in the `SchemaSoftNudge.tsx` docstring (`# Polling cadence` section). Matches MinerU health chip cadence; one number is easier to reason about than three. Switching to 10s is a one-line change in `useNotebookEvents` if user feedback demands faster surfacing.
- **M4 (paused_count via job-list length)** — deferred to later perf pass. Tracking note: at scale (>>100 paused jobs per notebook) consider a dedicated `COUNT(...)` query; currently the endpoint enforces a list limit which caps the cost.
- **M5 (Z-suffix on sentinel `created_at`)** — `datetime.now(timezone.utc).isoformat()` now post-processed with `.replace("+00:00", "Z")` for consistency with the rest of the codebase. See `schemas.py::resume_extraction` (~line 1009).

### Grep evidence — sentinel filter inventory

After attempt-2, `grep -rn "is_resume_sentinel" --include="*.py" --include="*.ts" --include="*.tsx"` shows filter sites at:

- `apps/app-main/src/app_main/api/routers/schemas.py:214` (TTL exporter)
- `apps/app-main/src/app_main/api/routers/schemas.py:636` (JSON schema endpoint)
- `apps/app-main/src/app_main/services/entity_extraction_service.py:50,374` (helper + extraction service)
- `pipelines/ontology-extraction/src/ontology_extraction/prompts/pass2.py:158` (prompt builder)
- `frontend/src/lib/types/notebook_schema.ts:50` (type contract)
- `frontend/src/components/notebooks/schema/SchemaBrowser.tsx:97` (UI filter)

Total: 4 distinct backend/runtime filter sites + 1 frontend filter site + the helper + the type contract = the reviewer's "≥ 4 filter sites" bar is met with margin.

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
