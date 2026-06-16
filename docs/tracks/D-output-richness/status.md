# Track D — Output Rijkdom (status)

## Phase D.0 — Foundation (complete)

**Branch**: `track/d-foundation`
**Commits**:
- `57e4ace` — feat(track-d/shared): export contracts + external_ids stub + JobType.EXPORT_OBSIDIAN
- `8ff7364` — feat(track-d/repo): notebook-scoped entity+relation projections for export

**Outcomes**:
- `shared.models.export` — `ExportFilter`, `ExportReport`,
  `ObsidianExportRequest`, `JsonlExportRequest`,
  `NetworkxExportRequest`, `NetworkxFormat` Literal.
- `shared.utils.external_ids.resolve_external_ids` — V1 stub.
- `JobType.EXPORT_OBSIDIAN` — sole async export surface for V1
  (Q-D-2: JSONL/NetworkX stay sync-only).
- `EntityRepository.list_entities_for_notebook` — notebook-scoped
  projection with `embedding` omitted (Q-D-1).
- `EntityRepository.list_relations_for_notebook` — two-phase with the
  entity-id intersection that implements Q-D-4 (silent drop of edges
  into filtered entities).

**Tests**:
- `packages/shared`: 154 → 199 (+45)
- `packages/surrealdb-service`: 98 → 108 (+10 docker)
- `packages/job-queue`: 38 → 38 (no regression)
- `apps/app-main`: 508 → 508 (no regression)

**Self-review**: `docs/tracks/D-output-richness/reviews/phase-D.0-self-review.md`

**Ready for review.** Next: D.3 (NetworkX exporter) per Q-D-9 phase order.

---

## Phase D.3 — NetworkX 7-format export (DONE)

**Branch**: `track/d-networkx-export`
**Commits**: `d0f3d87..dd9ae33`

**Delivered**:
- `NetworkxExportService` — builds a `networkx.DiGraph` from the D.0
  notebook-scoped projections and serialises to any of GraphML / GEXF
  / GML / JSON-tree / edge-list / adjacency-list / pickle.
- Attribute flattening contract documented in module docstring:
  `type_tags` → CSV string, `properties` → JSON-encoded string. Keeps
  the XML serialisers happy and keeps the attribute shape identical
  across all 7 formats so downstream pandas pipelines don't branch.
- `POST /api/notebooks/{id}/export-networkx` router under
  `apps/app-main/src/app_main/api/routers/exports.py` (new router file
  that D.1 + D.2 will extend).
- Per-format Content-Type + filename extension dispatched from a
  single `_FORMAT_TABLE` so the writer and the HTTP response can't
  fall out of sync.
- DI wiring: `get_networkx_export_service()` in `dependencies.py`.
- Frontend `NetworkxExportMenu` dropdown wired into the notebook
  Schema tab next to the existing TTL download.
- Playwright spec covers GraphML + JSON-tree (the two AC formats).

**Decisions documented in the self-review**:
- Q-D-8 enforced: telemetry payload is counts + format only; the
  service test searches the JSON repr for `entity:` and `entity_id`
  substrings.
- D.0 Minor #1 (status filter): chose the cheaper Python-side post-
  filter over a SurrealQL change. Recommend D.0 SurrealQL patch when
  D.1/D.2 land so all three exporters share the gate.

**Tests**:
- `apps/app-main`: 508 → 533 (+25: 19 service + 6 router)
- `packages/shared`: 199 → 199 (unchanged, no shared-model edits)
- `frontend e2e (track-d)`: 0 → 2 (GraphML + JSON-tree)

**Self-review**: `docs/tracks/D-output-richness/reviews/phase-D.3-self-review.md`

**Ready for review.** Next: D.1 (Obsidian) per Q-D-9 phase order.

---

## Phase D.1a — Obsidian zip export (DONE)

**Branch**: `track/d-obsidian-zip`
**Commits**: see PR; commit-range filled in self-review after push.

**Delivered**:
- `ObsidianExportService` (`apps/app-main/src/app_main/services/obsidian_export_service.py`) builds a flat
  Obsidian vault for a notebook, with one `.md` per surviving entity
  plus a `README.md` index, and zips it into an in-memory archive.
- `POST /api/notebooks/{notebook_id}/export-obsidian` in
  `apps/app-main/src/app_main/api/routers/exports.py` — streams the
  zip back as `application/zip`. `mode="vault_path"` returns 501 (D.1b).
- `get_obsidian_export_service()` factory in `dependencies.py` wired
  with the entity repo + settings service (settings ready for D.1b).
- Snapshot baseline fixture
  `apps/app-main/tests/fixtures/obsidian_export_golden.md` with
  reviewer-pinned frontmatter shape.

**Decisions honoured (pre-resolved)**:
- Q-D-3: reused `shared.utils.external_ids.resolve_external_ids` stub
  (D.0 ship → `external_ids: []` in V1).
- Q-D-4: relations to filtered-out targets are silently dropped, never
  rendered as broken `[[…]]`. Covered by
  `test_broken_wikilinks_silently_dropped`.
- Q-D-5: filename collisions get `-2`, `-3`, … suffix. Covered by
  `test_filename_collision_appends_suffix`.
- Q-D-7: `io.BytesIO` + `StreamingResponse` for the zip surface.
- Q-D-8: telemetry payload carries counts only. Covered by a recursive
  walker assertion in
  `test_telemetry_emits_export_obsidian_with_counts_only`.
- Q-D-10: `aliases: []` in V1 (matches golden).

**D.0 follow-up #1 (status not in archived/merged)**:
Same Python-side post-filter as D.3. Documented in the module
docstring + self-review. Promotion to SurrealQL deferred to D.2 so all
three exporters can share the gate in one swing (matching the D.3
self-review recommendation).

**Tests**:
- `apps/app-main`: 536 → 552 (+16: 12 service + 4 router)
- `packages/shared`: 199 → 199 (no shared-model edits)

**Self-review**: `docs/tracks/D-output-richness/reviews/phase-D.1a-self-review.md`

**Ready for review.** Next: D.1b (direct-write vault mode) per
Q-D-9 phase order.

---

## Phase D.1b — Obsidian direct-write-to-vault (DONE)

**Branch**: `track/d-obsidian-vault`
**Commit range**: `043bcf3..HEAD` (single commit `7ff80c6`)

**Delivered**:
- `ObsidianExportService._export_to_vault` + `_write_to_vault`
  (`apps/app-main/src/app_main/services/obsidian_export_service.py`)
  implement the `mode="vault_path"` branch. The vault is built in
  memory (same logic as zip mode), then each file is written to
  `<Settings.vault_path>/<Settings.vault_entities_folder>/` via
  tempfile + `os.replace` (POSIX atomic rename per file).
- `VaultPathNotConfigured` exception surfaces a friendly 400 at the
  router boundary when `Settings.vault_path` /
  `Settings.vault_entities_folder` is missing.
- `POST /api/notebooks/{notebook_id}/export-obsidian` now dispatches
  on mode: zip streams as before, vault_path returns the
  `ExportReport` as JSON. Filesystem failures map to 500 with
  `entities_written` + `failed_file` in the body so the client knows
  where the batch stopped.
- `JobType.EXPORT_OBSIDIAN` handler registered in
  `apps/app-main/src/app_main/handlers.py`. Always uses
  `mode="vault_path"` — auto-pipeline entry point.

**Safety guards (defense-in-depth on top of D.1a's `_safe_entity_stem`)**:
- `vault_path` must be absolute (rejects `./relative/vault`).
- `vault_path` must exist as a directory and be writable.
- `vault_entities_folder` cannot escape `vault_path` (rejects
  `../../etc`).
- Each per-file write resolves the target path and verifies it's a
  child of the target dir before opening the tmp file.

**Decisions honoured (pre-resolved)**:
- Q-D-6: overwrite is default for `<entities_folder>/`. User-added
  files outside the export's filename set are preserved. Covered by
  `test_vault_path_overwrite_existing_md` (a `user_added.md` file
  inside the entities folder survives an export pass).
- Q-D-8: telemetry payload carries `mode: "vault_path"` +
  `vault_path_redacted: True`. Raw path NEVER in the payload.
  Covered by `test_vault_path_telemetry_redacts_path` (recursive walk
  asserting `str(tmp_path)` is absent in every string value).

**Atomicity**: per-file (not whole-batch) — documented in module
docstring + self-review. A mid-batch failure leaves earlier files
written and propagates the exception with
`{"entities_written": N, "failed_file": "<name>"}` in `exc.args`.

**Tests**:
- `apps/app-main`: 554 → 566 (+12: 7 service + 3 router + 2 handler).
  All passing in 1:38.
- `packages/job-queue`: 38 → 38 (no regression). All passing in 41s.
- `packages/shared`: unchanged.

**D.0 follow-up #1 (status not in archived/merged)**: still deferred
to D.2 per the D.1a + D.3 self-reviews — final exporter lands before
the SurrealQL promotion so all three exporters share the gate in one
swing.

**Self-review**: `docs/tracks/D-output-richness/reviews/phase-D.1b-self-review.md`

**Ready for review.** Next per plan: D.2 (JSONL stream export).

---

## Phase D.1c — Obsidian export UI dialog + E2E (DONE)

**Branch**: `track/d-obsidian-dialog`
**Commit range**: `9f5aa54..d951b6b` (4 commits on top of `main`)

**Delivered**:
- ``GET /api/notebooks/{id}/export-preview`` (counts-only) in
  ``apps/app-main/src/app_main/api/routers/exports.py``. Reuses the
  D.0 ``list_entities_for_notebook`` + ``list_relations_for_notebook``
  with the same ``min_connections`` post-filter the Obsidian service
  applies, plus a Q-D-4-style silent drop of relations whose
  endpoints didn't survive so the preview matches what the export
  emits. 404 on unknown notebook, 400 on out-of-range confidence
  (``Query()`` doesn't auto-apply the ExportFilter Pydantic bound).
- ``frontend/src/lib/types/exports.ts`` -- TypeScript mirrors of the
  Pydantic shapes (``ExportFilter``, ``ObsidianExportRequest``,
  ``ExportReport``, ``ObsidianExportMode``, ``ExportPreviewCounts``).
- ``frontend/src/lib/hooks/use-obsidian-export.ts`` -- React Query
  mutation hook. Branches on response Content-Type: zip ->
  ``URL.createObjectURL`` + hidden ``<a>`` click; JSON -> parse
  ``ExportReport``, expose via ``lastReport``, fire success toast.
- ``frontend/src/lib/utils/content-disposition.ts`` -- pure parser
  for the ``Content-Disposition`` header. Handles RFC 6266 quoted,
  unquoted token, and RFC 5987 ``filename*=UTF-8''...`` forms with
  §4.3 precedence.
- ``frontend/src/lib/hooks/use-export-preview.ts`` -- React Query
  fetch against ``/export-preview`` with 300ms debounce via
  ``use-debounce`` and ``staleTime: 30s``.
- ``frontend/src/components/notebooks/exports/ExportPreviewCounts.tsx``
  -- presentational widget with loading skeleton + error fallback +
  ``aria-live="polite"`` for screen-reader updates.
- ``frontend/src/components/notebooks/exports/ObsidianExportDialog.tsx``
  -- main dialog. Mode toggle (Tabs, Vault disabled w/ tooltip when
  ``Settings.vault_path`` empty), three sliders, two switches,
  comma-separated entity-types input, live preview counts, error
  banner, vault-mode success state.
- ``NotebookHeader.tsx`` -- new "Export Obsidian" button + dialog
  open-state next to Archive.

**Tests**:
- ``apps/app-main``: +4 router tests (``test_export_preview.py``). All
  pass in 67s under ``uv run --package app-main pytest``. Existing
  exports + obsidian service tests: 34/34 still pass.
- ``frontend`` parser unit (Playwright runner): 7/7 pass in ~1.3s.
- ``frontend`` E2E (``obsidian-export.spec.ts``): 2 tests, listed
  clean but not executed in sandbox (requires Next.js dev server
  running; no live backend dependency since all routes are mocked).
- TypeScript: ``npx tsc --noEmit`` clean.
- ESLint: only pre-existing warnings in unrelated files; no new
  warnings from the D.1c additions.

**Mental-inversion regression checks embedded**:
1. **Debounce**: E2E presses ArrowRight 5x on the slider; asserts the
   preview-counts refetch fires AT MOST twice (and at least once
   after the debounce settles). Removing the debounce makes the
   assertion fail with delta=5.
2. **Filename parser**: unit spec has a dedicated test for the
   RFC 5987 ``filename*=UTF-8''my-file.zip`` form. A simplified
   ``/filename="([^"]+)"/`` parser returns ``null`` here and fails.
3. **Vault-path disabled state**: E2E asserts the Tab is
   ``toBeDisabled()`` AND that clicking it doesn't fire the mutation
   AND that the vault-path-display label stays hidden. The dialog's
   ``handleExport`` short-circuits on the same condition, so even a
   force-click on the disabled tab can't reach the mutation.

**Self-review**: `docs/tracks/D-output-richness/reviews/phase-D.1c-self-review.md`

**Ready for review.** Next per plan: D.2 (JSONL stream export).

---

## Phase D.2 — JSONL streaming export + endpoint + button (DONE)

**Branch**: `track/d-jsonl-export`
**Commit range**: `78e46e6..feeda75` (7 commits on top of `main`)

**Delivered**:
- `JsonlExportService`
  (`apps/app-main/src/app_main/services/jsonl_export_service.py`)
  streams a zip containing `entities.jsonl` + `relations.jsonl`. Build-
  then-stream (Q-D-7): one Pydantic `model_dump(mode="json",
  exclude={"embedding"})` per entity, written line-by-line into the
  open `ZipFile` member via `archive.open(..., "w")`, then yielded in
  16KB chunks. The `embedding` exclusion is the privacy + memory
  invariant (Q-D-1).
- Per-line entity shape: `{id, canonical_name, entity_type, type_tags,
  primary_type, confidence, properties, source_documents,
  extracted_at}`. Per-line relation shape: `{id, source_entity,
  target_entity, relation_type, confidence, properties,
  source_documents}` -- the `in`/`out` -> `source_entity`/
  `target_entity` rename is for Neo4j `apoc.load.json` + LangChain RAG
  loader compatibility.
- `POST /api/notebooks/{id}/export-jsonl` in
  `apps/app-main/src/app_main/api/routers/exports.py` -- StreamingResponse
  with `application/zip` and `Content-Disposition: attachment;
  filename="<safe>.jsonl.zip"`. The filename uses the same B.2b-
  derived `_FILENAME_UNSAFE_RE` sanitisation as the other Track-D
  export endpoints.
- `get_jsonl_export_service()` DI factory in `dependencies.py`.
- `JsonlExportRequest` TypeScript mirror in
  `frontend/src/lib/types/exports.ts`.
- `useJsonlExport` hook in `frontend/src/lib/hooks/use-jsonl-export.ts`
  -- React Query mutation, always blob-downloads (no JSON branch --
  the JSONL surface has no vault-path mode). Reuses the shared
  `parseFilenameFromContentDisposition` helper from D.1c.
- `JsonlExportButton`
  (`frontend/src/components/notebooks/exports/JsonlExportButton.tsx`)
  -- single button + lightweight `Popover` (NOT a Dialog -- plan
  directive). Three sliders (`min_connections`, `min_confidence`,
  `min_relation_confidence`), two switches (`include_orphans`,
  `include_archived`), live `ExportPreviewCounts` from D.1c, error
  banner, Download/Cancel buttons. Closes on success via the hook's
  `onSuccess` callback.
- `NotebookHeader.tsx` -- "Export JSONL" button added next to "Export
  Obsidian".

**Filter pipeline (D.1c canonical)**:
- SurrealQL gate (D.0) -> status post-filter (`EXCLUDED_ENTITY_STATUSES`
  imported from `obsidian_export_service`, shared symbol) ->
  `_apply_min_connections_filter` (delegated to the static method on
  `ObsidianExportService` so any future tuning lands on both paths
  simultaneously). Q-D-4 silent drop of relations whose endpoints
  didn't survive.

**Tests**:
- `apps/app-main`: 552 -> 567 (+15: 11 service + 4 router). Existing
  62 export tests still pass. Service tests cover line-shape +
  embedding-absent assertion + min_confidence/min_connections/status
  filters + Q-D-4 silent drop + empty-notebook + streaming-yields-
  multiple-chunks + tracemalloc <200MB + single-call telemetry on
  happy + failure paths.
- `frontend e2e (track-d)`: +1 spec
  (`jsonl-export.spec.ts`) -- listed clean. Mocks preview + export
  endpoints, asserts popover opens, slider triggers debounced
  refetch, switch toggles round-trip into POST payload, download
  fires with parsed filename, popover closes on success. Pageerror
  watchdog asserted empty.
- TypeScript: `npx tsc --noEmit` clean.

**Mental-inversion regression checks embedded**:
1. **Status filter removed** -> `test_status_archived_and_merged_excluded`
   asserts only `entity:active` survives a mixed-status fixture; a
   regression that skipped `EXCLUDED_ENTITY_STATUSES` would land
   tombstones in the JSONL.
2. **min_connections filter removed** ->
   `test_min_connections_filter_drops_isolated_entities` asserts the
   isolated `entity:island` is dropped; a regression that skipped the
   shared staticmethod would include it.
3. **Full materialisation before zip** ->
   `test_streaming_yields_multiple_chunks` counts yields with a 5000-
   entity fixture; a single-yield generator (e.g. one that
   `return buf.getvalue()` instead of looping chunk_size) fails the
   `chunk_count > 1` assertion. The same test also smoke-checks
   `tracemalloc.get_traced_memory()[1] < 200MB`.
4. **Embedding accidentally serialised** -> `test_entity_line_shape`
   explicitly asserts `"embedding" not in line`; a regression that
   flipped `exclude={"embedding"}` to `exclude=set()` would still
   produce valid JSON but with the 768-float vector included.
5. **Telemetry fires multiple times** ->
   `test_metrics_emitted_once_per_export` asserts exactly one
   `record_metric` call. A naive `await record_metric(...)` inside
   the per-line loop would fail with `len(events) == 5000`.
6. **Notebook-name path injection** -> the router uses
   `_safe_filename(notebook_id, "jsonl.zip")` which strips `:`/`/`/
   etc. via `_FILENAME_UNSAFE_RE` (same regex the other export
   endpoints already use). Tested via `test_jsonl_happy_path`
   asserting `notebook_abc.jsonl.zip` (colon -> underscore).

**Self-review**: `docs/tracks/D-output-richness/reviews/phase-D.2-self-review.md`

**Ready for review.** Next per plan: D.4 (export job UX polish) or
D.0 SurrealQL promotion (status filter consolidation).

---

## Phase D.4 — Track integration + RETRO (DONE)

**Branch**: `track/d-integration-retro`
**Commit range**: starts at `ee465dd` (`docs(track-d): E2E evidence`)

**Delivered (docs-only — no code changes)**:
- `docs/tracks/D-output-richness/E2E_EVIDENCE.md` (NEW) — sandbox-
  limited live-smoke acknowledgement + what the 76+ test surface
  covers + manual smoke checklist for an operator with live external
  tools (Obsidian, Neo4j+APOC, Gephi). Mirrors Track B RETRO §"Live-
  test recommendation" deferral pattern.
- `ARCHITECTURE.md` — new §7 "Knowledge graph export surfaces
  (Track D — output richness)" documenting the three export
  endpoints, services, `export.*` metric event types, the shared
  filter pipeline (EXCLUDED_ENTITY_STATUSES +
  _apply_min_connections_filter + Q-D-4 endpoint intersection), the
  Pydantic + TypeScript model surface, and the telemetry redaction
  contract. §7 "Further reading" renumbered to §8 with D-track RETRO
  + troubleshooting links appended.
- `docs/FEATURE_ROADMAP.md` Track D — status block + phase table
  (D.0–D.4 all ✅) + restated past-tense vision paragraph. Cross-
  refs to RETRO + E2E_EVIDENCE + status.md.
- `docs/tracks/D-output-richness/RETRO.md` (NEW) — 7 "what worked"
  + 4 "what hurt" + 6 "recommendations for future tracks" (exceeds
  required 5/3/5). Closes Track B RETRO entries 2, 3, 6, 7 with
  operational evidence. Phase-by-phase attempt count (2/6 first-try
  APPROVEDs; 4/6 needed attempt-2 — D.1a, D.1c, D.2, D.3).
- `docs/troubleshooting/exports.md` (NEW) — failure-mode diagnostics
  for Obsidian zip + vault_path, JSONL + Neo4j apoc.load.json,
  NetworkX 7-format (per-format limitations table), preview-vs-
  export drift. Linked from `docs/troubleshooting/index.md`.
- `docs/tracks/D-output-richness/reviews/phase-D.4-self-review.md`
  (NEW) — AC-by-AC checklist + deliverables ledger + cross-refs to
  Track-B RETRO entries this track validated.

**Sandbox limitation acknowledged honestly**: AC #1's "run end-to-end
against the corpus notebook" is partially satisfied via the
deterministic test surface (proves the export pipelines produce
contract-matching bytes) but the live-tool consumer-side smoke
(Obsidian wikilink rendering, Neo4j apoc.load.json parse, Gephi
GraphML import) is deferred to operator session per E2E_EVIDENCE.md
manual checklist. Same trade-off Track B made (see B's RETRO
§"Live-test recommendation").

**Tests**: no code changes, no test deltas.

**Self-review**: `docs/tracks/D-output-richness/reviews/phase-D.4-self-review.md`

**Ready for review. Track D CLOSED.**
