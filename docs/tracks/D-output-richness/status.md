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
