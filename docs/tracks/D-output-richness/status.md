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
