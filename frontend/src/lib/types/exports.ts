/**
 * Track-D export contracts -- mirrors ``shared.models.export`` so the
 * dialog (D.1c) and the obsidian-export hook can type-check the request
 * + response shapes against the Pydantic source of truth.
 *
 * The shapes here are the on-the-wire JSON, NOT the Pydantic model
 * class. Optional fields omit ``Field(default=None)`` defaults: the
 * frontend sends either a value or omits the key (FastAPI treats both
 * the same).
 *
 * Keep this file structural and dependency-free -- it is imported by
 * hooks AND components AND e2e specs, so adding a runtime import here
 * would balloon every consumer's bundle.
 */

/**
 * Filter knob set applied before entities/relations are emitted.
 *
 * Defaults match ``ExportFilter`` in Pydantic:
 *   - ``min_connections=5``
 *   - ``min_confidence=0.9``
 *   - ``min_relation_confidence=undefined`` (inherits ``min_confidence``)
 *   - ``entity_types=undefined`` (all types)
 *   - ``include_orphans=false``
 *   - ``include_archived=false``
 *
 * Bounds (Pydantic-enforced):
 *   - ``min_connections >= 0``
 *   - ``min_confidence`` in ``[0.0, 1.0]``
 *   - ``min_relation_confidence`` in ``[0.0, 1.0]`` when set
 */
export interface ExportFilter {
  min_connections: number
  min_confidence: number
  min_relation_confidence?: number
  entity_types?: string[]
  include_orphans: boolean
  include_archived: boolean
}

/**
 * Obsidian export delivery mode. Two literals:
 *   - ``zip``: server streams an archive back, browser downloads it.
 *   - ``vault_path``: server writes the vault to ``Settings.vault_path``
 *     directly and returns an ``ExportReport`` JSON body.
 */
export type ObsidianExportMode = 'zip' | 'vault_path'

/**
 * Request body for ``POST /api/notebooks/{id}/export-obsidian``.
 */
export interface ObsidianExportRequest {
  mode: ObsidianExportMode
  filter: ExportFilter
}

/**
 * Server-side report returned by export handlers. Used by the
 * vault_path branch (the zip branch returns the archive blob, not
 * this).
 *
 * Per Q-D-8 the payload is counts + filter snapshot only -- no entity
 * IDs leak through this surface so the dialog can render the success
 * banner without PII concerns.
 */
export interface ExportReport {
  entities_written: number
  relations_written: number
  files_written: number
  bytes_written: number
  duration_ms: number
  filter_applied: ExportFilter
  metadata?: Record<string, unknown>
}

/**
 * Response shape for ``GET /api/notebooks/{id}/export-preview``.
 *
 * Counts-only (matches the Python ``ExportPreviewCounts`` Pydantic
 * model) so the dialog can poll on every slider change without
 * worrying about PII leakage.
 */
export interface ExportPreviewCounts {
  entity_count: number
  relation_count: number
}

/**
 * Request body for ``POST /api/notebooks/{id}/export-jsonl`` (Phase D.2).
 *
 * The JSONL surface is sync-only in V1 and always returns an
 * ``application/zip`` body containing ``entities.jsonl`` +
 * ``relations.jsonl``. There's no mode field -- the only knob is the
 * filter -- so the request payload is a thin wrapper around
 * ``ExportFilter``.
 */
export interface JsonlExportRequest {
  filter: ExportFilter
}

/**
 * Request body for ``POST /api/notebooks/{id}/export/okf`` (Track OKF.2).
 *
 * The OKF endpoint reuses the Track-D ``ObsidianExportRequest`` Pydantic
 * model server-side, so the wire body carries both ``mode`` and ``filter``.
 * OKF only ever emits an ``application/zip`` bundle (there is no vault_path
 * write mode), so the hook always sends ``mode: 'zip'`` -- the field exists
 * purely to satisfy the shared request contract.
 */
export interface OkfExportRequest {
  filter: ExportFilter
}

/**
 * OKF omitted-field ledger (OKF-D3) surfaced in the ``metadata`` block of
 * the export report. Maps each dropped substrate field to a human-readable
 * reason it has no OKF representation. Copied verbatim from
 * ``OKF_OMITTED_FIELDS`` server-side so the lossy-by-design loss is
 * explicit, never silent.
 */
export type OkfOmittedFields = Record<string, string>

/**
 * ``metadata`` block of the OKF ExportReport. OKF packs its bundle-specific
 * counts + the omitted-field ledger here rather than widening the shared
 * ``ExportReport`` shape.
 */
export interface OkfExportMetadata {
  notes_written?: number
  sources_written?: number
  dropped_relations?: number
  omitted_fields?: OkfOmittedFields
  okf_version?: string
  [key: string]: unknown
}

/**
 * OKF export report — the JSON payload the server serialises into the
 * ``X-OKF-Export-Report`` response header alongside the zip body. It is the
 * same ``ExportReport`` shape as the Obsidian/JSONL exports, narrowed so the
 * dialog can read the OKF-specific ``metadata`` (esp. ``omitted_fields``)
 * without an ``unknown`` cast.
 */
export interface OkfExportReport extends Omit<ExportReport, 'metadata'> {
  metadata?: OkfExportMetadata
}

/**
 * Body returned by ``POST /api/notebooks/{id}/export/okf`` when the notebook
 * is large enough to be deferred to the job queue (HTTP 202). The export runs
 * asynchronously; poll the job-status surface with ``job_id``.
 */
export interface OkfExportDeferred {
  job_id: string
  status: string
  entity_count: number
  threshold: number
}
