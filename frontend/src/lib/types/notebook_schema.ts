/**
 * TypeScript mirrors of the Phase B.3a JSON contracts.
 *
 * Hand-typed against
 * `apps/app-main/src/app_main/api/routers/schemas.py` —
 * `NotebookSchemaResponse` and `Pass1ResultView`.
 *
 * Kept narrow on purpose: the backend response is curated for the
 * Schema-tab UI, not a full mirror of `NotebookSchema` / `Pass1Result`.
 * When B.3b lands the edit mutations, extend (or split) this module
 * rather than coupling to the DB model.
 */

export interface PropertyDef {
  name: string
  description?: string | null
  data_type: string
  required: boolean
}

export interface EntityTypeNode {
  name: string
  description?: string | null
  /** Optional parent type — drives the tree hierarchy. */
  parent_type?: string | null
  properties: PropertyDef[]
}

export interface ExtensionView {
  /**
   * `extension_id` is assigned by B.1c when the LLM proposes an
   * extension. Older rows may not carry it (the storage shape is
   * FLEXIBLE), in which case the B.3b accept/reject buttons fall
   * back to matching by `type_name`.
   */
  extension_id?: string | null
  type_name: string
  parent_type?: string | null
  description?: string | null
  rationale?: string | null
  properties: PropertyDef[]
  /**
   * B.3c resume-sentinel marker. Older rows / non-sentinel entries
   * lack this field entirely. The SchemaBrowser filters items where
   * `is_resume_sentinel === true` so users never see the marker as a
   * tree node — it carries no schema content. The backend already
   * filters sentinels at the JSON-schema endpoint (defence-in-depth
   * here in case a future endpoint forgets).
   */
  is_resume_sentinel?: boolean
}

export interface NotebookSchemaResponse {
  notebook_id: string
  base_ontology: string
  base_ontology_types: EntityTypeNode[]
  accepted_extensions: ExtensionView[]
  pending_extensions: ExtensionView[]
  /** Type names hidden via the B.3b soft-delete (DELETE /schema/types/...). */
  excluded_types: string[]
  coverage_pct: number
  review_required: boolean
  soft_nudge_dismissed: boolean
}

export interface Pass1ResultView {
  id: string
  source_id: string
  /**
   * `null` when the underlying source has been deleted but the
   * append-only pass1_results row survived. The UI renders this as
   * "(deleted source)".
   */
  source_title?: string | null
  schema_attempted: string
  detected_schema: string
  confidence_in_choice: number
  coverage_pct: number
  uncovered_concept_count: number
  /** ISO-8601 string from the backend; `null` for legacy rows. */
  created?: string | null
}
