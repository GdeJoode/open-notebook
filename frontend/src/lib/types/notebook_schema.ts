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

/** A Pass-1 proposal in the notebook's queue. */
export interface ExtensionView {
  /**
   * `extension_id` is assigned by B.1c when the LLM proposes an
   * extension. Older rows may not carry it (the storage shape is
   * FLEXIBLE); accept/reject match by `type_name`.
   */
  extension_id?: string | null
  type_name: string
  parent_type?: string | null
  description?: string | null
  rationale?: string | null
  properties: PropertyDef[]
}

/**
 * OE.5: the ontologies are shared by every notebook. The response carries the
 * declared base ontology's types (read-only; edits happen in the ontology
 * editor) and the Pass-1 proposal queue.
 */
export interface NotebookSchemaResponse {
  notebook_id: string
  base_ontology: string
  base_ontology_types: EntityTypeNode[]
  pending_extensions: ExtensionView[]
  coverage_pct: number
  review_required: boolean
  soft_nudge_dismissed: boolean
}

/** `GET /schema/extensions/{type}/accept`: where a proposal goes by default. */
export interface AcceptTarget {
  type_name: string
  parent_type: string | null
  /** The ontology that introduces the parent; `schema_core` when none does. */
  default_ontology: string
  ontologies: string[]
  /** Where the name (or an alias of it) is already a shared type. Then the
   *  proposal is marked as covered instead of accepted. */
  defined_in: { ontology: string; type: string }[]
}

/** `POST /schema/extensions/{type}/accept`: the queue and what was published
 *  (`null` when the proposal had already been accepted). */
export interface AcceptExtensionResponse {
  notebook_schema: NotebookSchemaResponse
  published: {
    name: string
    version: string
    warnings: unknown[]
    retype: {
      changes: number
      collisions: number
      flag_only: number
      caused_by_this_edit: number
      collisions_caused_by_this_edit: number
    }
  } | null
}

// ---------------------------------------------------------------------------
// Phase B.3d — re-extract prompt contracts
// ---------------------------------------------------------------------------
//
// `NotebookEventView` is declared in
// `frontend/src/lib/api/notebook-schema.ts` as the canonical projection
// returned by the B.3c `/events` endpoint. B.3d intentionally does NOT
// re-declare it here — duplicating the shape across modules drifted
// during attempt 1 (B.3d's version added `payload` + `notebook_id` that
// the B.3c contract does not actually return). Banners that need a
// `payload` bag should fetch it via a follow-up endpoint instead.

export interface ReextractCandidatesResponse {
  notebook_id: string
  source_ids: string[]
  count: number
}

export interface ReextractResponse {
  notebook_id: string
  jobs_enqueued: number
  source_ids: string[]
  enqueued_source_ids: string[]
  skipped_source_ids: string[]
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
