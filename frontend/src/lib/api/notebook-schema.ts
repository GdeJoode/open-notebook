/**
 * API client for the B.3a notebook-schema endpoints.
 *
 * Mirrors the convention in `lib/api/sources.ts`: a single exported
 * object literal with one method per endpoint, returning the typed
 * response body (axios envelope unwrapped at the call-site).
 */

import apiClient from './client'
import type {
  NotebookSchemaResponse,
  Pass1ResultView,
} from '@/lib/types/notebook_schema'

/**
 * Payload contracts for the B.3b schema-edit mutations.
 *
 * Hand-typed against `RenameTypeRequest` / `MergeTypesRequest` /
 * `SplitTypeRequest` in `apps/app-main/src/app_main/api/routers/schemas.py`.
 * Kept here (rather than under `lib/types/`) because they are mutation-
 * scoped and don't leak into render code — co-locating with the client
 * keeps the surface narrow.
 */
export interface RenameTypeRequest {
  old_name: string
  new_name: string
}

export interface MergeTypesRequest {
  type_names: string[]
  merged_name: string
}

export interface SplitTypeRequest {
  type_name: string
  into: string[]
  criterion: string
}

/**
 * View-model of a single notebook event surfaced by the B.3c
 * SchemaSoftNudge banner. Mirrors `NotebookEventView` in
 * `apps/app-main/src/app_main/api/routers/notebook_events.py`.
 */
export interface NotebookEventView {
  id: string
  event_type: string
  message?: string | null
  source_id?: string | null
  created_at?: string | null
  read_at?: string | null
}

/** Echo response for POST /schema/review_required (B.3c). */
export interface ReviewRequiredResponse {
  notebook_id: string
  review_required: boolean
}

/** Echo response for POST /schema/dismiss_nudge (B.3c). */
export interface DismissNudgeResponse {
  notebook_id: string
  soft_nudge_dismissed: boolean
}

/** Result of POST /extraction/resume (B.3c). */
export interface ResumeExtractionResponse {
  notebook_id: string
  resumed_count: number
  sentinel_added: boolean
}

/** Response shape for GET /extraction/paused (B.3c). */
export interface PausedExtractionStatus {
  notebook_id: string
  paused_count: number
  paused_source_ids: string[]
}

/** Echo response for POST /events/{id}/mark_read (B.3c). */
export interface MarkReadResponse {
  event_id: string
  success: boolean
}

const nbPath = (id: string) => `/notebooks/${encodeURIComponent(id)}`

export const notebookSchemaApi = {
  /**
   * Fetch the notebook's effective schema (base ontology + accepted +
   * pending extensions + flags). Returns the bare-base-ontology defaults
   * when the notebook has not been pass-1-processed yet.
   */
  get: async (notebookId: string): Promise<NotebookSchemaResponse> => {
    const response = await apiClient.get<NotebookSchemaResponse>(
      `${nbPath(notebookId)}/schema`,
    )
    return response.data
  },

  /**
   * Fetch per-source pass-1 results for the notebook, newest-first.
   * Each row already carries `source_title`; the table can render
   * without a second round trip.
   */
  listPass1Results: async (notebookId: string): Promise<Pass1ResultView[]> => {
    const response = await apiClient.get<Pass1ResultView[]>(
      `${nbPath(notebookId)}/pass1_results`,
    )
    return response.data
  },

  // --------------------------------------------------------------------
  // Phase B.3b edit ops
  // --------------------------------------------------------------------
  //
  // Each mutation returns the FULL `NotebookSchemaResponse` so the
  // caller can replace the React Query cache directly without a follow-
  // up GET — guaranteeing UI updates within one round-trip (the AC#4
  // "Schema tab updates within 200ms" guarantee).

  /** Accept a pending extension by type name. */
  acceptExtension: async (
    notebookId: string,
    typeName: string,
  ): Promise<NotebookSchemaResponse> => {
    const response = await apiClient.post<NotebookSchemaResponse>(
      `${nbPath(notebookId)}/schema/extensions/${encodeURIComponent(typeName)}/accept`,
    )
    return response.data
  },

  /** Reject (drop) a pending extension by type name. */
  rejectExtension: async (
    notebookId: string,
    typeName: string,
  ): Promise<NotebookSchemaResponse> => {
    const response = await apiClient.post<NotebookSchemaResponse>(
      `${nbPath(notebookId)}/schema/extensions/${encodeURIComponent(typeName)}/reject`,
    )
    return response.data
  },

  /** Record a rename as a synonym in `accepted_extensions`. */
  renameType: async (
    notebookId: string,
    payload: RenameTypeRequest,
  ): Promise<NotebookSchemaResponse> => {
    const response = await apiClient.post<NotebookSchemaResponse>(
      `${nbPath(notebookId)}/schema/rename`,
      payload,
    )
    return response.data
  },

  /** Record a merge of N types into one. */
  mergeTypes: async (
    notebookId: string,
    payload: MergeTypesRequest,
  ): Promise<NotebookSchemaResponse> => {
    const response = await apiClient.post<NotebookSchemaResponse>(
      `${nbPath(notebookId)}/schema/merge`,
      payload,
    )
    return response.data
  },

  /** Record a split of one type into N new ones. */
  splitType: async (
    notebookId: string,
    payload: SplitTypeRequest,
  ): Promise<NotebookSchemaResponse> => {
    const response = await apiClient.post<NotebookSchemaResponse>(
      `${nbPath(notebookId)}/schema/split`,
      payload,
    )
    return response.data
  },

  /** Soft-delete a type by adding it to `excluded_types`. */
  deleteType: async (
    notebookId: string,
    typeName: string,
  ): Promise<NotebookSchemaResponse> => {
    const response = await apiClient.delete<NotebookSchemaResponse>(
      `${nbPath(notebookId)}/schema/types/${encodeURIComponent(typeName)}`,
    )
    return response.data
  },

  /**
   * B.3c — toggle the notebook's `review_required` flag. When enabled,
   * the next extraction halts at the Pass-1 boundary until the user
   * resumes via `/extraction/resume`.
   */
  setReviewRequired: async (
    notebookId: string,
    enabled: boolean,
  ): Promise<ReviewRequiredResponse> => {
    const response = await apiClient.post<ReviewRequiredResponse>(
      `/notebooks/${encodeURIComponent(notebookId)}/schema/review_required`,
      { enabled },
    )
    return response.data
  },

  /**
   * B.3c — dismiss the soft-nudge banner for this notebook. The flag
   * persists until the orchestrator re-arms it (coverage drops again).
   */
  dismissNudge: async (notebookId: string): Promise<DismissNudgeResponse> => {
    const response = await apiClient.post<DismissNudgeResponse>(
      `/notebooks/${encodeURIComponent(notebookId)}/schema/dismiss_nudge`,
    )
    return response.data
  },

  /**
   * B.3c — resume paused extraction for this notebook. Adds the resume
   * sentinel if the review-gate predicate would still fire, and moves
   * any `PAUSED_FOR_REVIEW` jobs back to `QUEUED`.
   */
  resumeExtraction: async (
    notebookId: string,
  ): Promise<ResumeExtractionResponse> => {
    const response = await apiClient.post<ResumeExtractionResponse>(
      `/notebooks/${encodeURIComponent(notebookId)}/extraction/resume`,
    )
    return response.data
  },

  /**
   * B.3c — list paused extraction jobs for this notebook. Drives the
   * `ExtractionPausedBanner` — the banner shows when `paused_count > 0`.
   */
  listPausedExtraction: async (
    notebookId: string,
  ): Promise<PausedExtractionStatus> => {
    const response = await apiClient.get<PausedExtractionStatus>(
      `/notebooks/${encodeURIComponent(notebookId)}/extraction/paused`,
    )
    return response.data
  },

  /**
   * B.3c — list notebook events for the soft-nudge banner. Pass
   * `unread=true` from the polling loop; pass `type` as a comma-
   * separated list to filter.
   */
  listEvents: async (
    notebookId: string,
    opts: { types?: string[]; unread?: boolean; limit?: number } = {},
  ): Promise<NotebookEventView[]> => {
    const params: Record<string, string> = {}
    if (opts.types && opts.types.length > 0) {
      params.type = opts.types.join(',')
    }
    if (opts.unread) {
      params.unread = 'true'
    }
    if (typeof opts.limit === 'number') {
      params.limit = String(opts.limit)
    }
    const response = await apiClient.get<NotebookEventView[]>(
      `/notebooks/${encodeURIComponent(notebookId)}/events`,
      { params },
    )
    return response.data
  },

  /**
   * B.3c — mark a single notebook event read. Idempotent.
   */
  markEventRead: async (
    notebookId: string,
    eventId: string,
  ): Promise<MarkReadResponse> => {
    const response = await apiClient.post<MarkReadResponse>(
      `/notebooks/${encodeURIComponent(
        notebookId,
      )}/events/${encodeURIComponent(eventId)}/mark_read`,
    )
    return response.data
  },
}

// NOTE: A `getTtlUrl` helper used to live here, returning an absolute
// URL so a plain `<a download>` could trigger the export. We removed it
// for two reasons:
//   1. `TtlDownloadButton` already drives the download through the
//      shared `apiClient` (Blob path), so the bearer-token interceptor
//      fires automatically when `OPEN_NOTEBOOK_PASSWORD` is set.
//   2. Returning a raw URL would side-step that interceptor — a
//      footgun for future callers who may not realise the URL is
//      unauthenticated unless cookies happen to be in play.
// Re-introduce only if both the auth model and an alternative download
// path are re-evaluated together.
