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

export const notebookSchemaApi = {
  /**
   * Fetch the notebook's effective schema (base ontology + accepted +
   * pending extensions + flags). Returns the bare-base-ontology defaults
   * when the notebook has not been pass-1-processed yet.
   */
  get: async (notebookId: string): Promise<NotebookSchemaResponse> => {
    const response = await apiClient.get<NotebookSchemaResponse>(
      `/notebooks/${encodeURIComponent(notebookId)}/schema`,
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
      `/notebooks/${encodeURIComponent(notebookId)}/pass1_results`,
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
