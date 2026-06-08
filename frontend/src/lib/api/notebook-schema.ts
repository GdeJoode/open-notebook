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

  /**
   * Resolve the absolute TTL download URL for a notebook's schema.
   *
   * Returned as a string rather than a Blob fetch so the
   * `<TtlDownloadButton>` can drive the download with a plain
   * `<a download>` — which lets the browser handle the file save
   * dialog natively (including filename hint from the
   * `Content-Disposition` header). Going via axios + Blob would
   * round-trip the file through JS memory for no benefit.
   *
   * Note: callers still need a valid auth header for the download to
   * succeed when `OPEN_NOTEBOOK_PASSWORD` is set. The `<a>` route
   * relies on the user's browser session having the bearer token; in
   * the auth-disabled local-dev case this is a non-issue.
   */
  getTtlUrl: async (notebookId: string): Promise<string> => {
    const { getApiUrl } = await import('@/lib/config')
    const apiUrl = await getApiUrl()
    return `${apiUrl}/api/notebooks/${encodeURIComponent(notebookId)}/schema.ttl`
  },
}
