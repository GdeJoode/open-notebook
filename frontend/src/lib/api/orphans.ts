/**
 * API client for the B.5b orphans-dashboard endpoints.
 *
 * Mirrors the convention in `lib/api/notebook-schema.ts`: a single
 * exported object literal, one method per endpoint, returning the
 * typed body (axios envelope unwrapped at the call-site).
 */

import apiClient from './client'
import type { OrphansResponse, ReconnectResponse } from '@/lib/types/orphans'

const nbPath = (id: string) => `/notebooks/${encodeURIComponent(id)}`

export const orphansApi = {
  /**
   * Fetch pending + archived orphan entities for a notebook. Returns
   * an empty `items` list when the notebook has no orphans.
   */
  list: async (notebookId: string): Promise<OrphansResponse> => {
    const response = await apiClient.get<OrphansResponse>(
      `${nbPath(notebookId)}/orphans`,
    )
    return response.data
  },

  /**
   * Trigger a manual reconnect for one orphan entity. The backend
   * runs B.5a synchronously for that orphan and returns the new
   * status so the UI can flip the row in place.
   */
  reconnect: async (
    notebookId: string,
    entityId: string,
  ): Promise<ReconnectResponse> => {
    const response = await apiClient.post<ReconnectResponse>(
      `${nbPath(notebookId)}/orphans/${encodeURIComponent(entityId)}/reconnect`,
    )
    return response.data
  },
}
