/**
 * API client for the Track F.1 per-notebook quality-audit endpoints.
 * Mirrors `lib/api/orphans.ts`.
 */

import apiClient from './client'
import type { AuditRunResponse } from '@/lib/types/audit'

const nbPath = (id: string) => `/notebooks/${encodeURIComponent(id)}`

export const auditApi = {
  /** Fetch the most recent audit run (empty findings if never run). */
  latest: async (notebookId: string): Promise<AuditRunResponse> => {
    const response = await apiClient.get<AuditRunResponse>(
      `${nbPath(notebookId)}/audit`,
    )
    return response.data
  },

  /** Run the six LLM-free checks now and return the fresh run. */
  run: async (notebookId: string): Promise<AuditRunResponse> => {
    const response = await apiClient.post<AuditRunResponse>(
      `${nbPath(notebookId)}/audit`,
    )
    return response.data
  },

  /** Run the on-demand deep checks (conflicting facts + provenance gaps). */
  runDeep: async (notebookId: string): Promise<AuditRunResponse> => {
    const response = await apiClient.post<AuditRunResponse>(
      `${nbPath(notebookId)}/audit/deep`,
    )
    return response.data
  },
}
