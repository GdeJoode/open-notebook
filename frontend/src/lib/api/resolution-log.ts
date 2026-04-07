import apiClient from './client'

export interface ResolutionLogEntry {
  id: string
  entity_a_text: string
  entity_b_text: string
  entity_a_label: string
  entity_b_label: string
  match: boolean
  confidence: number
  match_method: string
  match_reasoning: string
  iterations: number
  source_document_a?: string | null
  source_document_b?: string | null
  source_section_a?: string | null
  source_section_b?: string | null
  matched_by_model?: string | null
  match_timestamp?: string | null
  status: 'pending' | 'accepted' | 'rejected' | 'auto_accepted'
  reviewed_at?: string | null
  source_id?: string | null
}

export interface PaginatedResolutionLog {
  items: ResolutionLogEntry[]
  total: number
  limit: number
  offset: number
}

export interface ResolutionLogStats {
  total: number
  matches: number
  pending: number
  accepted: number
  rejected: number
  auto_accepted: number
}

export interface ResolutionLogFilters {
  limit?: number
  offset?: number
  status?: string
  source_id?: string
  match_method?: string
  match_only?: boolean
}

export const resolutionLogApi = {
  list: async (params?: ResolutionLogFilters) => {
    const response = await apiClient.get<PaginatedResolutionLog>('/resolution-log', { params })
    return response.data
  },

  getStats: async () => {
    const response = await apiClient.get<ResolutionLogStats>('/resolution-log/stats')
    return response.data
  },

  getMethods: async () => {
    const response = await apiClient.get<string[]>('/resolution-log/methods')
    return response.data
  },

  updateStatus: async (id: string, status: 'accepted' | 'rejected') => {
    const response = await apiClient.patch(`/resolution-log/${id}/status`, null, {
      params: { status },
    })
    return response.data
  },

  bulkUpdateStatus: async (ids: string[], status: 'accepted' | 'rejected') => {
    const response = await apiClient.post<{ updated: number }>('/resolution-log/bulk-status', {
      ids,
      status,
    })
    return response.data
  },
}
