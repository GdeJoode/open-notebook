import apiClient from './client'

export interface StrategyInfo {
  name: string
  implemented: boolean
}

export interface SummaryListItem {
  id: string
  source_id: string
  strategy: string
  document_summary: string | null
  num_layers: number
  num_input_chunks: number
  node_count: number
  created: string
}

export interface SummaryNode {
  text: string
  layer: number
  source_chunk_ids: string[]
  children_ids: string[]
  node_id: string | null
  section_title: string | null
  metadata: Record<string, unknown>
}

export interface SummaryDetail extends SummaryListItem {
  summary_nodes: SummaryNode[]
  metadata: Record<string, unknown>
}

export interface GenerateSummaryRequest {
  source_id: string
  strategy: string
  config?: Record<string, unknown>
}

export const summariesApi = {
  listStrategies: async () => {
    const response = await apiClient.get<StrategyInfo[]>('/summaries/strategies')
    return response.data
  },

  list: async (sourceId?: string) => {
    const response = await apiClient.get<SummaryListItem[]>('/summaries', {
      params: sourceId ? { source_id: sourceId } : undefined,
    })
    return response.data
  },

  get: async (id: string) => {
    const response = await apiClient.get<SummaryDetail>(`/summaries/${id}`)
    return response.data
  },

  generate: async (data: GenerateSummaryRequest) => {
    const response = await apiClient.post<SummaryDetail>('/summaries/generate', data)
    return response.data
  },

  delete: async (id: string) => {
    await apiClient.delete(`/summaries/${id}`)
  },
}
