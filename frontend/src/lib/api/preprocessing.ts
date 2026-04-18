import { apiClient } from './client'

export interface DocumentClassification {
  document_type: string
  has_toc: boolean
  has_hierarchical_structure: boolean
  language: string
  domain: string
  key_topics: string[]
  formality_level: string
  suggested_ontologies: string[]
}

export interface PreprocessingResult {
  id?: string
  source_id: string
  naive_summary: string
  classification: DocumentClassification
  filtered_chunk_ids: string[]
  removed_chunk_ids: string[]
  total_chunks: number
  noise_stats: Record<string, number>
  structure_stats: Record<string, unknown>
  created?: string
}

export const preprocessingApi = {
  run: async (
    sourceId: string,
    privacy?: 'public' | 'internal' | 'confidential',
  ): Promise<PreprocessingResult> => {
    const { data } = await apiClient.post('/preprocessing/run', {
      source_id: sourceId,
      ...(privacy ? { privacy } : {}),
    })
    return data
  },

  get: async (sourceId: string): Promise<PreprocessingResult | null> => {
    const { data } = await apiClient.get(`/preprocessing/${sourceId}`)
    return data
  },

  update: async (
    sourceId: string,
    body: {
      classification?: Partial<DocumentClassification>
      filtered_chunk_ids?: string[]
      removed_chunk_ids?: string[]
    }
  ): Promise<PreprocessingResult> => {
    const { data } = await apiClient.patch(`/preprocessing/${sourceId}`, body)
    return data
  },
}
