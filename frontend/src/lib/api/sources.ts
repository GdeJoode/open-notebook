import type { AxiosResponse } from 'axios'

import apiClient from './client'
import {
  SourceListResponse,
  SourceDetailResponse,
  SourceResponse,
  SourceStatusResponse,
  CreateSourceRequest,
  UpdateSourceRequest,
  ExtractionResultResponse,
  RunEntitiesOptions,
} from '@/lib/types/api'

export const sourcesApi = {
  list: async (params?: {
    notebook_id?: string
    limit?: number
    offset?: number
    sort_by?: 'created' | 'updated'
    sort_order?: 'asc' | 'desc'
  }) => {
    const response = await apiClient.get<SourceListResponse[]>('/sources', { params })
    return response.data
  },

  get: async (id: string) => {
    const response = await apiClient.get<SourceDetailResponse>(`/sources/${id}`)
    return response.data
  },

  create: async (data: CreateSourceRequest & { file?: File }) => {
    // Always use FormData to match backend expectations
    const formData = new FormData()
    
    // Add basic fields
    formData.append('type', data.type)
    
    if (data.notebooks !== undefined) {
      formData.append('notebooks', JSON.stringify(data.notebooks))
    }
    if (data.notebook_id) {
      formData.append('notebook_id', data.notebook_id)
    }
    if (data.title) {
      formData.append('title', data.title)
    }
    if (data.url) {
      formData.append('url', data.url)
    }
    if (data.content) {
      formData.append('content', data.content)
    }
    if (data.transformations !== undefined) {
      formData.append('transformations', JSON.stringify(data.transformations))
    }
    
    const dataWithFile = data as CreateSourceRequest & { file?: File }
    if (dataWithFile.file instanceof File) {
      formData.append('file', dataWithFile.file)
    }
    
    formData.append('embed', String(data.embed ?? false))
    formData.append('delete_source', String(data.delete_source ?? false))
    formData.append('async_processing', String(data.async_processing ?? false))

    if (data.processing_overrides && Object.keys(data.processing_overrides).length > 0) {
      formData.append('processing_overrides', JSON.stringify(data.processing_overrides))
    }
    
    const response = await apiClient.post<SourceResponse>('/sources', formData)
    return response.data
  },

  update: async (id: string, data: UpdateSourceRequest) => {
    const response = await apiClient.put<SourceListResponse>(`/sources/${id}`, data)
    return response.data
  },

  delete: async (id: string) => {
    await apiClient.delete(`/sources/${id}`)
  },

  status: async (id: string) => {
    const response = await apiClient.get<SourceStatusResponse>(`/sources/${id}/status`)
    return response.data
  },

  upload: async (file: File, notebook_id: string) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('notebook_id', notebook_id)
    formData.append('type', 'upload')
    formData.append('async_processing', 'true')
    
    const response = await apiClient.post<SourceResponse>('/sources', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  retry: async (id: string) => {
    const response = await apiClient.post<SourceResponse>(`/sources/${id}/retry`)
    return response.data
  },

  runSummaries: async (id: string) => {
    const response = await apiClient.post<{ command_id: string | null; status: string }>(`/sources/${id}/run-summaries`)
    return response.data
  },

  runEntities: async (id: string, options?: RunEntitiesOptions) => {
    const response = await apiClient.post<{ command_id: string | null; status: string }>(
      `/sources/${id}/run-entities`, options || {}
    )
    return response.data
  },

  getExtractionResult: async (id: string): Promise<ExtractionResultResponse> => {
    const response = await apiClient.get<ExtractionResultResponse>(
      `/sources/${id}/extraction-result`
    )
    return response.data
  },

  runFiltering: async (id: string, options?: {
    dedup_enabled?: boolean
    dedup_similarity_threshold?: number
    fuzzy_dedup_enabled?: boolean
    fuzzy_similarity_threshold?: number
    embedding_dedup_enabled?: boolean
    embedding_similarity_threshold?: number
    edge_prediction_enabled?: boolean
  }) => {
    const response = await apiClient.post<{
      source_id: string
      entities_before: number
      entities_after: number
      entities_removed: number
      merge_groups: number
      predicted_edges: number
    }>(`/sources/${id}/run-filtering`, options || {})
    return response.data
  },

  runEmbed: async (id: string) => {
    const response = await apiClient.post<{ command_id: string | null; status: string }>(`/sources/${id}/run-embed`)
    return response.data
  },

  getProcessingLogs: async (id: string) => {
    const response = await apiClient.get<Array<{ level: string; message: string; timestamp: number }>>(
      `/sources/${id}/processing-logs`
    )
    return response.data
  },

  downloadFile: async (id: string): Promise<AxiosResponse<Blob>> => {
    return apiClient.get(`/sources/${id}/download`, {
      responseType: 'blob',
    })
  },

  getImageUrl: async (sourceId: string, filename: string) => {
    const { getApiUrl } = await import('@/lib/config')
    const apiUrl = await getApiUrl()
    return `${apiUrl}/api/sources/${sourceId}/images/${filename}`
  },

  getChunks: async (id: string) => {
    const response = await apiClient.get<{
      chunks: Array<{
        id: string
        text: string
        order: number
        physical_page: number
        printed_page: number | null
        chapter: string | null
        paragraph_number: number | null
        element_type: string
        positions: number[][]
        metadata: Record<string, unknown>
        is_content: boolean
      }>
      total_chunks: number
      has_spatial_data: boolean
    }>(`/sources/${id}/chunks`)
    return response.data
  },

  getPdfUrl: async (id: string) => {
    // For react-pdf-highlighter PdfLoader, we need an absolute URL
    // because it loads PDFs directly in the browser, not through axios
    const { getApiUrl } = await import('@/lib/config')
    const apiUrl = await getApiUrl()
    return `${apiUrl}/api/sources/${id}/pdf`
  },
}
