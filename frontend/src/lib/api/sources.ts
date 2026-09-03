import type { AxiosResponse } from 'axios'

import apiClient from './client'
import {
  SourceListResponse,
  SourceDetailResponse,
  SourceResponse,
  SourceStatusResponse,
  CreateSourceRequest,
  UpdateSourceRequest,
  RunEntitiesOptions,
  RelatedSourceHybrid,
} from '@/lib/types/api'

export type PrivacyLevel = 'public' | 'internal' | 'confidential'

export interface DoclingPipelineConfig {
  // Privacy & model routing
  privacy?: PrivacyLevel
  // Parser engine override for this reparse run (A.2). When undefined,
  // the backend uses the global parser_engine setting. Empty string in
  // the form maps to undefined here (see PipelineConfigPanel).
  parser_engine?: 'simple' | 'docling' | 'mineru' | 'auto'
  // Auto-mode confidence threshold override (A.2). Only meaningful
  // when parser_engine === 'auto'.
  docling_min_confidence?: number
  // Docling settings
  docling_ocr_engine?: string
  docling_ocr_languages?: string[]
  docling_table_mode?: string
  docling_pipeline?: string
  docling_vlm_model?: string
  docling_auto_export_images?: boolean
  docling_image_scale?: number
  // Full docling conversion config (I.D-2). Page images off, enrichment on
  // by default. Picture classification undefined => follow the VLM toggle.
  docling_generate_page_images?: boolean
  docling_do_code_enrichment?: boolean
  docling_do_formula_enrichment?: boolean
  docling_do_picture_classification?: boolean
  docling_chunking_enabled?: boolean
  docling_chunking_method?: string
  docling_chunking_max_tokens?: number
}

export interface StructureGraphNode {
  id: string
  self_ref: string | null
  element_type: string | null
  text: string | null
  page: number | null
  level: number | null
  sequence: number | null
  bbox: number[] | null
  // Owning chunk id for leaf nodes (drives the bbox-highlight wiring); null on
  // section nodes.
  chunk_id: string | null
}

export interface StructureGraphEdge {
  source: string
  target: string
  type: 'parent_of' | 'next_node' | 'derived_from'
}

export interface StructureGraphResponse {
  nodes: StructureGraphNode[]
  edges: StructureGraphEdge[]
  total_nodes: number
  truncated: boolean
}

export const DEFAULT_PIPELINE_CONFIG: DoclingPipelineConfig = {
  privacy: 'internal',
  docling_ocr_engine: 'easyocr',
  docling_ocr_languages: ['en', 'nl'],
  docling_table_mode: 'accurate',
  docling_pipeline: 'vlm',
  docling_vlm_model: 'granite-docling-258m',
  docling_auto_export_images: true,
  docling_image_scale: 2.0,
  docling_generate_page_images: false,
  // Off by default to preserve pre-I.D-2 behavior (these enrichments were never
  // forwarded to docling before, so the effective default was off). Users opt in.
  docling_do_code_enrichment: false,
  docling_do_formula_enrichment: false,
  // docling_do_picture_classification intentionally left undefined so the
  // backend keeps it coupled to the VLM toggle unless the user overrides it.
  docling_chunking_enabled: true,
  docling_chunking_method: 'hybrid',
  docling_chunking_max_tokens: 512,
}

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
    formData.append('private', String(data.private ?? false))

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

  reprocess: async (id: string, overrides?: DoclingPipelineConfig) => {
    const response = await apiClient.post<{
      command_id: string | null
      status: string
      overrides_applied: string[]
    }>(`/sources/${id}/reprocess`, overrides || {})
    return response.data
  },

  runPreprocessing: async (id: string, privacy?: 'public' | 'internal' | 'confidential') => {
    const response = await apiClient.post<Record<string, unknown>>('/preprocessing/run', {
      source_id: id,
      ...(privacy ? { privacy } : {}),
    })
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

  getImages: async (id: string) => {
    const response = await apiClient.get<{ images: string[] }>(
      `/sources/${id}/images`
    )
    return response.data
  },

  updateChunk: async (sourceId: string, chunkId: string, data: {
    text?: string
    element_type?: string
    positions?: number[][]
    is_content?: boolean
    chapter?: string | null
  }) => {
    const response = await apiClient.patch(`/sources/${sourceId}/chunks/${chunkId}`, data)
    return response.data
  },

  deleteChunk: async (sourceId: string, chunkId: string) => {
    await apiClient.delete(`/sources/${sourceId}/chunks/${chunkId}`)
  },

  createChunk: async (sourceId: string, data: {
    text: string
    element_type?: string
    physical_page: number
    positions?: number[][]
    is_content?: boolean
  }) => {
    const response = await apiClient.post(`/sources/${sourceId}/chunks`, data)
    return response.data
  },

  mergeChunk: async (
    sourceId: string,
    chunkId: string,
    targetChunkId?: string,
  ) => {
    const response = await apiClient.post(
      `/sources/${sourceId}/chunks/${chunkId}/merge`,
      { target_chunk_id: targetChunkId ?? null },
    )
    return response.data
  },

  splitChunk: async (sourceId: string, chunkId: string, cursorOffset: number) => {
    const response = await apiClient.post<{ chunks: unknown[] }>(
      `/sources/${sourceId}/chunks/${chunkId}/split`,
      { cursorOffset },
    )
    return response.data
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

  // Document structure graph (Track I.F). Returns Sigma/graphology-friendly
  // nodes + edges. `page_limit` defaults to 100 on the backend (hard cap 500).
  getStructureGraph: async (
    id: string,
    opts?: { page?: number; pageLimit?: number }
  ) => {
    const params: Record<string, number> = {}
    if (opts?.page != null) params.page = opts.page
    if (opts?.pageLimit != null) params.page_limit = opts.pageLimit
    const response = await apiClient.get<StructureGraphResponse>(
      `/sources/${id}/structure-graph`,
      { params }
    )
    return response.data
  },

  getPdfUrl: async (id: string) => {
    const { getApiUrl } = await import('@/lib/config')
    const apiUrl = await getApiUrl()
    return `${apiUrl}/api/sources/${id}/pdf`
  },

  getPagePreviewUrl: async (id: string, page: number, dpi = 150) => {
    const { getApiUrl } = await import('@/lib/config')
    const apiUrl = await getApiUrl()
    return `${apiUrl}/api/sources/${id}/page-preview?page=${page}&dpi=${dpi}`
  },

  // Hybrid related-sources retrieval (Track R.3/R.5). Fuses the dense
  // embedding signal and the KG-proximity signal via RRF, KG-prominent by
  // default. Each result carries per-signal provenance + the KG driving
  // entities (the "why matched" lineage). Returns [] (not 404) for a source
  // with no aggregate embedding and no shared entities.
  getRelatedHybrid: async (
    id: string,
    opts?: { k?: number; preset?: 'kg-heavy' | 'balanced' }
  ): Promise<RelatedSourceHybrid[]> => {
    const params: Record<string, string | number> = {}
    if (opts?.k != null) params.k = opts.k
    if (opts?.preset != null) params.preset = opts.preset
    const response = await apiClient.get<RelatedSourceHybrid[]>(
      `/sources/${id}/related-hybrid`,
      { params }
    )
    return response.data
  },

  getPageCount: async (id: string) => {
    const response = await apiClient.get<{
      page_count: number
      pages: Array<{
        page_number: number
        width: number
        height: number
      }>
    }>(`/sources/${id}/page-count`)
    return response.data
  },
}
