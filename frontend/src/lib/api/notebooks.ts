import apiClient from './client'
import { NotebookResponse, CreateNotebookRequest, UpdateNotebookRequest } from '@/lib/types/api'

/** Conflict row carried by NotebookMergeReport (B.6). */
export interface NotebookMergeConflict {
  normalized_name: string
  conflicting_type_tags: string[]
  source_notebook_ids: string[]
}

/** Response shape from POST /api/notebooks/merge (B.6). */
export interface NotebookMergeReport {
  entities_merged: number
  relations_created: number
  conflicts: NotebookMergeConflict[]
  source_notebook_ids: string[]
  target_notebook_id: string
  dry_run: boolean
}

/** Request body for POST /api/notebooks/merge (B.6). */
export interface NotebookMergeRequest {
  source_ids: string[]
  target_id: string
  type_match_required?: boolean
  dry_run?: boolean
}

export const notebooksApi = {
  list: async (params?: { archived?: boolean; order_by?: string }) => {
    const response = await apiClient.get<NotebookResponse[]>('/notebooks', { params })
    return response.data
  },

  get: async (id: string) => {
    const response = await apiClient.get<NotebookResponse>(`/notebooks/${id}`)
    return response.data
  },

  create: async (data: CreateNotebookRequest) => {
    const response = await apiClient.post<NotebookResponse>('/notebooks', data)
    return response.data
  },

  update: async (id: string, data: UpdateNotebookRequest) => {
    const response = await apiClient.put<NotebookResponse>(`/notebooks/${id}`, data)
    return response.data
  },

  delete: async (id: string) => {
    await apiClient.delete(`/notebooks/${id}`)
  },

  addSource: async (notebookId: string, sourceId: string) => {
    const response = await apiClient.post(`/notebooks/${notebookId}/sources/${sourceId}`)
    return response.data
  },

  removeSource: async (notebookId: string, sourceId: string) => {
    const response = await apiClient.delete(`/notebooks/${notebookId}/sources/${sourceId}`)
    return response.data
  },

  /**
   * Cross-notebook entity/relation merge (B.6).
   *
   * Server-side aggregates entities + relations from `source_ids` into
   * `target_id`. Same call with `dry_run=true` returns the counters
   * without writing — used by the dialog's Preview button.
   */
  merge: async (data: NotebookMergeRequest) => {
    const response = await apiClient.post<NotebookMergeReport>(
      '/notebooks/merge',
      data,
    )
    return response.data
  },
}