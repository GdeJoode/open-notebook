import apiClient from './client'

export interface VaultImportItem {
  id: string
  canonical_name: string
  aliases: string[]
  entity_type: string
  source_file: string
  status: string
  imported_at: string
}

export interface VaultExportItem {
  id: string
  canonical_name: string
  aliases: string[]
  entity_type: string
  source_ids: string[]
  status: string
  created_at: string
}

export interface VaultSyncResult {
  new: number
  changed: number
  unchanged: number
  total_files: number
}

export interface ApplyResult {
  accepted: number
  rejected: number
  dictionary_size?: number
}

export interface WriteResult {
  written: number
  rejected: number
}

export const vaultApi = {
  sync: async () => {
    const response = await apiClient.post<VaultSyncResult>('/vault/sync')
    return response.data
  },

  getPendingImports: async () => {
    const response = await apiClient.get<VaultImportItem[]>('/vault/imports/pending')
    return response.data
  },

  applyImports: async (acceptedIds: string[], rejectedIds: string[]) => {
    const response = await apiClient.post<ApplyResult>('/vault/imports/apply', {
      accepted_ids: acceptedIds,
      rejected_ids: rejectedIds,
    })
    return response.data
  },

  getPendingExports: async () => {
    const response = await apiClient.get<VaultExportItem[]>('/vault/exports/pending')
    return response.data
  },

  writeExports: async (acceptedIds: string[], rejectedIds: string[]) => {
    const response = await apiClient.post<WriteResult>('/vault/exports/write', {
      accepted_ids: acceptedIds,
      rejected_ids: rejectedIds,
    })
    return response.data
  },

  getDictionary: async () => {
    const response = await apiClient.get<Record<string, string>>('/vault/dictionary')
    return response.data
  },
}
