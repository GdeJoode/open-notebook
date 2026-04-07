import apiClient from './client'

export interface ZoteroStatus {
  connected: boolean
  library_type?: string
  auto_push?: boolean
  default_collection?: string | null
  sync_interval_minutes?: number
  last_sync?: string | null
  library_version?: number
  sync_status?: string
}

export interface ZoteroConnectConfig {
  library_id: string
  api_key: string
  library_type?: string
  default_collection?: string
  auto_push?: boolean
  sync_interval_minutes?: number
}

export interface ZoteroPushResult {
  zotero_key: string
  zotero_version: number
  action: 'created' | 'duplicate_found' | 'updated'
  title?: string
}

export interface ZoteroSyncResult {
  pushed: number
  pulled: number
  conflicts: number
  deleted: number
  new_version: number
}

export interface ZoteroCollection {
  key: string
  name: string
}

export const zoteroApi = {
  getStatus: async () => {
    const response = await apiClient.get<ZoteroStatus>('/zotero/status')
    return response.data
  },

  connect: async (config: ZoteroConnectConfig) => {
    const response = await apiClient.post<{ status: string; library_version: number }>(
      '/zotero/connect',
      config
    )
    return response.data
  },

  disconnect: async () => {
    const response = await apiClient.delete('/zotero/disconnect')
    return response.data
  },

  getCollections: async () => {
    const response = await apiClient.get<ZoteroCollection[]>('/zotero/collections')
    return response.data
  },

  pushSource: async (sourceId: string, collection?: string) => {
    const response = await apiClient.post<ZoteroPushResult>(
      `/zotero/sources/${sourceId}/push`,
      { collection }
    )
    return response.data
  },

  sync: async () => {
    const response = await apiClient.post<ZoteroSyncResult>('/zotero/sync')
    return response.data
  },
}
