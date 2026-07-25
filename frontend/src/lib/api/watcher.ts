import apiClient from './client'

export interface WatcherActivity {
  name: string
  status: 'processed' | 'errored'
  when: string
}

export interface WatcherStatus {
  enabled: boolean
  paths: string[]
  default_notebook_id: string | null
  debounce_seconds: number
  recent: WatcherActivity[]
}

/** Session-authed inbox file-watcher status (Track G.6, read-only). */
export const watcherApi = {
  getStatus: async (): Promise<WatcherStatus> => {
    const res = await apiClient.get<WatcherStatus>('/watcher/status')
    return res.data
  },
}
