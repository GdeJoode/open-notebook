import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { zoteroApi, ZoteroConnectConfig } from '@/lib/api/zotero'
import { toast } from 'sonner'

const ZOTERO_KEYS = {
  status: ['zotero', 'status'] as const,
  collections: ['zotero', 'collections'] as const,
}

export function useZoteroStatus() {
  return useQuery({
    queryKey: ZOTERO_KEYS.status,
    queryFn: () => zoteroApi.getStatus(),
    staleTime: 30_000,
    retry: false,
  })
}

export function useZoteroCollections(enabled = true) {
  return useQuery({
    queryKey: ZOTERO_KEYS.collections,
    queryFn: () => zoteroApi.getCollections(),
    enabled,
    retry: false,
  })
}

export function useZoteroConnect() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (config: ZoteroConnectConfig) => zoteroApi.connect(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['zotero'] })
      toast.success('Connected to Zotero')
    },
    onError: (error: Error) => {
      toast.error(`Connection failed: ${error.message}`)
    },
  })
}

export function useZoteroDisconnect() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => zoteroApi.disconnect(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['zotero'] })
      toast.success('Disconnected from Zotero')
    },
  })
}

export function useZoteroPush() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ sourceId, collection }: { sourceId: string; collection?: string }) =>
      zoteroApi.pushSource(sourceId, collection),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['zotero'] })
      if (data.action === 'duplicate_found') {
        toast.info(`Already in Zotero: ${data.title || data.zotero_key}`)
      } else {
        toast.success(`Pushed to Zotero: ${data.title || data.zotero_key}`)
      }
    },
    onError: () => {
      toast.error('Failed to push to Zotero')
    },
  })
}

export function useZoteroSync() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => zoteroApi.sync(),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['zotero'] })
      toast.success(
        `Sync complete: ${data.pushed} pushed, ${data.pulled} pulled` +
        (data.conflicts > 0 ? `, ${data.conflicts} conflicts` : '')
      )
    },
    onError: () => {
      toast.error('Zotero sync failed')
    },
  })
}
