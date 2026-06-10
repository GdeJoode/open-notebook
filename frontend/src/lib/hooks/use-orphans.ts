/**
 * React Query hooks for the B.5b orphans dashboard.
 *
 * - `useNotebookOrphans(notebookId)` -- fetches the pending + archived
 *   orphan rows. Powers the OrphansDashboard table.
 *
 * - `useReconnectOrphan(notebookId)` -- triggers a manual reconnect.
 *   On success, invalidates the orphan query so the table refreshes
 *   from the source of truth. We deliberately invalidate (instead of
 *   replacing the cache directly) because the reconnect response only
 *   carries the per-entity outcome; the surrounding counts may have
 *   shifted in unrelated ways during the round-trip.
 */

import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationResult,
} from '@tanstack/react-query'

import { orphansApi } from '@/lib/api/orphans'
import type { OrphansResponse, ReconnectResponse } from '@/lib/types/orphans'
import { useToast } from '@/lib/hooks/use-toast'

export const NOTEBOOK_ORPHANS_QUERY_KEY = (id: string) =>
  ['notebook-orphans', id] as const

export function useNotebookOrphans(notebookId: string) {
  return useQuery<OrphansResponse>({
    queryKey: NOTEBOOK_ORPHANS_QUERY_KEY(notebookId),
    queryFn: () => orphansApi.list(notebookId),
    enabled: !!notebookId,
  })
}

export function useReconnectOrphan(
  notebookId: string,
): UseMutationResult<ReconnectResponse, Error, string> {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  return useMutation({
    mutationFn: (entityId: string) =>
      orphansApi.reconnect(notebookId, entityId),
    onSuccess: (data) => {
      queryClient.invalidateQueries({
        queryKey: NOTEBOOK_ORPHANS_QUERY_KEY(notebookId),
      })
      toast({
        title:
          data.reconnected > 0
            ? 'Orphan reconnected'
            : 'Reconnect attempt recorded',
        description:
          data.reconnected > 0
            ? 'The entity is back in the active graph.'
            : 'The orphan stays pending; another attempt may succeed.',
      })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to reconnect orphan',
        variant: 'destructive' as const,
      })
    },
  })
}
