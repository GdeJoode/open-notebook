/**
 * TanStack Query hooks for the entity-resolution review hub (Track K.6).
 *
 * Mirrors `use-model-routing.ts`: query hooks for the K.3 plan + K.5 candidate
 * queue, mutations for apply / overlay add+remove that invalidate the queues and
 * surface toast errors.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import {
  ApplyClusterInput,
  entityResolutionApi,
  OverlayCreateInput,
} from '@/lib/api/entity-resolution'
import { useToast } from '@/lib/hooks/use-toast'

export const RESOLUTION_QUERY_KEYS = {
  plan: (notebookId?: string) =>
    ['entity-resolution', 'plan', notebookId ?? 'global'] as const,
  candidates: (notebookId?: string) =>
    ['entity-resolution', 'candidates', notebookId ?? 'global'] as const,
}

export function useMergePlan(notebookId?: string) {
  return useQuery({
    queryKey: RESOLUTION_QUERY_KEYS.plan(notebookId),
    queryFn: () => entityResolutionApi.plan(notebookId),
  })
}

export function useMergeCandidates(notebookId?: string) {
  return useQuery({
    queryKey: RESOLUTION_QUERY_KEYS.candidates(notebookId),
    queryFn: () => entityResolutionApi.candidates(notebookId),
  })
}

/** Apply reviewed clusters. Destructive — only called behind the confirm step. */
export function useApplyMerge(notebookId?: string) {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: (clusters: ApplyClusterInput[]) =>
      entityResolutionApi.apply(clusters),
    onSuccess: (res) => {
      queryClient.invalidateQueries({
        queryKey: RESOLUTION_QUERY_KEYS.plan(notebookId),
      })
      queryClient.invalidateQueries({
        queryKey: RESOLUTION_QUERY_KEYS.candidates(notebookId),
      })
      toast({
        title: 'Merged',
        description: `${res.applied} cluster${res.applied === 1 ? '' : 's'} merged${
          res.skipped ? `, ${res.skipped} skipped` : ''
        }`,
      })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to apply merge',
        variant: 'destructive',
      })
    },
  })
}

/** Add a force-merge/force-split overlay (also the reject→split path). */
export function useAddOverlay(notebookId?: string) {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: (rule: OverlayCreateInput) =>
      entityResolutionApi.addOverlay(rule),
    onSuccess: (_data, rule) => {
      queryClient.invalidateQueries({
        queryKey: RESOLUTION_QUERY_KEYS.plan(notebookId),
      })
      queryClient.invalidateQueries({
        queryKey: RESOLUTION_QUERY_KEYS.candidates(notebookId),
      })
      toast({
        title: rule.kind === 'split' ? 'Kept apart' : 'Force-merged',
        description:
          rule.kind === 'split'
            ? 'These will not be proposed again'
            : 'These will be merged on the next scan',
      })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to save overlay rule',
        variant: 'destructive',
      })
    },
  })
}

export function useRemoveOverlay(notebookId?: string) {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: (id: string) => entityResolutionApi.removeOverlay(id),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: RESOLUTION_QUERY_KEYS.candidates(notebookId),
      })
      toast({ title: 'Removed', description: 'Overlay rule removed' })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to remove overlay rule',
        variant: 'destructive',
      })
    },
  })
}
