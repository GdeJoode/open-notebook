/**
 * TanStack Query hooks for the model-routing config + health (Track J.5).
 *
 * Mirrors `use-models.ts`: query hooks for the route list + health, a mutation
 * for chain upserts that invalidates the list and surfaces toast errors. Health
 * polls on an interval like `use-mineru-health.ts` so circuit-state chips stay
 * current.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { modelRoutingApi } from '@/lib/api/model-routing'
import { useToast } from '@/lib/hooks/use-toast'
import { ModelRouteUpdate, RoutingTask } from '@/lib/types/model-routing'

export const ROUTING_QUERY_KEYS = {
  routes: ['model-routes'] as const,
  health: ['model-routes', 'health'] as const,
  summary: (notebookId?: string) =>
    ['model-routes', 'summary', notebookId ?? 'global'] as const,
}

export function useModelRoutes() {
  return useQuery({
    queryKey: ROUTING_QUERY_KEYS.routes,
    queryFn: () => modelRoutingApi.list(),
  })
}

export function useModelRouteHealth() {
  return useQuery({
    queryKey: ROUTING_QUERY_KEYS.health,
    queryFn: () => modelRoutingApi.health(),
    refetchInterval: 30_000,
    staleTime: 25_000,
    retry: false,
  })
}

/**
 * Recent-routing summary (Track J.6), optionally scoped to a notebook. Polls so
 * the cloud-vs-local + fallback counts stay current after ingestion runs.
 */
export function useRoutingSummary(notebookId?: string) {
  return useQuery({
    queryKey: ROUTING_QUERY_KEYS.summary(notebookId),
    queryFn: () => modelRoutingApi.summary(notebookId),
    refetchInterval: 30_000,
    staleTime: 25_000,
    retry: false,
  })
}

export function useUpdateModelRoute() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: ({
      task,
      update,
    }: {
      task: RoutingTask
      update: ModelRouteUpdate
    }) => modelRoutingApi.update(task, update),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ROUTING_QUERY_KEYS.routes })
      toast({
        title: 'Saved',
        description: 'Provider chain updated',
      })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to update provider chain',
        variant: 'destructive',
      })
    },
  })
}
