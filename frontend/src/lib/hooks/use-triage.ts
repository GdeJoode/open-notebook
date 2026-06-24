/**
 * TanStack Query hooks for the triage UI surface (Track Q.5).
 *
 * Mirrors `use-entity-resolution.ts`: a query for the review queue + backbone
 * warnings, and mutations for the queue decision / entity-override toggle that
 * invalidate the queue and (for the override) the KG entity caches so the
 * status re-renders.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import {
  DecisionInput,
  EntityPinInput,
  triageApi,
} from '@/lib/api/triage'
import { useToast } from '@/lib/hooks/use-toast'

export const TRIAGE_QUERY_KEYS = {
  queue: (all = false) => ['triage', 'queue', all] as const,
  backbone: (ids: string[]) => ['triage', 'backbone', ...ids] as const,
}

/** The review queue (open rows by default). */
export function useTriageQueue(includeAll = false) {
  return useQuery({
    queryKey: TRIAGE_QUERY_KEYS.queue(includeAll),
    queryFn: () => triageApi.queue(includeAll),
  })
}

/** Backbone warnings for a fixed set of entity ids (read-time recompute). */
export function useBackboneWarnings(entityIds: string[]) {
  return useQuery({
    queryKey: TRIAGE_QUERY_KEYS.backbone(entityIds),
    queryFn: () => triageApi.backboneWarnings(entityIds),
    enabled: entityIds.length > 0,
  })
}

/** Decide one queue row (pin status + close it). */
export function useDecideQueueRow() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: ({
      queueId,
      body,
    }: {
      queueId: string
      body: DecisionInput
    }) => triageApi.decide(queueId, body),
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ['triage', 'queue'] })
      queryClient.invalidateQueries({ queryKey: ['knowledge-graph'] })
      toast({
        title: 'Decided',
        description: `Pinned ${res.status}`,
      })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to record decision',
        variant: 'destructive',
      })
    },
  })
}

/** Toggle the manual_override pin on an entity (KG detail panel). */
export function useSetEntityOverride() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: ({
      entityId,
      body,
    }: {
      entityId: string
      body: EntityPinInput
    }) => triageApi.setEntityOverride(entityId, body),
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ['triage', 'queue'] })
      queryClient.invalidateQueries({ queryKey: ['knowledge-graph'] })
      toast({
        title: res.manual_override ? 'Pinned' : 'Released',
        description: res.manual_override
          ? `Status pinned${res.status ? ` to ${res.status}` : ''}`
          : 'Status returned to automation',
      })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to update override',
        variant: 'destructive',
      })
    },
  })
}
