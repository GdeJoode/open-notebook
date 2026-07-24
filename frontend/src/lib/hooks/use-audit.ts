/**
 * React Query hooks for the Track F.1 audit widget (mirrors use-orphans).
 *
 * - `useNotebookAudit(id)` — fetches the latest audit run.
 * - `useRunAudit(id)` — re-runs the checks; on success invalidates the latest
 *   query so the widget refreshes from the source of truth.
 */

import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationResult,
} from '@tanstack/react-query'

import { auditApi } from '@/lib/api/audit'
import type { AuditRunResponse } from '@/lib/types/audit'
import { useToast } from '@/lib/hooks/use-toast'

export const NOTEBOOK_AUDIT_QUERY_KEY = (id: string) =>
  ['notebook-audit', id] as const

export function useNotebookAudit(notebookId: string) {
  return useQuery<AuditRunResponse>({
    queryKey: NOTEBOOK_AUDIT_QUERY_KEY(notebookId),
    queryFn: () => auditApi.latest(notebookId),
    enabled: !!notebookId,
  })
}

export function useRunAudit(
  notebookId: string,
): UseMutationResult<AuditRunResponse, Error, void> {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  return useMutation({
    mutationFn: () => auditApi.run(notebookId),
    onSuccess: (data) => {
      // Seed the cache with the fresh run, then invalidate so any other
      // subscriber refetches the canonical latest.
      queryClient.setQueryData(NOTEBOOK_AUDIT_QUERY_KEY(notebookId), data)
      queryClient.invalidateQueries({
        queryKey: NOTEBOOK_AUDIT_QUERY_KEY(notebookId),
      })
      const errors = data.findings.filter((f) => f.severity === 'error').length
      toast({
        title: 'Audit complete',
        description: data.findings.length
          ? `${data.findings.length} finding${
              data.findings.length === 1 ? '' : 's'
            }${errors ? ` (${errors} error${errors === 1 ? '' : 's'})` : ''}.`
          : 'No issues found.',
      })
    },
    onError: () => {
      toast({
        title: 'Audit failed',
        description: 'Could not run the quality audit.',
        variant: 'destructive' as const,
      })
    },
  })
}
