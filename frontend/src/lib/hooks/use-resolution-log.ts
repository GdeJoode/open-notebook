import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { resolutionLogApi, ResolutionLogFilters } from '@/lib/api/resolution-log'
import { QUERY_KEYS } from '@/lib/api/query-client'
import { toast } from 'sonner'

export function useResolutionLog(filters?: ResolutionLogFilters) {
  return useQuery({
    queryKey: QUERY_KEYS.resolutionLog(filters as Record<string, unknown>),
    queryFn: () => resolutionLogApi.list(filters),
  })
}

export function useResolutionLogStats() {
  return useQuery({
    queryKey: QUERY_KEYS.resolutionLogStats,
    queryFn: () => resolutionLogApi.getStats(),
  })
}

export function useResolutionLogMethods() {
  return useQuery({
    queryKey: QUERY_KEYS.resolutionLogMethods,
    queryFn: () => resolutionLogApi.getMethods(),
  })
}

export function useUpdateResolutionStatus() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ id, status }: { id: string; status: 'accepted' | 'rejected' }) =>
      resolutionLogApi.updateStatus(id, status),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: ['resolution-log'] })
      toast.success(`Match ${variables.status}`)
    },
    onError: () => {
      toast.error('Failed to update status')
    },
  })
}

export function useBulkUpdateResolutionStatus() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ ids, status }: { ids: string[]; status: 'accepted' | 'rejected' }) =>
      resolutionLogApi.bulkUpdateStatus(ids, status),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['resolution-log'] })
      toast.success(`${data.updated} entries updated`)
    },
    onError: () => {
      toast.error('Failed to update entries')
    },
  })
}
