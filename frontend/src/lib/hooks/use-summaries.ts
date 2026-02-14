import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { summariesApi, GenerateSummaryRequest } from '@/lib/api/summaries'
import { QUERY_KEYS } from '@/lib/api/query-client'
import { toast } from 'sonner'

export function useStrategies() {
  return useQuery({
    queryKey: QUERY_KEYS.strategies,
    queryFn: () => summariesApi.listStrategies(),
  })
}

export function useSummaries(sourceId?: string) {
  return useQuery({
    queryKey: QUERY_KEYS.summaries(sourceId),
    queryFn: () => summariesApi.list(sourceId),
  })
}

export function useSummary(id: string) {
  return useQuery({
    queryKey: QUERY_KEYS.summary(id),
    queryFn: () => summariesApi.get(id),
    enabled: !!id,
  })
}

export function useGenerateSummary() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: GenerateSummaryRequest) => summariesApi.generate(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['summaries'] })
      toast.success('Summary generated successfully')
    },
    onError: () => {
      toast.error('Failed to generate summary')
    },
  })
}
