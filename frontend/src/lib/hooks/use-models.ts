import { useQuery } from '@tanstack/react-query'
import { modelsApi } from '@/lib/api/models'

export const MODEL_QUERY_KEYS = {
  models: ['models'] as const,
}

/** De modellen uit het register; alleen lezen (ADR 0007). */
export function useModels() {
  return useQuery({
    queryKey: MODEL_QUERY_KEYS.models,
    queryFn: () => modelsApi.list(),
  })
}
