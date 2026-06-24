import { useQuery } from '@tanstack/react-query'
import { knowledgeGraphApi } from '@/lib/api/knowledge-graph'
import { QUERY_KEYS } from '@/lib/api/query-client'

export function useEntities(filters?: {
  limit?: number
  offset?: number
  entity_type?: string
  status?: string
}) {
  return useQuery({
    queryKey: QUERY_KEYS.entities(filters),
    queryFn: () => knowledgeGraphApi.listEntities(filters),
  })
}

export function useEntity(id: string) {
  return useQuery({
    queryKey: QUERY_KEYS.entity(id),
    queryFn: () => knowledgeGraphApi.getEntity(id),
    enabled: !!id,
  })
}

export function useEntityTypes() {
  return useQuery({
    queryKey: QUERY_KEYS.entityTypes,
    queryFn: () => knowledgeGraphApi.getEntityTypes(),
  })
}

export function useGraphData(filters?: {
  entity_type?: string
  limit?: number
}) {
  return useQuery({
    queryKey: QUERY_KEYS.graphData(filters),
    queryFn: () => knowledgeGraphApi.getGraphData(filters),
  })
}

export function useSearchEntities(query: string) {
  return useQuery({
    queryKey: QUERY_KEYS.entitySearch(query),
    queryFn: () => knowledgeGraphApi.searchEntities(query),
    enabled: query.length > 0,
  })
}
