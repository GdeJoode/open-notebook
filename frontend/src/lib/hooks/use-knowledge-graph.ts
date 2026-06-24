import { useQuery } from '@tanstack/react-query'
import { knowledgeGraphApi } from '@/lib/api/knowledge-graph'
import { QUERY_KEYS } from '@/lib/api/query-client'

export function useEntities(filters?: {
  limit?: number
  offset?: number
  entity_type?: string
  status?: string
  source_id?: string
}) {
  return useQuery({
    queryKey: QUERY_KEYS.entities(filters),
    queryFn: () => knowledgeGraphApi.listEntities(filters),
  })
}

/**
 * Source-scoped entity listing for the source-detail Entities tab. Thin wrapper
 * over {@link useEntities} that pins `source_id` and disables the query until a
 * source id is known. Reuses the same query key/cache as the KG entity list.
 */
export function useSourceEntities(
  sourceId: string | undefined,
  filters?: { limit?: number; offset?: number },
) {
  return useQuery({
    queryKey: QUERY_KEYS.entities({ ...filters, source_id: sourceId }),
    queryFn: () =>
      knowledgeGraphApi.listEntities({ ...filters, source_id: sourceId }),
    enabled: !!sourceId,
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
