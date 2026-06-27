import { useQuery } from '@tanstack/react-query'
import { knowledgeGraphApi } from '@/lib/api/knowledge-graph'
import { sourcesApi } from '@/lib/api/sources'
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

/**
 * Fetch the document↔entity (`mentions`) projection for the document-graph view
 * (Track U.4). `minWeight` collapses to the named-only skeleton via the backend
 * `min_weight` query param (0.3 preset). Returns the raw edge list; the view
 * derives nodes + joins source titles.
 */
export function useDocumentGraph(minWeight = 0) {
  return useQuery({
    queryKey: QUERY_KEYS.documentGraph({ min_weight: minWeight }),
    queryFn: () =>
      knowledgeGraphApi.getDocumentGraph({ min_weight: minWeight }),
  })
}

/**
 * Fetch every source (across all notebooks) so the document graph can label its
 * `source:` nodes by title. The document-graph endpoint carries only record
 * ids, so the title map is joined client-side. Isolated documents (papers with
 * 0 active entities) are surfaced from this list, not the edge set.
 */
export function useAllSources() {
  return useQuery({
    queryKey: QUERY_KEYS.sources(undefined),
    queryFn: () => sourcesApi.list({ limit: 500 }),
  })
}

export function useSearchEntities(query: string) {
  return useQuery({
    queryKey: QUERY_KEYS.entitySearch(query),
    queryFn: () => knowledgeGraphApi.searchEntities(query),
    enabled: query.length > 0,
  })
}
