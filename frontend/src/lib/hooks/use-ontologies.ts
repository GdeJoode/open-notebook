import { useQuery } from '@tanstack/react-query'
import { ontologiesApi } from '@/lib/api/ontologies'
import { QUERY_KEYS } from '@/lib/api/query-client'

export function useOntologies() {
  return useQuery({
    queryKey: QUERY_KEYS.ontologies,
    queryFn: () => ontologiesApi.list(),
  })
}

export function useOntology(name: string) {
  return useQuery({
    queryKey: QUERY_KEYS.ontology(name),
    queryFn: () => ontologiesApi.get(name),
    enabled: !!name,
  })
}
