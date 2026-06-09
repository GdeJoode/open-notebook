/**
 * React Query hooks for the B.3a notebook-schema endpoints.
 *
 * Two hooks:
 *
 * - `useNotebookSchema(notebookId)` — fetches the effective schema
 *   (base ontology + extensions + flags). Powers the SchemaBrowser
 *   tree and the PendingExtensionsPanel.
 *
 * - `usePass1Results(notebookId)` — fetches per-source pass-1 results.
 *   Powers the CoverageStatsTable.
 *
 * Cache keys live next to other QUERY_KEYS in `lib/api/query-client.ts`
 * — added there alongside this hook so any consumer can invalidate
 * after a B.3b mutation lands.
 */

import { useQuery } from '@tanstack/react-query'
import { notebookSchemaApi } from '@/lib/api/notebook-schema'

export const NOTEBOOK_SCHEMA_QUERY_KEY = (id: string) =>
  ['notebook-schema', id] as const

export const PASS1_RESULTS_QUERY_KEY = (id: string) =>
  ['notebook-schema', id, 'pass1_results'] as const

export function useNotebookSchema(notebookId: string) {
  return useQuery({
    queryKey: NOTEBOOK_SCHEMA_QUERY_KEY(notebookId),
    queryFn: () => notebookSchemaApi.get(notebookId),
    enabled: !!notebookId,
  })
}

export function usePass1Results(notebookId: string) {
  return useQuery({
    queryKey: PASS1_RESULTS_QUERY_KEY(notebookId),
    queryFn: () => notebookSchemaApi.listPass1Results(notebookId),
    enabled: !!notebookId,
  })
}
