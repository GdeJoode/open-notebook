/**
 * React Query hooks for the B.3a/B.3b notebook-schema endpoints.
 *
 * Read hooks (B.3a):
 *
 * - `useNotebookSchema(notebookId)` — fetches the effective schema
 *   (base ontology + extensions + flags). Powers the SchemaBrowser
 *   tree and the PendingExtensionsPanel.
 *
 * - `usePass1Results(notebookId)` — fetches per-source pass-1 results.
 *   Powers the CoverageStatsTable.
 *
 * Mutation hooks (B.3b) — one per edit op:
 *
 * - `useAcceptExtension(notebookId)` / `useRejectExtension(notebookId)`
 * - `useRenameType(notebookId)` / `useMergeTypes(notebookId)`
 * - `useSplitType(notebookId)` / `useDeleteType(notebookId)`
 *
 * Each mutation:
 *  1. Posts to the corresponding endpoint.
 *  2. Replaces the React Query cache directly with the response body
 *     (the endpoint returns the full updated `NotebookSchemaResponse`)
 *     to guarantee a single-roundtrip refresh — the AC#4 "200ms"
 *     guarantee. We deliberately do NOT call `invalidateQueries` after
 *     because it would force a second GET; the response is the
 *     canonical post-write state.
 *  3. Surfaces success / failure toasts. Toast wording mirrors the
 *     `use-notebooks` hooks for consistency.
 */

import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationResult,
} from '@tanstack/react-query'
import {
  notebookSchemaApi,
  type MergeTypesRequest,
  type RenameTypeRequest,
  type SplitTypeRequest,
} from '@/lib/api/notebook-schema'
import type { NotebookSchemaResponse } from '@/lib/types/notebook_schema'
import { useToast } from '@/lib/hooks/use-toast'

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

// ---------------------------------------------------------------------------
// Internal: shared success / error handlers for the B.3b mutations.
// ---------------------------------------------------------------------------

function buildMutationHandlers(
  notebookId: string,
  successDescription: string,
  errorDescription: string,
) {
  return (
    queryClient: ReturnType<typeof useQueryClient>,
    toast: ReturnType<typeof useToast>['toast'],
  ) => ({
    onSuccess: (data: NotebookSchemaResponse) => {
      // Replace the cache directly — guaranteed single-roundtrip refresh.
      queryClient.setQueryData(NOTEBOOK_SCHEMA_QUERY_KEY(notebookId), data)
      // Pass1 results don't change on schema-only ops; no invalidate.
      toast({
        title: 'Success',
        description: successDescription,
      })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: errorDescription,
        variant: 'destructive' as const,
      })
    },
  })
}

// ---------------------------------------------------------------------------
// Mutation hooks
// ---------------------------------------------------------------------------

export function useAcceptExtension(
  notebookId: string,
): UseMutationResult<NotebookSchemaResponse, Error, string> {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  return useMutation({
    mutationFn: (typeName: string) =>
      notebookSchemaApi.acceptExtension(notebookId, typeName),
    ...buildMutationHandlers(
      notebookId,
      'Extension accepted',
      'Failed to accept extension',
    )(queryClient, toast),
  })
}

export function useRejectExtension(
  notebookId: string,
): UseMutationResult<NotebookSchemaResponse, Error, string> {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  return useMutation({
    mutationFn: (typeName: string) =>
      notebookSchemaApi.rejectExtension(notebookId, typeName),
    ...buildMutationHandlers(
      notebookId,
      'Extension rejected',
      'Failed to reject extension',
    )(queryClient, toast),
  })
}

export function useRenameType(
  notebookId: string,
): UseMutationResult<NotebookSchemaResponse, Error, RenameTypeRequest> {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  return useMutation({
    mutationFn: (payload: RenameTypeRequest) =>
      notebookSchemaApi.renameType(notebookId, payload),
    ...buildMutationHandlers(
      notebookId,
      'Type renamed',
      'Failed to rename type',
    )(queryClient, toast),
  })
}

export function useMergeTypes(
  notebookId: string,
): UseMutationResult<NotebookSchemaResponse, Error, MergeTypesRequest> {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  return useMutation({
    mutationFn: (payload: MergeTypesRequest) =>
      notebookSchemaApi.mergeTypes(notebookId, payload),
    ...buildMutationHandlers(
      notebookId,
      'Types merged',
      'Failed to merge types',
    )(queryClient, toast),
  })
}

export function useSplitType(
  notebookId: string,
): UseMutationResult<NotebookSchemaResponse, Error, SplitTypeRequest> {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  return useMutation({
    mutationFn: (payload: SplitTypeRequest) =>
      notebookSchemaApi.splitType(notebookId, payload),
    ...buildMutationHandlers(
      notebookId,
      'Type split',
      'Failed to split type',
    )(queryClient, toast),
  })
}

export function useDeleteType(
  notebookId: string,
): UseMutationResult<NotebookSchemaResponse, Error, string> {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  return useMutation({
    mutationFn: (typeName: string) =>
      notebookSchemaApi.deleteType(notebookId, typeName),
    ...buildMutationHandlers(
      notebookId,
      'Type deleted',
      'Failed to delete type',
    )(queryClient, toast),
  })
}
