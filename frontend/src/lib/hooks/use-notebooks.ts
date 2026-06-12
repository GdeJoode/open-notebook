import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  notebooksApi,
  type NotebookMergeReport,
  type NotebookMergeRequest,
} from '@/lib/api/notebooks'
import { QUERY_KEYS } from '@/lib/api/query-client'
import { useToast } from '@/lib/hooks/use-toast'
import { CreateNotebookRequest, UpdateNotebookRequest } from '@/lib/types/api'

export function useNotebooks(archived?: boolean) {
  return useQuery({
    queryKey: [...QUERY_KEYS.notebooks, { archived }],
    queryFn: () => notebooksApi.list({ archived, order_by: 'updated desc' }),
  })
}

export function useNotebook(id: string) {
  return useQuery({
    queryKey: QUERY_KEYS.notebook(id),
    queryFn: () => notebooksApi.get(id),
    enabled: !!id,
  })
}

export function useCreateNotebook() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: (data: CreateNotebookRequest) => notebooksApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.notebooks })
      toast({
        title: 'Success',
        description: 'Notebook created successfully',
      })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to create notebook',
        variant: 'destructive',
      })
    },
  })
}

export function useUpdateNotebook() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: UpdateNotebookRequest }) =>
      notebooksApi.update(id, data),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.notebooks })
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.notebook(id) })
      toast({
        title: 'Success',
        description: 'Notebook updated successfully',
      })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to update notebook',
        variant: 'destructive',
      })
    },
  })
}

/**
 * Mutation hook for the B.6 cross-notebook merge endpoint.
 *
 * The mutation never auto-invalidates queries because the merge writes
 * into a target notebook and the only consumers (graph view, entity
 * list) refetch on focus / explicit invalidation. The dry-run path
 * uses the same hook — callers pass `dry_run: true` and read the
 * returned report off the mutation result.
 *
 * Toasts:
 *  - success (non-dry-run): "Merged N entities, M relations"
 *  - success (dry-run): no toast (the dialog renders the counts inline)
 *  - error: generic destructive toast; the dialog stays open for retry
 */
export function useMergeNotebooks() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation<NotebookMergeReport, Error, NotebookMergeRequest>({
    mutationFn: (data) => notebooksApi.merge(data),
    onSuccess: (report, variables) => {
      // Invalidate the target notebook's entity / graph data so the
      // UI reflects the merge — only on commit, never on preview.
      if (!variables.dry_run) {
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.notebooks })
        queryClient.invalidateQueries({
          queryKey: QUERY_KEYS.notebook(variables.target_id),
        })
        toast({
          title: 'Merge complete',
          description: `Merged ${report.entities_merged} entities, ${report.relations_created} relations.`,
        })
      }
    },
    onError: () => {
      toast({
        title: 'Merge failed',
        description: 'Failed to merge notebooks. Please try again.',
        variant: 'destructive',
      })
    },
  })
}

export function useDeleteNotebook() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: (id: string) => notebooksApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.notebooks })
      toast({
        title: 'Success',
        description: 'Notebook deleted successfully',
      })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to delete notebook',
        variant: 'destructive',
      })
    },
  })
}