import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { sourcesApi } from '@/lib/api/sources'
import { QUERY_KEYS } from '@/lib/api/query-client'
import { useToast } from '@/lib/hooks/use-toast'
import {
  CreateSourceRequest,
  UpdateSourceRequest,
  SourceResponse,
  SourceStatusResponse
} from '@/lib/types/api'
import { isPollableStage } from '@/lib/pipeline/processing-stage'

/** Poll cadence (ms) for an in-flight source's pipeline stage. */
const PIPELINE_POLL_MS = 2000

/**
 * `refetchInterval` decision for {@link useSourcePipeline}. Exported so the
 * stop/continue logic is unit-testable without rendering the hook: returns
 * `2000` while the source's `processing_stage` can still advance on its own
 * (including when the stage is not yet known) and `false` once it is terminal
 * (`complete`/`failed`) or gated (`awaiting_schema_review`).
 */
export function pipelineRefetchInterval(query: {
  state: { data?: SourceResponse }
}): number | false {
  const stage = query.state.data?.processing_stage
  return isPollableStage(stage) ? PIPELINE_POLL_MS : false
}

export function useSources(notebookId?: string) {
  return useQuery({
    queryKey: QUERY_KEYS.sources(notebookId),
    queryFn: () => sourcesApi.list({ notebook_id: notebookId }),
    enabled: !!notebookId,
    staleTime: 5 * 1000, // 5 seconds - more responsive for real-time source updates
    refetchOnWindowFocus: true, // Refetch when user comes back to the tab
  })
}

export function useSource(id: string) {
  return useQuery({
    queryKey: QUERY_KEYS.source(id),
    queryFn: () => sourcesApi.get(id),
    enabled: !!id,
    staleTime: 30 * 1000, // 30 seconds - shorter stale time for more responsive updates
    refetchOnWindowFocus: true, // Refetch when user comes back to the tab
  })
}

/**
 * Poll GET `/sources/{id}` as the pipeline-stage spine (Track UX.1).
 *
 * Unlike {@link useSource} (30s stale, no polling) and {@link useSourceStatus}
 * (the job axis), this hook treats `processing_stage` as the source of truth for
 * per-document progress and refetches every 2s while the stage can still advance
 * (undefined ⇒ keep polling), stopping at `complete`/`failed`/
 * `awaiting_schema_review`. It shares the `source(id)` query cache with
 * {@link useSource} so mounting both does not duplicate requests.
 */
export function useSourcePipeline(id: string, enabled = true) {
  return useQuery({
    queryKey: QUERY_KEYS.source(id),
    queryFn: () => sourcesApi.get(id),
    enabled: !!id && enabled,
    refetchInterval: pipelineRefetchInterval,
    staleTime: 0, // stage must be fresh to drive live progress
    retry: (failureCount, error) => {
      // Don't retry on 404 (source not found) — mirror useSourceStatus.
      const axiosError = error as { response?: { status?: number } }
      if (axiosError?.response?.status === 404) {
        return false
      }
      return failureCount < 3
    },
  })
}

export function useCreateSource() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: (data: CreateSourceRequest) => sourcesApi.create(data),
    onSuccess: (result: SourceResponse, variables) => {
      // Invalidate queries for all relevant notebooks with immediate refetch
      if (variables.notebooks) {
        variables.notebooks.forEach(notebookId => {
          queryClient.invalidateQueries({
            queryKey: QUERY_KEYS.sources(notebookId),
            refetchType: 'active' // Refetch active queries immediately
          })
        })
      } else if (variables.notebook_id) {
        queryClient.invalidateQueries({
          queryKey: QUERY_KEYS.sources(variables.notebook_id),
          refetchType: 'active'
        })
      }

      // Invalidate general sources query too with immediate refetch
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.sources(),
        refetchType: 'active'
      })

      // Show different messages based on processing mode
      if (variables.async_processing) {
        toast({
          title: 'Source Queued',
          description: 'Source submitted for background processing. You can monitor progress in the sources list.',
        })
      } else {
        toast({
          title: 'Success',
          description: 'Source added successfully',
        })
      }
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to add source',
        variant: 'destructive',
      })
    },
  })
}

export function useUpdateSource() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: UpdateSourceRequest }) =>
      sourcesApi.update(id, data),
    onSuccess: (_, { id }) => {
      // Invalidate ALL sources queries (both general and notebook-specific)
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sources() })
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.source(id) })
      toast({
        title: 'Success',
        description: 'Source updated successfully',
      })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to update source',
        variant: 'destructive',
      })
    },
  })
}

export function useDeleteSource() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: (id: string) => sourcesApi.delete(id),
    onSuccess: (_, id) => {
      // Invalidate ALL sources queries (both general and notebook-specific)
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sources() })
      // Also invalidate the specific source
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.source(id) })
      toast({
        title: 'Success',
        description: 'Source deleted successfully',
      })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to delete source',
        variant: 'destructive',
      })
    },
  })
}

export function useFileUpload() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: ({ file, notebookId }: { file: File; notebookId: string }) =>
      sourcesApi.upload(file, notebookId),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ 
        queryKey: QUERY_KEYS.sources(variables.notebookId) 
      })
      toast({
        title: 'Success',
        description: 'File uploaded successfully',
      })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to upload file',
        variant: 'destructive',
      })
    },
  })
}

export function useSourceStatus(sourceId: string, enabled = true) {
  return useQuery({
    queryKey: QUERY_KEYS.sourceStatus(sourceId),
    queryFn: () => sourcesApi.status(sourceId),
    enabled: !!sourceId && enabled,
    refetchInterval: (query) => {
      // Auto-refresh every 2 seconds while processing
      const data = query.state.data as SourceStatusResponse | undefined
      // Keep polling if status is active OR if we haven't received data yet
      if (!data || data.status === 'running' || data.status === 'queued' || data.status === 'new') {
        return 2000
      }
      // Stop polling once completed, failed, or unknown
      return false
    },
    staleTime: 0, // Always consider status data stale for real-time updates
    retry: (failureCount, error) => {
      // Don't retry on 404 (source not found)
      const axiosError = error as { response?: { status?: number } }
      if (axiosError?.response?.status === 404) {
        return false
      }
      return failureCount < 3
    },
  })
}

export function useRetrySource() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: (sourceId: string) => sourcesApi.retry(sourceId),
    onSuccess: (result, sourceId) => {
      // Invalidate status query to refetch latest status
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.sourceStatus(sourceId)
      })
      // Invalidate ALL sources queries to refresh the UI
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sources() })
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.source(sourceId) })

      toast({
        title: 'Source Retry Queued',
        description: 'The source has been requeued for processing.',
      })
    },
    onError: () => {
      toast({
        title: 'Retry Failed',
        description: 'Failed to retry source processing. Please try again.',
        variant: 'destructive',
      })
    },
  })
}

export function useAddSourcesToNotebook() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: async ({ notebookId, sourceIds }: { notebookId: string; sourceIds: string[] }) => {
      const { notebooksApi } = await import('@/lib/api/notebooks')

      // Use Promise.allSettled to handle partial failures gracefully
      const results = await Promise.allSettled(
        sourceIds.map(sourceId => notebooksApi.addSource(notebookId, sourceId))
      )

      // Count successes and failures
      const successes = results.filter(r => r.status === 'fulfilled').length
      const failures = results.filter(r => r.status === 'rejected').length

      return { successes, failures, total: sourceIds.length }
    },
    onSuccess: (result, { notebookId, sourceIds }) => {
      // Invalidate ALL sources queries to refresh all lists
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sources() })
      // Specifically invalidate the notebook's sources
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sources(notebookId) })
      // Invalidate each affected source
      sourceIds.forEach(sourceId => {
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.source(sourceId) })
      })

      // Show appropriate toast based on results
      if (result.failures === 0) {
        toast({
          title: 'Success',
          description: `${result.successes} source${result.successes > 1 ? 's' : ''} added to notebook`,
        })
      } else if (result.successes === 0) {
        toast({
          title: 'Error',
          description: 'Failed to add sources to notebook',
          variant: 'destructive',
        })
      } else {
        toast({
          title: 'Partial Success',
          description: `${result.successes} source${result.successes > 1 ? 's' : ''} added, ${result.failures} failed`,
          variant: 'default',
        })
      }
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to add sources to notebook',
        variant: 'destructive',
      })
    },
  })
}

export function useRemoveSourceFromNotebook() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: async ({ notebookId, sourceId }: { notebookId: string; sourceId: string }) => {
      // This will call the API we created
      const { notebooksApi } = await import('@/lib/api/notebooks')
      return notebooksApi.removeSource(notebookId, sourceId)
    },
    onSuccess: (_, { notebookId, sourceId }) => {
      // Invalidate ALL sources queries to refresh all lists
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sources() })
      // Specifically invalidate the notebook's sources
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sources(notebookId) })
      // Also invalidate the specific source
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.source(sourceId) })

      toast({
        title: 'Success',
        description: 'Source removed from notebook successfully',
      })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to remove source from notebook',
        variant: 'destructive',
      })
    },
  })
}

export function useSourceChunks(sourceId: string, enabled = true) {
  return useQuery({
    queryKey: QUERY_KEYS.sourceChunks(sourceId),
    queryFn: () => sourcesApi.getChunks(sourceId),
    enabled: !!sourceId && enabled,
    staleTime: 60 * 1000,
  })
}

/**
 * Hybrid related-sources retrieval for a source (Track R.5).
 *
 * Returns the fused dense + KG ranking. A source with no aggregate embedding
 * and no shared entities returns [] from the backend (not an error), which the
 * consuming component renders as a friendly empty state.
 */
export function useRelatedSources(
  sourceId: string,
  opts?: { k?: number; preset?: 'kg-heavy' | 'balanced'; enabled?: boolean }
) {
  return useQuery({
    queryKey: QUERY_KEYS.sourceRelated(sourceId),
    queryFn: () =>
      sourcesApi.getRelatedHybrid(sourceId, { k: opts?.k, preset: opts?.preset }),
    enabled: !!sourceId && (opts?.enabled ?? true),
    staleTime: 60 * 1000,
  })
}

export function useUpdateChunk(sourceId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ chunkId, data }: {
      chunkId: string
      data: { text?: string; element_type?: string; positions?: number[][]; is_content?: boolean; chapter?: string | null }
    }) => sourcesApi.updateChunk(sourceId, chunkId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sourceChunks(sourceId) })
    },
  })
}

export function useDeleteChunk(sourceId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (chunkId: string) => sourcesApi.deleteChunk(sourceId, chunkId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sourceChunks(sourceId) })
    },
  })
}

export function useCreateChunk(sourceId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: {
      text: string; element_type?: string; physical_page: number;
      positions?: number[][]; is_content?: boolean
    }) => sourcesApi.createChunk(sourceId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sourceChunks(sourceId) })
    },
  })
}

export function useMergeChunk(sourceId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ chunkId, targetChunkId }: {
      chunkId: string
      targetChunkId?: string
    }) => sourcesApi.mergeChunk(sourceId, chunkId, targetChunkId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sourceChunks(sourceId) })
    },
  })
}

export function useSplitChunk(sourceId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ chunkId, cursorOffset }: {
      chunkId: string
      cursorOffset: number
    }) => sourcesApi.splitChunk(sourceId, chunkId, cursorOffset),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sourceChunks(sourceId) })
    },
  })
}
