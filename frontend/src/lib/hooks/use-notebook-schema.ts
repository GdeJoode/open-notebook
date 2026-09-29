/**
 * React Query hooks for the B.3a/B.3b notebook-schema endpoints.
 *
 * Read hooks (B.3a):
 *
 * - `useNotebookSchema(notebookId)` — fetches the schema view (declared
 *   base ontology + pending proposals + flags). Powers the SchemaBrowser
 *   and the PendingExtensionsPanel.
 *
 * - `usePass1Results(notebookId)` — fetches per-source pass-1 results.
 *   Powers the CoverageStatsTable.
 *
 * The proposal queue (OE.5):
 *
 * - `useAcceptTarget` / `useAcceptPreview` / `useAcceptExtension` — Accept
 *   publishes the proposal to a SHARED ontology through the ontology
 *   editor's gate (preview first; a blocking finding refuses it).
 * - `useRejectExtension` — drops it and remembers the "no".
 *
 * Both write the returned queue straight into the cache. The per-notebook
 * edit ops (rename / merge / split / delete) are gone: those are edits of the
 * shared ontology, made in the ontology editor.
 */

import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationResult,
} from '@tanstack/react-query'
import {
  notebookSchemaApi,
  type NotebookEventView,
  type ReextractRequest,
} from '@/lib/api/notebook-schema'
import { QUERY_KEYS } from '@/lib/api/query-client'
import type {
  AcceptExtensionResponse,
  NotebookSchemaResponse,
  ReextractCandidatesResponse,
  ReextractResponse,
} from '@/lib/types/notebook_schema'
import { useToast } from '@/lib/hooks/use-toast'

export const NOTEBOOK_SCHEMA_QUERY_KEY = (id: string) =>
  ['notebook-schema', id] as const

export const PASS1_RESULTS_QUERY_KEY = (id: string) =>
  ['notebook-schema', id, 'pass1_results'] as const

export const NOTEBOOK_EVENTS_QUERY_KEY = (
  id: string,
  opts: { types?: string[]; unread?: boolean } = {},
) =>
  [
    'notebook-events',
    id,
    {
      // Sort the types list so the cache key is stable regardless of
      // call-site ordering.
      types: opts.types ? [...opts.types].sort() : undefined,
      unread: !!opts.unread,
    },
  ] as const

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

/**
 * Poll `GET /api/notebooks/{id}/events` every 30s for the soft-nudge
 * banner. Mirrors the cadence the MinerU health chip uses.
 *
 * Design notes:
 *
 * - `retry: false` because a transient 500 is meaningful state for the
 *   banner — silently retrying would hide a backend bug.
 * - `staleTime: 25_000` keeps the data fresh until just before the next
 *   refetch, avoiding double-fetches on quick re-mounts.
 * - The hook takes `enabled` so callers can disable polling when the
 *   notebook id is not yet known (e.g. SSR / first paint).
 */
export function useNotebookEvents(
  notebookId: string,
  opts: {
    types?: string[]
    unread?: boolean
    enabled?: boolean
    refetchInterval?: number
  } = {},
) {
  const {
    types,
    unread = true,
    enabled = true,
    refetchInterval = 30_000,
  } = opts
  return useQuery({
    queryKey: NOTEBOOK_EVENTS_QUERY_KEY(notebookId, { types, unread }),
    queryFn: () =>
      notebookSchemaApi.listEvents(notebookId, { types, unread }),
    enabled: !!notebookId && enabled,
    refetchInterval,
    staleTime: Math.max(0, refetchInterval - 5_000),
    retry: false,
  })
}

/**
 * Mark a single event read. On success invalidates every events query
 * for the notebook so the banner disappears immediately.
 */
export function useMarkEventRead(notebookId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (eventId: string) =>
      notebookSchemaApi.markEventRead(notebookId, eventId),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ['notebook-events', notebookId],
      })
    },
  })
}

// ---------------------------------------------------------------------------
// The proposal queue (OE.5)
// ---------------------------------------------------------------------------

export interface AcceptVariables {
  typeName: string
  ontology: string
  /** The curator's parent; `undefined` keeps the proposed one, `""` is none. */
  parentType?: string
}

/** Where a proposal goes by default, and the ontologies it can go to. */
export function useAcceptTarget(notebookId: string, typeName: string | null) {
  return useQuery({
    queryKey: ['notebook-schema', notebookId, 'accept-target', typeName] as const,
    queryFn: () => notebookSchemaApi.getAcceptTarget(notebookId, typeName as string),
    enabled: !!notebookId && !!typeName,
  })
}

/** What accepting would do to the shared ontologies. Writes nothing. */
export function useAcceptPreview(notebookId: string) {
  return useMutation({
    mutationFn: ({ typeName, ontology, parentType }: AcceptVariables) =>
      notebookSchemaApi.previewAccept(notebookId, typeName, {
        ontology,
        parent_type: parentType,
      }),
  })
}

/**
 * Publish the proposal to a shared ontology. Errors are left to the caller,
 * which shows the gate's findings (422) or the reason (409) in place.
 */
export function useAcceptExtension(
  notebookId: string,
): UseMutationResult<AcceptExtensionResponse, Error, AcceptVariables> {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  return useMutation({
    mutationFn: ({ typeName, ontology, parentType }: AcceptVariables) =>
      notebookSchemaApi.acceptExtension(notebookId, typeName, {
        ontology,
        parent_type: parentType,
      }),
    onSuccess: (data, { typeName, ontology }) => {
      queryClient.setQueryData(NOTEBOOK_SCHEMA_QUERY_KEY(notebookId), data.notebook_schema)
      // The shared ontology changed: the browse views and the editor follow.
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.ontologies })
      queryClient.invalidateQueries({ queryKey: ['ontology-editor'] })
      toast({
        title: 'Published',
        description: data.published
          ? `${typeName} is now a type of ${ontology} (version ${data.published.version})`
          : `${typeName} was already accepted`,
      })
    },
  })
}

/** Take out a proposal a shared ontology already defines (no publish). */
export function useMarkCovered(notebookId: string) {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  return useMutation({
    mutationFn: (typeName: string) => notebookSchemaApi.markCovered(notebookId, typeName),
    onSuccess: (data, typeName) => {
      queryClient.setQueryData(NOTEBOOK_SCHEMA_QUERY_KEY(notebookId), data.notebook_schema)
      toast({ title: 'Already covered', description: `${typeName} left the queue` })
    },
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
    onSuccess: (data: NotebookSchemaResponse) => {
      queryClient.setQueryData(NOTEBOOK_SCHEMA_QUERY_KEY(notebookId), data)
      toast({ title: 'Success', description: 'Extension rejected' })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to reject extension',
        variant: 'destructive',
      })
    },
  })
}

/**
 * Dismiss the soft-nudge for this notebook. Optimistically updates the
 * schema cache so the banner hides without waiting for the next refetch.
 */
export function useDismissNudge(notebookId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => notebookSchemaApi.dismissNudge(notebookId),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: NOTEBOOK_SCHEMA_QUERY_KEY(notebookId),
      })
      queryClient.invalidateQueries({
        queryKey: ['notebook-events', notebookId],
      })
    },
  })
}

/** Toggle `review_required` for this notebook. */
export function useToggleReviewRequired(notebookId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (enabled: boolean) =>
      notebookSchemaApi.setReviewRequired(notebookId, enabled),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: NOTEBOOK_SCHEMA_QUERY_KEY(notebookId),
      })
    },
  })
}

export const PAUSED_EXTRACTION_QUERY_KEY = (id: string) =>
  ['notebook-schema', id, 'paused-extraction'] as const

/**
 * Resume extraction for paused sources in this notebook. Invalidates
 * sources + jobs + paused-status caches so the workspace view updates.
 *
 * Declared after `PAUSED_EXTRACTION_QUERY_KEY` so the cache-key
 * reference resolves at module-evaluation time without relying on
 * function-scope hoisting (the previous ordering was runtime-safe but
 * awkward to read).
 */
export function useResumeExtraction(notebookId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => notebookSchemaApi.resumeExtraction(notebookId),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: NOTEBOOK_SCHEMA_QUERY_KEY(notebookId),
      })
      queryClient.invalidateQueries({
        queryKey: PAUSED_EXTRACTION_QUERY_KEY(notebookId),
      })
      queryClient.invalidateQueries({
        queryKey: ['sources', notebookId],
      })
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}

/**
 * Poll `GET /extraction/paused` to know whether any source is paused.
 * Drives the `ExtractionPausedBanner` — the banner shows when
 * `paused_count > 0`.
 */
export function usePausedExtraction(notebookId: string) {
  return useQuery({
    queryKey: PAUSED_EXTRACTION_QUERY_KEY(notebookId),
    queryFn: () => notebookSchemaApi.listPausedExtraction(notebookId),
    enabled: !!notebookId,
    refetchInterval: 30_000,
    staleTime: 25_000,
    retry: false,
  })
}

// ---------------------------------------------------------------------------
// Phase B.3d — re-extract prompt
// ---------------------------------------------------------------------------
//
// The ReextractPromptBanner polls `schema_changed` events via the
// shared B.3c `useNotebookEvents` hook above (with
// `types: ['schema_changed'], unread: true`). The hooks below cover
// the bits the B.3c surface does NOT already provide: the candidate
// list endpoint and the re-extract mutation.
//
// We intentionally do NOT duplicate `useSchemaChangedEvents` or
// `useMarkEventRead` here — those live above (B.3c) and the banner
// consumes them directly. Keeping a single source of truth avoids the
// two-hook cache-bust drift that bit attempt 1 of this phase.

export const REEXTRACT_CANDIDATES_QUERY_KEY = (id: string) =>
  ['notebook-schema', id, 'reextract-candidates'] as const

/**
 * Fetch the affected-source list for the banner. Lazy — only enabled
 * when the caller asks for it (the banner queries once when the user
 * clicks "Re-extract selected" to populate the multi-select).
 */
export function useReextractCandidates(
  notebookId: string,
  options: { enabled?: boolean } = {},
) {
  return useQuery({
    queryKey: REEXTRACT_CANDIDATES_QUERY_KEY(notebookId),
    queryFn: () => notebookSchemaApi.listReextractCandidates(notebookId),
    enabled: !!notebookId && (options.enabled ?? true),
  })
}

/**
 * Submit the re-extract request. On success:
 *
 *   1. Invalidates the events query so the banner disappears (the
 *      events the user acted on are marked-read by the markEventRead
 *      flow; this is belt-and-braces in case the banner is dismissed
 *      via the action itself rather than via Later).
 *   2. Toasts the per-source enqueued / skipped split so the user
 *      sees what happened without staring at the workspace column.
 */
export function useReextractMutation(
  notebookId: string,
): UseMutationResult<ReextractResponse, Error, ReextractRequest> {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  return useMutation({
    mutationFn: (payload: ReextractRequest) =>
      notebookSchemaApi.enqueueReextract(notebookId, payload),
    onSuccess: (data) => {
      // Bust the events poll so the banner refetches with the latest
      // read_at state. Source list isn't invalidated because re-extract
      // doesn't change the candidate set. We invalidate the entire
      // `notebook-events` namespace so all event-type filters (B.3c
      // soft-nudge AND B.3d schema_changed) pick up the change — the
      // B.3c key structure is opaque to this hook.
      queryClient.invalidateQueries({
        queryKey: ['notebook-events', notebookId],
      })
      const skipped = data.skipped_source_ids.length
      const desc =
        skipped > 0
          ? `${data.jobs_enqueued} job(s) enqueued, ${skipped} already running`
          : `${data.jobs_enqueued} re-extract job(s) enqueued`
      toast({ title: 'Re-extract queued', description: desc })
    },
    onError: () => {
      toast({
        title: 'Error',
        description: 'Failed to enqueue re-extract jobs',
        variant: 'destructive',
      })
    },
  })
}

// Re-export types the B.3d banner consumes — keeping callers free of
// `@/lib/types/...` imports for the shapes a hook produces.
export type { NotebookEventView, ReextractCandidatesResponse }
