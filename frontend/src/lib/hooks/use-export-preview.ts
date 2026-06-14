/**
 * D.1c -- React Query hook for the debounced preview-counts endpoint.
 *
 * The dialog calls this with the current filter on every render. The
 * hook holds a 300ms-debounced copy of the filter internally so rapid
 * slider drags don't fire one request per pixel; the React Query cache
 * (``staleTime: 30s``) further suppresses identical-filter refetches.
 *
 * Per the D.1c plan, ``min_connections`` is the only filter knob the
 * AC explicitly tests for debounce behaviour, but every knob shares
 * the same code path -- ``useDebounce`` is applied to the entire
 * filter object so the dialog can keep its three sliders + two
 * switches + entity-types input cheap.
 */

'use client'

import { useQuery } from '@tanstack/react-query'
import { useDebounce } from 'use-debounce'

import type { ExportFilter, ExportPreviewCounts } from '@/lib/types/exports'

/** ms to wait after the last filter change before firing a request. */
const DEBOUNCE_MS = 300

/** Tan-Stack ``staleTime`` for preview counts. */
const STALE_TIME_MS = 30_000

/** Build a React Query key from notebook id + the debounced filter.
 *
 * The filter object is included directly (not stringified) so React
 * Query's structural-equality comparison can detect "same filter" hits
 * across re-renders. Keys are pure data so they survive serialisation
 * by the devtools.
 */
function buildQueryKey(notebookId: string, filter: ExportFilter) {
  return [
    'export-preview',
    notebookId,
    {
      mc: filter.min_connections,
      mco: filter.min_confidence,
      mrc: filter.min_relation_confidence ?? null,
      et: filter.entity_types ?? null,
      io: filter.include_orphans,
      ia: filter.include_archived,
    },
  ] as const
}

/**
 * Public hook: poll ``GET /api/notebooks/{id}/export-preview`` with a
 * 300ms debounce against ``filter``.
 *
 * @param notebookId - Notebook record ID (e.g. ``notebook:abc``).
 * @param filter - The dialog's current filter. Whenever this changes
 *   the hook waits 300ms before firing the next request.
 */
export function useExportPreview(notebookId: string, filter: ExportFilter) {
  // Debounce the WHOLE filter object so every knob (slider, switch,
  // entity-types input) flows through the same wait. ``use-debounce``
  // returns ``[value]`` so we destructure -- ``debouncedFilter`` is
  // always defined.
  const [debouncedFilter] = useDebounce(filter, DEBOUNCE_MS)

  return useQuery<ExportPreviewCounts>({
    queryKey: buildQueryKey(notebookId, debouncedFilter),
    queryFn: async () => {
      const apiClient = (await import('@/lib/api/client')).default
      const params = new URLSearchParams()
      params.append('min_connections', String(debouncedFilter.min_connections))
      params.append('min_confidence', String(debouncedFilter.min_confidence))
      if (debouncedFilter.min_relation_confidence !== undefined) {
        params.append(
          'min_relation_confidence',
          String(debouncedFilter.min_relation_confidence),
        )
      }
      if (debouncedFilter.entity_types) {
        // Repeatable query param: ``?entity_types=A&entity_types=B``.
        for (const t of debouncedFilter.entity_types) {
          params.append('entity_types', t)
        }
      }
      params.append('include_orphans', String(debouncedFilter.include_orphans))
      params.append('include_archived', String(debouncedFilter.include_archived))

      const response = await apiClient.get<ExportPreviewCounts>(
        `/notebooks/${encodeURIComponent(notebookId)}/export-preview?${params.toString()}`,
      )
      return response.data
    },
    enabled: !!notebookId,
    staleTime: STALE_TIME_MS,
    // Don't blow up the dialog if the preview endpoint is briefly
    // unavailable -- the dialog renders an error fallback and the
    // Export button stays enabled.
    retry: 1,
  })
}
