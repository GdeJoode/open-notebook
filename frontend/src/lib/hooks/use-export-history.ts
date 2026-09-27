/**
 * PC.5c -- the newest export run of one format, for the "last export" line.
 *
 * Reads ``GET /api/notebooks/{id}/exports/history?format=…&limit=1``, the
 * reader of the ``export.*`` metrics every exporter writes. ``staleTime: 0`` so
 * reopening an export surface after a run shows that run.
 */

'use client'

import { useQuery } from '@tanstack/react-query'

import type { ExportHistoryEntry, ExportHistoryFormat } from '@/lib/types/exports'

export function useLastExport(notebookId: string, format: ExportHistoryFormat) {
  return useQuery<ExportHistoryEntry | null>({
    queryKey: ['export-history', notebookId, format],
    queryFn: async () => {
      const apiClient = (await import('@/lib/api/client')).default
      const response = await apiClient.get<ExportHistoryEntry[]>(
        `/notebooks/${encodeURIComponent(notebookId)}/exports/history`,
        { params: { format, limit: 1 } },
      )
      return response.data[0] ?? null
    },
    enabled: !!notebookId,
    staleTime: 0,
    retry: 1,
  })
}
