/**
 * OKF import mutation hook (Track OKF — import UI follow-up).
 *
 * The inverse of ``use-okf-export``: uploads an OKF v0.1 bundle (a ``.zip``) to
 * ``POST /api/notebooks/{id}/import/okf`` as multipart form-data and returns the
 * server's :type:`OkfImportReport` (created/matched counts + the non-silent
 * skip / dangling-link ledger).
 *
 * A malformed bundle *container* (not a zip / unreadable archive) is a 422 and
 * surfaces as an error toast; a malformed *concept* inside a valid bundle is
 * skipped non-silently and reported in the 200 body (never an error), so the
 * dialog renders it in the ledger rather than failing the whole import.
 */

'use client'

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'

import { useToast } from '@/lib/hooks/use-toast'
import type { OkfImportReport } from '@/lib/types/exports'

interface MutateVars {
  file: File
  applyDedup: boolean
}

interface MutateOptions {
  onSuccess?: (report: OkfImportReport) => void
  onError?: (error: Error) => void
}

interface UseOkfImportResult {
  mutate: (vars: MutateVars, opts?: MutateOptions) => void
  isPending: boolean
  error: Error | null
  /** The report from the most-recent successful import, else ``null``. */
  lastReport: OkfImportReport | null
}

export function useOkfImport(notebookId: string): UseOkfImportResult {
  const { toast } = useToast()
  const [lastReport, setLastReport] = useState<OkfImportReport | null>(null)

  const mutation = useMutation({
    mutationFn: async ({
      file,
      applyDedup,
    }: MutateVars): Promise<OkfImportReport> => {
      // Lazy-import so the bearer-token interceptor fires (same pattern as
      // use-okf-export).
      const apiClient = (await import('@/lib/api/client')).default

      const form = new FormData()
      form.append('file', file)

      const response = await apiClient.post<OkfImportReport>(
        `/notebooks/${encodeURIComponent(notebookId)}/import/okf`,
        form,
        {
          params: { apply_dedup: applyDedup },
          headers: { 'Content-Type': 'multipart/form-data' },
        },
      )
      return response.data
    },
    onSuccess: (report) => {
      setLastReport(report)
      const created =
        report.entities_created + report.notes_created + report.sources_created
      toast({
        title: 'Import complete',
        description: `Imported ${created} new concept${created === 1 ? '' : 's'}${
          report.skipped.length
            ? `, ${report.skipped.length} skipped`
            : ''
        }.`,
      })
    },
    onError: (err: Error) => {
      toast({
        title: 'Import failed',
        description: err.message || 'Unable to import the OKF bundle.',
        variant: 'destructive',
      })
    },
  })

  return {
    mutate: mutation.mutate as unknown as UseOkfImportResult['mutate'],
    isPending: mutation.isPending,
    error: mutation.error,
    lastReport,
  }
}
