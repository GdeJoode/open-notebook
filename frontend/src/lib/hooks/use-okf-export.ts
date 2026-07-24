/**
 * Track OKF.5 — Open Knowledge Format export mutation hook.
 *
 * Mirrors ``use-obsidian-export`` / ``use-jsonl-export`` but for the OKF
 * v0.1 bundle endpoint ``POST /api/notebooks/{id}/export/okf``. Two response
 * shapes are handled:
 *
 *   * ``Content-Type: application/zip`` (HTTP 200) → the bundle. The browser
 *     downloads the blob via a hidden ``<a>``; the ExportReport (incl. the
 *     lossy-by-design **omitted-field ledger**) rides in the
 *     ``X-OKF-Export-Report`` response header, parsed here and exposed via
 *     ``lastReport`` so the dialog can show the user exactly what the bundle
 *     dropped.
 *   * ``Content-Type: application/json`` (HTTP 202) → the notebook was large
 *     enough to be deferred to the job queue; the body carries
 *     ``{job_id, status, ...}``. Surfaced via ``lastDeferred`` + a toast.
 *
 * Unlike Obsidian there is no vault_path write mode — OKF only ever emits a
 * zip — so the hook always sends ``mode: 'zip'`` to satisfy the shared
 * request contract.
 *
 * ``parseOkfExportReport`` is exported so the unit-test spec can pin the
 * header-JSON parsing in a Node-only context without dragging in React Query.
 */

'use client'

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import type { AxiosResponse } from 'axios'

import { useToast } from '@/lib/hooks/use-toast'
import { parseFilenameFromContentDisposition } from '@/lib/utils/content-disposition'
import type {
  ExportFilter,
  OkfExportDeferred,
  OkfExportReport,
} from '@/lib/types/exports'

export { parseFilenameFromContentDisposition }

/**
 * Parse the ``X-OKF-Export-Report`` header value into an
 * :type:`OkfExportReport`. The server serialises the report as compact
 * single-line JSON (safe as a header value).
 *
 * Returns ``null`` for a missing, empty, or unparseable header so the dialog
 * can still confirm the download succeeded even if the ledger is unreadable —
 * the zip is the SSOT, the report is a nicety.
 */
export function parseOkfExportReport(
  header: string | undefined | null,
): OkfExportReport | null {
  if (!header || !header.trim()) return null
  try {
    return JSON.parse(header) as OkfExportReport
  } catch {
    return null
  }
}

/** Mutation result shape exposed to the dialog. Discriminated union so the
 *  dialog can branch on the inline-zip vs deferred-job path. */
export type OkfExportResult =
  | { kind: 'zip'; filename: string; report: OkfExportReport | null }
  | { kind: 'deferred'; deferred: OkfExportDeferred }

interface MutateOptions {
  onSuccess?: (result: OkfExportResult) => void
  onError?: (error: Error) => void
}

interface UseOkfExportResult {
  /** Trigger the export. ``opts`` forward to React Query's per-call options
   *  so the dialog can branch on success without subscribing to ``data``. */
  mutate: (filter: ExportFilter, opts?: MutateOptions) => void
  isPending: boolean
  error: Error | null
  /** ExportReport (incl. omitted-field ledger) from the most-recent zip
   *  success, or ``null`` before the first success / on the deferred path. */
  lastReport: OkfExportReport | null
  /** Job descriptor from the most-recent deferred (202) response, else
   *  ``null``. */
  lastDeferred: OkfExportDeferred | null
}

/** Pull a header off an Axios response, case-insensitively (Playwright's
 *  ``route.fulfill`` sometimes preserves the original casing). */
function extractHeader(
  response: AxiosResponse<unknown>,
  name: string,
): string | undefined {
  const headers = (response.headers ?? {}) as Record<string, string>
  const lower = name.toLowerCase()
  return headers[lower] ?? headers[name] ?? headers[name.toUpperCase()]
}

/** Trigger the browser download for a zip blob. Not exported — it touches
 *  ``document`` + ``URL`` so it can't run in a node-only context. */
function triggerBlobDownload(blob: Blob, filename: string): void {
  const objectUrl = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = objectUrl
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(objectUrl)
}

/**
 * React Query mutation hook for the OKF export.
 *
 * Branches on the response Content-Type so the dialog stays agnostic to the
 * inline-zip vs deferred-job path.
 */
export function useOkfExport(notebookId: string): UseOkfExportResult {
  const { toast } = useToast()
  const [lastReport, setLastReport] = useState<OkfExportReport | null>(null)
  const [lastDeferred, setLastDeferred] = useState<OkfExportDeferred | null>(
    null,
  )

  const mutation = useMutation({
    mutationFn: async (filter: ExportFilter): Promise<OkfExportResult> => {
      // Lazy-import so the bearer-token interceptor fires (same pattern as
      // use-obsidian-export / use-jsonl-export).
      const apiClient = (await import('@/lib/api/client')).default

      // ``responseType: 'blob'`` lets axios hand us the binary zip without
      // trying to parse it; the JSON (202) branch is a Blob we text-decode
      // manually below.
      const response = await apiClient.post<Blob>(
        `/notebooks/${encodeURIComponent(notebookId)}/export/okf`,
        // OKF reuses the ObsidianExportRequest contract server-side, but only
        // ever emits a zip — mode is fixed to 'zip'.
        { mode: 'zip', filter },
        { responseType: 'blob' },
      )

      const contentType = (
        (response.headers?.['content-type'] as string | undefined) ?? ''
      ).toLowerCase()

      if (contentType.startsWith('application/zip')) {
        const disposition = extractHeader(response, 'content-disposition')
        const filename =
          parseFilenameFromContentDisposition(disposition) ?? 'okf-bundle.zip'
        const report = parseOkfExportReport(
          extractHeader(response, 'x-okf-export-report'),
        )
        triggerBlobDownload(response.data, filename)
        return { kind: 'zip', filename, report }
      }

      if (contentType.startsWith('application/json')) {
        // Deferred (202): body is the job descriptor. Blob → text → JSON so
        // onSuccess receives the parsed value, not a Promise.
        const text = await response.data.text()
        const deferred = JSON.parse(text) as OkfExportDeferred
        return { kind: 'deferred', deferred }
      }

      throw new Error(
        `Unexpected response content-type: ${contentType || 'unknown'}`,
      )
    },
    onSuccess: (result) => {
      if (result.kind === 'zip') {
        setLastReport(result.report)
        setLastDeferred(null)
        // The browser download is the visible confirmation; no toast (matches
        // the Obsidian zip branch).
      } else {
        setLastDeferred(result.deferred)
        setLastReport(null)
        toast({
          title: 'Export queued',
          description: `This notebook is large (${result.deferred.entity_count} entities); the OKF bundle is being built in the background.`,
        })
      }
    },
    onError: (err: Error) => {
      toast({
        title: 'Export failed',
        description: err.message || 'Unable to export the notebook.',
        variant: 'destructive',
      })
    },
  })

  return {
    mutate: mutation.mutate as unknown as UseOkfExportResult['mutate'],
    isPending: mutation.isPending,
    error: mutation.error,
    lastReport,
    lastDeferred,
  }
}
