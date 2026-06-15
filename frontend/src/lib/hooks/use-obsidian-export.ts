/**
 * D.1c -- Obsidian export mutation hook.
 *
 * Wraps the ``POST /api/notebooks/{id}/export-obsidian`` call with the
 * dual-branch response handling the dialog needs:
 *
 *   * ``Content-Type: application/zip`` → blob download via a hidden
 *     ``<a>`` element. Filename pulled from ``Content-Disposition``.
 *   * ``Content-Type: application/json`` → parse ``ExportReport``,
 *     expose via ``lastReport``, fire a success toast.
 *
 * Errors are surfaced via a destructive toast AND ``error`` so the
 * dialog can render an inline banner without losing user state.
 *
 * The filename parser ``parseFilenameFromContentDisposition`` is
 * exported so the unit-test spec can pin its behaviour -- regex parsers
 * are easy to over-simplify (see the mental-inversion test for the
 * RFC 5987 ``filename*=UTF-8''...`` form).
 */

'use client'

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import type { AxiosResponse } from 'axios'

import { useToast } from '@/lib/hooks/use-toast'
import { parseFilenameFromContentDisposition } from '@/lib/utils/content-disposition'
import type {
  ExportReport,
  ObsidianExportRequest,
} from '@/lib/types/exports'

// Re-export so existing import paths against
// ``use-obsidian-export`` still resolve. The actual implementation
// lives in ``lib/utils/content-disposition.ts`` so unit tests can
// import it without dragging in React Query / 'use client'.
export { parseFilenameFromContentDisposition }

/** Mutation result shape exposed to the dialog. Discriminated union so
 *  the dialog can react to the zip vs vault_path branch without
 *  inspecting the mutation's internal state. */
export type ObsidianExportResult =
  | { kind: 'zip'; filename: string }
  | { kind: 'vault_path'; report: ExportReport }

/** Mutation options forwarded by the dialog's ``mutate(req, opts)`` call.
 *  React Query v5 exposes these as the second argument of ``mutate``;
 *  we narrow them to the per-call ``onSuccess`` / ``onError`` we
 *  actually use. */
interface MutateOptions {
  onSuccess?: (result: ObsidianExportResult) => void
  onError?: (error: Error) => void
}

interface UseObsidianExportResult {
  /** Trigger the export. Forward ``opts`` to React Query's per-call
   *  options so the dialog can branch on success without subscribing
   *  to ``data`` re-renders. */
  mutate: (request: ObsidianExportRequest, opts?: MutateOptions) => void
  /** Mutation in flight. */
  isPending: boolean
  /** Last error, ``null`` when the last attempt succeeded or no
   *  attempt has been made. */
  error: Error | null
  /** ExportReport from the most-recent ``mode="vault_path"`` success.
   *  ``null`` for the zip branch (no report body) or before the first
   *  success. The dialog reads this to render the success banner. */
  lastReport: ExportReport | null
}

/**
 * Pull the filename header from an Axios response. Axios normalises
 * headers to lowercase but Playwright's ``route.fulfill`` sometimes
 * passes the casing through, so we check both spellings -- same trick
 * as ``NetworkxExportMenu``.
 */
function extractContentDisposition(
  response: AxiosResponse<unknown>,
): string | undefined {
  const headers = (response.headers ?? {}) as Record<string, string>
  return (
    headers['content-disposition'] ?? headers['Content-Disposition']
  )
}

/** Trigger the browser download for a zip blob.
 *
 * Pulled into its own function so the hook stays linear: hook does
 * mutation, this helper handles the DOM side-effect. The helper is
 * NOT exported -- it touches ``document`` and ``URL`` so it can't be
 * unit-tested in a node-only context anyway.
 */
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
 * React Query mutation hook for the Obsidian export.
 *
 * Branches on the response Content-Type so the dialog doesn't need to
 * know whether the user picked zip vs vault_path -- the mutation
 * surface looks the same either way.
 */
export function useObsidianExport(
  notebookId: string,
): UseObsidianExportResult {
  const { toast } = useToast()
  const [lastReport, setLastReport] = useState<ExportReport | null>(null)

  const mutation = useMutation({
    mutationFn: async (request: ObsidianExportRequest) => {
      // Lazy-import so the bearer-token interceptor fires automatically
      // (same trick as NetworkxExportMenu).
      const apiClient = (await import('@/lib/api/client')).default

      // ``responseType: 'blob'`` lets axios hand us the binary zip
      // payload without trying to parse it. For the JSON branch we
      // still get a Blob and parse it manually below -- the cost is
      // negligible and avoids needing two different request paths.
      const response = await apiClient.post<Blob>(
        `/notebooks/${encodeURIComponent(notebookId)}/export-obsidian`,
        request,
        { responseType: 'blob' },
      )

      // Axios normalises content-type to lowercase. Compare with a
      // ``startsWith`` because the server may attach a charset
      // (e.g. ``application/json; charset=utf-8``).
      const contentType = (
        (response.headers?.['content-type'] as string | undefined) ?? ''
      ).toLowerCase()

      if (contentType.startsWith('application/zip')) {
        const disposition = extractContentDisposition(response)
        const filename =
          parseFilenameFromContentDisposition(disposition) ??
          'obsidian-vault.zip'
        triggerBlobDownload(response.data, filename)
        return { kind: 'zip' as const, filename }
      }

      if (contentType.startsWith('application/json')) {
        // Blob -> text -> JSON. We do this synchronously inside the
        // mutationFn so onSuccess receives the parsed report rather
        // than a Promise.
        const text = await response.data.text()
        const report = JSON.parse(text) as ExportReport
        return { kind: 'vault_path' as const, report }
      }

      throw new Error(
        `Unexpected response content-type: ${contentType || 'unknown'}`,
      )
    },
    onSuccess: (result) => {
      if (result.kind === 'vault_path') {
        setLastReport(result.report)
        toast({
          title: 'Export complete',
          description: `Wrote ${result.report.files_written} files (${result.report.entities_written} entities, ${result.report.relations_written} relations).`,
        })
      } else {
        // Zip mode: the download itself is the user-facing
        // confirmation. We don't toast here because the browser's
        // own download UI is the SSOT.
        setLastReport(null)
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
    // React Query's ``mutation.mutate`` already accepts
    // ``(variables, options)`` -- the narrower local type just hides
    // the variants we don't surface. ``as unknown as`` is the
    // minimum-friction cast since the runtime shape matches exactly.
    mutate: mutation.mutate as unknown as UseObsidianExportResult['mutate'],
    isPending: mutation.isPending,
    error: mutation.error,
    lastReport,
  }
}
