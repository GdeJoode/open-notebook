/**
 * D.2 — JSONL streaming export mutation hook.
 *
 * Wraps ``POST /api/notebooks/{id}/export-jsonl`` with the same blob-
 * download pattern as ``use-obsidian-export`` minus the JSON branch:
 * the JSONL endpoint is always ``application/zip``, never a JSON body
 * (the format has no on-disk vault-path mode like Obsidian's D.1b).
 *
 * Filename comes from ``Content-Disposition`` via the shared
 * ``parseFilenameFromContentDisposition`` helper added in D.1c — same
 * regex parser, same fallback name when the header is missing.
 *
 * Blob-URL cleanup mirrors ``use-obsidian-export``: create object URL,
 * trigger anchor click, remove anchor, revoke object URL on next tick.
 */

'use client'

import { useMutation } from '@tanstack/react-query'
import type { AxiosResponse } from 'axios'

import { useToast } from '@/lib/hooks/use-toast'
import { parseFilenameFromContentDisposition } from '@/lib/utils/content-disposition'
import type { JsonlExportRequest } from '@/lib/types/exports'

export { parseFilenameFromContentDisposition }

/** Mutation result -- always blob download for JSONL. */
export interface JsonlExportResult {
  kind: 'zip'
  filename: string
}

interface MutateOptions {
  onSuccess?: (result: JsonlExportResult) => void
  onError?: (error: Error) => void
}

interface UseJsonlExportResult {
  mutate: (request: JsonlExportRequest, opts?: MutateOptions) => void
  isPending: boolean
  error: Error | null
}

/** Pull the Content-Disposition header off an Axios response. */
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
 * Pulled into its own function so the hook stays linear: hook does the
 * mutation, this helper handles the DOM side-effect. Not exported --
 * it touches ``document`` + ``URL`` so it can't be unit-tested in a
 * node-only context.
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
 * React Query mutation hook for the JSONL export.
 *
 * Always blob-downloads -- there's no JSON branch (the JSONL surface
 * doesn't have an on-disk write mode like Obsidian's D.1b). The popover
 * closes on success via the consumer's ``onSuccess`` callback; this
 * hook stays presentation-agnostic.
 */
export function useJsonlExport(notebookId: string): UseJsonlExportResult {
  const { toast } = useToast()

  const mutation = useMutation({
    mutationFn: async (request: JsonlExportRequest) => {
      // Lazy-import keeps the bearer-token interceptor active (same
      // pattern as use-obsidian-export).
      const apiClient = (await import('@/lib/api/client')).default

      // ``responseType: 'blob'`` lets axios hand us the binary zip
      // payload without trying to parse it.
      const response = await apiClient.post<Blob>(
        `/notebooks/${encodeURIComponent(notebookId)}/export-jsonl`,
        request,
        { responseType: 'blob' },
      )

      const contentType = (
        (response.headers?.['content-type'] as string | undefined) ?? ''
      ).toLowerCase()

      if (!contentType.startsWith('application/zip')) {
        throw new Error(
          `Unexpected response content-type: ${contentType || 'unknown'}`,
        )
      }

      const disposition = extractContentDisposition(response)
      const filename =
        parseFilenameFromContentDisposition(disposition) ??
        'notebook-jsonl.zip'
      triggerBlobDownload(response.data, filename)

      return { kind: 'zip' as const, filename }
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
    mutate: mutation.mutate as unknown as UseJsonlExportResult['mutate'],
    isPending: mutation.isPending,
    error: mutation.error,
  }
}
