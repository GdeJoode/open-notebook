'use client'

import { useState } from 'react'
import { Download } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { useToast } from '@/lib/hooks/use-toast'

interface TtlDownloadButtonProps {
  notebookId: string
}

/**
 * Trigger a browser download of the notebook's effective schema as
 * Turtle (`GET /api/notebooks/{id}/schema.ttl`).
 *
 * The implementation uses the Blob-fetch route rather than a plain
 * `<a download>` so that:
 *   1. The bearer-token auth header on `apiClient` is included
 *      automatically — direct `<a>` navigation can't add headers.
 *   2. We can surface a toast on failure rather than the browser
 *      dumping a raw JSON error into a new tab.
 *
 * The download is staged via `URL.createObjectURL` + a temporary
 * `<a>` element so the browser's native Save dialog kicks in. The
 * blob URL is revoked immediately afterwards to keep memory tidy.
 *
 * Keyboard accessibility: this is a plain `<button>` — Space and
 * Enter both activate it via native button semantics.
 */
export function TtlDownloadButton({ notebookId }: TtlDownloadButtonProps) {
  const [isDownloading, setIsDownloading] = useState(false)
  const { toast } = useToast()

  const handleDownload = async () => {
    setIsDownloading(true)
    try {
      // Go through the shared api-client so the bearer-token interceptor
      // fires automatically; the path is relative to the auto-injected
      // `${apiUrl}/api` baseURL.
      const apiClient = (await import('@/lib/api/client')).default
      const response = await apiClient.get<Blob>(
        `/notebooks/${encodeURIComponent(notebookId)}/schema.ttl`,
        { responseType: 'blob' },
      )
      const blob = response.data
      const filename = parseFilename(
        response.headers?.['content-disposition'],
        notebookId,
      )

      const objectUrl = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = objectUrl
      anchor.download = filename
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      URL.revokeObjectURL(objectUrl)
    } catch (err) {
      console.error('Failed to download schema TTL', err)
      toast({
        title: 'Download failed',
        description: 'Could not export schema as Turtle. Please try again.',
        variant: 'destructive',
      })
    } finally {
      setIsDownloading(false)
    }
  }

  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      onClick={handleDownload}
      disabled={isDownloading}
      aria-label="Download schema as Turtle"
      data-testid="ttl-download-button"
    >
      <Download className="h-4 w-4" />
      {isDownloading ? 'Downloading…' : 'Download TTL'}
    </Button>
  )
}

/**
 * Extract the `filename="..."` token from a Content-Disposition header.
 *
 * Falls back to a sensible default if the header is missing or
 * malformed — preserves the user's ability to save *something*
 * sensible even if the backend headers drift.
 */
function parseFilename(
  contentDisposition: string | undefined,
  notebookId: string,
): string {
  if (contentDisposition) {
    const match = /filename="([^"]+)"/.exec(contentDisposition)
    if (match) {
      return match[1]
    }
  }
  // Mirror the backend's `_safe_filename` rules at a high level: replace
  // colons with underscores so older save dialogs accept the suggestion.
  const safeId = notebookId.replace(/[:/\\\r\n\t"']+/g, '_')
  return `${safeId}.ttl`
}
