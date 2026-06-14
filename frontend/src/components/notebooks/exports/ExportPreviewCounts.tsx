'use client'

/**
 * D.1c -- Live preview-counts widget for the Obsidian export dialog.
 *
 * Reads ``useExportPreview`` and renders one of three states:
 *
 *   - **loading**: small skeleton row so the dialog doesn't jump
 *     between heights when the counts arrive.
 *   - **error**: muted-foreground fallback that explains the preview
 *     couldn't load but DOES NOT block the export -- the dialog's
 *     Export button remains enabled. The actual export runs against a
 *     fresh request so a stale preview never silently disables it.
 *   - **loaded**: ``"This will export N entities, M relations."``
 *
 * Pure presentational. The hook owns the request lifecycle; this
 * component just maps state → JSX. Test seams live in the hook.
 */

import { AlertCircle } from 'lucide-react'

import { Skeleton } from '@/components/ui/skeleton'
import { useExportPreview } from '@/lib/hooks/use-export-preview'
import type { ExportFilter } from '@/lib/types/exports'

interface ExportPreviewCountsProps {
  notebookId: string
  filter: ExportFilter
}

export function ExportPreviewCounts({
  notebookId,
  filter,
}: ExportPreviewCountsProps) {
  const { data, isLoading, isError } = useExportPreview(notebookId, filter)

  return (
    <div
      className="rounded-md border border-border bg-card px-3 py-2"
      data-testid="export-preview-counts"
      role="status"
      aria-live="polite"
    >
      {isLoading && (
        <div data-testid="export-preview-loading">
          <Skeleton className="h-4 w-56" />
        </div>
      )}

      {!isLoading && isError && (
        <div
          className="flex items-center gap-2 text-sm text-muted-foreground"
          data-testid="export-preview-error"
        >
          <AlertCircle className="h-4 w-4" aria-hidden="true" />
          <span>Preview unavailable. Export will still run.</span>
        </div>
      )}

      {!isLoading && !isError && data && (
        <div className="text-sm" data-testid="export-preview-loaded">
          This will export{' '}
          <strong data-testid="export-preview-entity-count">
            {data.entity_count}
          </strong>{' '}
          {data.entity_count === 1 ? 'entity' : 'entities'},{' '}
          <strong data-testid="export-preview-relation-count">
            {data.relation_count}
          </strong>{' '}
          {data.relation_count === 1 ? 'relation' : 'relations'}.
        </div>
      )}
    </div>
  )
}
