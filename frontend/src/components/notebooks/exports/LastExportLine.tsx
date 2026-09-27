'use client'

/**
 * LastExportLine — PC.5c. The newest run of this export, as its exporter
 * recorded it: when, what it wrote, or that it failed.
 *
 * Renders nothing while loading, on a read error, or before the first run —
 * it is context, never a reason to block an export.
 */

import { useLastExport } from '@/lib/hooks/use-export-history'
import type { ExportHistoryEntry, ExportHistoryFormat } from '@/lib/types/exports'

interface LastExportLineProps {
  notebookId: string
  format: ExportHistoryFormat
}

function when(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString()
}

function counts(e: ExportHistoryEntry): string {
  const parts = [
    `${e.entities_written} ${e.entities_written === 1 ? 'entity' : 'entities'}`,
    `${e.relations_written} ${e.relations_written === 1 ? 'relation' : 'relations'}`,
  ]
  if (e.files_written > 1) parts.push(`${e.files_written} files`)
  return parts.join(', ')
}

export function LastExportLine({ notebookId, format }: LastExportLineProps) {
  const { data } = useLastExport(notebookId, format)
  if (!data) return null
  return (
    <p
      className="text-xs text-muted-foreground"
      data-testid={`last-export-${format}`}
    >
      Last export: {when(data.created_at)} —{' '}
      {data.error ? (
        <span className="text-destructive">failed ({data.error})</span>
      ) : (
        counts(data)
      )}
    </p>
  )
}
