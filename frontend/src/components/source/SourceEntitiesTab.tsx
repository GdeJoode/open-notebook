'use client'

import { AlertCircle } from 'lucide-react'

import { useSourceEntities } from '@/lib/hooks/use-knowledge-graph'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { StatusBadge } from '@/components/knowledge-graph/StatusBadge'
import {
  sourceEntitiesView,
  toSourceEntityRow,
} from '@/components/source/source-entities'

interface SourceEntitiesTabProps {
  sourceId: string
}

/**
 * Read-only listing of the LIVE entities linked to a source (its
 * `source_documents`). Distinct from the orphaned extraction/run UI in
 * `sources/pipeline/tabs/EntitiesTab.tsx`, which reads the (possibly stale)
 * `extraction_result`; this tab reads the live `entity` table via the
 * source-scoped KG listing so the count matches the fixed `entity_count`.
 *
 * Reuses the Q.5 triage components (`StatusBadge`) and mirrors the KG entity
 * table layout. Renders loading / error / empty states explicitly.
 */
export function SourceEntitiesTab({ sourceId }: SourceEntitiesTabProps) {
  const { data, isLoading, isError, error } = useSourceEntities(sourceId, {
    limit: 500,
  })

  const view = sourceEntitiesView({
    isLoading,
    isError,
    items: data?.items,
  })

  if (view === 'loading') {
    return (
      <div
        className="flex items-center justify-center py-12"
        role="status"
        aria-live="polite"
        aria-label="Loading entities"
      >
        <LoadingSpinner />
      </div>
    )
  }

  if (view === 'error') {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Failed to load entities</AlertTitle>
        <AlertDescription>
          {error instanceof Error
            ? error.message
            : 'An unexpected error occurred while loading entities for this source.'}
        </AlertDescription>
      </Alert>
    )
  }

  if (view === 'empty') {
    return (
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>No entities</AlertTitle>
        <AlertDescription>
          This source has no extracted entities yet. Entities appear here once
          the knowledge-graph extraction has run for this source.
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <div className="border rounded-md overflow-auto">
      <table className="w-full text-sm" aria-label="Source entities">
        <thead className="sticky top-0 bg-background">
          <tr className="border-b bg-muted/50">
            <th scope="col" className="text-left p-3 font-medium">
              Name
            </th>
            <th scope="col" className="text-left p-3 font-medium">
              Type
            </th>
            <th scope="col" className="text-left p-3 font-medium">
              Status
            </th>
            <th scope="col" className="text-left p-3 font-medium">
              Degree
            </th>
            <th scope="col" className="text-left p-3 font-medium">
              Docs
            </th>
          </tr>
        </thead>
        <tbody>
          {(data?.items ?? []).map(toSourceEntityRow).map((row) => (
            <tr
              key={row.id}
              data-testid={`source-entity-row-${row.id}`}
              data-entity-name={row.name}
              className="border-b"
            >
              <td className="p-3 font-medium">{row.name}</td>
              <td className="p-3">
                {row.type ? (
                  <Badge variant="secondary">{row.type}</Badge>
                ) : (
                  <span className="text-xs text-muted-foreground">—</span>
                )}
              </td>
              <td className="p-3">
                {row.status ? (
                  <StatusBadge
                    status={row.status}
                    manualOverride={row.manualOverride}
                  />
                ) : (
                  <span className="text-xs text-muted-foreground">—</span>
                )}
              </td>
              <td className="p-3 text-muted-foreground tabular-nums">
                {row.degree ?? '—'}
              </td>
              <td className="p-3 text-muted-foreground tabular-nums">
                {row.docCount ?? '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
