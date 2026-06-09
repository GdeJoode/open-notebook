'use client'

import { useMemo } from 'react'

import { Badge } from '@/components/ui/badge'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { usePass1Results } from '@/lib/hooks/use-notebook-schema'

interface CoverageStatsTableProps {
  notebookId: string
}

/**
 * Per-source coverage table for the Schema tab.
 *
 * Reads `GET /api/notebooks/{id}/pass1_results` and renders one row
 * per *latest* pass-1 result per source.  The backend returns the
 * full append-only history newest-first; we dedupe by `source_id`
 * client-side so the table shows the current state without dropping
 * the audit history server-side.
 *
 * Loading / error / empty states are all surfaced explicitly because
 * the table is the primary feedback signal for "have any pass-1 runs
 * happened yet?".
 */
export function CoverageStatsTable({ notebookId }: CoverageStatsTableProps) {
  const { data, isLoading, isError } = usePass1Results(notebookId)

  const rows = useMemo(() => {
    if (!data) return []
    // Dedupe by source_id, keeping the newest (data is already
    // ordered newest-first by the backend).
    const seen = new Set<string>()
    const out = []
    for (const r of data) {
      if (seen.has(r.source_id)) continue
      seen.add(r.source_id)
      out.push(r)
    }
    return out
  }, [data])

  if (isLoading) {
    return (
      <div
        role="status"
        aria-live="polite"
        className="flex items-center justify-center py-8"
      >
        <LoadingSpinner />
      </div>
    )
  }

  if (isError) {
    return (
      <div
        role="alert"
        className="rounded-md border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"
      >
        Failed to load coverage statistics. Reload to retry.
      </div>
    )
  }

  if (rows.length === 0) {
    return (
      <div
        role="status"
        className="rounded-md border border-dashed p-6 text-sm text-muted-foreground"
        data-testid="coverage-empty-state"
      >
        No pass-1 runs yet for this notebook. Coverage statistics appear
        here once a source is processed.
      </div>
    )
  }

  return (
    <div
      className="overflow-x-auto rounded-md border"
      data-testid="coverage-stats-table"
    >
      <table className="w-full text-sm">
        <caption className="sr-only">
          Per-source pass-1 coverage statistics
        </caption>
        <thead className="bg-muted/50">
          <tr>
            <th scope="col" className="px-3 py-2 text-left font-medium">
              Source
            </th>
            <th scope="col" className="px-3 py-2 text-right font-medium">
              Coverage
            </th>
            <th scope="col" className="px-3 py-2 text-right font-medium">
              Uncovered concepts
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const coveragePct = Math.round(row.coverage_pct * 100)
            const isLow = row.coverage_pct < 0.7
            return (
              <tr
                key={row.id}
                className="border-t"
                data-testid={`coverage-row-${row.source_id}`}
              >
                <td className="px-3 py-2">
                  {row.source_title ?? (
                    <span className="italic text-muted-foreground">
                      (deleted source)
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 text-right tabular-nums">
                  <span className="inline-flex items-center gap-1.5">
                    {coveragePct}%
                    {isLow && (
                      <Badge variant="outline" className="text-[10px]">
                        low
                      </Badge>
                    )}
                  </span>
                </td>
                <td className="px-3 py-2 text-right tabular-nums">
                  {row.uncovered_concept_count}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
