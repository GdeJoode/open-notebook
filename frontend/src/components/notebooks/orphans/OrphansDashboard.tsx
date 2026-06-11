'use client'

import { useMemo, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import {
  useNotebookOrphans,
  useReconnectOrphan,
} from '@/lib/hooks/use-orphans'
import type { OrphanRow } from '@/lib/types/orphans'

interface OrphansDashboardProps {
  notebookId: string
}

type TabKey = 'pending' | 'archived'

/**
 * Per-notebook orphan-prune dashboard (B.5b).
 *
 * Reads `GET /api/notebooks/{id}/orphans` and renders two tabs:
 * pending (`orphan_status = "pending_reconnect"`) and archived
 * (`orphan_status = "archived"`). Each row carries a `[Reconnect]`
 * action for pending entries; archived rows are read-only.
 *
 * Loading / error / empty states are explicit so the UI carries the
 * "have any orphans accumulated yet?" signal without ambiguity.
 */
export function OrphansDashboard({ notebookId }: OrphansDashboardProps) {
  const { data, isLoading, isError } = useNotebookOrphans(notebookId)
  const reconnect = useReconnectOrphan(notebookId)
  const [activeTab, setActiveTab] = useState<TabKey>('pending')

  const rows = useMemo(() => {
    if (!data) return { pending: [] as OrphanRow[], archived: [] as OrphanRow[] }
    const pending: OrphanRow[] = []
    const archived: OrphanRow[] = []
    for (const r of data.items) {
      if (r.orphan_status === 'pending_reconnect') pending.push(r)
      else if (r.orphan_status === 'archived') archived.push(r)
    }
    return { pending, archived }
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

  if (isError || !data) {
    return (
      <div
        role="alert"
        className="rounded-md border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"
      >
        Failed to load orphans. Reload to retry.
      </div>
    )
  }

  const totalCount = data.pending_count + data.archived_count
  if (totalCount === 0) {
    return (
      <div
        role="status"
        className="rounded-md border border-dashed p-6 text-sm text-muted-foreground"
        data-testid="orphans-empty-state"
      >
        No orphan entities for this notebook. Entities the orphan-connector
        cannot link to the graph show up here.
      </div>
    )
  }

  const visibleRows = activeTab === 'pending' ? rows.pending : rows.archived

  return (
    <div data-testid="orphans-dashboard" className="flex flex-col gap-3">
      <div
        role="tablist"
        aria-label="Orphan status"
        className="flex gap-2 border-b"
      >
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'pending'}
          aria-controls="orphans-tabpanel"
          onClick={() => setActiveTab('pending')}
          data-testid="orphans-tab-pending"
          className={`-mb-px border-b-2 px-3 py-2 text-sm font-medium ${
            activeTab === 'pending'
              ? 'border-primary text-primary'
              : 'border-transparent text-muted-foreground hover:text-foreground'
          }`}
        >
          Pending
          <Badge variant="secondary" className="ml-2 text-[10px]">
            {data.pending_count}
          </Badge>
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'archived'}
          aria-controls="orphans-tabpanel"
          onClick={() => setActiveTab('archived')}
          data-testid="orphans-tab-archived"
          className={`-mb-px border-b-2 px-3 py-2 text-sm font-medium ${
            activeTab === 'archived'
              ? 'border-primary text-primary'
              : 'border-transparent text-muted-foreground hover:text-foreground'
          }`}
        >
          Archived
          <Badge variant="secondary" className="ml-2 text-[10px]">
            {data.archived_count}
          </Badge>
        </button>
      </div>

      <div id="orphans-tabpanel" role="tabpanel">
        {visibleRows.length === 0 ? (
          <div
            role="status"
            className="rounded-md border border-dashed p-6 text-sm text-muted-foreground"
            data-testid={`orphans-empty-${activeTab}`}
          >
            {activeTab === 'pending'
              ? 'No pending orphans. Good news -- everything is linked.'
              : 'No archived orphans yet.'}
          </div>
        ) : (
          <div
            className="overflow-x-auto rounded-md border"
            data-testid={`orphans-table-${activeTab}`}
          >
            <table className="w-full text-sm">
              <caption className="sr-only">
                Orphan entities ({activeTab})
              </caption>
              <thead className="bg-muted/50">
                <tr>
                  <th scope="col" className="px-3 py-2 text-left font-medium">
                    Entity
                  </th>
                  <th scope="col" className="px-3 py-2 text-left font-medium">
                    Type
                  </th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">
                    Attempts
                  </th>
                  <th scope="col" className="px-3 py-2 text-left font-medium">
                    First orphaned
                  </th>
                  {activeTab === 'pending' && (
                    <th
                      scope="col"
                      className="px-3 py-2 text-right font-medium"
                    >
                      <span className="sr-only">Actions</span>
                    </th>
                  )}
                </tr>
              </thead>
              <tbody>
                {visibleRows.map((row) => {
                  const firstOrphanedDisplay = row.first_orphaned_at
                    ? new Date(row.first_orphaned_at).toLocaleDateString()
                    : '—'
                  const isReconnectingThisRow =
                    reconnect.isPending &&
                    reconnect.variables === row.id
                  return (
                    <tr
                      key={row.id}
                      className="border-t"
                      data-testid={`orphans-row-${row.id}`}
                    >
                      <td className="px-3 py-2 font-medium">
                        {row.canonical_name}
                      </td>
                      <td className="px-3 py-2 text-muted-foreground">
                        {row.entity_type ?? '—'}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {row.reconnect_attempts}
                      </td>
                      <td className="px-3 py-2 text-muted-foreground">
                        {firstOrphanedDisplay}
                      </td>
                      {activeTab === 'pending' && (
                        <td className="px-3 py-2 text-right">
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            disabled={isReconnectingThisRow}
                            onClick={() => reconnect.mutate(row.id)}
                            aria-label={`Reconnect orphan ${row.canonical_name}`}
                            data-testid={`reconnect-${row.id}`}
                          >
                            {isReconnectingThisRow
                              ? 'Reconnecting…'
                              : 'Reconnect'}
                          </Button>
                        </td>
                      )}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
