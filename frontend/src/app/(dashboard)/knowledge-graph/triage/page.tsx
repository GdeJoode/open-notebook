'use client'

import { useMemo, useState } from 'react'
import Link from 'next/link'
import { AlertTriangle, ArrowLeft, CheckCircle2, ClipboardList } from 'lucide-react'

import { AppShell } from '@/components/layout/AppShell'
import { EmptyState } from '@/components/common/EmptyState'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { useDecideQueueRow, useTriageQueue } from '@/lib/hooks/use-triage'
import type { QueueItem } from '@/lib/api/triage'
import { buildDecision, groupByReason } from '@/lib/utils/triage'

/**
 * Triage review queue (Q.5). Lists the open queue rows surfaced by the
 * after-extraction pipeline — grouped by reason ("unsure — operator decides",
 * "isolated core actor", "candidate active", "match conflict") — with a decision
 * control per row (promote → active, keep reference, mark override). Deciding a
 * row pins the entity (override-wins) and closes it; the queue re-fetches.
 *
 * Reuses the resolution-hub structure: loading skeletons, an error alert with
 * retry, an explicit empty state, and a locally-dismissed set so a decided row
 * disappears instantly while the query revalidates.
 */
export default function TriageQueuePage() {
  const queueQuery = useTriageQueue(false)
  const decide = useDecideQueueRow()

  // Hide rows the operator just decided so the queue updates instantly while the
  // query re-fetches in the background (mirrors the resolution hub's `dismissed`).
  const [dismissed, setDismissed] = useState<Set<string>>(new Set())

  const openItems = useMemo(
    () => (queueQuery.data?.items ?? []).filter((i) => !dismissed.has(i.id)),
    [queueQuery.data, dismissed],
  )
  const groups = useMemo(() => groupByReason(openItems), [openItems])

  const onDecide = (item: QueueItem, action: 'promote' | 'keep' | 'override') => {
    decide.mutate({
      queueId: item.id,
      body: buildDecision(action, item.structural_degree > 0 ? 'active' : undefined),
    })
    setDismissed((prev) => new Set(prev).add(item.id))
  }

  return (
    <AppShell>
      <div className="flex-1 overflow-y-auto">
        <div className="p-6 space-y-6 max-w-5xl">
          <div>
            <Link
              href="/knowledge-graph"
              className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              Knowledge Graph
            </Link>
            <h1 className="text-2xl font-bold mt-2">Triage queue</h1>
            <p className="text-muted-foreground mt-1">
              Edge cases the after-extraction triage flagged for a human:
              isolated core actors, well-connected reference candidates, unsure
              types, and match conflicts. Decide each — promote it into the active
              graph, keep it as reference, or pin it by hand. Decisions stick
              across re-runs.
            </p>
          </div>

          {queueQuery.isLoading ? (
            <div className="space-y-4" aria-busy="true" data-testid="triage-loading">
              <Skeleton className="h-8 w-64" />
              <Skeleton className="h-32 w-full" />
              <Skeleton className="h-32 w-full" />
            </div>
          ) : queueQuery.isError ? (
            <Alert variant="destructive" data-testid="triage-error">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Failed to load the triage queue</AlertTitle>
              <AlertDescription>
                Could not fetch the review queue. Check that the API is reachable
                and try again.
                <div className="mt-3">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => queueQuery.refetch()}
                  >
                    Retry
                  </Button>
                </div>
              </AlertDescription>
            </Alert>
          ) : openItems.length === 0 ? (
            <EmptyState
              icon={CheckCircle2}
              title="Nothing to triage"
              description="The review queue is empty. New edge cases appear here after the next extraction runs."
            />
          ) : (
            <div className="space-y-8" data-testid="triage-queue">
              <div className="flex items-center gap-2">
                <ClipboardList className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">
                  {openItems.length} open row{openItems.length === 1 ? '' : 's'}
                </span>
              </div>

              {groups.map((group) => (
                <section key={group.reason} className="space-y-3">
                  <div className="flex items-center gap-2">
                    <h2 className="text-lg font-semibold">{group.reason}</h2>
                    <Badge variant="secondary">{group.items.length}</Badge>
                  </div>

                  <div className="border rounded-md overflow-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-muted/50">
                        <tr className="border-b text-left">
                          <th className="p-3 font-medium">Name</th>
                          <th className="p-3 font-medium">Type</th>
                          <th className="p-3 font-medium">Degree</th>
                          <th className="p-3 font-medium">Docs</th>
                          <th className="p-3 font-medium">Reason</th>
                          <th className="p-3 font-medium text-right">Decision</th>
                        </tr>
                      </thead>
                      <tbody>
                        {group.items.map((item) => (
                          <tr
                            key={item.id}
                            data-testid={`triage-row-${item.id}`}
                            className="border-b last:border-b-0"
                          >
                            <td className="p-3 font-medium">
                              <Link
                                href={`/knowledge-graph?entity=${encodeURIComponent(item.entity)}`}
                                className="text-primary hover:underline"
                              >
                                {item.name || item.entity}
                              </Link>
                            </td>
                            <td className="p-3">
                              <Badge variant="outline">{item.type || '—'}</Badge>
                            </td>
                            <td className="p-3 text-muted-foreground tabular-nums">
                              {item.structural_degree}
                            </td>
                            <td className="p-3 text-muted-foreground tabular-nums">
                              {item.doc_count}
                            </td>
                            <td className="p-3 text-xs text-muted-foreground max-w-xs">
                              {item.reason}
                            </td>
                            <td className="p-3">
                              <div className="flex justify-end gap-1.5">
                                <Button
                                  size="sm"
                                  variant="default"
                                  disabled={decide.isPending}
                                  aria-label={`Promote ${item.name || item.entity} to active`}
                                  onClick={() => onDecide(item, 'promote')}
                                >
                                  Promote
                                </Button>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  disabled={decide.isPending}
                                  aria-label={`Keep ${item.name || item.entity} as reference`}
                                  onClick={() => onDecide(item, 'keep')}
                                >
                                  Keep reference
                                </Button>
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  disabled={decide.isPending}
                                  aria-label={`Pin ${item.name || item.entity} by hand`}
                                  onClick={() => onDecide(item, 'override')}
                                >
                                  Override
                                </Button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              ))}
            </div>
          )}
        </div>
      </div>
    </AppShell>
  )
}
