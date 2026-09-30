'use client'

import { useMemo, useState } from 'react'
import Link from 'next/link'
import { AlertTriangle, ArrowLeft, CheckCircle2, ClipboardList } from 'lucide-react'

import { EmptyState } from '@/components/common/EmptyState'
import { NextStepLink } from '@/components/common/NextStepLink'
import { PageBody } from '@/components/layout/PageBody'
import { PageHeader } from '@/components/layout/PageHeader'
import { EntityTypeLabel } from '@/components/knowledge-graph/EntityTypeLabel'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import {
  useDecideQueueRow,
  useDecideTooiRow,
  useTriageQueue,
} from '@/lib/hooks/use-triage'
import type { QueueItem, TooiDecisionInput } from '@/lib/api/triage'
import { useMinWidth } from '@/lib/hooks/use-media-query'
import {
  buildDecision,
  docsTooltipLines,
  groupByReason,
  isTooiRow,
  tooiCandidateLabel,
} from '@/lib/utils/triage'

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
  const decideTooi = useDecideTooiRow()

  // Hide rows the operator just decided so the queue updates instantly while the
  // query re-fetches in the background (mirrors the resolution hub's `dismissed`).
  const [dismissed, setDismissed] = useState<Set<string>>(new Set())

  const openItems = useMemo(
    () => (queueQuery.data?.items ?? []).filter((i) => !dismissed.has(i.id)),
    [queueQuery.data, dismissed],
  )
  const groups = useMemo(() => groupByReason(openItems), [openItems])
  // WB.5: a table from md up, a card per row below it.
  const isMd = useMinWidth('md')

  const onDecide = (item: QueueItem, action: 'promote' | 'keep' | 'override') => {
    decide.mutate({
      queueId: item.id,
      body: buildDecision(action, item.structural_degree > 0 ? 'active' : undefined),
    })
    setDismissed((prev) => new Set(prev).add(item.id))
  }

  // A TOOI row is answered with link/dismiss and never pins status (migration 83).
  const onDecideTooi = (item: QueueItem, body: TooiDecisionInput) => {
    decideTooi.mutate({ queueId: item.id, body })
    setDismissed((prev) => new Set(prev).add(item.id))
  }

  // The pieces a row and a card share: the same data, links and decisions.
  const nameLink = (item: QueueItem) => (
    <Link
      href={`/knowledge-graph?entity=${encodeURIComponent(item.entity)}`}
      className="text-primary hover:underline max-md:inline-flex max-md:min-h-10 max-md:items-center"
    >
      {item.name || item.entity}
    </Link>
  )

  const docsCount = (item: QueueItem) => (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          className="cursor-help underline decoration-dotted underline-offset-4"
          data-testid={`triage-docs-${item.id}`}
          aria-label={`${item.doc_count} documents: ${docsTooltipLines(item).join('; ')}`}
        >
          {item.doc_count}
        </span>
      </TooltipTrigger>
      <TooltipContent side="right" className="max-w-sm">
        <ul className="space-y-0.5 text-xs">
          {docsTooltipLines(item).map((line, idx) => (
            // Two uploads of one file share a display title.
            <li key={`${idx}-${line}`}>{line}</li>
          ))}
        </ul>
      </TooltipContent>
    </Tooltip>
  )

  const reasonCell = (item: QueueItem) =>
    isTooiRow(item) ? (
      <ul className="space-y-1" data-testid={`tooi-candidates-${item.id}`}>
        {(item.candidates ?? []).map((c) => (
          <li key={c.uri} className="flex items-center justify-between gap-2">
            <span>{tooiCandidateLabel(c)}</span>
            <Button
              size="sm"
              variant="outline"
              disabled={decideTooi.isPending}
              aria-label={`Link ${item.name || item.entity} to ${c.label}`}
              onClick={() => onDecideTooi(item, { action: 'link', uri: c.uri })}
            >
              Link
            </Button>
          </li>
        ))}
      </ul>
    ) : (
      item.reason
    )

  const decisionButtons = (item: QueueItem, className: string) =>
    isTooiRow(item) ? (
      <div className={className}>
        <Button
          size="sm"
          variant="ghost"
          disabled={decideTooi.isPending}
          aria-label={`Dismiss the TOOI candidates for ${item.name || item.entity}`}
          onClick={() => onDecideTooi(item, { action: 'dismiss' })}
        >
          Dismiss
        </Button>
      </div>
    ) : (
      <div className={className}>
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
    )

  return (
    <PageBody>
      <PageHeader
        back={
          <>
            <Link
              href="/knowledge-graph"
              className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground max-md:min-h-10"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              Knowledge Graph
            </Link>
            <NextStepLink href="/knowledge-graph/resolution" data-testid="triage-next">
              Next: Duplicates
            </NextStepLink>
          </>
        }
        title="Triage queue"
        subtitle={
          <>
            Edge cases the after-extraction triage flagged for a human:
            isolated core actors, well-connected reference candidates, unsure
            types, and match conflicts. Decide each — promote it into the active
            graph, keep it as reference, or pin it by hand. TOOI link candidates
            are answered separately: link the right organisation, or dismiss —
            neither changes the entity&apos;s status. Decisions stick across re-runs.
          </>
        }
      />

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
          action={
            <NextStepLink href="/knowledge-graph/resolution" primary data-testid="triage-empty-next">
              Review duplicates
            </NextStepLink>
          }
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

              {isMd ? (
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
                        <td className="p-3 font-medium">{nameLink(item)}</td>
                        <td className="p-3">
                          <EntityTypeLabel canonical={item.entity_type} display={item.type} />
                        </td>
                        <td className="p-3 text-muted-foreground tabular-nums">
                          {item.structural_degree}
                        </td>
                        <td className="p-3 text-muted-foreground tabular-nums">
                          {docsCount(item)}
                        </td>
                        <td className="p-3 text-xs text-muted-foreground max-w-xs">
                          {reasonCell(item)}
                        </td>
                        <td className="p-3">
                          {decisionButtons(item, 'flex justify-end gap-1.5')}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              ) : (
              <ul className="space-y-2">
                {group.items.map((item) => (
                  <li
                    key={item.id}
                    data-testid={`triage-card-${item.id}`}
                    className="space-y-2 rounded-md border p-3 text-sm"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="font-medium">{nameLink(item)}</span>
                      <EntityTypeLabel canonical={item.entity_type} display={item.type} />
                    </div>
                    <div className="flex gap-4 text-muted-foreground tabular-nums">
                      <span>Degree {item.structural_degree}</span>
                      <span>Docs {docsCount(item)}</span>
                    </div>
                    <div className="text-xs text-muted-foreground">{reasonCell(item)}</div>
                    {decisionButtons(item, 'flex flex-wrap gap-1.5')}
                  </li>
                ))}
              </ul>
              )}
            </section>
          ))}
        </div>
      )}
    </PageBody>
  )
}
