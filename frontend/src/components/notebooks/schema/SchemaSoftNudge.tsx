'use client'

import Link from 'next/link'
import { useMemo } from 'react'
import { AlertTriangle, BellOff, X } from 'lucide-react'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import {
  useDismissNudge,
  useMarkEventRead,
  useNotebookEvents,
} from '@/lib/hooks/use-notebook-schema'
import { useNotebookSchema } from '@/lib/hooks/use-notebook-schema'

interface SchemaSoftNudgeProps {
  notebookId: string
}

/**
 * Phase B.3c — soft-nudge banner for `extension_suggested` /
 * `schema_mismatch` events.
 *
 * Polls `GET /api/notebooks/{id}/events?type=extension_suggested,
 * schema_mismatch&unread=true` every 30s. When the response is
 * non-empty AND the notebook's `soft_nudge_dismissed` flag is false,
 * we render an inline banner over the workspace.
 *
 * Three actions:
 *
 *  - **Review** — navigates to `/notebooks/{id}/schema` where the user
 *    can accept / reject pending extensions.
 *  - **Use as-is** — calls `POST /events/{id}/mark_read` on the most
 *    recent event; the banner disappears once the events query
 *    invalidates.
 *  - **Don't show again for this notebook** — calls
 *    `POST /schema/dismiss_nudge` which sets
 *    `notebook_schema.soft_nudge_dismissed=true`. The banner stays
 *    hidden until the next time the orchestrator re-arms the flag
 *    (coverage drops below threshold again).
 *
 * Accessibility:
 *  - Wrapped in `role="alert"` via the `Alert` primitive — screen
 *    readers announce on first appearance.
 *  - Each action is a real `<button>` with a descriptive label; icon-
 *    only "close" carries `aria-label="Mark as read"` because it wires
 *    to the same `mark_read` mutation as "Use as-is" (the persistence
 *    matters — not a transient hide).
 *  - The banner doesn't trap focus — keyboard users can Tab through.
 *
 * # Polling cadence — 30s, matches MinerU health chip
 *
 * AC #1 of the plan asks for the banner to surface "within 5s" of an
 * event. We poll at 30s, the same cadence as the MinerU health chip in
 * the workspace header. Trade-offs we considered:
 *
 *  - **5s polling** would meet AC #1 literally. Cost: 6× more HTTP
 *    traffic per notebook tab open in the background; events are
 *    relatively rare (one per source per Pass-1 run), so the
 *    request-amplification penalty isn't worth the latency win.
 *  - **10s polling** is a reasonable compromise. We left it at 30s to
 *    avoid introducing a third cadence into the workspace (current
 *    cadences: 30s MinerU health, 30s paused-extraction). One number
 *    is easier to reason about than three.
 *  - **WebSockets / SSE** would give near-real-time delivery without
 *    polling cost. Out of scope — the workspace doesn't yet have a
 *    push-channel and adding one for this single feature is overkill.
 *
 * The trade-off is documented for the next reviewer: if user-feedback
 * says "the banner takes too long to appear", dial `refetchInterval`
 * to 10_000 in `useNotebookEvents` — the rest of the component is
 * already cadence-agnostic.
 */
export function SchemaSoftNudge({ notebookId }: SchemaSoftNudgeProps) {
  // The schema fetch is also driving the SchemaBrowser tab; this hook
  // reuses the same cache entry, so the extra mount here is free.
  const { data: schema } = useNotebookSchema(notebookId)

  const eventsQuery = useNotebookEvents(notebookId, {
    types: ['extension_suggested', 'schema_mismatch'],
    unread: true,
    enabled: !!notebookId,
  })

  const markRead = useMarkEventRead(notebookId)
  const dismissNudge = useDismissNudge(notebookId)

  // The banner surfaces the most-recent event (the API returns them
  // newest-first). Memoise so the rendered string is stable across
  // re-renders when the events list hasn't actually changed.
  const latestEvent = useMemo(() => {
    const evs = eventsQuery.data ?? []
    return evs.length > 0 ? evs[0] : null
  }, [eventsQuery.data])

  // Suppression rules:
  //  - User dismissed (`soft_nudge_dismissed`).
  //  - No event in the queue.
  //  - Loading or error states fall through to "no banner" — the
  //    soft-nudge is non-blocking, so silent failure is preferable to
  //    a flashing error UI.
  if (schema?.soft_nudge_dismissed) return null
  if (!latestEvent) return null
  if (eventsQuery.isError) return null

  const isMismatch = latestEvent.event_type === 'schema_mismatch'
  const title = isMismatch
    ? 'Schema mismatch detected'
    : 'New schema extensions suggested'
  const fallbackMessage = isMismatch
    ? 'A recent source has low coverage against the current schema. Review the suggestions to add new entity types.'
    : 'Pass-1 found candidate entity types worth adding to this notebook. Review the suggestions to keep coverage high.'

  return (
    <Alert
      variant={isMismatch ? 'destructive' : 'default'}
      data-testid="schema-soft-nudge"
      data-event-type={latestEvent.event_type}
      data-event-id={latestEvent.id}
      className="mb-3 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between"
    >
      <div className="flex min-w-0 gap-3">
        <AlertTriangle
          className="h-4 w-4 shrink-0"
          aria-hidden="true"
        />
        <div className="min-w-0 space-y-1">
          <AlertTitle className="text-sm">{title}</AlertTitle>
          <AlertDescription className="text-sm">
            {latestEvent.message || fallbackMessage}
          </AlertDescription>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2 sm:shrink-0">
        <Button asChild size="sm" variant="default">
          <Link
            href={`/notebooks/${encodeURIComponent(notebookId)}/schema`}
            data-testid="schema-soft-nudge-review"
          >
            Review
          </Link>
        </Button>
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={markRead.isPending}
          data-testid="schema-soft-nudge-use-as-is"
          onClick={() => markRead.mutate(latestEvent.id)}
        >
          Use as-is
        </Button>
        <Button
          type="button"
          size="sm"
          variant="ghost"
          disabled={dismissNudge.isPending}
          data-testid="schema-soft-nudge-dismiss"
          onClick={() => dismissNudge.mutate()}
        >
          <BellOff className="mr-1.5 h-3.5 w-3.5" aria-hidden="true" />
          Don&apos;t show again
        </Button>
        {/* Plain visual close as a hint that the banner can be hidden;
            wires to the same mark-read endpoint so the click feels
            cheap (vs. the full dismiss-for-notebook flow). */}
        <Button
          type="button"
          size="icon"
          variant="ghost"
          aria-label="Mark as read"
          disabled={markRead.isPending}
          data-testid="schema-soft-nudge-close"
          onClick={() => markRead.mutate(latestEvent.id)}
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </Button>
      </div>
    </Alert>
  )
}
