'use client'

import Link from 'next/link'
import { formatDistanceToNow } from 'date-fns'
import { AlertTriangle, ArrowRight, CheckCircle2, RefreshCw } from 'lucide-react'

import { AppShell } from '@/components/layout/AppShell'
import { EmptyState } from '@/components/common/EmptyState'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import type { AttentionGroup } from '@/lib/api/attention'
import { useAttention } from '@/lib/hooks/use-attention'
import { setLastSeen } from '@/lib/werkbak/last-seen'

/**
 * `/` — the Werkbak (Track WB.1): what needs the owner, across every notebook,
 * read from `GET /api/attention`. One card per group the endpoint returns,
 * with its count, the top items and one action linking to where it is
 * resolved. "Since your last visit" lists what arrived after the report the
 * owner last marked as seen (stored per browser), or over the last 7 days
 * before the first mark; it only moves on "Mark as seen", so working through
 * the list and coming back keeps it.
 */
export default function WerkbakPage() {
  const { data, isLoading, isError, error, refetch, isFetching } = useAttention({ fresh: true })

  const shown = (g: AttentionGroup) => g.count > 0 || !!g.error
  const needsYou = data?.groups.filter((g) => g.needs_action && shown(g)) ?? []
  const inProgress = data?.groups.filter((g) => !g.needs_action && shown(g)) ?? []
  const arrived = data?.arrived.filter(shown) ?? []

  return (
    <AppShell>
      <div className="flex-1 overflow-y-auto">
        <div className="p-6 space-y-8 max-w-6xl">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold">The Tray</h1>
              <p className="text-base text-muted-foreground mt-1">
                What needs you, across every notebook.
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => refetch()}
              disabled={isFetching}
              aria-label="Refresh The Tray"
            >
              <RefreshCw className={isFetching ? 'animate-spin' : undefined} />
              Refresh
            </Button>
          </div>

          {isLoading ? (
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3" data-testid="werkbak-loading">
              {[0, 1, 2].map((i) => (
                <Skeleton key={i} className="h-40 w-full" />
              ))}
            </div>
          ) : isError || !data ? (
            <Alert variant="destructive" data-testid="werkbak-error">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Could not load The Tray</AlertTitle>
              <AlertDescription>
                <p>{error instanceof Error ? error.message : 'The request failed.'}</p>
                <Button variant="outline" size="sm" className="mt-2" onClick={() => refetch()}>
                  Try again
                </Button>
              </AlertDescription>
            </Alert>
          ) : (
            <>
              <section aria-labelledby="werkbak-since" className="space-y-4">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <h2 id="werkbak-since" className="text-lg font-semibold">
                      Since your last visit
                    </h2>
                    <p className="text-sm text-muted-foreground" data-testid="werkbak-since-note">
                      {data.since_default
                        ? 'The last 7 days: nothing marked as seen on this browser yet.'
                        : `Marked as seen ${formatDistanceToNow(new Date(data.since), { addSuffix: true })}.`}
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setLastSeen(data.generated_at)}
                    data-testid="werkbak-mark-seen"
                  >
                    <CheckCircle2 />
                    Mark as seen
                  </Button>
                </div>
                {arrived.length > 0 ? (
                  <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                    {arrived.map((g) => (
                      <GroupCard key={g.key} group={g} />
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground" data-testid="werkbak-nothing-new">
                    Nothing new.
                  </p>
                )}
              </section>

              <section aria-labelledby="werkbak-needs-you" className="space-y-4">
                <h2 id="werkbak-needs-you" className="text-lg font-semibold">
                  Needs you
                  {(data.total > 0 || data.partial) && (
                    <span className="ml-2 font-normal text-muted-foreground" data-testid="werkbak-total">
                      ({data.total}
                      {data.partial && '+'})
                      {data.partial && <span className="sr-only"> some groups did not load</span>}
                    </span>
                  )}
                </h2>
                {needsYou.length === 0 ? (
                  <div data-testid="werkbak-empty">
                    <EmptyState
                      icon={CheckCircle2}
                      title="Nothing needs you"
                      description="No failed or stopped sources, proposals, reviews or failed jobs."
                    />
                  </div>
                ) : (
                  <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                    {needsYou.map((g) => (
                      <GroupCard key={g.key} group={g} />
                    ))}
                  </div>
                )}
              </section>

              {inProgress.length > 0 && (
                <section aria-labelledby="werkbak-in-progress" className="space-y-4">
                  <h2 id="werkbak-in-progress" className="text-lg font-semibold">
                    In progress
                  </h2>
                  <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                    {inProgress.map((g) => (
                      <GroupCard key={g.key} group={g} />
                    ))}
                  </div>
                </section>
              )}
            </>
          )}
        </div>
      </div>
    </AppShell>
  )
}

function GroupCard({ group }: { group: AttentionGroup }) {
  const more = group.item_total - group.items.length
  const elsewhere = group.needs_action && !group.error ? group.count - group.counted : 0
  return (
    <Card className="gap-3 py-4" data-testid={`werkbak-group-${group.key}`}>
      <div className="flex items-start justify-between gap-3 px-4">
        <div className="min-w-0">
          <h3 className="text-base font-semibold">{group.title}</h3>
          {group.detail && <p className="text-sm text-muted-foreground">{group.detail}</p>}
          {elsewhere > 0 && (
            <p className="text-sm text-muted-foreground" data-testid={`werkbak-elsewhere-${group.key}`}>
              {elsewhere} already counted with their source
            </p>
          )}
        </div>
        {!group.error && (
          <Badge
            variant={group.needs_action ? 'default' : 'secondary'}
            className="text-sm tabular-nums"
            data-testid={`werkbak-count-${group.key}`}
          >
            {group.count}
          </Badge>
        )}
      </div>

      {group.error ? (
        <p role="alert" className="px-4 text-sm text-destructive">
          Could not load this group: {group.error}
        </p>
      ) : (
        <ul className="px-4 space-y-1">
          {group.items.map((item) => (
            <li key={item.id}>
              <Link
                href={item.link}
                className="-mx-2 block rounded-md px-2 py-1.5 text-sm hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                data-testid={`werkbak-item-${item.id}`}
              >
                <span className="block truncate font-medium">{item.label}</span>
                {(item.detail || item.at) && (
                  <span className="block truncate text-muted-foreground">
                    {item.detail}
                    {item.detail && item.at && ' · '}
                    {item.at && formatDistanceToNow(new Date(item.at), { addSuffix: true })}
                  </span>
                )}
              </Link>
            </li>
          ))}
          {more > 0 && <li className="text-sm text-muted-foreground">and {more} more</li>}
        </ul>
      )}

      {group.action && group.link && (
        <div className="px-4">
          <Button asChild size="sm" variant={group.needs_action ? 'default' : 'outline'}>
            <Link href={group.link} data-testid={`werkbak-action-${group.key}`}>
              {group.action}
              <ArrowRight />
            </Link>
          </Button>
        </div>
      )}
    </Card>
  )
}
