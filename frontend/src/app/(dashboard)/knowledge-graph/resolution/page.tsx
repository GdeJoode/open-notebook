'use client'

import { useMemo, useState } from 'react'
import Link from 'next/link'

import { AppShell } from '@/components/layout/AppShell'
import { EmptyState } from '@/components/common/EmptyState'
import { MergeClusterCard } from '@/components/resolution/MergeClusterCard'
import { OverlayEditor } from '@/components/resolution/OverlayEditor'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import {
  useAddOverlay,
  useApplyMerge,
  useMergeCandidates,
  useMergePlan,
} from '@/lib/hooks/use-entity-resolution'
import type {
  ApplyClusterInput,
  MergeCandidate,
  MergeCluster,
} from '@/lib/api/entity-resolution'
import {
  buildCandidateRejectOverlay,
  buildRejectOverlay,
  candidateToApplyCluster,
} from '@/lib/utils/entity-resolution'
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react'

const PAGE_SIZE = 10

export default function EntityResolutionPage() {
  const planQuery = useMergePlan()
  const candidatesQuery = useMergeCandidates()
  const applyMerge = useApplyMerge()
  const addOverlay = useAddOverlay()

  const [page, setPage] = useState(0)
  // Locally hide clusters the user has just acted on so the queue updates
  // instantly while the plan/candidates queries re-fetch in the background.
  const [dismissed, setDismissed] = useState<Set<string>>(new Set())

  const dismiss = (key: string) =>
    setDismissed((prev) => new Set(prev).add(key))

  const clusters = useMemo(
    () =>
      (planQuery.data?.clusters ?? []).filter(
        (c) => !dismissed.has(`plan:${c.winner_id}`),
      ),
    [planQuery.data, dismissed],
  )

  const reviewCandidates = useMemo(
    () =>
      (candidatesQuery.data?.review ?? []).filter(
        (c) => !dismissed.has(`cand:${c.id_a}:${c.id_b}`),
      ),
    [candidatesQuery.data, dismissed],
  )

  const isLoading = planQuery.isLoading || candidatesQuery.isLoading
  const isError = planQuery.isError && candidatesQuery.isError

  const totalPages = Math.ceil(clusters.length / PAGE_SIZE)
  const pageClusters = clusters.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE)

  const onApprove = (cluster: ApplyClusterInput) => {
    applyMerge.mutate([cluster])
    dismiss(`plan:${cluster.winner_id}`)
  }

  const onReject = (cluster: MergeCluster) => {
    const rule = buildRejectOverlay(cluster)
    if (rule) addOverlay.mutate(rule)
    dismiss(`plan:${cluster.winner_id}`)
  }

  const onApproveCandidate = (candidate: MergeCandidate) => {
    applyMerge.mutate([candidateToApplyCluster(candidate)])
    dismiss(`cand:${candidate.id_a}:${candidate.id_b}`)
  }

  const onRejectCandidate = (candidate: MergeCandidate) => {
    addOverlay.mutate(buildCandidateRejectOverlay(candidate))
    dismiss(`cand:${candidate.id_a}:${candidate.id_b}`)
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
            <h1 className="text-2xl font-bold mt-2">Entity Resolution</h1>
            <p className="text-muted-foreground mt-1">
              Review proposed merges of duplicate entities, manage manual
              merge/split rules, and keep the graph free of fragmentation.
              Merging is destructive — you confirm each one.
            </p>
          </div>

          {isLoading ? (
            <div className="space-y-4" aria-busy="true" data-testid="resolution-loading">
              <Skeleton className="h-8 w-64" />
              <Skeleton className="h-40 w-full" />
              <Skeleton className="h-40 w-full" />
            </div>
          ) : isError ? (
            <Alert variant="destructive" data-testid="resolution-error">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Failed to load the resolution queue</AlertTitle>
              <AlertDescription>
                Could not fetch merge proposals. Check that the API is reachable
                and try again.
                <div className="mt-3">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      planQuery.refetch()
                      candidatesQuery.refetch()
                    }}
                  >
                    Retry
                  </Button>
                </div>
              </AlertDescription>
            </Alert>
          ) : (
            <>
              <section className="space-y-4">
                <div className="flex items-center gap-2">
                  <h2 className="text-lg font-semibold">Proposed merges</h2>
                  <Badge variant="secondary">{clusters.length}</Badge>
                </div>

                {clusters.length === 0 ? (
                  <EmptyState
                    icon={CheckCircle2}
                    title="No duplicates pending"
                    description="There are no proposed merges to review. New duplicates appear here after the next resolution scan."
                  />
                ) : (
                  <>
                    {pageClusters.map((cluster) => (
                      <MergeClusterCard
                        key={cluster.winner_id}
                        cluster={cluster}
                        onApprove={onApprove}
                        onReject={onReject}
                        disabled={applyMerge.isPending || addOverlay.isPending}
                      />
                    ))}

                    {totalPages > 1 && (
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">
                          Page {page + 1} of {totalPages}
                        </span>
                        <div className="flex gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            disabled={page === 0}
                            onClick={() => setPage((p) => p - 1)}
                            aria-label="Previous page"
                          >
                            <ChevronLeft className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            disabled={page >= totalPages - 1}
                            onClick={() => setPage((p) => p + 1)}
                            aria-label="Next page"
                          >
                            <ChevronRight className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </section>

              {reviewCandidates.length > 0 && (
                <section className="space-y-3">
                  <div className="flex items-center gap-2">
                    <h2 className="text-lg font-semibold">
                      Fuzzy candidates (review)
                    </h2>
                    <Badge variant="secondary">{reviewCandidates.length}</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Near-duplicate pairs the matcher is unsure about. These are
                    never merged automatically — approve to merge, or keep apart.
                  </p>
                  {reviewCandidates.map((c) => (
                    <Card key={`${c.id_a}:${c.id_b}`} className="p-4">
                      <div className="flex items-center justify-between gap-3">
                        <div className="min-w-0 space-y-1">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium truncate">
                              {c.name_a}
                            </span>
                            <span className="text-muted-foreground">↔</span>
                            <span className="text-sm font-medium truncate">
                              {c.name_b}
                            </span>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {c.entity_type} · {(c.score * 100).toFixed(0)}% ·{' '}
                            {c.method}
                          </div>
                        </div>
                        <div className="flex gap-2 shrink-0">
                          <Button
                            size="sm"
                            disabled={applyMerge.isPending}
                            onClick={() => onApproveCandidate(c)}
                          >
                            Merge
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            disabled={addOverlay.isPending}
                            onClick={() => onRejectCandidate(c)}
                          >
                            Keep apart
                          </Button>
                        </div>
                      </div>
                    </Card>
                  ))}
                </section>
              )}

              <OverlayEditor
                onSubmit={(rule) => addOverlay.mutate(rule)}
                disabled={addOverlay.isPending}
              />
            </>
          )}
        </div>
      </div>
    </AppShell>
  )
}
