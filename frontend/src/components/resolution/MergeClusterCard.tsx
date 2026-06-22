'use client'

import { useMemo, useState } from 'react'

import type { MergeCluster } from '@/lib/api/entity-resolution'
import {
  buildApplyCluster,
  describeMerge,
} from '@/lib/utils/entity-resolution'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Crown, GitMerge, Scissors, X } from 'lucide-react'

interface MergeClusterCardProps {
  cluster: MergeCluster
  /** Approve → confirm → apply. Receives the winner-resolved apply payload. */
  onApprove: (cluster: ReturnType<typeof buildApplyCluster>) => void
  /** Reject → reversible force-split overlay (never a hard delete). */
  onReject: (cluster: MergeCluster) => void
  disabled?: boolean
}

/**
 * One proposed merge cluster: member surface forms as chips, an editable
 * proposed winner, and approve/reject controls. Approving opens a confirmation
 * step that shows exactly what will merge before `POST /apply` is issued — the
 * merge is destructive (AC2). Reject records a force-split so the pair is never
 * re-proposed (AC3).
 */
export function MergeClusterCard({
  cluster,
  onApprove,
  onReject,
  disabled = false,
}: MergeClusterCardProps) {
  const memberIds = useMemo(
    () => [cluster.winner_id, ...cluster.loser_ids],
    [cluster.winner_id, cluster.loser_ids],
  )
  const [winnerId, setWinnerId] = useState(cluster.winner_id)
  const [confirmOpen, setConfirmOpen] = useState(false)

  // Map member id → surface form for chip labels. The plan returns surface
  // forms positionally aligned with [winner, ...losers].
  const labelFor = (id: string, index: number) =>
    cluster.member_surface_forms[index] ?? id

  const applyCluster = buildApplyCluster(cluster, winnerId)
  const summary = describeMerge(applyCluster)

  return (
    <Card data-testid={`merge-cluster-${cluster.winner_id}`}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <GitMerge className="h-4 w-4 text-muted-foreground shrink-0" />
              <span className="font-medium truncate">
                {cluster.new_canonical}
              </span>
              <Badge variant="secondary" className="text-[10px] shrink-0">
                {cluster.entity_type}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {cluster.size} surface forms · {cluster.total_source_docs} source
              {cluster.total_source_docs === 1 ? ' doc' : ' docs'}
            </p>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1.5">
            Members (click to pick the surviving entity)
          </p>
          <div className="flex flex-wrap gap-1.5">
            {memberIds.map((id, index) => {
              const isWinner = id === winnerId
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => setWinnerId(id)}
                  disabled={disabled}
                  aria-pressed={isWinner}
                  aria-label={`${isWinner ? 'Surviving entity' : 'Make surviving entity'}: ${labelFor(id, index)}`}
                  className="inline-flex"
                >
                  <Badge
                    variant={isWinner ? 'default' : 'outline'}
                    className="gap-1 cursor-pointer"
                  >
                    {isWinner && <Crown className="h-3 w-3" aria-hidden="true" />}
                    {labelFor(id, index)}
                  </Badge>
                </button>
              )
            })}
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            className="gap-1.5"
            disabled={disabled || summary.loserCount === 0}
            onClick={() => setConfirmOpen(true)}
          >
            <GitMerge className="h-3.5 w-3.5" />
            Approve merge
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="gap-1.5"
            disabled={disabled}
            onClick={() => onReject(cluster)}
            aria-label="Reject — keep these entities separate"
          >
            <Scissors className="h-3.5 w-3.5" />
            Keep separate
          </Button>
        </div>
      </CardContent>

      {/* Destructive-merge confirmation step (AC2). */}
      <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <AlertDialogContent data-testid="merge-confirm-dialog">
          <AlertDialogHeader>
            <AlertDialogTitle>Merge {summary.loserCount + 1} entities?</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-2 text-sm">
                <p>
                  This is destructive. The entities below will be merged into a
                  single canonical entity. The losers are marked merged
                  (reversible), and their relations re-point to the winner.
                </p>
                <div className="rounded-md border p-2 space-y-1">
                  <div className="flex items-center gap-1.5 text-xs">
                    <Crown className="h-3 w-3 text-primary" aria-hidden="true" />
                    <span className="font-medium">Survives:</span>
                    <span>{summary.winnerLabel}</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Merged away: {summary.loserCount} entit
                    {summary.loserCount === 1 ? 'y' : 'ies'}
                  </div>
                  <div className="flex flex-wrap gap-1 pt-1">
                    {memberIds.map((id, index) =>
                      id === winnerId ? null : (
                        <Badge
                          key={id}
                          variant="outline"
                          className="gap-1 text-[10px]"
                        >
                          <X className="h-2.5 w-2.5" aria-hidden="true" />
                          {labelFor(id, index)}
                        </Badge>
                      ),
                    )}
                  </div>
                </div>
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              data-testid="merge-confirm-apply"
              onClick={() => {
                onApprove(applyCluster)
                setConfirmOpen(false)
              }}
            >
              Merge entities
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  )
}
