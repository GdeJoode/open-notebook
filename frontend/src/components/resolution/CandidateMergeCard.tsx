'use client'

import { useState } from 'react'

import type {
  ApplyClusterInput,
  MergeCandidate,
} from '@/lib/api/entity-resolution'
import {
  candidateMethodLabel,
  candidateToApplyCluster,
  candidateTypeLabel,
  isCrossTypeCandidate,
} from '@/lib/utils/entity-resolution'
import { MergeConfirmDialog } from '@/components/resolution/MergeConfirmDialog'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'

interface CandidateMergeCardProps {
  candidate: MergeCandidate
  /** Approve → confirm → apply. Receives the candidate's apply payload. */
  onApprove: (cluster: ApplyClusterInput) => void
  /** Reject → reversible force-split overlay (keep the pair apart). */
  onReject: (candidate: MergeCandidate) => void
  disabled?: boolean
}

/**
 * One fuzzy/embedding merge candidate (K.5 review band). The "Merge" action is
 * destructive, so — like the cluster card — it opens the shared confirmation
 * gate showing the surviving winner and the loser by surface form before any
 * `POST /apply` is issued. "Keep apart" records a reversible force-split.
 */
export function CandidateMergeCard({
  candidate,
  onApprove,
  onReject,
  disabled = false,
}: CandidateMergeCardProps) {
  const [confirmOpen, setConfirmOpen] = useState(false)
  const applyCluster = candidateToApplyCluster(candidate)

  // A cross-type candidate has byte-identical names on both sides — the two
  // entities exist separately only because they were typed differently. Without
  // the types rendered inline the card reads as the same name twice, and the
  // Merge button below is one click from a destructive apply.
  const crossType = isCrossTypeCandidate(candidate)

  // The apply payload puts the winner at name_a / loser at name_b.
  const winnerLabel =
    candidate.winner_id === candidate.id_a ? candidate.name_a : candidate.name_b
  const loserLabel =
    candidate.loser_id === candidate.id_b ? candidate.name_b : candidate.name_a

  return (
    <Card className="p-4" data-testid={`merge-candidate-${candidate.id_a}:${candidate.id_b}`}>
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium truncate">
              {candidate.name_a}
              {crossType && (
                <span className="ml-1 font-normal text-muted-foreground">
                  ({candidate.entity_type})
                </span>
              )}
            </span>
            <span className="text-muted-foreground">↔</span>
            <span className="text-sm font-medium truncate">
              {candidate.name_b}
              {crossType && (
                <span className="ml-1 font-normal text-muted-foreground">
                  ({candidate.entity_type_b})
                </span>
              )}
            </span>
          </div>
          <div className="text-xs text-muted-foreground">
            {candidateTypeLabel(candidate)} ·{' '}
            {(candidate.score * 100).toFixed(0)}% · {candidateMethodLabel(candidate)}
          </div>
        </div>
        <div className="flex gap-2 shrink-0">
          <Button
            size="sm"
            disabled={disabled}
            onClick={() => setConfirmOpen(true)}
          >
            Merge
          </Button>
          <Button
            size="sm"
            variant="outline"
            disabled={disabled}
            onClick={() => onReject(candidate)}
            aria-label="Keep these entities apart"
          >
            Keep apart
          </Button>
        </div>
      </div>

      {/* Same destructive-merge confirmation gate as the cluster card (AC2). */}
      <MergeConfirmDialog
        open={confirmOpen}
        onOpenChange={setConfirmOpen}
        applyCluster={applyCluster}
        winnerLabel={winnerLabel}
        loserLabels={[loserLabel]}
        onConfirm={onApprove}
        confirmDisabled={disabled}
      />
    </Card>
  )
}
