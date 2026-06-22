'use client'

import type { ApplyClusterInput } from '@/lib/api/entity-resolution'
import { describeMerge } from '@/lib/utils/entity-resolution'
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
import { Crown, X } from 'lucide-react'

interface MergeConfirmDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  /** The resolved apply payload that confirming will POST to /apply. */
  applyCluster: ApplyClusterInput
  /**
   * Surface forms for the winner + each loser, positionally [winner, ...losers].
   * Falls back to the apply payload's `member_surface_forms`, then ids.
   */
  winnerLabel: string
  loserLabels: string[]
  /** Issued only by the confirm action — the merge is destructive. */
  onConfirm: (cluster: ApplyClusterInput) => void
  /** Disable the confirm action while an apply is in flight (double-submit guard). */
  confirmDisabled?: boolean
}

/**
 * The single destructive-merge confirmation gate shared by every merge entry
 * point (cluster card + fuzzy candidate). Lists the surviving winner and the
 * loser(s) by surface form, and only the confirm action issues `POST /apply`
 * (AC2: a merge is destructive — confirm + show what merges before applying).
 */
export function MergeConfirmDialog({
  open,
  onOpenChange,
  applyCluster,
  winnerLabel,
  loserLabels,
  onConfirm,
  confirmDisabled = false,
}: MergeConfirmDialogProps) {
  const summary = describeMerge(applyCluster)

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent data-testid="merge-confirm-dialog">
        <AlertDialogHeader>
          <AlertDialogTitle>
            Merge {summary.loserCount + 1} entities?
          </AlertDialogTitle>
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
                  <span>{winnerLabel}</span>
                </div>
                <div className="text-xs text-muted-foreground">
                  Merged away: {summary.loserCount} entit
                  {summary.loserCount === 1 ? 'y' : 'ies'}
                </div>
                <div className="flex flex-wrap gap-1 pt-1">
                  {loserLabels.map((label, index) => (
                    <Badge
                      key={`${label}:${index}`}
                      variant="outline"
                      className="gap-1 text-[10px]"
                    >
                      <X className="h-2.5 w-2.5" aria-hidden="true" />
                      {label}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            data-testid="merge-confirm-apply"
            disabled={confirmDisabled}
            onClick={() => {
              onConfirm(applyCluster)
              onOpenChange(false)
            }}
          >
            Merge entities
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
