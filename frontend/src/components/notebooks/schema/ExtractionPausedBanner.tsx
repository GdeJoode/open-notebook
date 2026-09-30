'use client'

import { useState } from 'react'
import Link from 'next/link'
import { PauseCircle } from 'lucide-react'

import { ConfirmDialog } from '@/components/common/ConfirmDialog'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import {
  useNotebookSchema,
  usePausedExtraction,
  useResumeExtraction,
} from '@/lib/hooks/use-notebook-schema'

interface ExtractionPausedBannerProps {
  notebookId: string
  /** Off on the Schema tab itself, where the link would point at the page. */
  showSchemaLink?: boolean
}

/**
 * Phase B.3c — banner shown when one or more entity-extraction jobs in
 * this notebook are parked in `PAUSED_FOR_REVIEW` because the user
 * enabled `review_required`.
 *
 * Two actions:
 *
 *  - **Resume with current schema** — calls
 *    `POST /extraction/resume`. If the review-gate predicate would
 *    still fire (no accepted extensions yet), the backend appends a
 *    sentinel marker to satisfy the gate; either way the paused jobs
 *    are re-queued.
 *  - **Open schema editor** — link to `/notebooks/{id}/schema` where
 *    the user can accept the pending extensions explicitly (the
 *    "approve before continuing" path).
 *
 * The banner polls `GET /extraction/paused` every 30s. We only render
 * when `paused_count > 0` so the workspace stays uncluttered for the
 * common case.
 *
 * WB.3: Resume is the secondary action; accepting the proposals is the way
 * through. While nothing is accepted (`review_gate_active`, or the schema is
 * not read) Resume asks first and says it turns the review gate off for this
 * notebook for good.
 */
export function ExtractionPausedBanner({
  notebookId,
  showSchemaLink = true,
}: ExtractionPausedBannerProps) {
  const pausedQuery = usePausedExtraction(notebookId)
  const resume = useResumeExtraction(notebookId)
  const schema = useNotebookSchema(notebookId)
  const [confirming, setConfirming] = useState(false)

  if (pausedQuery.isError) return null
  const pausedCount = pausedQuery.data?.paused_count ?? 0
  if (pausedCount === 0) return null
  // Unknown (not read, or a server without the field) counts as "may turn off".
  const gateTurnsOff = schema.data?.review_gate_active !== false

  return (
    <Alert
      variant="default"
      data-testid="extraction-paused-banner"
      data-paused-count={pausedCount}
      className="mb-3 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between"
    >
      <div className="flex min-w-0 gap-3">
        <PauseCircle className="h-4 w-4 shrink-0" aria-hidden="true" />
        <div className="min-w-0 space-y-1">
          <AlertTitle className="text-sm">
            Extraction paused for review
          </AlertTitle>
          <AlertDescription className="text-sm">
            {pausedCount === 1
              ? '1 source is waiting for schema review before extraction can continue.'
              : `${pausedCount} sources are waiting for schema review before extraction can continue.`}
          </AlertDescription>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2 sm:shrink-0">
        {showSchemaLink && (
          <Button asChild size="sm" variant="default">
            <Link
              href={`/notebooks/${encodeURIComponent(notebookId)}/schema`}
              data-testid="extraction-paused-open-schema"
            >
              Open schema editor
            </Link>
          </Button>
        )}
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={resume.isPending}
          data-testid="extraction-paused-resume"
          onClick={() => (gateTurnsOff ? setConfirming(true) : resume.mutate())}
        >
          {resume.isPending ? 'Resuming…' : 'Resume with current schema'}
        </Button>
      </div>
      <ConfirmDialog
        open={confirming}
        onOpenChange={setConfirming}
        title="Turn off schema review for this notebook?"
        description={
          'Nothing is accepted yet. Resuming now extracts the waiting sources with ' +
          'the current schema and turns off the review gate for this notebook for ' +
          'good: later sources are extracted without waiting for a schema review. ' +
          'Accept the proposals you want first, then resume.'
        }
        confirmText="Turn off review and resume"
        confirmVariant="destructive"
        onConfirm={() => {
          setConfirming(false)
          resume.mutate()
        }}
      />
    </Alert>
  )
}
