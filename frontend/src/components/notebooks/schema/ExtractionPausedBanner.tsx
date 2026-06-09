'use client'

import Link from 'next/link'
import { PauseCircle } from 'lucide-react'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import {
  usePausedExtraction,
  useResumeExtraction,
} from '@/lib/hooks/use-notebook-schema'

interface ExtractionPausedBannerProps {
  notebookId: string
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
 */
export function ExtractionPausedBanner({
  notebookId,
}: ExtractionPausedBannerProps) {
  const pausedQuery = usePausedExtraction(notebookId)
  const resume = useResumeExtraction(notebookId)

  if (pausedQuery.isError) return null
  const pausedCount = pausedQuery.data?.paused_count ?? 0
  if (pausedCount === 0) return null

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
        <Button
          type="button"
          size="sm"
          variant="default"
          disabled={resume.isPending}
          data-testid="extraction-paused-resume"
          onClick={() => resume.mutate()}
        >
          {resume.isPending ? 'Resuming…' : 'Resume with current schema'}
        </Button>
        <Button asChild size="sm" variant="outline">
          <Link
            href={`/notebooks/${encodeURIComponent(notebookId)}/schema`}
            data-testid="extraction-paused-open-schema"
          >
            Open schema editor
          </Link>
        </Button>
      </div>
    </Alert>
  )
}
