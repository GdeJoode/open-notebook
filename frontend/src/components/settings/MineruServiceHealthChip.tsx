'use client'

/**
 * Polled health indicator for the MinerU service.
 *
 * Three states:
 * - Loading (first fetch only): gray skeleton chip.
 * - Healthy: green chip "MinerU online".
 * - Unhealthy: red chip "MinerU offline" + tooltip with the error
 *   message from the backend.
 *
 * a11y: rendered as role="status" with aria-live="polite" so the
 * status changes announce to screen readers without interrupting.
 */

import { Badge } from '@/components/ui/badge'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { useMineruHealth } from '@/lib/hooks/use-mineru-health'
import { CheckCircle2, CircleAlert, Loader2 } from 'lucide-react'

export function MineruServiceHealthChip() {
  const { data, isLoading, error } = useMineruHealth()

  // First-load skeleton: only on isLoading + no cached data. Once we
  // have data from a previous poll we always render a real state.
  if (isLoading && !data) {
    return (
      <Badge
        variant="secondary"
        className="gap-1 text-[10px]"
        role="status"
        aria-live="polite"
        aria-label="MinerU service status: checking"
        data-testid="mineru-health-chip"
        data-state="checking"
      >
        <Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" />
        MinerU: checking
      </Badge>
    )
  }

  // Network failure on the request itself (rare, since the backend
  // contract says always-200). Treat as offline.
  const healthy = data?.healthy === true && !error
  const errorText = data?.error ?? (error instanceof Error ? error.message : null)

  if (!healthy) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant="destructive"
            className="gap-1 text-[10px] cursor-help"
            role="status"
            aria-live="polite"
            aria-label={`MinerU service status: offline${errorText ? `, ${errorText}` : ''}`}
            data-testid="mineru-health-chip"
            data-state="offline"
          >
            <CircleAlert className="h-3 w-3" aria-hidden="true" />
            MinerU: offline
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          {errorText ? errorText : 'MinerU service unreachable'}
        </TooltipContent>
      </Tooltip>
    )
  }

  const versionSuffix = data?.version ? ` v${data.version}` : ''
  return (
    <Badge
      variant="default"
      className="gap-1 text-[10px] bg-green-700 hover:bg-green-700"
      role="status"
      aria-live="polite"
      aria-label={`MinerU service status: online${versionSuffix}`}
      data-testid="mineru-health-chip"
      data-state="online"
    >
      <CheckCircle2 className="h-3 w-3" aria-hidden="true" />
      MinerU: online{versionSuffix}
    </Badge>
  )
}
