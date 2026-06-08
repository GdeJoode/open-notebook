'use client'

import * as React from 'react'

import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

/**
 * Bucket boundaries used by the colour ramp.
 *
 * Picked to match the threshold the extraction pipeline uses elsewhere
 * (>= 0.8 = "high confidence", < 0.5 = "low confidence / review"). The
 * amber band in between is the soft-warning zone.
 */
const LOW_THRESHOLD = 0.5
const HIGH_THRESHOLD = 0.8

function clamp01(n: number): number {
  if (!Number.isFinite(n)) return 0
  if (n < 0) return 0
  if (n > 1) return 1
  return n
}

function colorClassFor(value: number): string {
  if (value < LOW_THRESHOLD) return 'bg-red-500'
  if (value < HIGH_THRESHOLD) return 'bg-amber-500'
  return 'bg-green-500'
}

export interface ConfidenceBarProps {
  /** Confidence score in [0, 1]. Out-of-range values are clamped. */
  value: number
  /** Optional caption shown next to the bar (e.g. "entity"). */
  label?: string
  /** Optional className applied to the outer wrapper. */
  className?: string
}

/**
 * Inline confidence indicator used on entity tiles and relation rows.
 *
 * Renders a thin horizontal bar with width = `value * 100%`, colour-coded
 * red / amber / green by bucket. Hovering reveals the exact numeric
 * value (two decimal places) — useful when reviewers need to compare
 * borderline candidates without opening the detail panel.
 */
export function ConfidenceBar({ value, label, className }: ConfidenceBarProps) {
  const safe = clamp01(value)
  const pct = safe * 100
  const tooltip = `Confidence: ${safe.toFixed(2)}`

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div
          className={cn('flex items-center gap-2', className)}
          data-testid="confidence-bar"
          data-confidence={safe.toFixed(2)}
          aria-label={tooltip}
        >
          {label && (
            <span className="text-xs text-muted-foreground whitespace-nowrap">
              {label}
            </span>
          )}
          <div
            className="relative h-1.5 w-full min-w-12 max-w-32 overflow-hidden rounded-full bg-muted"
            role="progressbar"
            aria-valuenow={Math.round(pct)}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            <div
              className={cn(
                'h-full rounded-full transition-all',
                colorClassFor(safe),
              )}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      </TooltipTrigger>
      <TooltipContent>{tooltip}</TooltipContent>
    </Tooltip>
  )
}
