'use client'

import { Lock } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { statusDisplay } from '@/lib/utils/triage'

export interface StatusBadgeProps {
  /** Raw status string (`active` / `reference` / `archived`); unknown → outline. */
  status: string | undefined | null
  /**
   * When true, render a lock glyph indicating the status is operator-pinned
   * (`manual_override`) so automation won't change it.
   */
  manualOverride?: boolean
  className?: string
}

/**
 * Triage status badge (Q.5). Maps the status to a label + `Badge` variant via
 * the pure {@link statusDisplay} helper; an unknown value degrades to an
 * "Unknown" outline badge rather than crashing. A locked badge (operator pin)
 * carries an aria-label so the pin state is announced.
 */
export function StatusBadge({
  status,
  manualOverride = false,
  className,
}: StatusBadgeProps) {
  const display = statusDisplay(status)
  return (
    <Badge
      variant={display.variant}
      className={className}
      data-testid="status-badge"
      data-status={status ?? 'unknown'}
      aria-label={
        manualOverride
          ? `Status ${display.label}, operator-pinned`
          : `Status ${display.label}`
      }
    >
      {manualOverride && (
        <Lock className="h-3 w-3" aria-hidden="true" />
      )}
      {display.label}
    </Badge>
  )
}
