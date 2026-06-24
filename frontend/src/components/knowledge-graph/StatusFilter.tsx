'use client'

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { TRIAGE_STATUSES, statusDisplay } from '@/lib/utils/triage'
import type { TriageStatus } from '@/lib/api/triage'

/** Sentinel the Select uses for "no status filter" (Radix forbids empty values). */
export const STATUS_FILTER_ALL = 'all'

export interface StatusFilterProps {
  /**
   * The active status filter, or `undefined` for "all statuses". The parent owns
   * the canonical value (it feeds the entity-list query); this component is a
   * controlled Select over it.
   */
  value: TriageStatus | undefined
  onChange: (value: TriageStatus | undefined) => void
  className?: string
}

/**
 * Status filter for the KG entity table (Q.5). Lets the operator narrow the
 * table to `active` / `reference` / `archived` (or all). Modeled on the
 * entity-type `Select` already in the KG page so it matches the existing filter
 * row visually; the trigger carries an aria-label for the icon-free control.
 */
export function StatusFilter({ value, onChange, className }: StatusFilterProps) {
  return (
    <Select
      value={value ?? STATUS_FILTER_ALL}
      onValueChange={(v) =>
        onChange(v === STATUS_FILTER_ALL ? undefined : (v as TriageStatus))
      }
    >
      <SelectTrigger
        className={`w-40 ${className ?? ''}`}
        aria-label="Filter entities by status"
        data-testid="status-filter"
      >
        <SelectValue placeholder="All statuses" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={STATUS_FILTER_ALL}>All statuses</SelectItem>
        {TRIAGE_STATUSES.map((s) => (
          <SelectItem key={s} value={s}>
            {statusDisplay(s).label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
