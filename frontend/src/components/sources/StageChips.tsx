'use client'

import Link from 'next/link'

import { cn } from '@/lib/utils'
import { BOARD_CHIPS, boardHref, chipCount } from '@/lib/sources/board'
import type { SourceCounts } from '@/lib/types/api'

interface StageChipsProps {
  /** `GET /sources/counts` for the current notebook filter. */
  counts: SourceCounts | undefined
  countsLoading: boolean
  countsError: boolean
  /** The chip the URL's filter is (see `activeChipKey`). */
  activeKey: string | undefined
  notebookId?: string | null
}

/**
 * The Sources board's stage filters (WB.3). Each chip is a link to the list
 * filtered by one stage, the stopped sources or all, and shows the server's
 * count for that filter. The counts are never taken from the rows on screen,
 * which are one page of the list.
 */
export function StageChips({
  counts,
  countsLoading,
  countsError,
  activeKey,
  notebookId,
}: StageChipsProps) {
  return (
    <nav aria-label="Filter by stage" className="flex flex-wrap items-center gap-1.5">
      {BOARD_CHIPS.map((chip) => {
        const active = chip.key === activeKey
        const count = chipCount(chip, counts)
        const waiting = chip.needsYou && (count ?? 0) > 0
        return (
          <Link
            key={chip.key}
            href={boardHref(chip.filter, notebookId)}
            aria-current={active ? 'true' : undefined}
            data-testid={`sources-chip-${chip.key}`}
            data-count={count ?? ''}
            className={cn(
              'inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-sm transition-colors',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
              active
                ? 'border-primary bg-primary text-primary-foreground'
                : 'border-border hover:bg-accent',
              !active && waiting && 'border-amber-500 text-amber-800 dark:text-amber-300'
            )}
          >
            <span>{chip.label}</span>
            {count !== undefined && (
              <span className="tabular-nums font-medium" aria-label={`${count} sources`}>
                {count}
              </span>
            )}
            {count === undefined && countsLoading && (
              <span className="text-muted-foreground" aria-hidden>
                …
              </span>
            )}
          </Link>
        )
      })}
      {countsError && (
        <span role="status" className="text-sm text-muted-foreground" data-testid="sources-counts-error">
          Counts unavailable
        </span>
      )}
    </nav>
  )
}
