/**
 * The Sources board's filter chips (Track WB.3), framework-free.
 *
 * A chip is a filter the list endpoint already has: one `?stage=`, or
 * `?stalled=true` (the stopped sources, the same server predicate as The
 * Tray's group). Its count is read from `GET /sources/counts`, which counts the
 * same rows; the page never counts rows itself.
 */

import type { ProcessingStage } from '@/lib/pipeline/processing-stage'
import type { SourceCounts } from '@/lib/types/api'

export type BoardFilter =
  | { kind: 'all' }
  | { kind: 'stage'; stage: ProcessingStage }
  | { kind: 'stalled' }

export interface BoardChip {
  key: string
  label: string
  filter: BoardFilter
  /** Failed, awaiting review and stopped sources wait for the owner. */
  needsYou: boolean
}

/** "Needs you" first (the order the board sorts in), then the pipeline order. */
export const BOARD_CHIPS: BoardChip[] = [
  { key: 'all', label: 'All', filter: { kind: 'all' }, needsYou: false },
  { key: 'failed', label: 'Failed', filter: { kind: 'stage', stage: 'failed' }, needsYou: true },
  {
    key: 'awaiting_schema_review',
    label: 'Awaiting review',
    filter: { kind: 'stage', stage: 'awaiting_schema_review' },
    needsYou: true,
  },
  { key: 'stalled', label: 'Stopped', filter: { kind: 'stalled' }, needsYou: true },
  { key: 'ingested', label: 'Ingested', filter: { kind: 'stage', stage: 'ingested' }, needsYou: false },
  { key: 'embedded', label: 'Embedded', filter: { kind: 'stage', stage: 'embedded' }, needsYou: false },
  { key: 'extracted', label: 'Extracted', filter: { kind: 'stage', stage: 'extracted' }, needsYou: false },
  { key: 'graphed', label: 'Graphed', filter: { kind: 'stage', stage: 'graphed' }, needsYou: false },
  { key: 'complete', label: 'Complete', filter: { kind: 'stage', stage: 'complete' }, needsYou: false },
]

/** The server's count for a chip; `undefined` while the counts are not read. */
export function chipCount(chip: BoardChip, counts: SourceCounts | undefined): number | undefined {
  if (!counts?.stages) return undefined
  switch (chip.filter.kind) {
    case 'all':
      return counts.total
    case 'stalled':
      return counts.stalled
    case 'stage':
      return counts.stages[chip.filter.stage]
  }
}

/** The chip the URL's filter is, if it is exactly one chip's. */
export function activeChipKey(stages: string[], stalled: boolean): string | undefined {
  if (stalled) return stages.length === 0 ? 'stalled' : undefined
  if (stages.length === 0) return 'all'
  if (stages.length === 1) return BOARD_CHIPS.find((c) => c.key === stages[0])?.key
  return undefined
}

/** The board's URL for a chip, keeping the notebook filter. */
export function boardHref(filter: BoardFilter, notebookId?: string | null): string {
  const params = new URLSearchParams()
  if (filter.kind === 'stage') params.append('stage', filter.stage)
  if (filter.kind === 'stalled') params.set('stalled', 'true')
  if (notebookId) params.set('notebook', notebookId)
  const query = params.toString()
  return query ? `/sources?${query}` : '/sources'
}
