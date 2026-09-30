// @vitest-environment jsdom
/**
 * WB.2: the triage page links on to Duplicates, and once the queue is empty
 * that is its primary action.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'

import { TooltipProvider } from '@/components/ui/tooltip'
import type { QueueItem } from '@/lib/api/triage'

const queue = vi.hoisted(() => ({ items: [] as unknown[] }))
vi.mock('next/link', () => ({
  default: ({ children, href, ...rest }: { children: ReactNode; href: string }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}))
vi.mock('@/lib/hooks/use-triage', () => ({
  useTriageQueue: () => ({ data: { items: queue.items }, isLoading: false, isError: false }),
  useDecideQueueRow: () => ({ mutate: vi.fn(), isPending: false }),
  useDecideTooiRow: () => ({ mutate: vi.fn(), isPending: false }),
}))

import TriageQueuePage from '../page'

const ROW: QueueItem = {
  id: 'triage_queue:t1',
  entity: 'entity:e1',
  name: 'Provincie Utrecht',
  type: 'organization',
  structural_degree: 0,
  doc_count: 1,
  reason: 'unsure',
  decision: 'open',
  batch_id: 'b1',
}

const mount = () =>
  render(
    <TooltipProvider>
      <TriageQueuePage />
    </TooltipProvider>,
  )

afterEach(() => {
  cleanup()
  queue.items = []
})

describe('Triage page — next step', () => {
  it('with rows open, links on to Duplicates from the header only', () => {
    queue.items = [ROW]
    mount()
    expect(screen.getByTestId('triage-row-triage_queue:t1')).toBeTruthy()
    expect(screen.getByTestId('triage-next').getAttribute('href')).toBe('/knowledge-graph/resolution')
    expect(screen.queryByTestId('triage-empty-next')).toBeNull()
  })

  it('once empty, Duplicates is the primary action', () => {
    mount()
    expect(screen.getByText('Nothing to triage')).toBeTruthy()
    expect(screen.getByTestId('triage-empty-next').getAttribute('href'))
      .toBe('/knowledge-graph/resolution')
  })
})

describe('Triage page — types (WB.3)', () => {
  it('shows the canonical type first and the stored label second', () => {
    queue.items = [{ ...ROW, entity_type: 'creative_work', type: 'Journal/ Publication' }]
    mount()
    const row = screen.getByTestId('triage-row-triage_queue:t1')
    expect(row.querySelector('[data-testid="entity-type-canonical"]')?.textContent)
      .toBe('creative_work')
    expect(row.querySelector('[data-testid="entity-type-label"]')?.textContent)
      .toBe('Journal/ Publication')
  })
})
