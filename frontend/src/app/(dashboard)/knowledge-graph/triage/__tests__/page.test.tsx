// @vitest-environment jsdom
/**
 * WB.2: the triage page links on to Duplicates, and once the queue is empty
 * that is its primary action.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import type { ReactNode } from 'react'

import { TooltipProvider } from '@/components/ui/tooltip'
import type { QueueItem } from '@/lib/api/triage'

const queue = vi.hoisted(() => ({ items: [] as unknown[], decide: vi.fn() }))
vi.mock('next/link', () => ({
  default: ({ children, href, ...rest }: { children: ReactNode; href: string }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}))
vi.mock('@/lib/hooks/use-triage', () => ({
  useTriageQueue: () => ({ data: { items: queue.items }, isLoading: false, isError: false }),
  useDecideQueueRow: () => ({ mutate: queue.decide, isPending: false }),
  useDecideTooiRow: () => ({ mutate: vi.fn(), isPending: false }),
}))

import { installViewport, removeViewport } from '@/lib/hooks/__tests__/viewport'
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

describe('Triage page below md (WB.5)', () => {
  afterEach(() => removeViewport())

  it('shows a card per row with the same data, link and decisions', () => {
    installViewport(390)
    queue.decide.mockReset()
    queue.items = [{ ...ROW, entity_type: 'organization', structural_degree: 3, doc_count: 2 }]
    mount()
    expect(screen.queryByRole('table')).toBeNull()
    const card = screen.getByTestId('triage-card-triage_queue:t1')
    expect(within(card).getByRole('link', { name: 'Provincie Utrecht' }).getAttribute('href'))
      .toBe('/knowledge-graph?entity=entity%3Ae1')
    expect(card.querySelector('[data-testid="entity-type-canonical"]')?.textContent).toBe('organization')
    expect(card.textContent).toContain('Degree 3')
    expect(within(card).getByTestId('triage-docs-triage_queue:t1').textContent).toBe('2')
    expect(card.textContent).toContain('unsure')
    fireEvent.click(within(card).getByRole('button', { name: 'Keep Provincie Utrecht as reference' }))
    expect(queue.decide).toHaveBeenCalledWith(expect.objectContaining({ queueId: 'triage_queue:t1' }))
    expect(within(card).getByRole('button', { name: 'Promote Provincie Utrecht to active' })).toBeTruthy()
    expect(within(card).getByRole('button', { name: 'Pin Provincie Utrecht by hand' })).toBeTruthy()
  })

  it('from md up it is the table', () => {
    installViewport(768)
    queue.items = [ROW]
    mount()
    expect(screen.getByTestId('triage-row-triage_queue:t1').tagName).toBe('TR')
    expect(screen.queryByTestId('triage-card-triage_queue:t1')).toBeNull()
  })
})
