// @vitest-environment jsdom
/**
 * WB.3: the Sources page as the pipeline board, against the sources-board fake
 * (the same double the e2e spec serves; its answers are the real endpoints',
 * see `sources-board.fake.ts`).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'

import {
  EXPECTED,
  NOTEBOOKS,
  fakeCounts,
  fakeList,
  sourceId,
} from '../../../../../e2e/track-wb/sources-board.fake'

const nav = vi.hoisted(() => ({ params: new URLSearchParams(), push: vi.fn() }))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: nav.push }),
  useSearchParams: () => nav.params,
}))
vi.mock('next/link', () => ({
  default: ({ children, href, ...rest }: { children: ReactNode; href: string }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}))

type ListParams = Record<string, unknown>
const api = vi.hoisted(() => ({ list: vi.fn(), counts: vi.fn(), delete: vi.fn() }))
vi.mock('@/lib/api/sources', () => ({ sourcesApi: api }))
vi.mock('@/lib/hooks/use-notebooks', () => ({
  useNotebooks: () => ({ data: NOTEBOOKS }),
}))

/** The client's params as the query string axios sends (repeated `stage`). */
function asQuery(params: ListParams = {}): URLSearchParams {
  const out = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined) continue
    if (Array.isArray(v)) v.forEach((x) => out.append(k, String(x)))
    else out.set(k, String(v))
  }
  return out
}

import SourcesPage from '../page'

function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <SourcesPage />
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  nav.params = new URLSearchParams()
  nav.push.mockReset()
  api.list.mockReset().mockImplementation(async (p: ListParams) => fakeList(asQuery(p)))
  api.counts.mockReset().mockImplementation(async (p?: ListParams) => fakeCounts(asQuery(p)))
})
afterEach(cleanup)

const rowIds = () =>
  screen.getAllByTestId(/^source-row-/).map((r) => r.getAttribute('data-testid')!.slice(11))

describe('Sources board', () => {
  it('lists needs-you first, in the server order, marked', async () => {
    mount()
    await screen.findByTestId(`source-row-${sourceId('gated')}`)
    expect(api.list.mock.calls[0][0]).toMatchObject({ sort_by: 'attention', sort_order: 'desc' })
    expect(rowIds()).toEqual(EXPECTED.attention_order.all.map(sourceId))
    const marked = screen
      .getAllByTestId(/^source-row-/)
      .filter((r) => r.getAttribute('data-needs-you') === 'true').length
    expect(marked).toBe(EXPECTED.counts.all.needs_you)
  })

  it('shows a source stopped at embedded differently from a complete one', async () => {
    mount()
    const stopped = await screen.findByTestId(`source-row-${sourceId('stopped')}`)
    const done = screen.getByTestId(`source-row-${sourceId('done1')}`)
    expect(within(stopped).getByTestId('pipeline-stage-label').textContent).toBe('Embed done')
    expect(within(stopped).getByTestId('source-stopped-badge').textContent).toBe('Stopped')
    expect(within(done).getByTestId('pipeline-stage-label').textContent).toBe('Complete')
    expect(within(done).queryByTestId('source-stopped-badge')).toBeNull()
    const states = (row: HTMLElement) =>
      [...row.querySelectorAll('[data-node-state]')].map((n) => n.getAttribute('data-node-state'))
    expect(states(stopped)).toEqual(['done', 'done', 'pending', 'pending', 'pending'])
    expect(states(done)).toEqual(['done', 'done', 'done', 'done', 'done'])
    // The fresh embedded source is not stopped: no badge, not marked.
    const fresh = screen.getByTestId(`source-row-${sourceId('fresh')}`)
    expect(within(fresh).queryByTestId('source-stopped-badge')).toBeNull()
    expect(fresh.getAttribute('data-needs-you')).toBe('false')
  })

  it('shows every chip count as the server counts it', async () => {
    mount()
    await waitFor(() =>
      expect(screen.getByTestId('sources-chip-all').getAttribute('data-count')).toBe(String(EXPECTED.counts.all.total)))
    const counts = EXPECTED.counts.all
    const want: Record<string, number> = {
      all: counts.total, stalled: counts.stalled, ...counts.stages,
    }
    for (const [key, n] of Object.entries(want)) {
      expect(screen.getByTestId(`sources-chip-${key}`).getAttribute('data-count'), key).toBe(String(n))
    }
    expect(screen.getByTestId('sources-chip-all').getAttribute('aria-current')).toBe('true')
  })

  it('filters by a chip, and a stopped filter asks the server for stalled', async () => {
    nav.params = new URLSearchParams('stalled=true')
    mount()
    await screen.findByTestId(`source-row-${sourceId('stopped')}`)
    expect(api.list.mock.calls[0][0]).toMatchObject({ stalled: true })
    expect(rowIds()).toEqual(['stopped', 'stopped2'].map(sourceId))
    expect(screen.getByTestId('sources-chip-stalled').getAttribute('aria-current')).toBe('true')
    expect(screen.getByTestId('sources-chip-embedded').getAttribute('href')).toBe(
      '/sources?stage=embedded')
  })

  it('scopes the list and the counts to the notebook filter', async () => {
    nav.params = new URLSearchParams('notebook=notebook:beta')
    mount()
    await screen.findByTestId(`source-row-${sourceId('stopped')}`)
    expect(api.list.mock.calls[0][0]).toMatchObject({ notebook_id: 'notebook:beta' })
    expect(api.counts).toHaveBeenCalledWith({ notebook_id: 'notebook:beta' })
    expect(rowIds()).toEqual(EXPECTED.attention_order['notebook:beta'].map(sourceId))
    await waitFor(() =>
      expect(screen.getByTestId('sources-chip-all').getAttribute('data-count')).toBe('4'))
    expect(screen.getByTestId('sources-chip-failed').getAttribute('href')).toBe(
      '/sources?stage=failed&notebook=notebook%3Abeta')
  })

  it('never shows the upload prefix', async () => {
    mount()
    await screen.findByTestId(`source-row-${sourceId('stopped')}`)
    const titles = screen.getAllByTestId(/^source-row-/).map((r) => r.textContent ?? '')
    expect(titles.some((t) => t.includes('Convenant Regio Deal Waterwegregio'))).toBe(true)
    expect(document.body.textContent).not.toMatch(/[0-9a-f]{12}__/)
  })

  it('has Add source in the header', async () => {
    mount()
    const add = await screen.findByTestId('sources-add')
    expect(add.getAttribute('href')).toBe('/sources/new')
  })

  it('keeps arrow keys and Enter on the table, not on a focused chip', async () => {
    mount()
    await screen.findByTestId(`source-row-${sourceId('gated')}`)
    const table = screen.getByRole('table', { name: 'Sources' })
    table.focus()
    fireEvent.keyDown(table, { key: 'ArrowDown' })
    fireEvent.keyDown(table, { key: 'Enter' })
    expect(nav.push).toHaveBeenCalledWith(`/sources/${sourceId('failed')}`)

    nav.push.mockReset()
    const chip = screen.getByTestId('sources-chip-complete')
    chip.focus()
    fireEvent.keyDown(chip, { key: 'Enter' })
    fireEvent.keyDown(chip, { key: 'ArrowDown' })
    expect(nav.push).not.toHaveBeenCalled()
  })

  it('names each chip by its label and count on the link itself (WB.3 m8)', async () => {
    mount()
    const total = EXPECTED.counts.all.total
    const chip = await screen.findByRole('link', { name: `All ${total} sources` })
    expect(chip.getAttribute('data-testid')).toBe('sources-chip-all')
    // aria-label on a generic span is ignored by assistive tech.
    expect(screen.getByRole('navigation', { name: 'Filter by stage' }).querySelector('span[aria-label]')).toBeNull()
  })

  it('marks the sorted column with aria-sort (WB.3 m8)', async () => {
    mount()
    await screen.findByTestId(`source-row-${sourceId('gated')}`)
    const header = (name: RegExp) =>
      screen.getAllByRole('columnheader').find((th) => name.test(th.textContent ?? ''))!
    expect(header(/^Stage/).getAttribute('aria-sort')).toBe('descending')
    expect(header(/^Created/).getAttribute('aria-sort')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /^Created/ }))
    await waitFor(() => expect(header(/^Created/).getAttribute('aria-sort')).toBe('descending'))
    expect(header(/^Stage/).getAttribute('aria-sort')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /^Created/ }))
    await waitFor(() => expect(header(/^Created/).getAttribute('aria-sort')).toBe('ascending'))
  })

  it('says so when the counts cannot be read, and shows no numbers', async () => {
    api.counts.mockReset().mockRejectedValue(new Error('down'))
    mount()
    await screen.findByTestId('sources-counts-error')
    expect(screen.getByTestId('sources-chip-all').getAttribute('data-count')).toBe('')
    // The list still loads.
    await screen.findByTestId(`source-row-${sourceId('gated')}`)
  })

  it('shows a loading state, then an error when the list fails', async () => {
    api.list.mockReset().mockRejectedValue(new Error('down'))
    mount()
    expect(screen.getByRole('status')).toBeTruthy()
    await screen.findByText('Failed to load sources')
  })
})
