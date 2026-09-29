// @vitest-environment jsdom
/**
 * WB.1: the Werkbak renders the groups `GET /api/attention` returns (the
 * fixture is the one the backend validates against its model), each linked to
 * where it is resolved, with loading, error and empty states. "Last seen" is
 * stored only on "Mark as seen", never on render, and the page works when
 * storage throws.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'

import fixture from '../../../../e2e/track-wb/attention.fixture.json'
import type { AttentionReport } from '@/lib/api/attention'

const get = vi.hoisted(() => vi.fn())
vi.mock('@/lib/api/attention', () => ({ attentionApi: { get } }))
vi.mock('@/components/layout/AppShell', () => ({
  AppShell: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}))
vi.mock('next/link', () => ({
  default: ({ children, href, ...rest }: { children: ReactNode; href: string }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}))

import WerkbakPage from '../page'

const REPORT = fixture as AttentionReport
const KEY = 'werkbak.lastSeen'

function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <WerkbakPage />
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  window.localStorage.clear()
  get.mockReset()
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('Werkbak', () => {
  it('shows a loading state, then one card per group that needs you', async () => {
    get.mockResolvedValue({ ...REPORT, arrived: [] })
    mount()
    expect(screen.getByTestId('werkbak-loading')).toBeTruthy()
    await screen.findByTestId('werkbak-group-triage')

    const shown = REPORT.groups.filter((g) => g.count > 0 || g.error).map((g) => g.key)
    for (const g of REPORT.groups) {
      const card = screen.queryByTestId(`werkbak-group-${g.key}`)
      expect(!!card, g.key).toBe(shown.includes(g.key))
    }
    const triage = screen.getByTestId('werkbak-group-triage')
    expect(within(triage).getByTestId('werkbak-count-triage').textContent).toBe('1238')
    expect(within(triage).getByText('and 1237 more')).toBeTruthy()
    const action = within(triage).getByTestId('werkbak-action-triage')
    expect(action.getAttribute('href')).toBe('/knowledge-graph/triage')
    expect(action.textContent).toContain('Triage 1238 entities')
    // Gated sources, per notebook, link to the Schema tab where review happens.
    const gated = screen.getByTestId('werkbak-group-sources_awaiting_review')
    expect(within(gated).getByTestId('werkbak-item-notebook:live').getAttribute('href'))
      .toBe('/notebooks/notebook%3Alive/schema')
    expect(within(gated).getByTestId('werkbak-action-sources_awaiting_review')
      .getAttribute('href')).toBe('/notebooks/notebook%3Alive/schema')
    expect(screen.getByRole('heading', { name: /Needs you/ }).textContent).toContain('(1257)')
    // Its source is listed as waiting for review: shown, not counted again.
    expect(screen.getByTestId('werkbak-elsewhere-extraction_paused').textContent)
      .toBe('1 already counted with their source')
    expect(screen.queryByTestId('werkbak-elsewhere-schema_proposals')).toBeNull()
  })

  it('a failed group shows its error and no count; the others still render', async () => {
    get.mockResolvedValue(REPORT)
    mount()
    const card = await screen.findByTestId('werkbak-group-merge_candidates')
    expect(within(card).getByRole('alert').textContent).toContain('timed out after 5 s')
    expect(within(card).queryByTestId('werkbak-count-merge_candidates')).toBeNull()
    expect(screen.getByTestId('werkbak-group-retype')).toBeTruthy()
  })

  it('shows the empty state when nothing needs you', async () => {
    get.mockResolvedValue({
      ...REPORT,
      total: 0,
      groups: REPORT.groups.map((g) => ({ ...g, count: 0, items: [], action: '', error: null })),
      arrived: [],
    })
    mount()
    expect(await screen.findByTestId('werkbak-empty')).toBeTruthy()
    expect(screen.getByText('Nothing needs you')).toBeTruthy()
  })

  it('shows an error with a retry when the read fails', async () => {
    get.mockRejectedValueOnce(new Error('Network Error')).mockResolvedValue(REPORT)
    mount()
    const alert = await screen.findByTestId('werkbak-error')
    expect(alert.textContent).toContain('Network Error')
    fireEvent.click(within(alert).getByRole('button', { name: 'Try again' }))
    expect(await screen.findByTestId('werkbak-group-triage')).toBeTruthy()
  })

  it('without a last visit: no arrivals, and nothing is stored on render', async () => {
    get.mockResolvedValue({ ...REPORT, since: null, arrived: [] })
    mount()
    await screen.findByTestId('werkbak-group-triage')
    expect(get).toHaveBeenCalledWith(null)
    expect(screen.getByTestId('werkbak-since-note').textContent)
      .toContain('Mark as seen to list')
    expect(screen.queryByTestId('werkbak-group-new_sources')).toBeNull()
    expect(screen.queryByTestId('werkbak-nothing-new')).toBeNull()
    expect(window.localStorage.getItem(KEY)).toBeNull()
  })

  it('Mark as seen stores the report time and reads again since it', async () => {
    get.mockResolvedValueOnce({ ...REPORT, since: null, arrived: [] })
      .mockResolvedValue({
        ...REPORT,
        arrived: REPORT.arrived.map((g) => ({ ...g, count: 0, items: [] })),
      })
    mount()
    await screen.findByTestId('werkbak-group-triage')
    fireEvent.click(screen.getByTestId('werkbak-mark-seen'))
    expect(window.localStorage.getItem(KEY)).toBe(REPORT.generated_at)
    await waitFor(() => expect(get).toHaveBeenLastCalledWith(REPORT.generated_at))
    expect(await screen.findByTestId('werkbak-nothing-new')).toBeTruthy()
  })

  it('with a last visit: what arrived since, linked to where it is resolved', async () => {
    window.localStorage.setItem(KEY, REPORT.since as string)
    get.mockResolvedValue(REPORT)
    mount()
    const card = await screen.findByTestId('werkbak-group-new_sources')
    expect(get).toHaveBeenCalledWith(REPORT.since)
    expect(within(card).getByTestId('werkbak-item-source:api1').getAttribute('href'))
      .toBe('/sources/source%3Aapi1')
    const jobs = screen.getByTestId('werkbak-group-finished_jobs')
    expect(within(jobs).getByTestId('werkbak-item-job:f1').getAttribute('href'))
      .toBe('/sources/source%3Aapi1')
  })

  it('works when storage throws', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('denied')
    })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('denied')
    })
    get.mockResolvedValue({ ...REPORT, since: null, arrived: [] })
    mount()
    await screen.findByTestId('werkbak-group-triage')
    expect(get).toHaveBeenCalledWith(null)
    fireEvent.click(screen.getByTestId('werkbak-mark-seen'))
    expect(screen.getByTestId('werkbak-group-triage')).toBeTruthy()
  })
})
