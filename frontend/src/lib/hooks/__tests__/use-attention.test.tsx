// @vitest-environment jsdom
/**
 * WB.1 review: the sidebar (on every page) reads the cached report instead of
 * refetching per page, refreshes every 90 s while the tab is visible (not
 * while hidden); the Werkbak page reads it anew; a mutation that resolves an
 * item (here a triage decision) makes it read again.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, render, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider, focusManager } from '@tanstack/react-query'
import type { ReactNode } from 'react'

import fixture from '../../../../e2e/track-wb/attention.fixture.json'

const get = vi.hoisted(() => vi.fn())
const decide = vi.hoisted(() => vi.fn())
vi.mock('@/lib/api/attention', () => ({ attentionApi: { get } }))
vi.mock('@/lib/api/triage', () => ({ triageApi: { decide } }))
vi.mock('@/lib/hooks/use-toast', () => ({ useToast: () => ({ toast: vi.fn() }) }))

import { REFRESH_MS, useAttention } from '../use-attention'
import { useDecideQueueRow } from '../use-triage'

let client: QueryClient
const wrapper = ({ children }: { children: ReactNode }) => (
  <QueryClientProvider client={client}>{children}</QueryClientProvider>
)

function Badge() {
  useAttention()
  return null
}
function Page() {
  useAttention({ fresh: true })
  return null
}

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  get.mockReset().mockResolvedValue(fixture)
  decide.mockReset().mockResolvedValue({ queue_id: 'q', entity: 'e', status: 'active',
    manual_override: true, decision: 'active' })
})
afterEach(() => cleanup())

describe('useAttention', () => {
  it('the sidebar reuses the cached report across pages', async () => {
    const first = render(<Badge />, { wrapper })
    await waitFor(() => expect(get).toHaveBeenCalledTimes(1))
    first.unmount()
    render(<Badge />, { wrapper })
    await new Promise((r) => setTimeout(r, 20))
    expect(get).toHaveBeenCalledTimes(1)
  })

  it('the sidebar refreshes every 90 s while visible, not while hidden', async () => {
    vi.useFakeTimers()
    try {
      render(<Badge />, { wrapper })
      await act(() => vi.advanceTimersByTimeAsync(0))
      expect(get).toHaveBeenCalledTimes(1)
      await act(() => vi.advanceTimersByTimeAsync(REFRESH_MS - 1000))
      expect(get).toHaveBeenCalledTimes(1)
      await act(() => vi.advanceTimersByTimeAsync(1500))
      expect(get).toHaveBeenCalledTimes(2)

      // Hidden for 10 minutes: no refresh.
      act(() => focusManager.setFocused(false))
      await act(() => vi.advanceTimersByTimeAsync(10 * 60 * 1000))
      expect(get).toHaveBeenCalledTimes(2)
      // Back, with a report older than the refresh interval: one read on
      // focus, then the 90 s refresh again.
      act(() => focusManager.setFocused(true))
      await act(() => vi.advanceTimersByTimeAsync(0))
      expect(get).toHaveBeenCalledTimes(3)
      await act(() => vi.advanceTimersByTimeAsync(REFRESH_MS + 500))
      expect(get).toHaveBeenCalledTimes(4)
    } finally {
      focusManager.setFocused(undefined)
      vi.useRealTimers()
    }
  })

  it('an owner who keeps navigating still gets a read about every 90 s', async () => {
    // Every page renders its own sidebar: a navigation remounts it, and that
    // restarts the refresh interval. Navigate every 60 s for 10 minutes.
    vi.useFakeTimers()
    try {
      let view = render(<Badge />, { wrapper })
      await act(() => vi.advanceTimersByTimeAsync(0))
      for (let minute = 1; minute <= 10; minute++) {
        await act(() => vi.advanceTimersByTimeAsync(60 * 1000))
        view.unmount()
        view = render(<Badge />, { wrapper })
        await act(() => vi.advanceTimersByTimeAsync(0))
      }
      // A read once the report is past 90 s: at 0, 2, 4, 6, 8 and 10 min.
      expect(get).toHaveBeenCalledTimes(6)
    } finally {
      vi.useRealTimers()
    }
  })

  it('the Werkbak page reads anew when it opens', async () => {
    render(<Badge />, { wrapper })
    await waitFor(() => expect(get).toHaveBeenCalledTimes(1))
    render(<Page />, { wrapper })
    await waitFor(() => expect(get).toHaveBeenCalledTimes(2))
  })

  it('a triage decision makes the badges read again', async () => {
    render(<Badge />, { wrapper })
    await waitFor(() => expect(get).toHaveBeenCalledTimes(1))
    const { result } = renderHook(() => useDecideQueueRow(), { wrapper })
    act(() => {
      result.current.mutate({ queueId: 'q', body: { status: 'active' } })
    })
    await waitFor(() => expect(get).toHaveBeenCalledTimes(2))
  })
})
