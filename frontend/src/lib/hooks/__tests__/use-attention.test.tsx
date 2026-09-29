// @vitest-environment jsdom
/**
 * WB.1 review, WB.2: the sidebar (in the dashboard layout, mounted once)
 * refreshes the report every 90 s while the tab is visible (not while hidden)
 * as pages come and go; a remount within that window reuses the cached
 * report; The Tray page reads it anew; a mutation that resolves an item (here
 * a triage decision) makes it read again.
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

import { REFRESH_MS, RETRY_BACKOFF_MS, useAttention } from '../use-attention'
import { useDecideQueueRow } from '../use-triage'

// The fixture has a group that timed out (partial); these cadence tests use a
// report where every group loaded. The retry is in werkbak.test.tsx.
const LOADED = {
  ...fixture,
  partial: false,
  groups: fixture.groups.map((g) => ({ ...g, error: null })),
}

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
  get.mockReset().mockResolvedValue(LOADED)
  decide.mockReset().mockResolvedValue({ queue_id: 'q', entity: 'e', status: 'active',
    manual_override: true, decision: 'active' })
})
afterEach(() => cleanup())

describe('useAttention', () => {
  it('a remount within the refresh interval reuses the cached report', async () => {
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

  it('an owner who keeps navigating gets a read every 90 s', async () => {
    // As the dashboard layout does: the sidebar stays, the page changes.
    // Navigate every 60 s for 10 minutes.
    const Shell = ({ page }: { page: number }) => (
      <>
        <Badge />
        <p>page {page}</p>
      </>
    )
    vi.useFakeTimers()
    try {
      const view = render(<Shell page={0} />, { wrapper })
      await act(() => vi.advanceTimersByTimeAsync(0))
      for (let minute = 1; minute <= 10; minute++) {
        await act(() => vi.advanceTimersByTimeAsync(60 * 1000))
        view.rerender(<Shell page={minute} />)
        await act(() => vi.advanceTimersByTimeAsync(0))
      }
      // At 0, 90, 180, 270, 360, 450 and 540 s.
      expect(get).toHaveBeenCalledTimes(7)
    } finally {
      vi.useRealTimers()
    }
  })

  it('a report that stays partial backs off 5, 15, 45 s, then 90 s; a complete one resets', async () => {
    vi.useFakeTimers()
    try {
      get.mockResolvedValue(fixture) // merge candidates time out, every time
      render(<Badge />, { wrapper })
      await act(() => vi.advanceTimersByTimeAsync(0))
      await act(() => vi.advanceTimersByTimeAsync(10 * 60 * 1000))
      // Reads at 0, 5, 20, 65 s, then every 90 s: 155 ... 515 s.
      expect(get).toHaveBeenCalledTimes(9)

      // The next read (605 s) is complete: the streak resets, next at 695 s.
      get.mockResolvedValue(LOADED)
      await act(() => vi.advanceTimersByTimeAsync(10 * 1000)) // 610 s
      expect(get).toHaveBeenCalledTimes(10)
      // Partial again at 695 s: the backoff starts over at 5 s (700 s).
      get.mockResolvedValue(fixture)
      await act(() => vi.advanceTimersByTimeAsync(86 * 1000)) // 696 s
      expect(get).toHaveBeenCalledTimes(11)
      await act(() => vi.advanceTimersByTimeAsync(RETRY_BACKOFF_MS[0])) // 701 s
      expect(get).toHaveBeenCalledTimes(12)
    } finally {
      vi.useRealTimers()
    }
  })

  it('a focus reads a partial report once its retry delay has passed', async () => {
    vi.useFakeTimers()
    try {
      get.mockResolvedValue(fixture)
      render(<Badge />, { wrapper })
      await act(() => vi.advanceTimersByTimeAsync(1000))
      act(() => focusManager.setFocused(false)) // hidden before the 5 s retry fired
      await act(() => vi.advanceTimersByTimeAsync(5000))
      expect(get).toHaveBeenCalledTimes(1)
      act(() => focusManager.setFocused(true))
      await act(() => vi.advanceTimersByTimeAsync(0))
      expect(get).toHaveBeenCalledTimes(2)
    } finally {
      focusManager.setFocused(undefined)
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
