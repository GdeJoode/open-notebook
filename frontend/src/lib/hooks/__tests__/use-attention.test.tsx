// @vitest-environment jsdom
/**
 * WB.1 review: the sidebar (on every page) reads the cached report and does
 * not refetch on mount or focus, even once it is stale; the Werkbak page
 * reads it anew; a mutation that resolves an item (here a triage decision)
 * makes it read again.
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

import { useAttention } from '../use-attention'
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

  it('the sidebar does not refetch on focus, even when the report is stale', async () => {
    client.setQueryData(['attention', null], fixture, { updatedAt: Date.now() - 60 * 60 * 1000 })
    render(<Badge />, { wrapper })
    act(() => {
      focusManager.setFocused(false)
      focusManager.setFocused(true)
    })
    await new Promise((r) => setTimeout(r, 20))
    expect(get).not.toHaveBeenCalled()
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
