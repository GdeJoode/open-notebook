// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeAll, beforeEach } from 'vitest'
import { render, screen, cleanup, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { ApiKeys } from '../ApiKeys'
import type { AgentKey } from '@/lib/api/agent-keys'

const mintMutate = vi.fn()
const mintReset = vi.fn()
const revokeMutate = vi.fn()

let keysState: { data?: AgentKey[]; isLoading: boolean; isError: boolean }
let auditState: { data?: unknown; isLoading: boolean; isError: boolean }

vi.mock('@/lib/hooks/use-agent-keys', () => ({
  useAgentKeys: () => keysState,
  useMintAgentKey: () => ({
    mutate: mintMutate,
    reset: mintReset,
    isPending: false,
    isError: false,
    error: null,
  }),
  useRevokeAgentKey: () => ({ mutate: revokeMutate, isPending: false }),
  useKeyAuditLog: () => auditState,
}))

const KEY: AgentKey = {
  id: 'agent_keys:1',
  agent_id: 'research-bot',
  key_prefix: 'ak_abcdefg',
  label: 'CI bot',
  permission: 'write',
  rate_limit_rpm: null,
  revoked: false,
  created: '2026-07-24T10:00:00Z',
  last_used_at: null,
}

beforeAll(() => {
  // Radix Dialog/AlertDialog reach for pointer-capture APIs jsdom lacks.
  Element.prototype.hasPointerCapture = vi.fn(
    () => false,
  ) as unknown as typeof Element.prototype.hasPointerCapture
  Element.prototype.setPointerCapture =
    vi.fn() as unknown as typeof Element.prototype.setPointerCapture
  Element.prototype.releasePointerCapture =
    vi.fn() as unknown as typeof Element.prototype.releasePointerCapture
  Element.prototype.scrollIntoView = vi.fn()
})

afterEach(cleanup)
beforeEach(() => {
  mintMutate.mockReset()
  mintReset.mockReset()
  revokeMutate.mockReset()
  keysState = { data: [], isLoading: false, isError: false }
  auditState = { data: undefined, isLoading: false, isError: false }
})

describe('ApiKeys — render states', () => {
  it('renders the loading state', () => {
    keysState = { data: undefined, isLoading: true, isError: false }
    render(<ApiKeys />)
    expect(screen.getByRole('status')).toBeTruthy()
  })

  it('renders the error state', () => {
    keysState = { data: undefined, isLoading: false, isError: true }
    render(<ApiKeys />)
    expect(screen.getByTestId('api-keys-error')).toBeTruthy()
  })

  it('renders the empty state', () => {
    keysState = { data: [], isLoading: false, isError: false }
    render(<ApiKeys />)
    expect(screen.getByTestId('api-keys-empty')).toBeTruthy()
  })

  it('lists keys with agent id, permission, and prefix', () => {
    keysState = { data: [KEY], isLoading: false, isError: false }
    render(<ApiKeys />)
    const row = screen.getByTestId('api-key-row-agent_keys:1')
    expect(within(row).getByText('research-bot')).toBeTruthy()
    expect(within(row).getByText('write')).toBeTruthy()
    expect(within(row).getByText(/ak_abcdefg/)).toBeTruthy()
  })
})

describe('ApiKeys — generate (show-once)', () => {
  it('mints a key and reveals the plaintext exactly once', async () => {
    const user = userEvent.setup()
    // The mint mock resolves by invoking the caller's onSuccess with a plaintext.
    mintMutate.mockImplementation((_input, opts) =>
      opts?.onSuccess?.({ ...KEY, key: 'ak_SECRET_PLAINTEXT_VALUE' }),
    )
    render(<ApiKeys />)

    await user.click(screen.getByTestId('api-keys-generate'))
    await user.type(screen.getByLabelText('Agent ID'), 'research-bot')
    await user.click(screen.getByTestId('api-key-generate-submit'))

    // mint called with the typed agent id
    expect(mintMutate).toHaveBeenCalledTimes(1)
    expect(mintMutate.mock.calls[0][0]).toMatchObject({
      agent_id: 'research-bot',
      permission: 'read',
    })
    // plaintext revealed once
    expect(screen.getByTestId('api-key-plaintext').textContent).toContain(
      'ak_SECRET_PLAINTEXT_VALUE',
    )

    // Closing the dialog drops the plaintext — it is never rendered again.
    await user.click(screen.getByRole('button', { name: 'Done' }))
    expect(screen.queryByTestId('api-key-plaintext')).toBeNull()
    expect(mintReset).toHaveBeenCalled()
  })
})

describe('ApiKeys — revoke', () => {
  it('revokes a key by id after confirmation', async () => {
    const user = userEvent.setup()
    keysState = { data: [KEY], isLoading: false, isError: false }
    render(<ApiKeys />)

    await user.click(screen.getByTestId('revoke-agent_keys:1'))
    await user.click(screen.getByTestId('revoke-confirm-agent_keys:1'))

    expect(revokeMutate).toHaveBeenCalledWith('agent_keys:1')
  })

  it('does not offer revoke for an already-revoked key', () => {
    keysState = {
      data: [{ ...KEY, revoked: true }],
      isLoading: false,
      isError: false,
    }
    render(<ApiKeys />)
    expect(screen.queryByTestId('revoke-agent_keys:1')).toBeNull()
    expect(screen.getByText('revoked')).toBeTruthy()
  })
})

describe('ApiKeys — accessibility', () => {
  it('exposes aria-labels on the per-key controls', () => {
    keysState = { data: [KEY], isLoading: false, isError: false }
    render(<ApiKeys />)
    expect(
      screen.getByLabelText('View audit log for research-bot'),
    ).toBeTruthy()
    expect(screen.getByLabelText('Revoke key for research-bot')).toBeTruthy()
  })
})
