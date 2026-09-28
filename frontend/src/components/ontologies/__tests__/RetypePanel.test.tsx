// @vitest-environment jsdom
/**
 * OE.4: the retype panel against the API's shapes (stubbed at the module
 * boundary): the dry run shown, an apply of exactly that plan behind a
 * confirmation, collisions pointed at the resolution page, and every state.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

import type { RetypeDryRun } from '@/lib/api/ontologies'

const retype = vi.hoisted(() => ({ dryRun: vi.fn(), apply: vi.fn(), undo: vi.fn() }))

vi.mock('@/lib/api/ontologies', async (original) => ({
  ...(await original<typeof import('@/lib/api/ontologies')>()),
  retypeApi: retype,
}))
vi.mock('@/lib/hooks/use-toast', () => ({ useToast: () => ({ toast: vi.fn() }) }))

import { RetypePanel } from '../RetypePanel'

const row = (id: string, name: string, old: string, next: string) => ({
  id, name, label: 'Streekfonds', status: 'active', old, new: next, old_flag: null, new_flag: null,
})

function plan(overrides: Partial<RetypeDryRun> = {}): RetypeDryRun {
  const base = {
    total: 4,
    changes: [row('entity:1', 'Fonds Noord', 'concept', 'grant')],
    collisions: [{ ...row('entity:2', 'Fonds Zuid', 'concept', 'grant'), held_by: ['entity:3'] }],
    flag_only: [],
    fingerprint: 'fp-1',
    notebook_edits: [],
    ...overrides,
  }
  return { ...base, counts: { changes: base.changes.length, collisions: base.collisions.length,
    flag_only: base.flag_only.length } }
}

function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(<QueryClientProvider client={client}><RetypePanel onClose={vi.fn()} /></QueryClientProvider>)
}

async function applyConfirmed() {
  fireEvent.click(await screen.findByTestId('ontology-retype-apply'))
  fireEvent.click(await screen.findByTestId('confirm-action'))
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('RetypePanel', () => {
  it('shows the changes, and the collisions apart with a link to the resolution page', async () => {
    retype.dryRun.mockResolvedValue(plan())
    mount()
    const changes = await screen.findByTestId('ontology-retype-changes')
    expect(changes.textContent).toContain('Fonds Noord')
    expect(changes.textContent).toContain('concept → grant')
    expect(changes.textContent).not.toContain('Fonds Zuid')
    const collisions = screen.getByTestId('ontology-retype-collisions')
    expect(collisions.textContent).toContain('held by entity:3')
    expect(within(collisions).getByRole('link', { name: 'resolution page' })
      .getAttribute('href')).toBe('/knowledge-graph/resolution')
    expect(screen.getByTestId('ontology-retype-counts').textContent)
      .toContain('1 type change(s), 0 literal(s) to flag, 1 collision(s)')
  })

  it('applies only after a confirmation, with the fingerprint of the plan shown', async () => {
    retype.dryRun.mockResolvedValue(plan())
    retype.apply.mockResolvedValue({ written: 1, skipped: [], failed: [],
      rollback_id: 'pc4b_retype_20260928T120000000000Z.jsonl' })
    mount()
    fireEvent.click(await screen.findByTestId('ontology-retype-apply'))
    const confirm = await screen.findByTestId('confirm-action')
    expect(retype.apply).not.toHaveBeenCalled() // the dialog is open; nothing sent yet
    fireEvent.click(confirm)
    await waitFor(() => expect(retype.apply).toHaveBeenCalledWith('fp-1'))
    expect((await screen.findByTestId('ontology-retype-result')).textContent).toContain('Written 1')

    fireEvent.click(screen.getByTestId('ontology-retype-undo'))
    fireEvent.click(await screen.findByTestId('confirm-action'))
    await waitFor(() => expect(retype.undo)
      .toHaveBeenCalledWith('pc4b_retype_20260928T120000000000Z.jsonl'))
  })

  it('a refused apply (the plan changed) fetches and shows the new plan', async () => {
    retype.dryRun.mockResolvedValueOnce(plan())
      .mockResolvedValue(plan({ fingerprint: 'fp-2',
        changes: [row('entity:1', 'Fonds Noord', 'concept', 'grant'),
          row('entity:9', 'Fonds West', 'concept', 'grant')] }))
    retype.apply.mockRejectedValue({ response: { status: 409, data: { detail: 'plan changed' } } })
    mount()
    await applyConfirmed()
    await waitFor(() => expect(screen.getByTestId('ontology-retype-changes').textContent)
      .toContain('Fonds West'))
    expect(screen.queryByTestId('ontology-retype-result')).toBeNull()
    await applyConfirmed()
    await waitFor(() => expect(retype.apply).toHaveBeenLastCalledWith('fp-2'))
  })

  it('an empty plan offers nothing to apply', async () => {
    retype.dryRun.mockResolvedValue(plan({ changes: [], collisions: [] }))
    mount()
    await screen.findByTestId('ontology-retype-empty')
    expect(screen.queryByTestId('ontology-retype-apply')).toBeNull()
    expect(screen.queryByTestId('ontology-retype-collisions')).toBeNull()
  })

  it('notebook schema edits block the apply', async () => {
    retype.dryRun.mockResolvedValue(plan({ notebook_edits: ['notebook:a'] }))
    mount()
    const apply = await screen.findByTestId('ontology-retype-apply') as HTMLButtonElement
    expect(apply.disabled).toBe(true)
    expect(screen.getByRole('alert').textContent).toContain('notebook:a')
  })

  it('a failed dry run says so', async () => {
    retype.dryRun.mockRejectedValue({ response: { data: { detail: 'database down' } } })
    mount()
    expect((await screen.findByRole('alert')).textContent).toContain('database down')
    expect(screen.queryByTestId('ontology-retype-apply')).toBeNull()
  })
})
