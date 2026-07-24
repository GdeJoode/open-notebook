// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeAll, beforeEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { LibrarianSettings } from '../LibrarianSettings'

const updateSpy = vi.fn()
let notebookState: { data?: unknown; isLoading: boolean; isError: boolean }
let auditData: { run_id: string; created?: string } | undefined

vi.mock('@/lib/hooks/use-notebooks', () => ({
  useNotebook: () => notebookState,
  useUpdateNotebook: () => ({ mutate: updateSpy, isPending: false }),
}))
vi.mock('@/lib/hooks/use-audit', () => ({
  useNotebookAudit: () => ({ data: auditData }),
}))

beforeAll(() => {
  Element.prototype.hasPointerCapture = vi.fn(
    () => false,
  ) as unknown as typeof Element.prototype.hasPointerCapture
  Element.prototype.setPointerCapture =
    vi.fn() as unknown as typeof Element.prototype.setPointerCapture
  Element.prototype.releasePointerCapture =
    vi.fn() as unknown as typeof Element.prototype.releasePointerCapture
})

afterEach(cleanup)
beforeEach(() => {
  updateSpy.mockReset()
  notebookState = {
    data: { id: 'notebook:abc', librarian_enabled: false },
    isLoading: false,
    isError: false,
  }
  auditData = undefined
})

function renderSettings() {
  return render(<LibrarianSettings notebookId="notebook:abc" />)
}

describe('LibrarianSettings', () => {
  it('renders the loading state', () => {
    notebookState = { data: undefined, isLoading: true, isError: false }
    renderSettings()
    expect(screen.getByRole('status')).toBeTruthy()
  })

  it('renders the error state', () => {
    notebookState = { data: undefined, isLoading: false, isError: true }
    renderSettings()
    expect(screen.getByTestId('librarian-error')).toBeTruthy()
  })

  it('shows "Never run" before any audit', () => {
    renderSettings()
    expect(screen.getByTestId('librarian-last-run').textContent).toContain(
      'Never run',
    )
  })

  it('shows the last-run timestamp once an audit has run', () => {
    auditData = { run_id: 'run-1', created: '2026-07-24T10:00:00Z' }
    renderSettings()
    expect(screen.getByTestId('librarian-last-run').textContent).toContain(
      'Last audit',
    )
  })

  it('toggling persists the opt-in via the update mutation', async () => {
    const user = userEvent.setup()
    renderSettings()
    await user.click(screen.getByTestId('librarian-toggle'))
    expect(updateSpy).toHaveBeenCalledWith({
      id: 'notebook:abc',
      data: { librarian_enabled: true },
    })
  })
})
