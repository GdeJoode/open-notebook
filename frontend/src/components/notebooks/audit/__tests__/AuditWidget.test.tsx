// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { AuditWidget } from '../AuditWidget'
import type { AuditRunResponse } from '@/lib/types/audit'

/**
 * F.2 widget test — the hooks are mocked so this is a pure render/interaction
 * test of the four states (loading / error / empty / findings), severity
 * grouping, and the Re-run action.
 */

const mutateSpy = vi.fn()
let auditState: {
  data?: AuditRunResponse
  isLoading: boolean
  isError: boolean
}

vi.mock('@/lib/hooks/use-audit', () => ({
  useNotebookAudit: () => auditState,
  useRunAudit: () => ({ mutate: mutateSpy, isPending: false }),
}))

afterEach(cleanup)
beforeEach(() => {
  mutateSpy.mockReset()
  auditState = { data: undefined, isLoading: false, isError: false }
})

function renderWidget() {
  return render(<AuditWidget notebookId="notebook:abc" />)
}

const RESPONSE: AuditRunResponse = {
  notebook_id: 'notebook:abc',
  run_id: 'run-1',
  created: '2026-07-24T10:00:00Z',
  findings: [
    { check_id: 'citation_completeness', severity: 'error', title: 'No citation', subject: 'entity:e1' },
    { check_id: 'low_confidence_survivors', severity: 'warn', title: 'Barely survived' },
    { check_id: 'stale_sources', severity: 'info', title: 'Stale source' },
    { check_id: 'schema_drift', severity: 'info', title: 'Drift 40%' },
  ],
}

describe('AuditWidget', () => {
  it('renders the loading state', () => {
    auditState = { data: undefined, isLoading: true, isError: false }
    renderWidget()
    expect(screen.getByRole('status')).toBeTruthy()
  })

  it('renders the error state', () => {
    auditState = { data: undefined, isLoading: false, isError: true }
    renderWidget()
    expect(screen.getByTestId('audit-error')).toBeTruthy()
  })

  it('renders an explicit empty state (never blank)', () => {
    auditState = {
      data: { notebook_id: 'notebook:abc', run_id: 'run-1', findings: [] },
      isLoading: false,
      isError: false,
    }
    renderWidget()
    expect(screen.getByTestId('audit-empty')).toBeTruthy()
  })

  it('groups findings by severity with counts and links the subject', () => {
    auditState = { data: RESPONSE, isLoading: false, isError: false }
    renderWidget()

    // Counts: 1 error, 1 warn, 2 info.
    const counts = screen.getByTestId('audit-counts')
    expect(within(counts).getByText(/1 Errors/)).toBeTruthy()
    expect(within(counts).getByText(/1 Warnings/)).toBeTruthy()
    expect(within(counts).getByText(/2 Info/)).toBeTruthy()

    // Severity groups render.
    expect(screen.getByTestId('audit-group-error')).toBeTruthy()
    expect(screen.getByTestId('audit-group-info')).toBeTruthy()

    // A finding with a subject is a link; the last-run timestamp shows.
    expect(screen.getByTestId('audit-finding-link')).toBeTruthy()
    expect(screen.getByTestId('audit-last-run')).toBeTruthy()
  })

  it('re-runs the audit on button click', async () => {
    auditState = { data: RESPONSE, isLoading: false, isError: false }
    const user = userEvent.setup()
    renderWidget()
    await user.click(screen.getByTestId('audit-rerun'))
    expect(mutateSpy).toHaveBeenCalledTimes(1)
  })
})
