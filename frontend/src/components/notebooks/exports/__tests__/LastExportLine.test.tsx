// @vitest-environment jsdom
/**
 * PC.5c: the ``export.*`` metrics have a reader a person sees.
 *
 * Mounts the real line over the real hook; only the HTTP client is stubbed, so
 * the request the hook makes (path, format, limit) is what is asserted.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

import { LastExportLine } from '../LastExportLine'
import type { ExportHistoryEntry } from '@/lib/types/exports'

const get = vi.fn()
vi.mock('@/lib/api/client', () => ({ default: { get: (...a: unknown[]) => get(...a) } }))

const RUN: ExportHistoryEntry = {
  format: 'jsonl',
  created_at: '2026-09-27T10:00:00Z',
  error: null,
  entities_written: 12,
  relations_written: 1,
  files_written: 1,
  bytes_written: 400,
  duration_ms: 30,
  mode: null,
}

function mount(rows: ExportHistoryEntry[]) {
  get.mockResolvedValue({ data: rows })
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <LastExportLine notebookId="notebook:abc" format="jsonl" />
    </QueryClientProvider>,
  )
}

afterEach(() => {
  cleanup()
  get.mockReset()
})

describe('LastExportLine', () => {
  it('asks for the newest run of its own format', async () => {
    mount([RUN])
    await screen.findByTestId('last-export-jsonl')
    expect(get).toHaveBeenCalledWith(
      '/notebooks/notebook%3Aabc/exports/history',
      { params: { format: 'jsonl', limit: 1 } },
    )
  })

  it('shows what the run wrote', async () => {
    mount([RUN])
    const line = await screen.findByTestId('last-export-jsonl')
    expect(line.textContent).toContain('12 entities, 1 relation')
  })

  it('shows a failed run as failed, with its reason', async () => {
    mount([{ ...RUN, entities_written: 0, error: 'RuntimeError: boom' }])
    const line = await screen.findByTestId('last-export-jsonl')
    expect(line.textContent).toContain('failed (RuntimeError: boom)')
  })

  it('renders nothing before the first run', async () => {
    mount([])
    await vi.waitFor(() => expect(get).toHaveBeenCalled())
    expect(screen.queryByTestId('last-export-jsonl')).toBeNull()
  })
})
