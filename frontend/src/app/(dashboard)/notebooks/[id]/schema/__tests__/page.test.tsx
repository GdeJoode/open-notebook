// @vitest-environment jsdom
/**
 * WB.3 (review minors m1, m5): on the Schema tab the proposals come before the
 * paused-extraction banner (whose Resume is secondary), and "Next: The Tray"
 * shows only once no proposal is pending and extraction is not paused.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'

const state = vi.hoisted(() => ({
  pending: [] as unknown[],
  paused: { isSuccess: true, data: { paused_count: 0 } } as {
    isSuccess: boolean
    data?: { paused_count: number }
  },
}))
vi.mock('next/navigation', () => ({ useParams: () => ({ id: 'notebook%3Anb1' }) }))
vi.mock('next/link', () => ({
  default: ({ children, href, ...rest }: { children: ReactNode; href: string }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}))
vi.mock('@/lib/hooks/use-notebooks', () => ({
  useNotebook: () => ({ data: { id: 'notebook:nb1', name: 'Alpha' }, isLoading: false }),
}))
vi.mock('@/lib/hooks/use-notebook-schema', () => ({
  useNotebookSchema: () => ({
    data: { pending_extensions: state.pending, review_gate_active: true },
    isLoading: false,
    isError: false,
  }),
  usePausedExtraction: () => state.paused,
}))
const stub = vi.hoisted(() => (id: string) => () => <div data-testid={id} />)
vi.mock('../../../components/NotebookHeader', () => ({ NotebookHeader: stub('header') }))
vi.mock('@/components/notebooks/schema/SchemaBrowser', () => ({ SchemaBrowser: stub('browser') }))
vi.mock('@/components/notebooks/schema/CoverageStatsTable', () => ({ CoverageStatsTable: stub('coverage') }))
vi.mock('@/components/notebooks/orphans/OrphansDashboard', () => ({ OrphansDashboard: stub('orphans') }))
vi.mock('@/components/notebooks/audit/AuditWidget', () => ({ AuditWidget: stub('audit') }))
vi.mock('@/components/notebooks/audit/LibrarianSettings', () => ({ LibrarianSettings: stub('librarian') }))
vi.mock('@/components/notebooks/schema/PendingExtensionsPanel', () => ({
  PendingExtensionsPanel: stub('proposals'),
}))
vi.mock('@/components/notebooks/schema/ExtractionPausedBanner', () => ({
  ExtractionPausedBanner: stub('paused-banner'),
}))
vi.mock('@/components/notebooks/schema/TtlDownloadButton', () => ({ TtlDownloadButton: stub('ttl') }))
vi.mock('@/components/notebooks/exports/NetworkxExportMenu', () => ({ NetworkxExportMenu: stub('nx') }))

import NotebookSchemaPage from '../page'

beforeEach(() => {
  state.pending = []
  state.paused = { isSuccess: true, data: { paused_count: 0 } }
})
afterEach(cleanup)

describe('Schema tab', () => {
  it('puts the proposals before the paused-extraction banner', () => {
    state.pending = [{ type_name: 'RegioDeal', properties: [] }]
    render(<NotebookSchemaPage />)
    const proposals = screen.getByTestId('proposals')
    const banner = screen.getByTestId('paused-banner')
    expect(proposals.compareDocumentPosition(banner) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('offers The Tray once nothing is pending or paused', () => {
    render(<NotebookSchemaPage />)
    expect(screen.getByTestId('schema-next').getAttribute('href')).toBe('/')
  })

  it('does not offer The Tray while extraction is paused', () => {
    state.paused = { isSuccess: true, data: { paused_count: 1 } }
    render(<NotebookSchemaPage />)
    expect(screen.queryByTestId('schema-next')).toBeNull()
  })

  it('nor while the paused state is unknown', () => {
    state.paused = { isSuccess: false }
    render(<NotebookSchemaPage />)
    expect(screen.queryByTestId('schema-next')).toBeNull()
  })

  it('nor while proposals are pending', () => {
    state.pending = [{ type_name: 'RegioDeal', properties: [] }]
    render(<NotebookSchemaPage />)
    expect(screen.queryByTestId('schema-next')).toBeNull()
  })
})
