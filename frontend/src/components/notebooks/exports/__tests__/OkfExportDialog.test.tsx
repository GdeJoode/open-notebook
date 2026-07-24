// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeAll, beforeEach } from 'vitest'
import { render, screen, cleanup, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { OkfExportDialog } from '../OkfExportDialog'
import type { ExportFilter } from '@/lib/types/exports'
import type { OkfExportResult } from '@/lib/hooks/use-okf-export'

/**
 * OKF.5 dialog test — mirrors the Track-D export dialog coverage. The hook +
 * preview widget are mocked so this stays a pure render/interaction test:
 *
 *   * default state renders the lossy-by-design banner + every filter knob,
 *   * the Export button is keyboard-reachable + aria-labelled (a11y gate),
 *   * submit forwards the parsed filter to the hook,
 *   * a zip success surfaces the omitted-field ledger the server returned.
 */

// --- hook mock: controllable mutate + success payload ---------------------

interface MockMutateOptions {
  onSuccess?: (result: OkfExportResult) => void
}

const mutateSpy = vi.fn<
  (filter: ExportFilter, opts?: MockMutateOptions) => void
>()

vi.mock('@/lib/hooks/use-okf-export', () => ({
  useOkfExport: () => ({
    mutate: mutateSpy,
    isPending: false,
    error: null,
    lastReport: null,
    lastDeferred: null,
  }),
}))

// The dialog embeds ExportPreviewCounts, which calls useExportPreview. Stub it
// to a loaded state so the widget doesn't sit on its skeleton.
vi.mock('@/lib/hooks/use-export-preview', () => ({
  useExportPreview: () => ({
    data: { entity_count: 9, relation_count: 4 },
    isLoading: false,
    isError: false,
  }),
}))

// Radix Slider observes its track size; jsdom ships neither ResizeObserver nor
// pointer-capture, so polyfill them or the dialog crashes on mount.
beforeAll(() => {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver
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
  mutateSpy.mockReset()
})

const OMITTED = {
  'entity.embedding': 'Vector embedding has no OKF representation.',
  verdict_contradiction_edges: 'Verdict edges have no OKF representation.',
}

function renderDialog() {
  return render(
    <OkfExportDialog open onOpenChange={() => {}} notebookId="notebook:abc" />,
  )
}

describe('OkfExportDialog — default state (AC1)', () => {
  it('renders the title, lossy-by-design banner, and filter knobs', () => {
    renderDialog()

    expect(screen.getByText('Export to Open Knowledge Format')).toBeTruthy()

    // Lossy-by-design loss must be visible up-front, role=note for a11y.
    const banner = screen.getByTestId('okf-lossy-banner')
    expect(banner.getAttribute('role')).toBe('note')
    expect(within(banner).getByText(/not included/i)).toBeTruthy()

    // Shared filter knobs are present.
    expect(screen.getByTestId('okf-min-connections-slider')).toBeTruthy()
    expect(screen.getByTestId('okf-min-confidence-slider')).toBeTruthy()
    expect(screen.getByTestId('okf-min-relation-confidence-slider')).toBeTruthy()
    expect(screen.getByTestId('okf-include-orphans-switch')).toBeTruthy()
    expect(screen.getByTestId('okf-include-archived-switch')).toBeTruthy()
    expect(screen.getByTestId('okf-entity-types-input')).toBeTruthy()

    // No delivery-mode toggle — OKF is zip-only (unlike the Obsidian dialog).
    expect(screen.queryByTestId('mode-toggle')).toBeNull()
  })

  it('exposes an aria-labelled, enabled Export button (a11y gate)', () => {
    renderDialog()
    const submit = screen.getByTestId('okf-export-submit')
    expect(submit.getAttribute('aria-label')).toBe(
      'Download Open Knowledge Format bundle as zip',
    )
    expect((submit as HTMLButtonElement).disabled).toBe(false)
  })
})

describe('OkfExportDialog — submit forwards the filter (AC2)', () => {
  it('parses the comma-separated entity types before calling the hook', async () => {
    const user = userEvent.setup()
    renderDialog()

    await user.type(
      screen.getByTestId('okf-entity-types-input'),
      'Person, Organization',
    )
    await user.click(screen.getByTestId('okf-export-submit'))

    expect(mutateSpy).toHaveBeenCalledTimes(1)
    const [filterArg] = mutateSpy.mock.calls[0]
    expect(filterArg.entity_types).toEqual(['Person', 'Organization'])
    expect(filterArg.min_connections).toBe(5)
  })
})

describe('OkfExportDialog — success surfaces the omitted-field ledger (AC3)', () => {
  it('renders the itemised ledger returned by the export', async () => {
    // Drive the success branch through the mocked mutate's onSuccess callback.
    mutateSpy.mockImplementation((_filter, opts) => {
      opts?.onSuccess?.({
        kind: 'zip',
        filename: 'okf-bundle.zip',
        report: {
          entities_written: 9,
          relations_written: 4,
          files_written: 11,
          bytes_written: 2048,
          duration_ms: 17,
          filter_applied: {
            min_connections: 5,
            min_confidence: 0.9,
            include_orphans: false,
            include_archived: false,
          },
          metadata: {
            notes_written: 1,
            sources_written: 1,
            okf_version: '0.1',
            omitted_fields: OMITTED,
          },
        },
      })
    })

    const user = userEvent.setup()
    renderDialog()
    await user.click(screen.getByTestId('okf-export-submit'))

    expect(screen.getByTestId('okf-export-success')).toBeTruthy()

    const ledger = screen.getByTestId('okf-omitted-ledger')
    const rows = within(ledger).getAllByTestId('okf-omitted-row')
    expect(rows).toHaveLength(2)
    expect(within(ledger).getByText('entity.embedding')).toBeTruthy()
    expect(within(ledger).getByText('verdict_contradiction_edges')).toBeTruthy()

    // The filter form is replaced by the success state.
    expect(screen.queryByTestId('okf-export-submit')).toBeNull()
    expect(screen.getByTestId('okf-export-cancel').textContent).toContain('Done')
  })
})
