// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeAll, beforeEach } from 'vitest'
import { render, screen, cleanup, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { OkfImportDialog } from '../OkfImportDialog'
import type { OkfImportReport } from '@/lib/types/exports'

/**
 * OKF import dialog test — mirrors the export dialog coverage. The hook is
 * mocked so this stays a pure render/interaction test:
 *
 *   * default state renders the idempotent banner + file input; Import is
 *     disabled until a bundle is chosen,
 *   * choosing a file enables Import and forwards {file, applyDedup} to the hook,
 *   * a successful import surfaces the created/matched counts + the skip ledger.
 */

interface MockMutateOptions {
  onSuccess?: (report: OkfImportReport) => void
}

const mutateSpy = vi.fn<
  (vars: { file: File; applyDedup: boolean }, opts?: MockMutateOptions) => void
>()

vi.mock('@/lib/hooks/use-okf-import', () => ({
  useOkfImport: () => ({
    mutate: mutateSpy,
    isPending: false,
    error: null,
    lastReport: null,
  }),
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
beforeEach(() => mutateSpy.mockReset())

const REPORT: OkfImportReport = {
  entities_created: 4,
  entities_matched: 2,
  entities_failed: 0,
  relations_created: 3,
  notes_created: 1,
  notes_matched: 0,
  sources_created: 0,
  sources_matched: 1,
  dedup_merges: 0,
  import_source_id: 'source:okf-import',
  skipped: [{ path: 'log.md', reason: 'reserved filename' }],
  dangling_links: [],
}

function renderDialog() {
  return render(
    <OkfImportDialog open onOpenChange={() => {}} notebookId="notebook:abc" />,
  )
}

function makeZip(name = 'bundle.zip') {
  return new File([new Uint8Array([80, 75, 3, 4])], name, {
    type: 'application/zip',
  })
}

describe('OkfImportDialog', () => {
  it('renders the idempotent banner and disables Import with no file', () => {
    renderDialog()
    expect(screen.getByTestId('okf-import-idempotent-banner')).toBeTruthy()
    expect(
      (screen.getByTestId('okf-import-submit') as HTMLButtonElement).disabled,
    ).toBe(true)
  })

  it('enables Import and forwards the file + dedup flag on submit', async () => {
    const user = userEvent.setup()
    renderDialog()

    const input = screen.getByTestId('okf-import-file') as HTMLInputElement
    await user.upload(input, makeZip())

    const submit = screen.getByTestId('okf-import-submit') as HTMLButtonElement
    expect(submit.disabled).toBe(false)

    await user.click(submit)
    expect(mutateSpy).toHaveBeenCalledTimes(1)
    const [vars] = mutateSpy.mock.calls[0]
    expect(vars.file.name).toBe('bundle.zip')
    expect(vars.applyDedup).toBe(true) // default on
  })

  it('renders the report ledger on a successful import', async () => {
    const user = userEvent.setup()
    mutateSpy.mockImplementation((_vars, opts) => opts?.onSuccess?.(REPORT))
    renderDialog()

    await user.upload(
      screen.getByTestId('okf-import-file') as HTMLInputElement,
      makeZip(),
    )
    await user.click(screen.getByTestId('okf-import-submit'))

    const success = screen.getByTestId('okf-import-success')
    expect(within(success).getByText(/OKF bundle imported/i)).toBeTruthy()
    // The skip ledger surfaces the non-silently-skipped concept.
    const ledger = screen.getByTestId('okf-import-skipped-ledger')
    expect(within(ledger).getByText(/log\.md/)).toBeTruthy()
  })
})
