// @vitest-environment jsdom
import { useState } from 'react'
import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import {
  AdvancedIngestionSettings,
  type ParserEngineChoice,
  type OcrEngineChoice,
  type TableModeChoice,
} from '../AdvancedIngestionSettings'

// Radix Select relies on pointer-capture + scrollIntoView, neither of which
// jsdom implements. Polyfill them so the listbox opens under test.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false) as unknown as typeof Element.prototype.hasPointerCapture
  Element.prototype.setPointerCapture = vi.fn() as unknown as typeof Element.prototype.setPointerCapture
  Element.prototype.releasePointerCapture = vi.fn() as unknown as typeof Element.prototype.releasePointerCapture
})

beforeEach(() => vi.clearAllMocks())
afterEach(cleanup)

/**
 * The component is now CONTROLLED (Track UX.5 fix): the parent owns the field +
 * open state so a selection survives back-navigation. This harness plays the
 * parent, tracking state so the disclosure toggles and selections stick — and
 * exposes the derived overrides so the "all-Auto ⇒ {}" / "only changed fields"
 * contract stays covered at the point where the real component's callbacks fire.
 */
function ControlledHarness({
  onOverrides,
}: {
  onOverrides?: (o: Record<string, unknown>) => void
}) {
  const [open, setOpen] = useState(false)
  const [parserEngine, setParserEngine] = useState<ParserEngineChoice>('auto')
  const [ocrEngine, setOcrEngine] = useState<OcrEngineChoice>('auto')
  const [tableMode, setTableMode] = useState<TableModeChoice>('auto')

  const overrides: Record<string, unknown> = {}
  if (parserEngine !== 'auto') overrides.parser_engine = parserEngine
  if (ocrEngine !== 'auto') overrides.docling_ocr_engine = ocrEngine
  if (tableMode !== 'auto') overrides.docling_table_mode = tableMode
  onOverrides?.(overrides)

  return (
    <AdvancedIngestionSettings
      open={open}
      onOpenChange={setOpen}
      parserEngine={parserEngine}
      ocrEngine={ocrEngine}
      tableMode={tableMode}
      onParserEngineChange={setParserEngine}
      onOcrEngineChange={setOcrEngine}
      onTableModeChange={setTableMode}
    />
  )
}

function setup() {
  const onOverrides = vi.fn()
  const utils = render(<ControlledHarness onOverrides={onOverrides} />)
  const user = userEvent.setup({ pointerEventsCheck: 0 })
  return { onOverrides, user, ...utils }
}

const trigger = () => screen.getByRole('button', { name: 'Advanced ingestion settings' })

describe('AC4 — collapsed by default + a11y', () => {
  it('is collapsed by default (aria-expanded=false, no fields visible)', () => {
    setup()
    expect(trigger().getAttribute('aria-expanded')).toBe('false')
    expect(screen.queryByLabelText('Parser engine')).toBeNull()
  })

  it('is keyboard-reachable: Enter and Space toggle the disclosure', async () => {
    const { user } = setup()
    const t = trigger()
    t.focus()
    expect(document.activeElement).toBe(t)

    await user.keyboard('{Enter}')
    await waitFor(() => expect(trigger().getAttribute('aria-expanded')).toBe('true'))

    await user.keyboard(' ')
    await waitFor(() => expect(trigger().getAttribute('aria-expanded')).toBe('false'))
  })
})

describe('AC4 — Auto defaults + overrides only when changed', () => {
  it('derives {} while all fields are Auto (no overrides)', () => {
    const { onOverrides } = setup()
    expect(onOverrides).toHaveBeenLastCalledWith({})
  })

  it('every field defaults to Auto once expanded', async () => {
    const { user } = setup()
    await user.click(trigger())
    // Three Select triggers, all showing "Auto".
    const autos = await screen.findAllByText('Auto')
    expect(autos.length).toBeGreaterThanOrEqual(3)
  })

  it('contributes a processing_overrides key only for a field moved off Auto', async () => {
    const { onOverrides, user } = setup()
    await user.click(trigger())

    await user.click(screen.getByLabelText('Parser engine'))
    await user.click(await screen.findByRole('option', { name: 'Docling' }))

    await waitFor(() =>
      expect(onOverrides).toHaveBeenLastCalledWith({ parser_engine: 'docling' })
    )
  })

  it('combines multiple changed fields and omits the ones left on Auto', async () => {
    const { onOverrides, user } = setup()
    await user.click(trigger())

    await user.click(screen.getByLabelText('Parser engine'))
    await user.click(await screen.findByRole('option', { name: 'Simple' }))

    await user.click(screen.getByLabelText('Table mode'))
    await user.click(await screen.findByRole('option', { name: 'Fast' }))

    await waitFor(() =>
      expect(onOverrides).toHaveBeenLastCalledWith({
        parser_engine: 'simple',
        docling_table_mode: 'fast',
      })
    )
    // OCR engine was never touched ⇒ no docling_ocr_engine key.
    const last = onOverrides.mock.calls.at(-1)?.[0]
    expect(last).not.toHaveProperty('docling_ocr_engine')
  })
})

describe('controlled selection persists across a remount (back-navigation)', () => {
  it('keeps the parent-owned Docling value when the child unmounts and remounts', async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 })
    // A parent that can hide the child (mimicking the conditional Input step).
    function Parent() {
      const [mounted, setMounted] = useState(true)
      const [open, setOpen] = useState(false)
      const [parserEngine, setParserEngine] = useState<ParserEngineChoice>('auto')
      return (
        <div>
          <button type="button" onClick={() => setMounted((m) => !m)}>
            toggle-mount
          </button>
          {mounted && (
            <AdvancedIngestionSettings
              open={open}
              onOpenChange={setOpen}
              parserEngine={parserEngine}
              ocrEngine="auto"
              tableMode="auto"
              onParserEngineChange={setParserEngine}
              onOcrEngineChange={() => {}}
              onTableModeChange={() => {}}
            />
          )}
        </div>
      )
    }
    render(<Parent />)

    await user.click(screen.getByRole('button', { name: 'Advanced ingestion settings' }))
    await user.click(screen.getByLabelText('Parser engine'))
    await user.click(await screen.findByRole('option', { name: 'Docling' }))

    // Unmount then remount the child (Input → Organize → Back).
    await user.click(screen.getByRole('button', { name: 'toggle-mount' }))
    await user.click(screen.getByRole('button', { name: 'toggle-mount' }))

    // The parent owns both the open and the field state, so the disclosure is
    // still expanded AND the selection survived (no re-click needed).
    await waitFor(() =>
      expect(screen.getByLabelText('Parser engine').textContent).toContain('Docling')
    )
  })
})
