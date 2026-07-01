// @vitest-environment jsdom
import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { AdvancedIngestionSettings } from '../AdvancedIngestionSettings'

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

function setup() {
  const onOverridesChange = vi.fn()
  const utils = render(<AdvancedIngestionSettings onOverridesChange={onOverridesChange} />)
  const user = userEvent.setup({ pointerEventsCheck: 0 })
  return { onOverridesChange, user, ...utils }
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
  it('emits {} on mount (all Auto ⇒ no overrides)', () => {
    const { onOverridesChange } = setup()
    expect(onOverridesChange).toHaveBeenCalledWith({})
  })

  it('every field defaults to Auto once expanded', async () => {
    const { user } = setup()
    await user.click(trigger())
    // Three Select triggers, all showing "Auto".
    const autos = await screen.findAllByText('Auto')
    expect(autos.length).toBeGreaterThanOrEqual(3)
  })

  it('contributes a processing_overrides key only for a field moved off Auto', async () => {
    const { onOverridesChange, user } = setup()
    await user.click(trigger())

    await user.click(screen.getByLabelText('Parser engine'))
    await user.click(await screen.findByRole('option', { name: 'Docling' }))

    await waitFor(() =>
      expect(onOverridesChange).toHaveBeenLastCalledWith({ parser_engine: 'docling' })
    )
  })

  it('combines multiple changed fields and omits the ones left on Auto', async () => {
    const { onOverridesChange, user } = setup()
    await user.click(trigger())

    await user.click(screen.getByLabelText('Parser engine'))
    await user.click(await screen.findByRole('option', { name: 'Simple' }))

    await user.click(screen.getByLabelText('Table mode'))
    await user.click(await screen.findByRole('option', { name: 'Fast' }))

    await waitFor(() =>
      expect(onOverridesChange).toHaveBeenLastCalledWith({
        parser_engine: 'simple',
        docling_table_mode: 'fast',
      })
    )
    // OCR engine was never touched ⇒ no docling_ocr_engine key.
    const last = onOverridesChange.mock.calls.at(-1)?.[0]
    expect(last).not.toHaveProperty('docling_ocr_engine')
  })
})
