/**
 * Unit tests for the layered privacy-mode logic (Track J.5).
 *
 * As with the chain editor, the project's node-environment vitest can't render
 * the Radix radio group, so we test the pure wire-mapping + inherited-value
 * resolution that `PrivacyModeToggle` renders — the behavior behind AC4 (shows
 * the inherited value; correct cloud/private/inherit handling per scope).
 * Keyboard accessibility (Radix RadioGroup focus/arrow/Space) is covered by the
 * Playwright E2E suite.
 */

import { describe, expect, it } from 'vitest'

import {
  canInherit,
  fromWire,
  resolveInherited,
  toWire,
} from '@/lib/utils/privacy-mode'

describe('toWire / fromWire', () => {
  it('maps inherit <-> null and passes through concrete values', () => {
    expect(toWire('inherit')).toBeNull()
    expect(toWire('cloud')).toBe('cloud')
    expect(toWire('private')).toBe('private')
    expect(fromWire(null)).toBe('inherit')
    expect(fromWire(undefined)).toBe('inherit')
    expect(fromWire('cloud')).toBe('cloud')
    expect(fromWire('private')).toBe('private')
  })
})

describe('canInherit', () => {
  it('global scope cannot inherit; notebook/document can', () => {
    expect(canInherit('global')).toBe(false)
    expect(canInherit('notebook')).toBe(true)
    expect(canInherit('document')).toBe(true)
  })
})

describe('resolveInherited (the "inherits: <resolved>" hint)', () => {
  it('notebook inherits the global default', () => {
    expect(
      resolveInherited('notebook', { globalDefault: 'private' }),
    ).toBe('private')
  })

  it('document inherits the notebook value when the notebook sets one', () => {
    expect(
      resolveInherited('document', {
        globalDefault: 'cloud',
        notebookValue: 'private',
      }),
    ).toBe('private')
  })

  it('document falls back to the global default when the notebook inherits', () => {
    expect(
      resolveInherited('document', {
        globalDefault: 'cloud',
        notebookValue: null,
      }),
    ).toBe('cloud')
  })
})
