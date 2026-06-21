/**
 * Unit tests for the provider-chain reorder/enable logic (Track J.5).
 *
 * The project's vitest runs in a `node` environment with no jsdom/testing-
 * library (DOM/component behavior is covered by the Playwright E2E suite, and
 * adding a DOM test runner is out of scope for this track). So we test the pure
 * reorder/effective-chain helpers that `ProviderChainEditor` wires to its
 * up/down buttons — the load-bearing behavior behind AC2 (reorder callback) and
 * AC3 (disabled excluded from the effective chain).
 */

import { describe, expect, it } from 'vitest'

import { ProviderChainEntry } from '@/lib/types/model-routing'
import {
  effectiveChain,
  isFirst,
  isLast,
  moveDown,
  moveUp,
  toggleEnabled,
} from '@/lib/utils/provider-chain'

function chain(): ProviderChainEntry[] {
  return [
    { provider: 'nvidia', model_id: null, enabled: true },
    { provider: 'openai', model_id: null, enabled: true },
    { provider: 'ollama', model_id: null, enabled: true },
  ]
}

describe('moveUp', () => {
  it('swaps an entry with the one above it (reorder callback)', () => {
    const next = moveUp(chain(), 1)
    expect(next.map((e) => e.provider)).toEqual(['openai', 'nvidia', 'ollama'])
  })

  it('is a no-op at the head', () => {
    const original = chain()
    expect(moveUp(original, 0)).toBe(original)
  })

  it('does not mutate the input array', () => {
    const original = chain()
    moveUp(original, 1)
    expect(original.map((e) => e.provider)).toEqual([
      'nvidia',
      'openai',
      'ollama',
    ])
  })
})

describe('moveDown', () => {
  it('swaps an entry with the one below it', () => {
    const next = moveDown(chain(), 0)
    expect(next.map((e) => e.provider)).toEqual(['openai', 'nvidia', 'ollama'])
  })

  it('is a no-op at the tail', () => {
    const original = chain()
    expect(moveDown(original, original.length - 1)).toBe(original)
  })
})

describe('toggleEnabled', () => {
  it('flips the enabled flag at the index', () => {
    const next = toggleEnabled(chain(), 1)
    expect(next[1].enabled).toBe(false)
    expect(next[0].enabled).toBe(true)
  })
})

describe('effectiveChain', () => {
  it('excludes disabled providers (AC3)', () => {
    const withDisabled = toggleEnabled(chain(), 1)
    expect(effectiveChain(withDisabled).map((e) => e.provider)).toEqual([
      'nvidia',
      'ollama',
    ])
  })
})

describe('end-position helpers (button disabled state)', () => {
  it('isFirst is true only at index 0', () => {
    expect(isFirst(0)).toBe(true)
    expect(isFirst(1)).toBe(false)
  })

  it('isLast is true only at the final index', () => {
    expect(isLast(2, 3)).toBe(true)
    expect(isLast(1, 3)).toBe(false)
  })
})
