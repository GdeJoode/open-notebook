/**
 * Pure helpers for the provider-chain editor (Track J.5).
 *
 * The reorder / enable-toggle / effective-chain logic lives here as framework-
 * free functions so it can be unit-tested under the project's `node`-environment
 * vitest setup (which deliberately scopes unit tests to pure logic and pushes
 * DOM/component behavior to the Playwright E2E suite). `ProviderChainEditor`
 * imports these and wires them to accessible up/down buttons.
 */

import { ProviderChainEntry } from '@/lib/types/model-routing'

/** Move the entry at `index` one slot earlier; no-op at the head. */
export function moveUp(
  chain: ProviderChainEntry[],
  index: number,
): ProviderChainEntry[] {
  if (index <= 0 || index >= chain.length) return chain
  const next = [...chain]
  ;[next[index - 1], next[index]] = [next[index], next[index - 1]]
  return next
}

/** Move the entry at `index` one slot later; no-op at the tail. */
export function moveDown(
  chain: ProviderChainEntry[],
  index: number,
): ProviderChainEntry[] {
  if (index < 0 || index >= chain.length - 1) return chain
  const next = [...chain]
  ;[next[index], next[index + 1]] = [next[index + 1], next[index]]
  return next
}

/** Flip the `enabled` flag on the entry at `index`. */
export function toggleEnabled(
  chain: ProviderChainEntry[],
  index: number,
): ProviderChainEntry[] {
  if (index < 0 || index >= chain.length) return chain
  return chain.map((entry, i) =>
    i === index ? { ...entry, enabled: !entry.enabled } : entry,
  )
}

/**
 * The chain as it would actually be routed: only enabled entries, in order.
 * AC3 — a disabled provider is excluded from the effective chain shown.
 */
export function effectiveChain(
  chain: ProviderChainEntry[],
): ProviderChainEntry[] {
  return chain.filter((entry) => entry.enabled)
}

/** True when `index` is the first row (its up button must be disabled). */
export function isFirst(index: number): boolean {
  return index === 0
}

/** True when `index` is the last row (its down button must be disabled). */
export function isLast(index: number, length: number): boolean {
  return index === length - 1
}
