/**
 * Minimal localStorage polyfill for the node test environment.
 *
 * The frontend's Zustand stores use the `persist` middleware, which reaches for
 * `globalThis.localStorage` at module-eval time. Rather than pull in jsdom just
 * for a key/value map, we install an in-memory shim. Component/DOM behavior is
 * covered by the Playwright E2E suite, not here.
 */
class MemoryStorage implements Storage {
  private store = new Map<string, string>()

  get length(): number {
    return this.store.size
  }

  clear(): void {
    this.store.clear()
  }

  getItem(key: string): string | null {
    return this.store.has(key) ? (this.store.get(key) as string) : null
  }

  key(index: number): string | null {
    return Array.from(this.store.keys())[index] ?? null
  }

  removeItem(key: string): void {
    this.store.delete(key)
  }

  setItem(key: string, value: string): void {
    this.store.set(key, String(value))
  }
}

if (typeof globalThis.localStorage === 'undefined') {
  Object.defineProperty(globalThis, 'localStorage', {
    value: new MemoryStorage(),
    writable: true,
  })
}
