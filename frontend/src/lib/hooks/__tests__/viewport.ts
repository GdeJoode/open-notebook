/**
 * A `window.matchMedia` for jsdom that answers `(min-width: Npx)` queries for
 * a set viewport width, as a browser does, and tells listeners when it
 * changes. Only the queries `useMinWidth` asks are understood.
 */
type Listener = () => void

const listeners = new Map<string, Set<Listener>>()
let width = 1440

function matches(query: string): boolean {
  const m = /^\(min-width: (\d+)px\)$/.exec(query)
  if (!m) throw new Error(`viewport stub: unsupported query ${query}`)
  return width >= Number(m[1])
}

export function installViewport(initial: number): void {
  width = initial
  listeners.clear()
  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    writable: true,
    value: (query: string) => ({
      get matches() {
        return matches(query)
      },
      media: query,
      addEventListener: (_: string, fn: Listener) => {
        if (!listeners.has(query)) listeners.set(query, new Set())
        listeners.get(query)!.add(fn)
      },
      removeEventListener: (_: string, fn: Listener) => listeners.get(query)?.delete(fn),
    }),
  })
}

/** Resize: every listener whose query flips is called. */
export function setViewport(next: number): void {
  const before = new Map([...listeners.keys()].map((q) => [q, matches(q)]))
  width = next
  for (const [q, was] of before) {
    if (matches(q) !== was) listeners.get(q)?.forEach((fn) => fn())
  }
}

export function removeViewport(): void {
  listeners.clear()
  // @ts-expect-error jsdom has no matchMedia of its own
  delete window.matchMedia
}
