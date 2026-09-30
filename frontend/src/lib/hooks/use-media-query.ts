import { useSyncExternalStore } from 'react'

/** Tailwind's default breakpoints (the theme does not change them). */
export const BREAKPOINTS = { md: 768, lg: 1024, xl: 1280 } as const

export type Breakpoint = keyof typeof BREAKPOINTS

const query = (bp: Breakpoint) => `(min-width: ${BREAKPOINTS[bp]}px)`

/**
 * True while the viewport is at least `bp` wide: the same test as Tailwind's
 * `bp:` prefix. For layouts that differ in structure, not only in classes
 * (a table or cards, a column or a sheet), so only one of them is mounted.
 * Without `matchMedia` (server render, jsdom) it answers true: desktop.
 */
export function useMinWidth(bp: Breakpoint): boolean {
  return useSyncExternalStore(
    (onChange) => {
      if (typeof window === 'undefined' || !window.matchMedia) return () => undefined
      const mql = window.matchMedia(query(bp))
      mql.addEventListener('change', onChange)
      return () => mql.removeEventListener('change', onChange)
    },
    () => (typeof window === 'undefined' || !window.matchMedia ? true : window.matchMedia(query(bp)).matches),
    () => true,
  )
}
