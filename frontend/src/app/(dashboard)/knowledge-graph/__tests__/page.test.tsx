// @vitest-environment jsdom
/**
 * WB.3: the Knowledge Graph table opens on Active entities, highest degree
 * first (the server sorts: `sort=degree`), and shows canonical types with the
 * display label second.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'

const hooks = vi.hoisted(() => ({ useEntities: vi.fn() }))
vi.mock('next/link', () => ({
  default: ({ children, href, ...rest }: { children: ReactNode; href: string }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}))
vi.mock('next/dynamic', () => ({ default: () => () => null }))
vi.mock('@/lib/hooks/use-knowledge-graph', () => ({
  useEntities: hooks.useEntities,
  useEntityTypes: () => ({ data: [] }),
  useSearchEntities: () => ({ data: undefined }),
}))
vi.mock('@/lib/hooks/use-triage', () => ({
  useSetEntityOverride: () => ({ mutate: vi.fn(), isPending: false }),
}))

// The confidence slider measures itself.
globalThis.ResizeObserver ??= class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as unknown as typeof ResizeObserver

import KnowledgeGraphPage from '../page'

const ROWS = [
  { id: 'entity:a', name: 'Regio Zwolle', entity_type: 'administrative_area', primary_type: 'Gemeente',
    weight: 0, status: 'active', structural_degree: 9, doc_count: 3 },
  { id: 'entity:b', name: 'Rijk', entity_type: 'organization', primary_type: 'Organization',
    weight: 0, status: 'active', structural_degree: 4, doc_count: 2 },
]

afterEach(() => {
  cleanup()
  hooks.useEntities.mockReset()
})

describe('Knowledge Graph table defaults', () => {
  it('asks for active entities sorted by degree, and can sort by name', () => {
    hooks.useEntities.mockReturnValue({ data: { items: ROWS, total: 2 }, isLoading: false })
    render(<KnowledgeGraphPage />)
    expect(hooks.useEntities.mock.calls[0][0]).toMatchObject({ status: 'active', sort: 'degree' })
    expect(screen.getByTestId('status-filter').textContent).toContain('Active')
    expect(screen.getByTestId('kg-sort-degree').closest('th')?.getAttribute('aria-sort'))
      .toBe('descending')

    fireEvent.click(screen.getByTestId('kg-sort-name'))
    expect(hooks.useEntities.mock.calls.at(-1)?.[0]).toMatchObject({ status: 'active', sort: 'name' })
  })

  it('shows canonical types first, the display label second', () => {
    hooks.useEntities.mockReturnValue({ data: { items: ROWS, total: 2 }, isLoading: false })
    render(<KnowledgeGraphPage />)
    const a = screen.getByTestId('entity-row-entity:a')
    expect(a.querySelector('[data-testid="entity-type-canonical"]')?.textContent)
      .toBe('administrative_area')
    expect(a.querySelector('[data-testid="entity-type-label"]')?.textContent).toBe('Gemeente')
    // A label that only restates the canonical type is not repeated.
    const b = screen.getByTestId('entity-row-entity:b')
    expect(b.querySelector('[data-testid="entity-type-label"]')).toBeNull()
  })
})
