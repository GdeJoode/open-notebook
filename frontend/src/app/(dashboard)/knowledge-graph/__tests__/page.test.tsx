// @vitest-environment jsdom
/**
 * WB.3: the Knowledge Graph table opens on Active entities, highest degree
 * first (the server sorts: `sort=degree`), and shows canonical types with the
 * display label second.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import type { ReactNode } from 'react'

const hooks = vi.hoisted(() => ({ useEntities: vi.fn(), getEntity: vi.fn() }))
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
vi.mock('@/lib/api/knowledge-graph', () => ({
  knowledgeGraphApi: { getEntity: hooks.getEntity },
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

import { installViewport, removeViewport, setViewport } from '@/lib/hooks/__tests__/viewport'
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

describe('Knowledge Graph by width (WB.5)', () => {
  afterEach(() => removeViewport())
  const DETAIL = { ...ROWS[0], relations: [], source_ids: ['source:s1'] }

  it('below md: a card per entity that opens its details in a sheet', async () => {
    installViewport(390)
    hooks.useEntities.mockReturnValue({ data: { items: ROWS, total: 2 }, isLoading: false })
    hooks.getEntity.mockResolvedValue(DETAIL)
    render(<KnowledgeGraphPage />)
    expect(screen.queryByRole('table')).toBeNull()
    const card = screen.getByTestId('entity-card-entity:a')
    expect(card.tagName).toBe('BUTTON')
    expect(card.querySelector('[data-testid="entity-type-canonical"]')?.textContent)
      .toBe('administrative_area')
    expect(card.textContent).toContain('Degree 9')
    expect(card.textContent).toContain('Docs 3')
    // The sorts are there too.
    fireEvent.click(screen.getByTestId('kg-sort-name'))
    expect(hooks.useEntities.mock.calls.at(-1)?.[0]).toMatchObject({ sort: 'name' })

    fireEvent.click(card)
    const sheet = await screen.findByRole('dialog')
    expect(within(sheet).getByText('Regio Zwolle')).toBeTruthy()
    expect(within(sheet).getByTestId('triage-panel')).toBeTruthy()
    expect(screen.queryByTestId('entity-detail-panel')).toBeNull()
    fireEvent.click(within(sheet).getByRole('button', { name: 'Close' }))
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('from md the table; below lg the details are still a sheet', async () => {
    installViewport(768)
    hooks.useEntities.mockReturnValue({ data: { items: ROWS, total: 2 }, isLoading: false })
    hooks.getEntity.mockResolvedValue(DETAIL)
    render(<KnowledgeGraphPage />)
    fireEvent.click(screen.getByTestId('entity-row-entity:a'))
    expect(await screen.findByRole('dialog')).toBeTruthy()
  })

  it('from lg the details are a side panel, not a modal', async () => {
    installViewport(1024)
    hooks.useEntities.mockReturnValue({ data: { items: ROWS, total: 2 }, isLoading: false })
    hooks.getEntity.mockResolvedValue(DETAIL)
    render(<KnowledgeGraphPage />)
    fireEvent.click(screen.getByTestId('entity-row-entity:a'))
    const panel = await screen.findByTestId('entity-detail-panel')
    expect(within(panel).getByText('Regio Zwolle')).toBeTruthy()
    expect(screen.queryByRole('dialog')).toBeNull()
    fireEvent.click(within(panel).getByRole('button', { name: 'Close details' }))
    expect(screen.queryByTestId('entity-detail-panel')).toBeNull()
  })
})

describe('Knowledge Graph details: closing wins (WB.5 review m2)', () => {
  afterEach(() => removeViewport())

  it('a sheet closed while it loads stays closed when the answer arrives', async () => {
    installViewport(768)
    hooks.useEntities.mockReturnValue({ data: { items: ROWS, total: 2 }, isLoading: false })
    let answer!: (d: unknown) => void
    hooks.getEntity.mockReturnValue(new Promise((r) => { answer = r }))
    render(<KnowledgeGraphPage />)
    fireEvent.click(screen.getByTestId('entity-row-entity:a'))
    const sheet = await screen.findByRole('dialog')
    expect(within(sheet).getByText('Loading entity')).toBeTruthy()
    fireEvent.click(within(sheet).getByRole('button', { name: 'Close' }))
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    await act(async () => { answer({ ...ROWS[0], relations: [] }) })
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('a later click wins over an earlier one still loading', async () => {
    installViewport(1024)
    hooks.useEntities.mockReturnValue({ data: { items: ROWS, total: 2 }, isLoading: false })
    let first!: (d: unknown) => void
    hooks.getEntity
      .mockReturnValueOnce(new Promise((r) => { first = r }))
      .mockResolvedValueOnce({ ...ROWS[1], relations: [] })
    render(<KnowledgeGraphPage />)
    fireEvent.click(screen.getByTestId('entity-row-entity:a'))
    fireEvent.click(screen.getByTestId('entity-row-entity:b'))
    const panel = await screen.findByTestId('entity-detail-panel')
    await waitFor(() => expect(within(panel).getByText('Rijk')).toBeTruthy())
    await act(async () => { first({ ...ROWS[0], relations: [] }) })
    expect(within(screen.getByTestId('entity-detail-panel')).queryByText('Regio Zwolle')).toBeNull()
  })
})

describe('Knowledge Graph details across lg (WB.5 review m5)', () => {
  afterEach(() => removeViewport())

  it('an entity picked in the side panel does not open a sheet on a narrower window', async () => {
    installViewport(1024)
    hooks.useEntities.mockReturnValue({ data: { items: ROWS, total: 2 }, isLoading: false })
    hooks.getEntity.mockResolvedValue({ ...ROWS[0], relations: [] })
    render(<KnowledgeGraphPage />)
    fireEvent.click(screen.getByTestId('entity-row-entity:a'))
    await screen.findByTestId('entity-detail-panel')
    act(() => setViewport(768))
    expect(screen.queryByRole('dialog')).toBeNull()
    // A selection made below lg does open it.
    fireEvent.click(screen.getByTestId('entity-row-entity:a'))
    expect(await screen.findByRole('dialog')).toBeTruthy()
  })

  it('a read in flight when lg is crossed does not open the sheet', async () => {
    installViewport(1024)
    hooks.useEntities.mockReturnValue({ data: { items: ROWS, total: 2 }, isLoading: false })
    let answer!: (d: unknown) => void
    hooks.getEntity.mockReturnValue(new Promise((r) => { answer = r }))
    render(<KnowledgeGraphPage />)
    fireEvent.click(screen.getByTestId('entity-row-entity:a'))
    act(() => setViewport(768))
    await act(async () => { answer({ ...ROWS[0], relations: [] }) })
    expect(screen.queryByRole('dialog')).toBeNull()
  })
})

describe('Knowledge Graph relations (WB.5 review m7)', () => {
  afterEach(() => removeViewport())

  it('the other end of a relation is a button that opens it', async () => {
    installViewport(1024)
    hooks.useEntities.mockReturnValue({ data: { items: ROWS, total: 2 }, isLoading: false })
    hooks.getEntity.mockResolvedValueOnce({ ...ROWS[0], relations: [
      { id: 'relation:1', source: 'entity:a', target: 'entity:b', relation_type: 'part_of' },
      { id: 'relation:2', source: 'entity:c', target: 'entity:a', relation_type: 'funds' },
    ] }).mockResolvedValueOnce({ ...ROWS[1], relations: [] })
    render(<KnowledgeGraphPage />)
    fireEvent.click(screen.getByTestId('entity-row-entity:a'))
    const panel = await screen.findByTestId('entity-detail-panel')
    expect(within(panel).getByRole('button', { name: 'entity:c' })).toBeTruthy()
    fireEvent.click(within(panel).getByRole('button', { name: 'entity:b' }))
    expect(hooks.getEntity).toHaveBeenLastCalledWith('entity:b')
  })
})
