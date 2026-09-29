// @vitest-environment jsdom
/**
 * OE.5 review m7: the deep link into the ontology editor. An unknown `tab`
 * falls back to the default tab; a link followed while the page is open
 * selects its ontology, tab and type.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'

const nav = vi.hoisted(() => ({ params: new URLSearchParams() }))
vi.mock('next/navigation', () => ({ useSearchParams: () => nav.params }))
vi.mock('@/components/layout/AppShell', () => ({
  AppShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))
vi.mock('@/lib/hooks/use-ontologies', () => ({
  useOntologies: () => ({ data: [], isLoading: false, error: null }),
  useOntology: (name: string) => ({
    data: { name, version: '1.0.0', description: null, entity_type_count: 0,
      relationship_type_count: 0, concept_count: 0, authors: null, extends: null,
      entity_types: [], relationship_types: [], concepts: [] },
    isLoading: false, error: null,
  }),
}))
vi.mock('@/components/ontologies/OntologyEditor', () => ({
  OntologyEditor: ({ name, initialType, initialRetypeOpen }: {
    name: string; initialType?: string; initialRetypeOpen?: boolean
  }) => <div data-testid="editor">{`${name}|${initialType ?? ''}|${initialRetypeOpen}`}</div>,
}))

import OntologiesPage from '../page'

afterEach(() => cleanup())

const selectedTab = () =>
  screen.getAllByRole('tab').find((t) => t.getAttribute('aria-selected') === 'true')

describe('/ontologies deep link', () => {
  it('opens the named ontology on the Edit tab with the type and the retype', () => {
    nav.params = new URLSearchParams('name=deals&tab=edit&type=Project&retype=1')
    render(<OntologiesPage />)
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('deals')
    expect(selectedTab()?.textContent).toBe('Edit')
    expect(screen.getByTestId('editor').textContent).toBe('deals|Project|true')
  })

  it('an unknown tab falls back to the default tab', () => {
    nav.params = new URLSearchParams('name=deals&tab=bogus')
    render(<OntologiesPage />)
    expect(selectedTab()?.textContent).toContain('Entity Types')
  })

  it('a link followed while the page is open selects its ontology', () => {
    nav.params = new URLSearchParams('name=deals&tab=edit&type=Project')
    const { rerender } = render(<OntologiesPage />)
    nav.params = new URLSearchParams('name=government&tab=edit&type=Gemeente')
    rerender(<OntologiesPage />)
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('government')
    expect(screen.getByTestId('editor').textContent).toBe('government|Gemeente|false')
  })
})
