// @vitest-environment jsdom
/**
 * OE.5: the Schema tab no longer edits a per-notebook layer. Every type links
 * to the shared ontology's editor, and no rename / merge / split / delete
 * control is left.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

vi.mock('@/lib/hooks/use-toast', () => ({ useToast: () => ({ toast: vi.fn() }) }))

import type { NotebookSchemaResponse } from '@/lib/types/notebook_schema'
import { SchemaBrowser, ontologyEditorHref } from '../SchemaBrowser'

function schema(overrides: Partial<NotebookSchemaResponse> = {}): NotebookSchemaResponse {
  return {
    notebook_id: 'notebook:x',
    base_ontology: 'scholarly',
    base_ontology_types: [
      { name: 'Thesis', parent_type: 'CreativeWork', description: 'A thesis', properties: [] },
      { name: 'Researcher', parent_type: 'Person', properties: [] },
    ],
    pending_extensions: [],
    coverage_pct: 0,
    review_required: false,
    soft_nudge_dismissed: false,
    ...overrides,
  }
}

function mount(s: NotebookSchemaResponse) {
  const client = new QueryClient()
  render(<QueryClientProvider client={client}><SchemaBrowser schema={s} /></QueryClientProvider>)
}

afterEach(() => cleanup())

describe('SchemaBrowser', () => {
  it('links every type to its ontology in the editor', () => {
    mount(schema())
    const link = screen.getByTestId('schema-row-edit-Thesis')
    expect(link.getAttribute('href')).toBe('/ontologies?name=scholarly&tab=edit&type=Thesis')
    expect(link.getAttribute('aria-label')).toBe('Edit Thesis in the ontology editor')
    expect(screen.getByTestId('schema-open-editor').getAttribute('href'))
      .toBe('/ontologies?name=scholarly&tab=edit')
  })

  it('offers no per-notebook edit', () => {
    mount(schema())
    for (const op of ['rename', 'merge', 'split', 'delete']) {
      expect(screen.queryByTestId(`schema-row-${op}-Thesis`)).toBeNull()
    }
    expect(screen.queryByTestId('schema-row-actions-Thesis')).toBeNull()
    expect(screen.queryByText('Extension')).toBeNull()
  })

  it('a notebook without a declared base points to the editor', () => {
    mount(schema({ base_ontology: '', base_ontology_types: [] }))
    expect(screen.getByRole('status').textContent).toContain('declares no base ontology')
    expect(screen.getByTestId('schema-open-editor').getAttribute('href')).toBe('/ontologies')
  })

  it('encodes names in the link', () => {
    expect(ontologyEditorHref('deals', 'A & B')).toBe('/ontologies?name=deals&tab=edit&type=A+%26+B')
  })
})
