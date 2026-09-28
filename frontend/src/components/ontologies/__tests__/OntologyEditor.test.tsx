// @vitest-environment jsdom
/**
 * OE.3: the editor against the OE.2 API shapes (stubbed at the module
 * boundary). Each case puts the editor in the state a review found wrong.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

import type { Finding, OntologyDraft, StoredEntityType } from '@/lib/api/ontologies'

const api = vi.hoisted(() => ({
  getDraft: vi.fn(),
  validate: vi.fn(),
  versions: vi.fn(),
  putEntityType: vi.fn(),
  putRelationshipType: vi.fn(),
  preview: vi.fn(),
  publish: vi.fn(),
  get: vi.fn(),
}))

vi.mock('@/lib/api/ontologies', async (original) => ({
  ...(await original<typeof import('@/lib/api/ontologies')>()),
  ontologyEditApi: api,
  ontologiesApi: { get: api.get, list: vi.fn() },
}))
vi.mock('@/lib/hooks/use-toast', () => ({ useToast: () => ({ toast: vi.fn() }) }))

import { OntologyEditor } from '../OntologyEditor'
import { RelationshipTypesEditor } from '../RelationshipTypesEditor'
import { TypeForm } from '../TypeForm'

function draft(overrides: Partial<OntologyDraft> = {}): OntologyDraft {
  return {
    name: 'deals',
    exists: true,
    base_version: '1.0.0',
    ontology: {
      metadata: { name: 'deals', version: '1.0.0', extends: 'instruments',
        includes: ['policy_themes'] },
      entity_types: {
        Project: { name: 'Project', parent_type: 'GovernmentService', graph_relevance: 'active' },
      },
      relationship_types: {},
    },
    ...overrides,
  }
}

const BLOCK: Finding = {
  rule: 'orphan', ontology: 'deals', subject: 'Project', severity: 'block', weight: 1,
  detail: "'Project' reaches no schema.org base",
}

const ONTOLOGIES: Record<string, unknown> = {
  deals: { name: 'deals', version: '1.0.0', entity_types: [] },
  instruments: { name: 'instruments', entity_types: [{ name: 'Person', parent_type: 'Thing' }] },
  policy_themes: { name: 'policy_themes', entity_types: [{ name: 'BeleidsThema', parent_type: 'Concept' }] },
}

function mount(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return { client, ...render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>) }
}

const publishButton = () => screen.getByTestId('ontology-publish') as HTMLButtonElement

beforeEach(() => {
  api.getDraft.mockResolvedValue(draft())
  api.versions.mockResolvedValue([])
  api.validate.mockResolvedValue([])
  api.get.mockImplementation(async (name: string) => ONTOLOGIES[name])
})
afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('OntologyEditor', () => {
  it('shows a blocking finding, marks its type, and refuses to publish', async () => {
    api.validate.mockResolvedValue([BLOCK])
    mount(<OntologyEditor name="deals" />)
    expect((await screen.findByRole('alert')).textContent).toContain('orphan')
    expect((await screen.findByTestId('ontology-type-findings-Project')).textContent)
      .toContain('1 blocking')
    expect(publishButton().disabled).toBe(true)
  })

  it('allows a publish when nothing blocks', async () => {
    api.validate.mockResolvedValue([{ ...BLOCK, severity: 'warn' }])
    mount(<OntologyEditor name="deals" />)
    await screen.findByText(/1 warning/)
    await waitFor(() => expect(publishButton().disabled).toBe(false))
  })

  it('shows inherited types from extends AND includes, with their ontology (review M3)', async () => {
    mount(<OntologyEditor name="deals" />)
    const project = await screen.findByTestId('ontology-type-Project')
    expect(within(project).getByRole('button', { name: 'Move Project' })).toBeTruthy()
    const person = await screen.findByTestId('ontology-type-Person')
    expect(within(person).queryByRole('button', { name: /Move/ })).toBeNull()
    expect(person.textContent).toContain('from instruments')
    expect((await screen.findByTestId('ontology-type-BeleidsThema')).textContent)
      .toContain('from policy_themes')
  })

  it('a draft based on a version that is no longer active cannot be published (review M5)', async () => {
    api.get.mockImplementation(async (name: string) =>
      name === 'deals' ? { name: 'deals', version: '1.0.0+later', entity_types: [] } : ONTOLOGIES[name])
    mount(<OntologyEditor name="deals" />)
    expect((await screen.findByTestId('ontology-draft-stale')).textContent).toContain('1.0.0+later')
    expect(publishButton().disabled).toBe(true)
  })

  it('refuses to "add" a type that exists (review M1)', async () => {
    mount(<OntologyEditor name="deals" />)
    await screen.findByTestId('ontology-type-Project')
    fireEvent.change(screen.getByLabelText('New type name'), { target: { value: 'project' } })
    expect(screen.getByTestId('ontology-add-type-taken')).toBeTruthy()
    expect((screen.getByTestId('ontology-add-type') as HTMLButtonElement).disabled).toBe(true)
  })

  it('drops the preview when the draft changes (review M4)', async () => {
    api.preview.mockResolvedValue({
      findings: [], blocking: false, detection: [], detection_error: null, notebook_edits: 0,
      label_changes: [{ label: 'Project', before: 'programme', after: 'creative_work', entities: 3 }],
      retype: { changes: 3, collisions: 0, flag_only: 0, caused_by_this_edit: 3,
        collisions_caused_by_this_edit: 0 },
    })
    const { client } = mount(<OntologyEditor name="deals" />)
    await screen.findByTestId('ontology-type-Project')
    fireEvent.click(screen.getByTestId('ontology-preview'))
    await screen.findByTestId('ontology-preview-result')
    client.setQueryData(['ontology-editor', 'deals', 'draft'], draft({ base_version: '1.0.0' }))
    await waitFor(() => expect(screen.queryByTestId('ontology-preview-result')).toBeNull())
  })
})

describe('RelationshipTypesEditor', () => {
  it('keeps the source_types/target_types spelling and every other key (review B1)', () => {
    const onSave = vi.fn()
    render(<RelationshipTypesEditor onSave={onSave} onDelete={vi.fn()} relationships={{
      BEVAT_PROJECT: { name: 'BEVAT_PROJECT', description: 'x', skos: 'y',
        source_types: ['Deal'], target_types: ['Project'] },
    }} />)
    const row = screen.getByTestId('ontology-relationship-BEVAT_PROJECT')
    expect((within(row).getByLabelText('BEVAT_PROJECT domain') as HTMLInputElement).value).toBe('Deal')
    fireEvent.change(within(row).getByLabelText('BEVAT_PROJECT description'), { target: { value: 'z' } })
    fireEvent.click(within(row).getByRole('button', { name: 'Save' }))
    expect(onSave).toHaveBeenCalledWith({ name: 'BEVAT_PROJECT', description: 'z', skos: 'y',
      source_types: ['Deal'], target_types: ['Project'] })
  })

  it('a row follows the draft, so a save never writes stale endpoints back (review M2)', () => {
    const onSave = vi.fn()
    const { rerender } = render(<RelationshipTypesEditor onSave={onSave} onDelete={vi.fn()}
      relationships={{ R: { name: 'R', domain: ['Old'], range: [] } }} />)
    rerender(<RelationshipTypesEditor onSave={onSave} onDelete={vi.fn()}
      relationships={{ R: { name: 'R', domain: ['New'], range: [] } }} />)
    expect((screen.getByLabelText('R domain') as HTMLInputElement).value).toBe('New')
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    expect(onSave).toHaveBeenCalledWith({ name: 'R', domain: ['New'], range: [] })
  })
})

describe('TypeForm', () => {
  it('saves the whole type: unshown keys kept, emptied fields removed', () => {
    const onSave = vi.fn()
    const type: StoredEntityType = {
      name: 'Project', parent_type: 'GovernmentService', graph_relevance: 'active',
      aliases: ['Projectje'],
    }
    render(<TypeForm type={type} parentOptions={[]} onSave={onSave}
      onRename={vi.fn()} onDelete={vi.fn()} />)
    fireEvent.change(screen.getByLabelText('Aliases (comma-separated)'), { target: { value: '' } })
    fireEvent.change(screen.getByLabelText('Description'), { target: { value: 'Een project' } })
    fireEvent.click(screen.getByTestId('ontology-type-save'))
    expect(onSave).toHaveBeenCalledWith({
      name: 'Project', parent_type: 'GovernmentService', graph_relevance: 'active',
      description: 'Een project',
    })
  })

  it('an unrelated draft refresh does not wipe what is being typed (review minor 4)', () => {
    const type: StoredEntityType = { name: 'Project', description: 'a' }
    const { rerender } = render(<TypeForm type={type} parentOptions={[]} onSave={vi.fn()}
      onRename={vi.fn()} onDelete={vi.fn()} />)
    fireEvent.change(screen.getByLabelText('Description'), { target: { value: 'half typed' } })
    rerender(<TypeForm type={{ ...type }} parentOptions={[]} onSave={vi.fn()}
      onRename={vi.fn()} onDelete={vi.fn()} />)
    expect((screen.getByLabelText('Description') as HTMLTextAreaElement).value).toBe('half typed')
  })
})
