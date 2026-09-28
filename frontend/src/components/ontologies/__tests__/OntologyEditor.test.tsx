// @vitest-environment jsdom
/**
 * OE.3: the editor renders the draft, the gate's findings and the type tree,
 * and refuses a publish while a finding blocks. The API is stubbed at the
 * module boundary with the OE.2 response shapes.
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
import { TypeForm } from '../TypeForm'

const DRAFT: OntologyDraft = {
  name: 'deals',
  exists: true,
  base_version: '1.0.0',
  ontology: {
    metadata: { name: 'deals', version: '1.0.0', extends: 'instruments' },
    entity_types: {
      Project: { name: 'Project', parent_type: 'GovernmentService', graph_relevance: 'active' },
    },
  },
}

const BLOCK: Finding = {
  rule: 'orphan', ontology: 'deals', subject: 'Project', severity: 'block', weight: 1,
  detail: "'Project' reaches no schema.org base",
}

function mount(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

beforeEach(() => {
  api.getDraft.mockResolvedValue(DRAFT)
  api.versions.mockResolvedValue([])
  api.get.mockResolvedValue({
    name: 'instruments', entity_types: [{ name: 'Person', parent_type: 'Thing' }],
  })
})
afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('OntologyEditor', () => {
  it('shows a blocking finding and refuses to publish', async () => {
    api.validate.mockResolvedValue([BLOCK])
    mount(<OntologyEditor name="deals" />)
    expect((await screen.findByRole('alert')).textContent).toContain('orphan')
    expect((screen.getByTestId('ontology-publish') as HTMLButtonElement).disabled).toBe(true)
  })

  it('allows a publish when nothing blocks', async () => {
    api.validate.mockResolvedValue([{ ...BLOCK, severity: 'warn' }])
    mount(<OntologyEditor name="deals" />)
    await screen.findByText(/1 warning/)
    await waitFor(() =>
      expect((screen.getByTestId('ontology-publish') as HTMLButtonElement).disabled).toBe(false))
  })

  it('draws own types draggable and inherited ones as drop targets only', async () => {
    api.validate.mockResolvedValue([])
    mount(<OntologyEditor name="deals" />)
    const project = await screen.findByTestId('ontology-type-Project')
    expect(within(project).getByRole('button', { name: 'Move Project' })).toBeTruthy()
    const person = await screen.findByTestId('ontology-type-Person')
    expect(within(person).queryByRole('button', { name: /Move/ })).toBeNull()
    expect(person.textContent).toContain('inherited')
    expect(screen.getByTestId('ontology-type-GovernmentService').textContent).toContain('base')
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
})
