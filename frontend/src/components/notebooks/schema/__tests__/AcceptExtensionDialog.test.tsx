// @vitest-environment jsdom
/**
 * OE.5: accepting a Pass-1 proposal publishes it to a SHARED ontology through
 * the editor's gate. The API is stubbed at the module boundary with the
 * backend's shapes: the target (default ontology), the preview (findings,
 * `blocking`, `draft_open`), and the accept (the queue + the published
 * version, or a 422 `{message, findings}` / 409 refusal).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const api = vi.hoisted(() => ({
  getAcceptTarget: vi.fn(),
  previewAccept: vi.fn(),
  acceptExtension: vi.fn(),
  markCovered: vi.fn(),
  rejectExtension: vi.fn(),
}))

vi.mock('@/lib/api/notebook-schema', async (original) => ({
  ...(await original<typeof import('@/lib/api/notebook-schema')>()),
  notebookSchemaApi: api,
}))
vi.mock('@/lib/hooks/use-toast', () => ({ useToast: () => ({ toast: vi.fn() }) }))

import { AcceptExtensionDialog } from '../AcceptExtensionDialog'
import { PendingExtensionsPanel } from '../PendingExtensionsPanel'

const NB = 'notebook:x'
const PROPOSAL = { extension_id: 'pass1::streekfonds', type_name: 'Streekfonds',
  parent_type: 'Grant', rationale: 'a regional fund', properties: [] }

const preview = (overrides: Record<string, unknown> = {}) => ({
  findings: [], blocking: false, draft_open: false, detection: [], detection_error: null,
  label_changes: [{ label: 'Streekfonds', before: 'concept', after: 'grant', entities: 2 }],
  retype: { changes: 2, collisions: 0, flag_only: 0, caused_by_this_edit: 2,
    collisions_caused_by_this_edit: 0 },
  ...overrides,
})

const BLOCK = { rule: 'orphan', ontology: 'deals', subject: 'Streekfonds', severity: 'block',
  weight: 1, detail: "'Streekfonds' reaches no schema.org base" }

function mount(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

const button = (id: string) => screen.getByTestId(id) as HTMLButtonElement

beforeEach(() => {
  api.getAcceptTarget.mockResolvedValue({
    type_name: 'Streekfonds', parent_type: 'Grant', default_ontology: 'schema_core',
    ontologies: ['deals', 'schema_core'], defined_in: [],
  })
})
afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('AcceptExtensionDialog', () => {
  it('starts from the default ontology and says why', async () => {
    mount(<AcceptExtensionDialog notebookId={NB} extension={PROPOSAL} onOpenChange={vi.fn()} />)
    await waitFor(() =>
      expect((screen.getByTestId('accept-ontology') as HTMLSelectElement).value).toBe('schema_core'))
    expect(screen.getByTestId('accept-default-note').textContent).toContain('introduces the proposed parent')
    expect((screen.getByTestId('accept-parent') as HTMLInputElement).value).toBe('Grant')
    // Nothing can be published before a preview.
    expect(button('accept-publish').disabled).toBe(true)
  })

  it('previews, then publishes to the picked ontology with the chosen parent', async () => {
    api.previewAccept.mockResolvedValue(preview())
    api.acceptExtension.mockResolvedValue({
      notebook_schema: { notebook_id: NB, base_ontology: '', base_ontology_types: [],
        pending_extensions: [], coverage_pct: 0, review_required: false,
        soft_nudge_dismissed: false },
      published: { name: 'deals', version: '1.0.0+t', warnings: [], retype: { ...preview().retype, changes: 0,
        caused_by_this_edit: 0 } },
    })
    const onOpenChange = vi.fn()
    mount(<AcceptExtensionDialog notebookId={NB} extension={PROPOSAL} onOpenChange={onOpenChange} />)
    await waitFor(() => expect(button('accept-preview').disabled).toBe(false))
    fireEvent.change(screen.getByTestId('accept-ontology'), { target: { value: 'deals' } })
    fireEvent.change(screen.getByTestId('accept-parent'), { target: { value: 'Subsidie' } })
    fireEvent.click(button('accept-preview'))
    await screen.findByTestId('accept-preview-clean')
    expect(api.previewAccept).toHaveBeenCalledWith(NB, 'Streekfonds',
      { ontology: 'deals', parent_type: 'Subsidie' })
    await waitFor(() => expect(button('accept-publish').disabled).toBe(false))
    expect(button('accept-publish').textContent).toContain('deals')
    fireEvent.click(button('accept-publish'))
    await waitFor(() => expect(onOpenChange).toHaveBeenCalledWith(false))
    expect(api.acceptExtension).toHaveBeenCalledWith(NB, 'Streekfonds',
      { ontology: 'deals', parent_type: 'Subsidie' })
  })

  it('a blocking finding is shown and the publish stays disabled', async () => {
    api.previewAccept.mockResolvedValue(preview({ findings: [BLOCK], blocking: true }))
    mount(<AcceptExtensionDialog notebookId={NB} extension={PROPOSAL} onOpenChange={vi.fn()} />)
    await waitFor(() => expect(button('accept-preview').disabled).toBe(false))
    fireEvent.click(button('accept-preview'))
    expect((await screen.findByTestId('accept-preview-blocking')).textContent).toContain('orphan')
    expect(button('accept-publish').disabled).toBe(true)
  })

  it('an open draft of the ontology refuses it, with a link to the editor', async () => {
    api.previewAccept.mockResolvedValue(preview({ draft_open: true }))
    mount(<AcceptExtensionDialog notebookId={NB} extension={PROPOSAL} onOpenChange={vi.fn()} />)
    await waitFor(() => expect(button('accept-preview').disabled).toBe(false))
    fireEvent.click(button('accept-preview'))
    const alert = await screen.findByTestId('accept-draft-open')
    expect(alert.querySelector('a')?.getAttribute('href'))
      .toBe('/ontologies?name=schema_core&tab=edit')
    expect(button('accept-publish').disabled).toBe(true)
  })

  it("the gate's refusal on publish shows its findings and keeps the dialog open", async () => {
    api.previewAccept.mockResolvedValue(preview())
    api.acceptExtension.mockRejectedValue({ response: { status: 422, data: { detail: {
      message: 'the new type has blocking findings', findings: [BLOCK] } } } })
    const onOpenChange = vi.fn()
    mount(<AcceptExtensionDialog notebookId={NB} extension={PROPOSAL} onOpenChange={onOpenChange} />)
    await waitFor(() => expect(button('accept-preview').disabled).toBe(false))
    fireEvent.click(button('accept-preview'))
    await waitFor(() => expect(button('accept-publish').disabled).toBe(false))
    fireEvent.click(button('accept-publish'))
    const refused = await screen.findByTestId('accept-refused')
    expect(refused.textContent).toContain('blocking findings')
    expect(refused.textContent).toContain("'Streekfonds' reaches no schema.org base")
    expect(onOpenChange).not.toHaveBeenCalledWith(false)
  })

  it('changing the ontology drops the preview it no longer describes', async () => {
    api.previewAccept.mockResolvedValue(preview())
    mount(<AcceptExtensionDialog notebookId={NB} extension={PROPOSAL} onOpenChange={vi.fn()} />)
    await waitFor(() => expect(button('accept-preview').disabled).toBe(false))
    fireEvent.click(button('accept-preview'))
    await screen.findByTestId('accept-preview-result')
    fireEvent.change(screen.getByTestId('accept-ontology'), { target: { value: 'deals' } })
    await waitFor(() => expect(screen.queryByTestId('accept-preview-result')).toBeNull())
    expect(button('accept-publish').disabled).toBe(true)
  })

  it('a name already shared is marked covered, not published (review M3)', async () => {
    api.getAcceptTarget.mockResolvedValue({
      type_name: 'Gemeente', parent_type: 'Organization', default_ontology: 'schema_core',
      ontologies: ['deals', 'government'], defined_in: [{ ontology: 'government', type: 'Gemeente' }],
    })
    api.markCovered.mockResolvedValue({ notebook_schema: { notebook_id: NB, base_ontology: '',
      base_ontology_types: [], pending_extensions: [], coverage_pct: 0, review_required: false,
      soft_nudge_dismissed: false }, published: null })
    const onOpenChange = vi.fn()
    mount(<AcceptExtensionDialog notebookId={NB}
      extension={{ ...PROPOSAL, type_name: 'Gemeente' }} onOpenChange={onOpenChange} />)
    const shown = await screen.findByTestId('accept-already-defined')
    expect(shown.textContent).toContain('government')
    expect(shown.querySelector('a')?.getAttribute('href'))
      .toBe('/ontologies?name=government&tab=edit&type=Gemeente')
    expect(screen.queryByTestId('accept-publish')).toBeNull()
    expect(screen.queryByTestId('accept-preview')).toBeNull()
    fireEvent.click(button('accept-mark-covered'))
    await waitFor(() => expect(onOpenChange).toHaveBeenCalledWith(false))
    expect(api.markCovered).toHaveBeenCalledWith(NB, 'Gemeente')
    expect(api.acceptExtension).not.toHaveBeenCalled()
    expect(api.rejectExtension).not.toHaveBeenCalled()
  })

  it('a publish that moves entities offers their retype (review m4)', async () => {
    api.previewAccept.mockResolvedValue(preview())
    api.acceptExtension.mockResolvedValue({
      notebook_schema: { notebook_id: NB, base_ontology: '', base_ontology_types: [],
        pending_extensions: [], coverage_pct: 0, review_required: false,
        soft_nudge_dismissed: false },
      published: { name: 'deals', version: '1.0.0+t', warnings: [], retype: preview().retype },
    })
    const onOpenChange = vi.fn()
    mount(<AcceptExtensionDialog notebookId={NB} extension={PROPOSAL} onOpenChange={onOpenChange} />)
    await waitFor(() => expect(button('accept-preview').disabled).toBe(false))
    fireEvent.change(screen.getByTestId('accept-ontology'), { target: { value: 'deals' } })
    fireEvent.click(button('accept-preview'))
    await waitFor(() => expect(button('accept-publish').disabled).toBe(false))
    fireEvent.click(button('accept-publish'))
    const link = await screen.findByTestId('accept-retype-link')
    expect(link.getAttribute('href')).toBe('/ontologies?name=deals&tab=edit&retype=1')
    expect(onOpenChange).not.toHaveBeenCalledWith(false)
  })

  it('a proposal that cannot be loaded says so', async () => {
    api.getAcceptTarget.mockRejectedValue({ response: { data: { detail: 'No pending extension' } } })
    mount(<AcceptExtensionDialog notebookId={NB} extension={PROPOSAL} onOpenChange={vi.fn()} />)
    expect((await screen.findByRole('alert')).textContent).toContain('No pending extension')
  })
})

describe('PendingExtensionsPanel', () => {
  it('Accept opens the dialog instead of accepting in the notebook', async () => {
    mount(<PendingExtensionsPanel extensions={[PROPOSAL]} notebookId={NB} />)
    fireEvent.click(screen.getByTestId('accept-pass1::streekfonds'))
    await screen.findByTestId('accept-extension-dialog')
    expect(api.acceptExtension).not.toHaveBeenCalled()
  })

  it('shows the empty state', () => {
    mount(<PendingExtensionsPanel extensions={[]} notebookId={NB} />)
    expect(screen.getByTestId('pending-extensions-empty')).toBeTruthy()
  })
})
