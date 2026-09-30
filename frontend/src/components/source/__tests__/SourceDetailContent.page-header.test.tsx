// @vitest-environment jsdom
/**
 * WB.4 review round 1: on the source page (`back` given) the PageHeader and
 * its h1 are there while the source loads and when it cannot be read, not
 * only once it has loaded. The dialog (no `back`) has no page header.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

import type { SourceDetailResponse } from '@/lib/types/api'
import type { SourceInsightResponse } from '@/lib/api/insights'

// --- Data seam (mirrors SourceDetailContent.spine.test): the detail page loads
// the source via sourcesApi.get and lists insights via insightsApi. Drive both
// from mutable hoisted state so each test controls stage + output presence.
const H = vi.hoisted(() => {
  const state: {
    source: SourceDetailResponse | null
    insights: SourceInsightResponse[]
  } = { source: null, insights: [] }
  const sourcesApi = {
    get: vi.fn(async () => state.source),
    update: vi.fn(async () => state.source),
    retry: vi.fn(async () => state.source),
    reprocess: vi.fn(),
    runSummaries: vi.fn(),
    runEntities: vi.fn(),
    runPreprocessing: vi.fn(),
    downloadFile: vi.fn(),
    delete: vi.fn(),
  }
  const insightsApi = {
    listForSource: vi.fn(async () => state.insights),
    create: vi.fn(),
  }
  return { state, sourcesApi, insightsApi }
})

vi.mock('@/lib/api/sources', () => ({
  sourcesApi: H.sourcesApi,
  DEFAULT_PIPELINE_CONFIG: {},
}))
vi.mock('@/lib/api/insights', () => ({ insightsApi: H.insightsApi }))
vi.mock('@/lib/api/transformations', () => ({
  transformationsApi: { list: vi.fn(async () => []) },
}))
vi.mock('@/lib/api/embedding', () => ({
  embeddingApi: { embedContent: vi.fn() },
}))
vi.mock('@/lib/hooks/use-sources', () => ({
  useSourceChunks: () => ({ data: { total_chunks: 0, chunks: [] }, isLoading: false }),
  useSourcePipeline: () => ({ data: undefined }),
}))
vi.mock('@/lib/hooks/use-zotero', () => ({
  useZoteroStatus: () => ({ data: { connected: false } }),
  useZoteroPush: () => ({ mutate: vi.fn(), isPending: false }),
}))
vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn() }) }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('react-markdown', () => ({
  default: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
}))
vi.mock('remark-gfm', () => ({ default: () => {} }))
vi.mock('@/components/source/SourceEntitiesTab', () => ({
  SourceEntitiesTab: () => <div>entities</div>,
}))
vi.mock('@/components/source/RelatedSources', () => ({
  RelatedSources: () => <div>related</div>,
}))
vi.mock('@/components/source/NotebookAssociations', () => ({
  NotebookAssociations: () => <div>associations</div>,
}))
vi.mock('@/components/source/PdfChunkViewer', () => ({
  PdfChunkViewer: () => <div>chunks</div>,
}))
vi.mock('@/components/source/SourceInsightDialog', () => ({
  SourceInsightDialog: () => null,
}))
vi.mock('@/components/source/ParserEngineBadge', () => ({
  ParserEngineBadge: () => <span />,
}))
vi.mock('@/components/sources/pipeline/PipelineConfigPanel', () => ({
  PipelineConfigPanel: () => <div>config</div>,
}))

import { SourceDetailContent } from '../SourceDetailContent'

const back = <button type="button">Back to Sources</button>

beforeEach(() => {
  vi.clearAllMocks()
})
afterEach(cleanup)

const header = () => document.querySelector('[data-page-header]')

describe('source page header in every state', () => {
  it('shows the header with a fallback title while loading', () => {
    H.sourcesApi.get.mockImplementationOnce(() => new Promise(() => {}))
    render(<SourceDetailContent sourceId="source:abc" back={back} />)
    expect(header()).not.toBeNull()
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Source')
    expect(screen.getByRole('button', { name: 'Back to Sources' })).toBeTruthy()
  })

  it('shows the header when the source cannot be read', async () => {
    H.sourcesApi.get.mockRejectedValueOnce(new Error('404'))
    render(<SourceDetailContent sourceId="source:gone" back={back} />)
    await screen.findByText('Failed to load source details')
    expect(header()).not.toBeNull()
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Source')
  })

  it('shows the source title once loaded', async () => {
    H.state.source = { id: 'source:abc', title: 'A detail source', topics: [], asset: null,
      embedded: false, embedded_chunks: 0, insights_count: 0, entity_count: 0, relation_count: 0,
      created: '2026-07-01T00:00:00Z', updated: '2026-07-01T00:00:00Z', full_text: 'body' } as SourceDetailResponse
    render(<SourceDetailContent sourceId="source:abc" back={back} />)
    await screen.findByLabelText('Toggle pipeline detail')
    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('A detail source')
  })

  it('has no page header in the dialog', () => {
    H.sourcesApi.get.mockImplementationOnce(() => new Promise(() => {}))
    render(<SourceDetailContent sourceId="source:abc" />)
    expect(header()).toBeNull()
  })
})
