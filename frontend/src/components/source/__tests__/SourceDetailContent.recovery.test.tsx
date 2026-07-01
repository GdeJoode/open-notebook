// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

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

function makeSource(overrides: Partial<SourceDetailResponse> = {}): SourceDetailResponse {
  return {
    id: 'source:abc',
    title: 'A detail source',
    topics: [],
    // A file asset so Reprocess is available (its only guard is file_path).
    asset: { file_path: '/data/doc.pdf' },
    embedded: false,
    embedded_chunks: 0,
    insights_count: 0,
    entity_count: 0,
    relation_count: 0,
    created: '2026-07-01T00:00:00Z',
    updated: '2026-07-01T00:00:00Z',
    full_text: 'body',
    ...overrides,
  }
}

async function openMenu(
  source: SourceDetailResponse,
  insights: SourceInsightResponse[] = []
) {
  H.state.source = source
  H.state.insights = insights
  render(<SourceDetailContent sourceId={source.id} />)
  // Wait for the spine (proves the async load finished), then open the menu.
  await screen.findByLabelText('Toggle pipeline detail')
  await userEvent.click(screen.getByRole('button', { name: 'Source actions' }))
}

const RUNNERS = [
  /Run Preprocessing/,
  /Generate Summaries/,
  /Extract Entities/,
  /Embed Content/,
] as const

function isDisabled(name: RegExp): boolean {
  const item = screen.getByRole('menuitem', { name })
  return item.getAttribute('aria-disabled') === 'true'
}

const insight = (): SourceInsightResponse =>
  ({ id: 'insight:1', insight_type: 'summary', content: 'x', created: '2026-07-01T00:00:00Z' }) as SourceInsightResponse

beforeEach(() => {
  vi.clearAllMocks()
})
afterEach(cleanup)

describe('UX.6 AC1 — runners disabled on a complete source', () => {
  it('all four runners are disabled with a "Runs automatically" hint', async () => {
    await openMenu(
      makeSource({ processing_stage: 'complete', embedded: true, entity_count: 5 }),
      [insight()]
    )
    for (const r of RUNNERS) expect(isDisabled(r)).toBe(true)
    // The hint appears (one per disabled runner).
    expect(screen.getAllByText('Runs automatically').length).toBe(RUNNERS.length)
  })
})

describe('UX.6 AC1 — runners enabled on recovery stages', () => {
  it('a failed source enables every runner', async () => {
    await openMenu(
      makeSource({ processing_stage: 'failed', embedded: true, entity_count: 5 }),
      [insight()]
    )
    for (const r of RUNNERS) expect(isDisabled(r)).toBe(false)
    expect(screen.queryByText('Runs automatically')).toBeNull()
  })

  it('an awaiting_schema_review source enables every runner (output present)', async () => {
    await openMenu(
      makeSource({
        processing_stage: 'awaiting_schema_review',
        embedded: true,
        entity_count: 5,
      }),
      [insight()]
    )
    for (const r of RUNNERS) expect(isDisabled(r)).toBe(false)
  })
})

describe('UX.6 AC1 — runners enabled when their own output is missing', () => {
  it('a complete source with missing outputs enables only the missing-output runners', async () => {
    // Non-recovery stage, but: not embedded, no entities, no insights ⇒ those
    // three runners are enabled; nothing "runs automatically" for them.
    await openMenu(
      makeSource({ processing_stage: 'complete', embedded: false, entity_count: 0 }),
      []
    )
    expect(isDisabled(/Embed Content/)).toBe(false)
    expect(isDisabled(/Extract Entities/)).toBe(false)
    expect(isDisabled(/Generate Summaries/)).toBe(false)
    // Preprocessing is recovery-only (needs a recovery stage), so it stays
    // disabled here on the happy path.
    expect(isDisabled(/Run Preprocessing/)).toBe(true)
  })
})

describe('UX.6 AC2 — Reprocess is always available', () => {
  it('stays enabled on a complete source', async () => {
    await openMenu(
      makeSource({ processing_stage: 'complete', embedded: true, entity_count: 5 }),
      [insight()]
    )
    expect(isDisabled(/Reprocess Document/)).toBe(false)
  })

  it('stays enabled on a failed source', async () => {
    await openMenu(makeSource({ processing_stage: 'failed' }), [])
    expect(isDisabled(/Reprocess Document/)).toBe(false)
  })
})
