// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { SourceDetailResponse } from '@/lib/types/api'
import type { ProcessingStage } from '@/lib/pipeline/processing-stage'

// --- Data seam: the detail page loads the source via sourcesApi.get and polls
// in-flight sources via useSourcePipeline. Drive both from a single fixture so
// the spine renders purely from `processing_stage`. `vi.hoisted` keeps the mock
// state accessible from the hoisted `vi.mock` factory.
const H = vi.hoisted(() => {
  const state: { source: SourceDetailResponse | null } = { source: null }
  const sourcesApi = {
    get: vi.fn(async () => state.source),
    update: vi.fn(async () => state.source),
    retry: vi.fn(async () => state.source),
    downloadFile: vi.fn(),
    delete: vi.fn(),
    reprocess: vi.fn(),
    runSummaries: vi.fn(),
    runEntities: vi.fn(),
    runPreprocessing: vi.fn(),
  }
  return { state, sourcesApi }
})
const sourcesApi = H.sourcesApi

vi.mock('@/lib/api/sources', () => ({
  sourcesApi: H.sourcesApi,
  DEFAULT_PIPELINE_CONFIG: {},
}))
vi.mock('@/lib/api/insights', () => ({
  insightsApi: { listForSource: vi.fn(async () => []), create: vi.fn() },
}))
vi.mock('@/lib/api/transformations', () => ({
  transformationsApi: { list: vi.fn(async () => []) },
}))
vi.mock('@/lib/api/embedding', () => ({
  embeddingApi: { embedContent: vi.fn() },
}))

// The pipeline poller is mocked so the test drives the spine from the fixture
// and can assert the `enabled` gating (terminal/gated ⇒ no polling).
const { useSourcePipeline } = vi.hoisted(() => ({
  useSourcePipeline: vi.fn((_id: string, _enabled?: boolean) => ({
    data: undefined as SourceDetailResponse | undefined,
  })),
}))
vi.mock('@/lib/hooks/use-sources', () => ({
  useSourceChunks: () => ({ data: { total_chunks: 0, chunks: [] }, isLoading: false }),
  useSourcePipeline: (id: string, enabled?: boolean) => useSourcePipeline(id, enabled),
}))
vi.mock('@/lib/hooks/use-zotero', () => ({
  useZoteroStatus: () => ({ data: { connected: false } }),
  useZoteroPush: () => ({ mutate: vi.fn(), isPending: false }),
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('sonner', () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}))

// Heavy tab bodies / renderers are stubbed to keep the test focused on the spine
// and tab wiring (Radix mounts only the active tab's content).
vi.mock('react-markdown', () => ({
  default: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
}))
vi.mock('remark-gfm', () => ({ default: () => {} }))
vi.mock('@/components/source/SourceEntitiesTab', () => ({
  SourceEntitiesTab: () => <div data-testid="entities-tab-stub">entities</div>,
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
  ParserEngineBadge: () => <span data-testid="parser-badge" />,
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
    asset: { url: 'https://example.com' },
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

async function renderDetail(source: SourceDetailResponse) {
  H.state.source = source
  const utils = render(<SourceDetailContent sourceId={source.id} />)
  // Wait for the async fetchSource to resolve and the spine to mount.
  await screen.findByLabelText('Toggle pipeline detail')
  return utils
}

/** Read the collapsed detail-variant segment state for a spine node. */
function segmentState(container: HTMLElement, key: string): string | null | undefined {
  return container
    .querySelector(`[data-variant="detail"] [data-node-key="${key}"]`)
    ?.getAttribute('data-node-state')
}

beforeEach(() => {
  vi.clearAllMocks()
  useSourcePipeline.mockReturnValue({ data: undefined })
})
afterEach(cleanup)

describe('AC1 — detail header shows the collapsed 5-stage spine (not 4 output badges)', () => {
  const SPINE = ['ingest', 'embed', 'extract', 'graph', 'complete'] as const
  const EXPECTED: Record<
    Exclude<ProcessingStage, 'failed'>,
    Record<(typeof SPINE)[number], string>
  > = {
    ingested: { ingest: 'done', embed: 'pending', extract: 'pending', graph: 'pending', complete: 'pending' },
    embedded: { ingest: 'done', embed: 'done', extract: 'pending', graph: 'pending', complete: 'pending' },
    extracted: { ingest: 'done', embed: 'done', extract: 'done', graph: 'pending', complete: 'pending' },
    graphed: { ingest: 'done', embed: 'done', extract: 'done', graph: 'done', complete: 'pending' },
    complete: { ingest: 'done', embed: 'done', extract: 'done', graph: 'done', complete: 'done' },
    awaiting_schema_review: { ingest: 'done', embed: 'done', extract: 'gated', graph: 'pending', complete: 'pending' },
  }

  for (const stage of Object.keys(EXPECTED) as Array<keyof typeof EXPECTED>) {
    it(`stage=${stage} renders the spine filled up to that stage`, async () => {
      const { container } = await renderDetail(makeSource({ processing_stage: stage }))
      for (const key of SPINE) {
        expect(segmentState(container, key)).toBe(EXPECTED[stage][key])
      }
    })
  }

  it('no longer renders the old independent output badges', async () => {
    await renderDetail(makeSource({ processing_stage: 'complete' }))
    expect(screen.queryByText('Not extracted')).toBeNull()
    expect(screen.queryByText('Not embedded')).toBeNull()
    expect(screen.queryByText('No entities')).toBeNull()
    expect(screen.queryByText('No insights')).toBeNull()
  })

  it('expanding the spine reveals per-node counts', async () => {
    await renderDetail(
      makeSource({ processing_stage: 'complete', embedded_chunks: 24, entity_count: 18 })
    )
    await userEvent.click(screen.getByLabelText('Toggle pipeline detail'))
    expect(screen.getByText('24 chunks')).toBeTruthy()
    expect(screen.getByText('18 entities')).toBeTruthy()
  })
})

describe('AC2 — Graph node membership + node-click deep-links', () => {
  it('graphed source ⇒ Graph node done', async () => {
    const { container } = await renderDetail(
      makeSource({ processing_stage: 'graphed', entity_count: 5, relation_count: 3 })
    )
    expect(segmentState(container, 'graph')).toBe('done')
  })

  it('non-graphed source (extracted) ⇒ Graph node pending', async () => {
    const { container } = await renderDetail(
      makeSource({ processing_stage: 'extracted', entity_count: 5, relation_count: 0 })
    )
    expect(segmentState(container, 'graph')).toBe('pending')
  })

  it('clicking Extract navigates to the Entities tab', async () => {
    await renderDetail(makeSource({ processing_stage: 'graphed', entity_count: 5 }))
    await userEvent.click(screen.getByLabelText('Toggle pipeline detail'))
    await userEvent.click(screen.getByRole('button', { name: /^Extract:/ }))
    await waitFor(() =>
      expect(
        screen.getByRole('tab', { name: /Entities/i }).getAttribute('aria-selected')
      ).toBe('true')
    )
  })

  it('clicking Insights navigates to the Insights tab', async () => {
    await renderDetail(makeSource({ processing_stage: 'complete', insights_count: 2 }))
    await userEvent.click(screen.getByLabelText('Toggle pipeline detail'))
    await userEvent.click(screen.getByRole('button', { name: /^Insights:/ }))
    await waitFor(() =>
      expect(
        screen.getByRole('tab', { name: /Insights/i }).getAttribute('aria-selected')
      ).toBe('true')
    )
  })
})

describe('AC3 — gated + failed actions', () => {
  it('awaiting_schema_review renders a Review schema action that opens the Entities tab', async () => {
    await renderDetail(makeSource({ processing_stage: 'awaiting_schema_review' }))
    const btn = screen.getByRole('button', { name: /Review schema/i })
    await userEvent.click(btn)
    await waitFor(() =>
      expect(
        screen.getByRole('tab', { name: /Entities/i }).getAttribute('aria-selected')
      ).toBe('true')
    )
  })

  it('failed source renders a Retry action wired to sourcesApi.retry', async () => {
    await renderDetail(makeSource({ processing_stage: 'failed' }))
    const btn = screen.getByRole('button', { name: /Retry/i })
    await userEvent.click(btn)
    await waitFor(() => expect(sourcesApi.retry).toHaveBeenCalledWith('source:abc'))
  })
})

describe('AC5 — all 6 tabs mount + header dropdown still renders', () => {
  it('renders the 6 tab triggers and the actions dropdown', async () => {
    const { container } = await renderDetail(makeSource({ processing_stage: 'complete' }))
    const tabs = screen.getAllByRole('tab')
    expect(tabs).toHaveLength(6)
    const labels = tabs.map((t) => t.textContent)
    expect(labels.some((l) => l?.includes('Content'))).toBe(true)
    expect(labels.some((l) => l?.includes('Insights'))).toBe(true)
    expect(labels.some((l) => l?.includes('Chunks'))).toBe(true)
    expect(labels.some((l) => l?.includes('Entities'))).toBe(true)
    expect(labels.some((l) => l?.includes('Related'))).toBe(true)
    expect(labels.some((l) => l?.includes('Details'))).toBe(true)
    // The header dropdown trigger (manual runners live behind it; UX.6 reframes).
    expect(container.querySelector('[aria-haspopup="menu"]')).toBeTruthy()
  })
})

describe('bounded polling — terminal/gated detail pages stop polling', () => {
  it('complete source calls useSourcePipeline with enabled=false', async () => {
    await renderDetail(makeSource({ processing_stage: 'complete' }))
    for (const call of useSourcePipeline.mock.calls) {
      expect(call[1]).toBe(false)
    }
  })

  it('in-flight (embedded) source polls (enabled=true)', async () => {
    await renderDetail(makeSource({ processing_stage: 'embedded' }))
    expect(useSourcePipeline).toHaveBeenCalledWith('source:abc', true)
  })
})
