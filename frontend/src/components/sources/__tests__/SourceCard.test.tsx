// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { ProcessingStage } from '@/lib/pipeline/processing-stage'
import type { SourceListResponse } from '@/lib/types/api'

// The card fans out a per-source stage poll; mock it so the test drives the
// spine purely from the `source` prop and can assert the `enabled` gating.
const useSourcePipeline = vi.fn((_id: string, _enabled?: boolean) => ({
  data: undefined as SourceListResponse | undefined,
}))
vi.mock('@/lib/hooks/use-sources', () => ({
  useSourcePipeline: (id: string, enabled?: boolean) => useSourcePipeline(id, enabled),
}))

// ContextMode is a type-only import at runtime; stub the heavy page module.
vi.mock('@/app/(dashboard)/notebooks/[id]/page', () => ({}))

import { SourceCard } from '../SourceCard'

beforeEach(() => {
  useSourcePipeline.mockClear()
  useSourcePipeline.mockReturnValue({ data: undefined })
})
afterEach(cleanup)

function makeSource(overrides: Partial<SourceListResponse> = {}): SourceListResponse {
  return {
    id: 'source:abc',
    title: 'A test source',
    topics: [],
    asset: { url: 'https://example.com' },
    embedded: false,
    embedded_chunks: 0,
    insights_count: 0,
    entity_count: 0,
    relation_count: 0,
    created: '2026-07-01T00:00:00Z',
    updated: '2026-07-01T00:00:00Z',
    ...overrides,
  }
}

/** Read the rendered card-variant segment state for a spine node. */
function segmentState(container: HTMLElement, key: string): string | null | undefined {
  return container
    .querySelector(`[data-variant="card"] [data-node-key="${key}"]`)
    ?.getAttribute('data-node-state')
}

describe('SourceCard renders the 5-segment bar per processing_stage (AC1)', () => {
  const SPINE = ['ingest', 'embed', 'extract', 'graph', 'complete'] as const
  // jobStatus is absent in the test, so the current node resolves to `done`.
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
    it(`stage=${stage} fills the bar up to that stage`, () => {
      const { container } = render(
        <SourceCard source={makeSource({ processing_stage: stage })} />
      )
      for (const key of SPINE) {
        expect(segmentState(container, key)).toBe(EXPECTED[stage][key])
      }
      // The title is always present alongside the bar.
      expect(screen.getByText('A test source')).toBeTruthy()
    })
  }

  it('failed card renders a red segment (Ingest failed with no counts)', () => {
    const { container } = render(
      <SourceCard source={makeSource({ processing_stage: 'failed' })} />
    )
    expect(segmentState(container, 'ingest')).toBe('failed')
  })
})

describe('AC0 — parse-failure regression via the counts adapter', () => {
  it('{failed, 0 chunks, 0 entities} => Ingest failed (NOT Embed)', () => {
    const { container } = render(
      <SourceCard
        source={makeSource({
          processing_stage: 'failed',
          embedded_chunks: 0,
          entity_count: 0,
        })}
      />
    )
    expect(segmentState(container, 'ingest')).toBe('failed')
    expect(segmentState(container, 'embed')).toBe('pending')
  })

  it('{failed, 12 chunks, 0 entities} => Extract failed', () => {
    const { container } = render(
      <SourceCard
        source={makeSource({
          processing_stage: 'failed',
          embedded_chunks: 12,
          entity_count: 0,
        })}
      />
    )
    expect(segmentState(container, 'ingest')).toBe('done')
    expect(segmentState(container, 'embed')).toBe('done')
    expect(segmentState(container, 'extract')).toBe('failed')
  })
})

describe('gated + failed inline actions (AC2)', () => {
  it('awaiting_schema_review shows a Review schema affordance deep-linking to detail', async () => {
    const onClick = vi.fn()
    render(
      <SourceCard
        source={makeSource({ processing_stage: 'awaiting_schema_review' })}
        onClick={onClick}
      />
    )
    const btn = screen.getByRole('button', { name: /Review schema/i })
    await userEvent.click(btn)
    expect(onClick).toHaveBeenCalledWith('source:abc')
  })

  it('failed card shows the existing Retry path (onRetry)', async () => {
    const onRetry = vi.fn()
    render(
      <SourceCard source={makeSource({ processing_stage: 'failed' })} onRetry={onRetry} />
    )
    const btn = screen.getByRole('button', { name: /Retry — Ingest/i })
    await userEvent.click(btn)
    expect(onRetry).toHaveBeenCalledWith('source:abc')
  })
})

describe('bounded polling: terminal cards pass enabled=false (AC3)', () => {
  it('complete card calls useSourcePipeline with enabled=false', () => {
    render(<SourceCard source={makeSource({ processing_stage: 'complete' })} />)
    expect(useSourcePipeline).toHaveBeenCalledWith('source:abc', false)
    for (const call of useSourcePipeline.mock.calls) {
      expect(call[1]).toBe(false)
    }
  })

  it('failed card calls useSourcePipeline with enabled=false', () => {
    render(<SourceCard source={makeSource({ processing_stage: 'failed' })} />)
    expect(useSourcePipeline).toHaveBeenCalledWith('source:abc', false)
    for (const call of useSourcePipeline.mock.calls) {
      expect(call[1]).toBe(false)
    }
  })

  it('awaiting_schema_review (gated) also stops polling (enabled=false)', () => {
    render(<SourceCard source={makeSource({ processing_stage: 'awaiting_schema_review' })} />)
    for (const call of useSourcePipeline.mock.calls) {
      expect(call[1]).toBe(false)
    }
  })

  it('a non-terminal (ingested) card polls (enabled=true)', () => {
    render(<SourceCard source={makeSource({ processing_stage: 'ingested' })} />)
    expect(useSourcePipeline).toHaveBeenCalledWith('source:abc', true)
  })
})

describe('undefined processing_stage degrades to an all-pending bar (AC4)', () => {
  it('renders all-pending spine with the title still shown, no crash', () => {
    const { container } = render(<SourceCard source={makeSource({ processing_stage: undefined })} />)
    for (const key of ['ingest', 'embed', 'extract', 'graph', 'complete']) {
      expect(segmentState(container, key)).toBe('pending')
    }
    expect(screen.getByText('A test source')).toBeTruthy()
    // No error badge is surfaced for unknown-stage cards.
    expect(screen.queryByText(/failed/i)).toBeNull()
  })
})

describe('completed sources keep their badges + dropdown actions (AC5)', () => {
  it('renders insights + topic badges and the actions menu', () => {
    render(
      <SourceCard
        source={makeSource({
          processing_stage: 'complete',
          insights_count: 4,
          topics: ['alpha', 'beta', 'gamma'],
        })}
        onDelete={vi.fn()}
      />
    )
    // "4 insights" appears both on the mini-bar's Insights node and the badge.
    expect(screen.getAllByText('4 insights').length).toBeGreaterThan(0)
    expect(screen.getByText('alpha')).toBeTruthy()
    expect(screen.getByText('beta')).toBeTruthy()
    expect(screen.getByText('+1')).toBeTruthy()
    expect(screen.getByRole('button', { name: /Source actions/i })).toBeTruthy()
  })
})
