// @vitest-environment jsdom
import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { ProcessingStage } from '@/lib/pipeline/processing-stage'

// Radix Tabs/Select need pointer-capture + scrollIntoView (absent in jsdom).
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false) as unknown as typeof Element.prototype.hasPointerCapture
  Element.prototype.setPointerCapture = vi.fn() as unknown as typeof Element.prototype.setPointerCapture
  Element.prototype.releasePointerCapture = vi.fn() as unknown as typeof Element.prototype.releasePointerCapture
})

// --- Mock seam -------------------------------------------------------------
// The flow reads the spine from useSourcePipeline (processing_stage) and the
// current-node spinner from useSourceStatus (job axis). Drive both from a
// mutable hoisted state so a test can pin a stage before submitting.
const H = vi.hoisted(() => {
  const state = {
    stage: 'ingested' as ProcessingStage | undefined,
    jobStatus: 'running' as string,
    // Per-id stage for the multi-file poll.
    stageById: {} as Record<string, ProcessingStage>,
  }
  let createCalls = 0
  return {
    state,
    resetCreateCalls: () => {
      createCalls = 0
    },
    createMutateAsync: vi.fn(async () => {
      createCalls += 1
      const id = createCalls === 1 ? 'source:a' : createCalls === 2 ? 'source:b' : 'source:new'
      return { id, title: `Source ${id}` }
    }),
    retryMutate: vi.fn(),
    sourcesGet: vi.fn(async (id: string) => ({
      id,
      processing_stage: H.state.stageById[id],
    })),
    toast: vi.fn(),
    push: vi.fn(),
    // Stable references — returning fresh objects/arrays from a mocked hook on
    // every render would retrigger the form-reset effect and spin forever.
    settings: { default_embedding_option: 'ask' },
    notebooks: [] as unknown[],
  }
})

vi.mock('@/lib/hooks/use-sources', () => ({
  useCreateSource: () => ({ mutateAsync: H.createMutateAsync, isPending: false }),
  useSourcePipeline: () => ({
    data: {
      id: 'source:new',
      title: 'My source',
      processing_stage: H.state.stage,
      embedded_chunks: 0,
      entity_count: 0,
      insights_count: 0,
      relation_count: 0,
    },
  }),
  useSourceStatus: () => ({ data: { status: H.state.jobStatus } }),
  useRetrySource: () => ({ mutate: H.retryMutate }),
}))
vi.mock('@/lib/hooks/use-notebooks', () => ({
  useNotebooks: () => ({ data: H.notebooks, isLoading: false }),
}))
vi.mock('@/lib/hooks/use-settings', () => ({
  useSettings: () => ({ data: H.settings }),
}))
vi.mock('@/lib/hooks/use-toast', () => ({
  useToast: () => ({ toast: H.toast }),
}))
vi.mock('@/lib/api/sources', () => ({
  sourcesApi: { get: H.sourcesGet },
}))
// The log console opens an EventSource — stub it so the tracker mounts cleanly.
vi.mock('../ProcessingLogConsole', () => ({
  ProcessingLogConsole: () => <div data-testid="log-console">logs</div>,
}))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: H.push }),
  useSearchParams: () => ({ get: () => null }),
}))

import { CreateSourcePipeline } from '../CreateSourcePipeline'

beforeEach(() => {
  vi.clearAllMocks()
  H.resetCreateCalls()
  H.state.stage = 'ingested'
  H.state.jobStatus = 'running'
  H.state.stageById = {}
})
afterEach(cleanup)

/** Fill the Input + Organize config steps for a text source and submit. */
async function submitTextSource(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('tab', { name: /Text/i }))
  await user.type(
    screen.getByPlaceholderText('Give your source a descriptive title'),
    'My text source'
  )
  await user.type(
    screen.getByPlaceholderText('Paste or type your content here...'),
    'Some body content'
  )
  await user.click(screen.getByRole('button', { name: 'Next' }))
  await user.click(screen.getByRole('button', { name: 'Submit' }))
}

describe('AC1 — exactly 4 steps, no mandatory Config step', () => {
  it('renders Input, Organize, Processing, Done and nothing else', () => {
    render(<CreateSourcePipeline />)
    for (const label of ['Input', 'Organize', 'Processing', 'Done']) {
      expect(screen.getByRole('button', { name: label })).toBeTruthy()
    }
    // The old wizard steps are gone from the stepper.
    expect(screen.queryByRole('button', { name: 'Config' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Extract' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Embed' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Classification' })).toBeNull()
  })
})

describe('AC2 — live tracker mounts during processing (Embed before Extract)', () => {
  it('shows the live PipelineStatus tracker + streaming log beneath it', async () => {
    H.state.stage = 'ingested'
    const user = userEvent.setup({ pointerEventsCheck: 0 })
    const { container } = render(<CreateSourcePipeline />)

    await submitTextSource(user)

    await waitFor(() =>
      expect(container.querySelector('[data-variant="live"]')).toBeTruthy()
    )
    // Runtime order: Embed precedes Extract in the rendered spine.
    const live = container.querySelector('[data-variant="live"]') as HTMLElement
    const keys = Array.from(live.querySelectorAll('[data-node-key]'))
      .map((n) => n.getAttribute('data-node-key'))
      .filter((k) => k && k !== 'insights')
    expect(keys.indexOf('embed')).toBeLessThan(keys.indexOf('extract'))
    // The streaming log console is mounted under the tracker.
    expect(within(live).getByTestId('log-console')).toBeTruthy()
    // Not on Done yet.
    expect(screen.queryByText('Source Created Successfully')).toBeNull()
  })
})

describe('AC3 — transitions to Done on complete', () => {
  it('advances to the Done step when processing_stage === complete', async () => {
    H.state.stage = 'complete'
    H.state.jobStatus = 'completed'
    const user = userEvent.setup({ pointerEventsCheck: 0 })
    render(<CreateSourcePipeline />)

    await submitTextSource(user)

    await waitFor(() =>
      expect(screen.getByText('Source Created Successfully')).toBeTruthy()
    )
  })
})

describe('AC3 — parks on the gated node at awaiting_schema_review (no false Done)', () => {
  it('renders a Review schema action and does not advance to Done', async () => {
    H.state.stage = 'awaiting_schema_review'
    H.state.jobStatus = 'paused_for_review'
    const user = userEvent.setup({ pointerEventsCheck: 0 })
    const { container } = render(<CreateSourcePipeline />)

    await submitTextSource(user)

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /Review schema/i })).toBeTruthy()
    )
    expect(screen.queryByText('Source Created Successfully')).toBeNull()
    // The Extract node is parked (gated), not done.
    const extract = container.querySelector('[data-variant="live"] [data-node-key="extract"]')
    expect(extract?.getAttribute('data-node-state')).toBe('gated')
  })
})

describe('AC5 — no manual Start Entities / Start Embed triggers', () => {
  it('exposes no manual entity/embed run buttons in the create flow', async () => {
    H.state.stage = 'embedded'
    const user = userEvent.setup({ pointerEventsCheck: 0 })
    render(<CreateSourcePipeline />)

    await submitTextSource(user)

    await waitFor(() =>
      expect(document.querySelector('[data-variant="live"]')).toBeTruthy()
    )
    expect(screen.queryByRole('button', { name: /Start Entities/i })).toBeNull()
    expect(screen.queryByRole('button', { name: /Start Embed/i })).toBeNull()
  })
})

describe('multi-file batch — each entry driven by its OWN processing_stage', () => {
  // Real timers: the per-source poll runs on a 3s interval, so give the test
  // room. (Fake timers deadlock against userEvent's internal async here.)
  it('renders a per-source card whose bar reflects that source stage', async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 })
    // source:a is still embedding; source:b has completed.
    H.state.stageById = { 'source:a': 'embedded', 'source:b': 'complete' }

    const { container } = render(<CreateSourcePipeline />)

    // Upload two files.
    await user.click(screen.getByRole('tab', { name: /Upload/i }))
    const fileInput = container.querySelector('#file') as HTMLInputElement
    const f1 = new File(['one'], 'one.pdf', { type: 'application/pdf' })
    const f2 = new File(['two'], 'two.pdf', { type: 'application/pdf' })
    await user.upload(fileInput, [f1, f2])

    await user.click(screen.getByRole('button', { name: 'Next' }))
    await user.click(screen.getByRole('button', { name: 'Submit' }))

    // Batch prompt → individual sources kicks off the sequential create loop.
    await user.click(screen.getByRole('button', { name: /Individual Sources/i }))

    // Two per-source cards render (batch upload preserved).
    await waitFor(() =>
      expect(container.querySelectorAll('[data-variant="card"]').length).toBe(2)
    )

    // The poll fires on a 3s interval — wait for each card to reflect its OWN
    // polled stage (source:a embed done / extract pending; source:b all done).
    await waitFor(
      () => {
        const cards = container.querySelectorAll('[data-variant="card"]')
        expect(
          cards[0].querySelector('[data-node-key="embed"]')?.getAttribute('data-node-state')
        ).toBe('done')
        expect(
          cards[1].querySelector('[data-node-key="complete"]')?.getAttribute('data-node-state')
        ).toBe('done')
      },
      { timeout: 8000, interval: 200 }
    )

    // source:a's extract stage was never reached ⇒ still pending.
    const cards = container.querySelectorAll('[data-variant="card"]')
    expect(
      cards[0].querySelector('[data-node-key="extract"]')?.getAttribute('data-node-state')
    ).toBe('pending')
  }, 15000)
})

/** Upload two files and pick "Individual Sources" to start the batch loop. */
async function submitTwoFileBatch(
  user: ReturnType<typeof userEvent.setup>,
  container: HTMLElement
) {
  await user.click(screen.getByRole('tab', { name: /Upload/i }))
  const fileInput = container.querySelector('#file') as HTMLInputElement
  const f1 = new File(['one'], 'one.pdf', { type: 'application/pdf' })
  const f2 = new File(['two'], 'two.pdf', { type: 'application/pdf' })
  await user.upload(fileInput, [f1, f2])
  await user.click(screen.getByRole('button', { name: 'Next' }))
  await user.click(screen.getByRole('button', { name: 'Submit' }))
  await user.click(screen.getByRole('button', { name: /Individual Sources/i }))
}

describe('multi-file batch — schema gate settles the batch (no infinite poll, no wedge)', () => {
  it('stops polling a gated entry, surfaces a settled state, and wires its Review schema action', async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 })
    // source:a parks on the schema-review gate; source:b completes.
    H.state.stageById = {
      'source:a': 'awaiting_schema_review',
      'source:b': 'complete',
    }

    const { container } = render(<CreateSourcePipeline />)
    await submitTwoFileBatch(user, container)

    // The batch reaches a settled state (NOT wedged on "Processing…"): the
    // gated entry surfaces a Review schema action.
    const reviewButton = await screen.findByRole(
      'button',
      { name: /Review schema/i },
      { timeout: 8000 }
    )
    // Settled banner tells the user automatic work finished.
    expect(screen.getByText(/Automatic processing finished/i)).toBeTruthy()
    // Never falsely advanced to the Done summary.
    expect(screen.queryByText('Source Created Successfully')).toBeNull()

    // Polling has STOPPED for the settled batch: no further sourcesApi.get calls.
    const settledCount = H.sourcesGet.mock.calls.length
    expect(settledCount).toBeGreaterThan(0)
    await new Promise((r) => setTimeout(r, 3500))
    expect(H.sourcesGet.mock.calls.length).toBe(settledCount)

    // The per-entry Review schema action deep-links to THAT entry's source.
    await user.click(reviewButton)
    expect(H.push).toHaveBeenCalledWith('/sources/source:a')
  }, 15000)

  it('resumes polling a re-armed entry after Retry once the batch has fully settled', async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 })
    // source:a fails; source:b completes ⇒ settled batch with a retryable entry.
    H.state.stageById = { 'source:a': 'failed', 'source:b': 'complete' }

    const { container } = render(<CreateSourcePipeline />)
    await submitTwoFileBatch(user, container)

    const retryButton = await screen.findByRole(
      'button',
      { name: /Retry/i },
      { timeout: 8000 }
    )
    expect(screen.getByText(/Automatic processing finished/i)).toBeTruthy()

    // FULLY settle: advance PAST the clearing tick so the poll interval is gone.
    // Once no more get() calls land across a >3s window, the interval is proven
    // dead — this is the exact state in which the old (all-ids) key never
    // resumed. Freeze the get() count in that dead state.
    let frozenCount = 0
    await waitFor(
      async () => {
        const before = H.sourcesGet.mock.calls.length
        await new Promise((r) => setTimeout(r, 3500))
        expect(H.sourcesGet.mock.calls.length).toBe(before)
        frozenCount = before
      },
      { timeout: 12000, interval: 500 }
    )
    // Sanity: source:a WAS polled at least once before settling.
    expect(
      H.sourcesGet.mock.calls.some((c) => c[0] === 'source:a')
    ).toBe(true)

    // Retry re-arms source:a to `processing`; the pollable-id set changes from ''
    // to 'source:a', so the poll effect must re-fire and recreate the interval.
    await user.click(retryButton)
    expect(H.retryMutate).toHaveBeenCalledWith('source:a')

    // Polling actually RESUMED: a NEW get() for source:a lands beyond the frozen
    // (interval-dead) point. Asserting only that retryMutate fired would miss the
    // dead-interval bug this test guards against.
    await waitFor(
      () => {
        const resumed = H.sourcesGet.mock.calls
          .slice(frozenCount)
          .some((c) => c[0] === 'source:a')
        expect(resumed).toBe(true)
      },
      { timeout: 8000, interval: 200 }
    )
  }, 25000)
})

describe('advanced ingestion overrides survive Input→Organize→Back and reach the create call', () => {
  it('keeps Docling selected after back-navigation and sends processing_overrides', async () => {
    H.state.stage = 'ingested'
    const user = userEvent.setup({ pointerEventsCheck: 0 })
    render(<CreateSourcePipeline />)

    // Fill a text source on the Input step.
    await user.click(screen.getByRole('tab', { name: /Text/i }))
    await user.type(
      screen.getByPlaceholderText('Give your source a descriptive title'),
      'Overrides survive'
    )
    await user.type(
      screen.getByPlaceholderText('Paste or type your content here...'),
      'Body'
    )

    // Expand Advanced ingestion settings and pick the Docling parser.
    await user.click(screen.getByRole('button', { name: 'Advanced ingestion settings' }))
    await user.click(screen.getByLabelText('Parser engine'))
    await user.click(await screen.findByRole('option', { name: 'Docling' }))

    // Navigate Input → Organize → Back to Input (remounts the disclosure).
    await user.click(screen.getByRole('button', { name: 'Next' }))
    await user.click(screen.getByRole('button', { name: 'Back' }))

    // The disclosure is still expanded and Docling is still selected.
    await waitFor(() =>
      expect(screen.getByLabelText('Parser engine').textContent).toContain('Docling')
    )

    // Submitting sends the preserved override to the backend.
    await user.click(screen.getByRole('button', { name: 'Next' }))
    await user.click(screen.getByRole('button', { name: 'Submit' }))

    await waitFor(() =>
      expect(H.createMutateAsync).toHaveBeenCalledWith(
        expect.objectContaining({ processing_overrides: { parser_engine: 'docling' } })
      )
    )
  }, 15000)
})
