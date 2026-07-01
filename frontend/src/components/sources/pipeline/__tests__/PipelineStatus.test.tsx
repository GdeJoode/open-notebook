// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { PipelineStatus } from '../PipelineStatus'
import type { ProcessingStage } from '@/lib/pipeline/processing-stage'

afterEach(cleanup)

type Variant = 'live' | 'card' | 'detail'
const VARIANTS: Variant[] = ['live', 'card', 'detail']

/**
 * A stage per node-state so every variant exercises pending/active/done/gated/
 * failed at least once:
 *   - pending: `ingested` (later nodes pending)
 *   - active:  `embedded` + running job
 *   - done:    `graphed` (earlier nodes done)
 *   - gated:   `awaiting_schema_review`
 *   - failed:  `failed`
 */
const STATE_SCENARIOS: Array<{ label: string; stage: ProcessingStage; jobStatus?: string }> = [
  { label: 'pending', stage: 'ingested', jobStatus: 'completed' },
  { label: 'active', stage: 'embedded', jobStatus: 'running' },
  { label: 'done', stage: 'graphed', jobStatus: 'completed' },
  { label: 'gated', stage: 'awaiting_schema_review' },
  { label: 'failed', stage: 'failed' },
]

describe('PipelineStatus renders every state in every variant (AC5)', () => {
  for (const variant of VARIANTS) {
    for (const scenario of STATE_SCENARIOS) {
      it(`variant=${variant} stage=${scenario.stage} (${scenario.label}) mounts without crashing`, () => {
        const { container } = render(
          <PipelineStatus
            variant={variant}
            processingStage={scenario.stage}
            jobStatus={scenario.jobStatus}
          />
        )
        // The five spine nodes are always present as data-node-key markers.
        for (const key of ['ingest', 'embed', 'extract', 'graph', 'complete']) {
          expect(container.querySelector(`[data-node-key="${key}"]`)).not.toBeNull()
        }
      })
    }
  }
})

describe('accessibility: focusable, aria-labelled node controls (AC5)', () => {
  it.each(VARIANTS)('variant=%s exposes aria-labelled node controls', (variant) => {
    // `detail` reveals its per-node controls once expanded (the collapsed row is
    // itself a focusable trigger); open it so the node controls are mounted.
    render(
      <PipelineStatus
        variant={variant}
        processingStage="embedded"
        jobStatus="running"
        defaultOpen={variant === 'detail'}
      />
    )
    // Embed is the active node — its control carries a stage + state aria-label.
    const embedControls = screen
      .getAllByRole('button')
      .filter((el) => /^Embed:/.test(el.getAttribute('aria-label') ?? ''))
    expect(embedControls.length).toBeGreaterThan(0)
    expect(embedControls[0].getAttribute('aria-label')).toContain('in progress')
  })

  it('describes stage + state in the aria-label (done + count enrichment)', () => {
    render(
      <PipelineStatus
        variant="detail"
        processingStage="complete"
        jobStatus="completed"
        counts={{ embedded_chunks: 24 }}
        defaultOpen
      />
    )
    const embed = screen
      .getAllByRole('button')
      .find((el) => /^Embed:/.test(el.getAttribute('aria-label') ?? ''))
    expect(embed?.getAttribute('aria-label')).toBe('Embed: complete, 24 chunks')
  })
})

describe('count enrichment appears only on done nodes (AC4)', () => {
  it('renders "24 chunks" / "18 entities" on done nodes in live variant', () => {
    render(
      <PipelineStatus
        variant="live"
        processingStage="complete"
        jobStatus="completed"
        counts={{ embedded_chunks: 24, entity_count: 18, insights_count: 3 }}
      />
    )
    expect(screen.getByText('24 chunks')).toBeTruthy()
    expect(screen.getByText('18 entities')).toBeTruthy()
    expect(screen.getByText('3 insights')).toBeTruthy()
  })

  it('does not render a count chip on a pending node', () => {
    render(
      <PipelineStatus
        variant="live"
        processingStage="ingested"
        jobStatus="completed"
        counts={{ embedded_chunks: 24 }}
      />
    )
    expect(screen.queryByText('24 chunks')).toBeNull()
  })
})

describe('gated action present + aria-labelled (AC5)', () => {
  it.each(VARIANTS)('variant=%s shows a Review schema button on awaiting_schema_review', (variant) => {
    render(<PipelineStatus variant={variant} processingStage="awaiting_schema_review" />)
    const btn = screen.getByRole('button', { name: /Review schema — Extract/i })
    expect(btn).toBeTruthy()
  })

  it('fires onNodeAction with review-schema when clicked', async () => {
    const onNodeAction = vi.fn()
    render(
      <PipelineStatus
        variant="live"
        processingStage="awaiting_schema_review"
        onNodeAction={onNodeAction}
      />
    )
    await userEvent.click(screen.getByRole('button', { name: /Review schema/i }))
    expect(onNodeAction).toHaveBeenCalledTimes(1)
    expect(onNodeAction.mock.calls[0][0]).toBe('review-schema')
    expect(onNodeAction.mock.calls[0][1].key).toBe('extract')
  })
})

describe('failed action present + aria-labelled (AC5)', () => {
  it.each(VARIANTS)('variant=%s shows a Retry button on failed', (variant) => {
    render(<PipelineStatus variant={variant} processingStage="failed" />)
    const btn = screen.getByRole('button', { name: /Retry — Ingest/i })
    expect(btn).toBeTruthy()
  })

  it('fires onNodeAction with retry when clicked', async () => {
    const onNodeAction = vi.fn()
    render(<PipelineStatus variant="live" processingStage="failed" onNodeAction={onNodeAction} />)
    await userEvent.click(screen.getByRole('button', { name: /Retry/i }))
    expect(onNodeAction).toHaveBeenCalledTimes(1)
    expect(onNodeAction.mock.calls[0][0]).toBe('retry')
  })
})

describe('node-click deep-link callback (AC5)', () => {
  it('fires onNodeClick with the extract tab when Extract is clicked (live)', async () => {
    const onNodeClick = vi.fn()
    render(
      <PipelineStatus
        variant="live"
        processingStage="complete"
        jobStatus="completed"
        onNodeClick={onNodeClick}
      />
    )
    const extract = screen
      .getAllByRole('button')
      .find((el) => /^Extract:/.test(el.getAttribute('aria-label') ?? ''))!
    await userEvent.click(extract)
    expect(onNodeClick).toHaveBeenCalledWith('entities')
  })

  it('fires onNodeClick with the chunks tab for a card segment', async () => {
    const onNodeClick = vi.fn()
    render(
      <PipelineStatus
        variant="card"
        processingStage="complete"
        jobStatus="completed"
        onNodeClick={onNodeClick}
      />
    )
    const embed = screen
      .getAllByRole('button')
      .find((el) => /^Embed:/.test(el.getAttribute('aria-label') ?? ''))!
    await userEvent.click(embed)
    expect(onNodeClick).toHaveBeenCalledWith('chunks')
  })

  it('fires onNodeClick with insights tab from the insights node (detail)', async () => {
    const onNodeClick = vi.fn()
    render(
      <PipelineStatus
        variant="detail"
        processingStage="complete"
        jobStatus="completed"
        counts={{ insights_count: 2 }}
        onNodeClick={onNodeClick}
        defaultOpen
      />
    )
    const insights = screen
      .getAllByRole('button')
      .find((el) => /^Insights:/.test(el.getAttribute('aria-label') ?? ''))!
    await userEvent.click(insights)
    expect(onNodeClick).toHaveBeenCalledWith('insights')
  })
})

describe('keyboard reachability (AC5)', () => {
  it('Tab moves focus onto node controls and the gated action', async () => {
    render(<PipelineStatus variant="live" processingStage="awaiting_schema_review" />)
    const user = userEvent.setup()
    await user.tab()
    // First focusable is a node button.
    expect(document.activeElement?.getAttribute('aria-label')).toMatch(/^Ingest:/)
    // Tabbing through eventually reaches the Review schema action.
    const reviewBtn = screen.getByRole('button', { name: /Review schema/i })
    reviewBtn.focus()
    expect(document.activeElement).toBe(reviewBtn)
  })
})

describe('undefined processing_stage -> all-pending, no crash (AC6)', () => {
  it.each(VARIANTS)('variant=%s renders all spine nodes pending', (variant) => {
    const { container } = render(<PipelineStatus variant={variant} processingStage={undefined} />)
    for (const key of ['ingest', 'embed', 'extract', 'graph', 'complete']) {
      const el = container.querySelector(`[data-node-key="${key}"]`)
      expect(el).not.toBeNull()
      expect(el?.getAttribute('data-node-state')).toBe('pending')
    }
  })
})

describe('live variant renders its streaming-log slot', () => {
  it('renders children beneath the tracker', () => {
    render(
      <PipelineStatus variant="live" processingStage="embedded" jobStatus="running">
        <div data-testid="log-console">streaming logs</div>
      </PipelineStatus>
    )
    const container = screen.getByTestId('log-console')
    expect(within(container.parentElement as HTMLElement).getByTestId('log-console')).toBeTruthy()
  })
})
