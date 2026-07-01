import { describe, it, expect } from 'vitest'
import type { ProcessingStage } from './processing-stage'
import {
  derivePipelineNodes,
  SPINE_NODES,
  type PipelineJobStatus,
  type PipelineNode,
  type PipelineNodeKey,
  type PipelineNodeState,
} from './pipeline-stages'

/** Convenience: index the returned nodes by key. */
function byKey(nodes: PipelineNode[]): Record<PipelineNodeKey, PipelineNode> {
  return Object.fromEntries(nodes.map((n) => [n.key, n])) as Record<
    PipelineNodeKey,
    PipelineNode
  >
}

/** Spine nodes only (Insights filtered out), in returned order. */
function spineOf(nodes: PipelineNode[]): PipelineNode[] {
  return nodes.filter((n) => !n.parallel)
}

const SPINE_KEYS: PipelineNodeKey[] = ['ingest', 'embed', 'extract', 'graph', 'complete']

describe('SPINE_NODES ordering (AC1)', () => {
  it('is Ingest -> Embed -> Extract -> Graph -> Complete (Embed before Extract)', () => {
    expect(SPINE_NODES.map((n) => n.key)).toEqual(SPINE_KEYS)
    const embedIdx = SPINE_NODES.findIndex((n) => n.key === 'embed')
    const extractIdx = SPINE_NODES.findIndex((n) => n.key === 'extract')
    expect(embedIdx).toBeLessThan(extractIdx)
  })
})

describe('derivePipelineNodes structure (AC1)', () => {
  it('returns 5 spine nodes in runtime order + Insights flagged parallel (not inline)', () => {
    const nodes = derivePipelineNodes({ processingStage: 'ingested' })
    expect(spineOf(nodes).map((n) => n.key)).toEqual(SPINE_KEYS)

    const insights = nodes.find((n) => n.key === 'insights')
    expect(insights).toBeDefined()
    expect(insights?.parallel).toBe(true)
    // Insights must not appear inside the spine sequence.
    expect(SPINE_KEYS).not.toContain('insights')
    expect(spineOf(nodes).some((n) => n.key === 'insights')).toBe(false)
  })

  it('assigns deep-link tabs per node', () => {
    const nodes = byKey(derivePipelineNodes({ processingStage: 'complete' }))
    expect(nodes.ingest.deepLinkTab).toBe('chunks')
    expect(nodes.embed.deepLinkTab).toBe('chunks')
    expect(nodes.extract.deepLinkTab).toBe('entities')
    expect(nodes.graph.deepLinkTab).toBe('entities')
    expect(nodes.complete.deepLinkTab).toBeUndefined()
    expect(nodes.insights.deepLinkTab).toBe('insights')
  })
})

describe('exhaustive stage -> state mapping, job terminal (AC2)', () => {
  // With a terminal/absent job, the current node resolves to `done`.
  const cases: Array<{
    stage: ProcessingStage
    expect: Partial<Record<PipelineNodeKey, PipelineNodeState>>
  }> = [
    {
      stage: 'ingested',
      expect: { ingest: 'done', embed: 'pending', extract: 'pending', graph: 'pending', complete: 'pending' },
    },
    {
      stage: 'embedded',
      expect: { ingest: 'done', embed: 'done', extract: 'pending', graph: 'pending', complete: 'pending' },
    },
    {
      stage: 'extracted',
      expect: { ingest: 'done', embed: 'done', extract: 'done', graph: 'pending', complete: 'pending' },
    },
    {
      stage: 'awaiting_schema_review',
      expect: { ingest: 'done', embed: 'done', extract: 'gated', graph: 'pending', complete: 'pending' },
    },
    {
      stage: 'graphed',
      expect: { ingest: 'done', embed: 'done', extract: 'done', graph: 'done', complete: 'pending' },
    },
    {
      stage: 'complete',
      expect: { ingest: 'done', embed: 'done', extract: 'done', graph: 'done', complete: 'done' },
    },
    {
      stage: 'failed',
      expect: { ingest: 'failed', embed: 'pending', extract: 'pending', graph: 'pending', complete: 'pending' },
    },
  ]

  it.each(cases)('$stage maps correctly (job completed)', ({ stage, expect: exp }) => {
    const nodes = byKey(derivePipelineNodes({ processingStage: stage, jobStatus: 'completed' }))
    for (const [key, state] of Object.entries(exp)) {
      expect(nodes[key as PipelineNodeKey].state).toBe(state)
    }
  })

  it('awaiting_schema_review exposes a review-schema action on Extract', () => {
    const nodes = byKey(derivePipelineNodes({ processingStage: 'awaiting_schema_review' }))
    expect(nodes.extract.state).toBe('gated')
    expect(nodes.extract.action).toBe('review-schema')
  })

  it('failed exposes a retry action on the failed node', () => {
    const nodes = byKey(derivePipelineNodes({ processingStage: 'failed' }))
    expect(nodes.ingest.state).toBe('failed')
    expect(nodes.ingest.action).toBe('retry')
  })
})

describe('current node active vs done driven ONLY by jobStatus (AC3)', () => {
  const activeJobs: PipelineJobStatus[] = ['new', 'queued', 'running']
  const resolvedJobs: Array<PipelineJobStatus | undefined> = ['completed', undefined]

  // Flip ONLY jobStatus for a fixed stage and observe active <-> done.
  it.each(['ingested', 'embedded', 'extracted', 'graphed'] as ProcessingStage[])(
    'stage %s: current node is active while a job runs, done once resolved',
    (stage) => {
      const currentKey = SPINE_NODES[
        { ingested: 0, embedded: 1, extracted: 2, graphed: 3 }[
          stage as 'ingested' | 'embedded' | 'extracted' | 'graphed'
        ]
      ].key

      for (const job of activeJobs) {
        const nodes = byKey(derivePipelineNodes({ processingStage: stage, jobStatus: job }))
        expect(nodes[currentKey].state).toBe('active')
      }
      for (const job of resolvedJobs) {
        const nodes = byKey(derivePipelineNodes({ processingStage: stage, jobStatus: job }))
        expect(nodes[currentKey].state).toBe('done')
      }
    }
  )

  it('does not treat earlier nodes as active regardless of jobStatus', () => {
    const nodes = byKey(derivePipelineNodes({ processingStage: 'extracted', jobStatus: 'running' }))
    expect(nodes.ingest.state).toBe('done')
    expect(nodes.embed.state).toBe('done')
    expect(nodes.extract.state).toBe('active')
  })

  it('a mid-flight job failure localises to the current node (pre-overwrite window)', () => {
    const nodes = byKey(derivePipelineNodes({ processingStage: 'embedded', jobStatus: 'failed' }))
    expect(nodes.ingest.state).toBe('done')
    expect(nodes.embed.state).toBe('failed')
    expect(nodes.embed.action).toBe('retry')
  })

  it('gate is unaffected by an active job (paused_for_review)', () => {
    const nodes = byKey(
      derivePipelineNodes({ processingStage: 'awaiting_schema_review', jobStatus: 'paused_for_review' })
    )
    expect(nodes.extract.state).toBe('gated')
  })
})

describe('counts enrich done nodes only, never change state (AC4)', () => {
  it('graphed with entity_count=0 still renders Extract done (count 0 no state change)', () => {
    const nodes = byKey(
      derivePipelineNodes({
        processingStage: 'graphed',
        jobStatus: 'completed',
        counts: { entity_count: 0, embedded_chunks: 0 },
      })
    )
    expect(nodes.extract.state).toBe('done')
    expect(nodes.extract.count).toBe(0)
    expect(nodes.extract.countLabel).toBeUndefined() // no misleading "0 entities"
    expect(nodes.embed.state).toBe('done')
    expect(nodes.embed.count).toBe(0)
  })

  it('populates count labels on done nodes when counts are positive', () => {
    const nodes = byKey(
      derivePipelineNodes({
        processingStage: 'complete',
        jobStatus: 'completed',
        counts: { embedded_chunks: 24, entity_count: 18, insights_count: 3, graph_present: true },
      })
    )
    expect(nodes.embed.countLabel).toBe('24 chunks')
    expect(nodes.extract.countLabel).toBe('18 entities')
    expect(nodes.graph.countLabel).toBe('linked')
    expect(nodes.insights.countLabel).toBe('3 insights')
  })

  it('does not attach counts to pending nodes', () => {
    const nodes = byKey(
      derivePipelineNodes({
        processingStage: 'ingested',
        jobStatus: 'completed',
        counts: { embedded_chunks: 24, entity_count: 18 },
      })
    )
    expect(nodes.embed.state).toBe('pending')
    expect(nodes.embed.count).toBeUndefined()
    expect(nodes.extract.count).toBeUndefined()
  })
})

describe('Insights parallel node (never affects the spine)', () => {
  it('is done when insights_count > 0 irrespective of stage', () => {
    const nodes = byKey(
      derivePipelineNodes({ processingStage: 'embedded', counts: { insights_count: 5 } })
    )
    expect(nodes.insights.state).toBe('done')
    expect(nodes.insights.count).toBe(5)
    // Spine unaffected.
    expect(nodes.extract.state).toBe('pending')
  })

  it('is active when past Embed with a running job and no insights yet', () => {
    const nodes = byKey(
      derivePipelineNodes({ processingStage: 'extracted', jobStatus: 'running', counts: { insights_count: 0 } })
    )
    expect(nodes.insights.state).toBe('active')
  })

  it('is pending before Embed even with a running job', () => {
    const nodes = byKey(derivePipelineNodes({ processingStage: 'ingested', jobStatus: 'running' }))
    expect(nodes.insights.state).toBe('pending')
  })

  it('is pending at complete when no insights were produced', () => {
    const nodes = byKey(
      derivePipelineNodes({ processingStage: 'complete', jobStatus: 'completed', counts: { insights_count: 0 } })
    )
    expect(nodes.insights.state).toBe('pending')
  })
})

describe('undefined processing_stage -> all-pending spine (AC6)', () => {
  it('renders every spine node pending with no crash', () => {
    const nodes = derivePipelineNodes({ processingStage: undefined })
    for (const key of SPINE_KEYS) {
      expect(byKey(nodes)[key].state).toBe('pending')
    }
  })

  it('all-pending even with a running job and stray counts (no inference)', () => {
    const nodes = byKey(
      derivePipelineNodes({
        processingStage: undefined,
        jobStatus: 'running',
        counts: { embedded_chunks: 99, entity_count: 42 },
      })
    )
    for (const key of SPINE_KEYS) {
      expect(nodes[key].state).toBe('pending')
    }
  })
})
