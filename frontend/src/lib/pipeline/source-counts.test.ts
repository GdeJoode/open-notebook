import { describe, it, expect } from 'vitest'

import { toPipelineCounts } from './source-counts'
import { derivePipelineNodes } from './pipeline-stages'

/** Minimal count-bearing shape the adapter reads. */
function row(
  partial: Partial<{
    embedded_chunks: number
    entity_count: number
    insights_count: number
    relation_count: number
  }>
) {
  return {
    embedded_chunks: 0,
    entity_count: 0,
    insights_count: 0,
    relation_count: 0,
    ...partial,
  }
}

describe('toPipelineCounts maps 0 => undefined for count-gated stages (AC0)', () => {
  it('collapses the list-default zeros to undefined', () => {
    expect(toPipelineCounts(row({}))).toEqual({
      embedded_chunks: undefined,
      entity_count: undefined,
      insights_count: undefined,
      graph_present: false,
    })
  })

  it('passes positive counts through unchanged', () => {
    expect(
      toPipelineCounts(
        row({ embedded_chunks: 12, entity_count: 4, insights_count: 3 })
      )
    ).toEqual({
      embedded_chunks: 12,
      entity_count: 4,
      insights_count: 3,
      graph_present: false,
    })
  })

  it('derives graph_present from relation_count > 0', () => {
    expect(toPipelineCounts(row({ relation_count: 5 })).graph_present).toBe(true)
    expect(toPipelineCounts(row({ relation_count: 0 })).graph_present).toBe(false)
  })
})

describe('the adapter fixes the failed-path regression end to end (AC0)', () => {
  it('a {0,0} parse failure locates the failure at Ingest (not Embed)', () => {
    const nodes = derivePipelineNodes({
      processingStage: 'failed',
      counts: toPipelineCounts(row({ embedded_chunks: 0, entity_count: 0 })),
    })
    const byKey = Object.fromEntries(nodes.map((n) => [n.key, n.state]))
    expect(byKey.ingest).toBe('failed')
    expect(byKey.embed).toBe('pending')
  })

  it('a {12,0} embedded-then-failed source locates the failure at Extract', () => {
    const nodes = derivePipelineNodes({
      processingStage: 'failed',
      counts: toPipelineCounts(row({ embedded_chunks: 12, entity_count: 0 })),
    })
    const byKey = Object.fromEntries(nodes.map((n) => [n.key, n.state]))
    expect(byKey.ingest).toBe('done')
    expect(byKey.embed).toBe('done')
    expect(byKey.extract).toBe('failed')
  })

  it('passing raw list zeros WOULD mislabel Ingest as done (guards the seam)', () => {
    // This asserts precisely the bug the adapter exists to prevent: feeding the
    // verbatim list default {embedded_chunks:0, entity_count:0} mislocates the
    // failure at Embed because raw 0 !== undefined reads as "Ingest reached".
    const nodes = derivePipelineNodes({
      processingStage: 'failed',
      counts: { embedded_chunks: 0, entity_count: 0 },
    })
    const byKey = Object.fromEntries(nodes.map((n) => [n.key, n.state]))
    expect(byKey.ingest).toBe('done')
    expect(byKey.embed).toBe('failed')
  })
})
