import { describe, expect, it } from 'vitest'

import type { DocumentGraphEdge } from '@/lib/api/knowledge-graph'
import {
  buildBipartiteGraph,
  buildCollapsedGraph,
  summarizeGraph,
  type SourceTitleMap,
} from './document-graph'

/**
 * Fixture modelled on the U.1 edge inventory: the four convenanten share the
 * "Regio Deal" programme + "Regio" area, forming a K-clique once collapsed.
 * Two of them additionally share "brede welvaart". The two papers carry 0
 * active entities — they are isolated documents injected from the source list.
 */
const DOC_A = 'source:hjrb1cjvzv86oyaojblj' // Noord-Holland Noord
const DOC_B = 'source:dndibxmjveoxk7tfqfsl' // Midden-Limburg
const PAPER = 'source:bc6xax5piw8cntg5nomc' // Ali paper — isolated

const REGIO_DEAL = 'entity:regio-deal'
const BREDE_WELVAART = 'entity:brede-welvaart'

const edges: DocumentGraphEdge[] = [
  {
    id: 'mentions:1',
    source: DOC_A,
    target: REGIO_DEAL,
    weight: 1.336,
    concept_name: 'Regio Deal',
    concept_type: 'programme',
    document_frequency: 4,
  },
  {
    id: 'mentions:2',
    source: DOC_B,
    target: REGIO_DEAL,
    weight: 1.336,
    concept_name: 'Regio Deal',
    concept_type: 'programme',
    document_frequency: 4,
  },
  {
    id: 'mentions:3',
    source: DOC_A,
    target: BREDE_WELVAART,
    weight: 0.2,
    concept_name: 'brede welvaart',
    concept_type: 'topic',
    document_frequency: 4,
  },
  {
    id: 'mentions:4',
    source: DOC_B,
    target: BREDE_WELVAART,
    weight: 0.2,
    concept_name: 'brede welvaart',
    concept_type: 'topic',
    document_frequency: 4,
  },
]

const titles: SourceTitleMap = new Map([
  [DOC_A, 'Convenant Regio Deal Noord-Holland Noord'],
  [DOC_B, 'Convenant Regio Deal Midden-Limburg'],
  [PAPER, 'Cohesion Policy & Institutional Quality'],
])

describe('buildBipartiteGraph', () => {
  it('emits document nodes, concept nodes, and one mentions link per edge', () => {
    const model = buildBipartiteGraph(edges, titles)
    const docs = model.nodes.filter((n) => n.kind === 'document')
    const concepts = model.nodes.filter((n) => n.kind === 'concept')

    expect(docs.map((d) => d.id).sort()).toEqual([DOC_A, DOC_B].sort())
    expect(concepts.map((c) => c.id).sort()).toEqual(
      [REGIO_DEAL, BREDE_WELVAART].sort(),
    )
    expect(model.links).toHaveLength(4)
  })

  it('labels document nodes by title and concept nodes by concept_name', () => {
    const model = buildBipartiteGraph(edges, titles)
    const docA = model.nodes.find((n) => n.id === DOC_A)
    const regio = model.nodes.find((n) => n.id === REGIO_DEAL)
    expect(docA?.label).toBe('Convenant Regio Deal Noord-Holland Noord')
    expect(regio?.label).toBe('Regio Deal')
    expect(regio?.conceptType).toBe('programme')
  })

  it('carries the shared concept name as the per-edge why', () => {
    const model = buildBipartiteGraph(edges, titles)
    const link = model.links.find((l) => l.id === 'mentions:1')
    expect(link?.concepts).toEqual(['Regio Deal'])
    expect(link?.weight).toBeCloseTo(1.336)
  })

  it('injects isolated documents as standalone nodes (not dropped)', () => {
    const model = buildBipartiteGraph(edges, titles, [PAPER])
    const paper = model.nodes.find((n) => n.id === PAPER)
    expect(paper).toBeDefined()
    expect(paper?.kind).toBe('document')
    expect(paper?.degree).toBe(0)
  })

  it('falls back to a short id when a title is missing', () => {
    const model = buildBipartiteGraph(edges, new Map())
    const docA = model.nodes.find((n) => n.id === DOC_A)
    expect(docA?.label).toBe('hjrb1cjv…')
  })
})

describe('buildCollapsedGraph', () => {
  it('links two documents that share a concept (entity layer hidden)', () => {
    const model = buildCollapsedGraph(edges, titles)
    expect(model.nodes.every((n) => n.kind === 'document')).toBe(true)
    expect(model.nodes.map((n) => n.id).sort()).toEqual([DOC_A, DOC_B].sort())
    // A↔B share two concepts → a single aggregated doc↔doc link.
    expect(model.links).toHaveLength(1)
  })

  it('aggregates every shared concept into the doc↔doc link why + weight', () => {
    const model = buildCollapsedGraph(edges, titles)
    const link = model.links[0]
    expect(link.concepts.sort()).toEqual(['Regio Deal', 'brede welvaart'].sort())
    // weight is the summed per-concept contribution → thicker for richer overlap.
    expect(link.weight).toBeGreaterThan(1)
  })

  it('keeps isolated documents as zero-degree nodes', () => {
    const model = buildCollapsedGraph(edges, titles, [PAPER])
    const paper = model.nodes.find((n) => n.id === PAPER)
    expect(paper).toBeDefined()
    expect(paper?.degree).toBe(0)
  })

  it('does not self-link a document that mentions a singleton concept', () => {
    const singleton: DocumentGraphEdge[] = [
      {
        id: 'mentions:solo',
        source: DOC_A,
        target: 'entity:solo',
        weight: 0.5,
        concept_name: 'solo concept',
        concept_type: 'topic',
        document_frequency: 1,
      },
    ]
    const model = buildCollapsedGraph(singleton, titles)
    expect(model.links).toHaveLength(0)
    expect(model.nodes).toHaveLength(1)
  })
})

describe('summarizeGraph', () => {
  it('counts documents, concepts, links and isolated documents', () => {
    const model = buildBipartiteGraph(edges, titles, [PAPER])
    const summary = summarizeGraph(model)
    expect(summary.documentCount).toBe(3)
    expect(summary.conceptCount).toBe(2)
    expect(summary.linkCount).toBe(4)
    expect(summary.isolatedDocuments.map((d) => d.id)).toEqual([PAPER])
  })

  it('reports an empty graph without throwing', () => {
    const summary = summarizeGraph(buildBipartiteGraph([], new Map()))
    expect(summary).toEqual({
      documentCount: 0,
      conceptCount: 0,
      linkCount: 0,
      isolatedDocuments: [],
    })
  })
})
