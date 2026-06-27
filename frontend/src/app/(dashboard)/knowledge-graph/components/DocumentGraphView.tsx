'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import Graph from 'graphology'
import forceAtlas2 from 'graphology-layout-forceatlas2'
import Sigma from 'sigma'
import { FileText, Network, RefreshCw, AlertCircle } from 'lucide-react'

import { useAllSources, useDocumentGraph } from '@/lib/hooks/use-knowledge-graph'
import {
  buildBipartiteGraph,
  buildCollapsedGraph,
  summarizeGraph,
  type DocumentGraphModel,
  type SourceTitleMap,
} from '@/lib/knowledge-graph/document-graph'
import { Skeleton } from '@/components/ui/skeleton'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'

// Document nodes are indigo, concept nodes amber. Colour is NOT the only cue:
// document labels are prefixed with a ◆ glyph (a shape cue that survives
// greyscale / colour-blindness) and the accessible summary names each kind.
const DOCUMENT_COLOR = '#6366f1'
const CONCEPT_COLOR = '#f59e0b'
const DOCUMENT_GLYPH = '◆ '

// U.1 named-only preset: collapses to the "Regio Deal / Regio" skeleton.
const NAMED_ONLY_MIN_WEIGHT = 0.3

interface DocumentGraphViewProps {
  onNodeClick?: (nodeId: string) => void
}

function edgeThickness(weight: number): number {
  // weight ranges ~0.2–1.34 (bipartite) and higher once summed (collapsed);
  // map to a legible 1–6px band so overlap richness reads as thickness.
  return Math.max(1, Math.min(6, weight * 3))
}

export default function DocumentGraphView({ onNodeClick }: DocumentGraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sigmaRef = useRef<Sigma | null>(null)

  const [showEntities, setShowEntities] = useState(true)
  // Per-edge weight floor. 0 = full filtered graph; 0.3 = the U.1 named-only
  // skeleton ("Regio Deal" / "Regio" anchors only). Driven by the slider; the
  // preset button snaps to 0.3.
  const [minWeight, setMinWeight] = useState(0)
  const {
    data: graph,
    isLoading: graphLoading,
    error: graphError,
    refetch,
    isRefetching,
  } = useDocumentGraph(minWeight)
  const { data: sources, isLoading: sourcesLoading } = useAllSources()

  const titleMap: SourceTitleMap = useMemo(() => {
    const map = new Map<string, string>()
    for (const s of sources ?? []) {
      if (s.title) map.set(s.id, s.title)
    }
    return map
  }, [sources])

  // Every source id is a candidate document node so isolated sources (the
  // papers with 0 active entities) still render instead of being dropped.
  const allSourceIds = useMemo(
    () => (sources ?? []).map((s) => s.id),
    [sources],
  )

  const model: DocumentGraphModel | null = useMemo(() => {
    if (!graph) return null
    return showEntities
      ? buildBipartiteGraph(graph.edges, titleMap, allSourceIds)
      : buildCollapsedGraph(graph.edges, titleMap, allSourceIds)
  }, [graph, titleMap, allSourceIds, showEntities])

  const summary = useMemo(
    () => (model ? summarizeGraph(model) : null),
    [model],
  )

  // Render the model into Sigma whenever it changes.
  useEffect(() => {
    if (!containerRef.current || !model) return

    if (sigmaRef.current) {
      sigmaRef.current.kill()
      sigmaRef.current = null
    }

    if (model.nodes.length === 0) return

    const g = new Graph()
    const scale = Math.sqrt(model.nodes.length) * 12

    for (const node of model.nodes) {
      if (g.hasNode(node.id)) continue
      const isDoc = node.kind === 'document'
      g.addNode(node.id, {
        x: (Math.random() - 0.5) * scale,
        y: (Math.random() - 0.5) * scale,
        size: isDoc
          ? Math.max(7, Math.min(16, 7 + node.degree))
          : Math.max(4, Math.min(11, 4 + node.degree)),
        color: isDoc ? DOCUMENT_COLOR : CONCEPT_COLOR,
        label: isDoc ? `${DOCUMENT_GLYPH}${node.label}` : node.label,
      })
    }

    for (const link of model.links) {
      if (!g.hasNode(link.source) || !g.hasNode(link.target)) continue
      if (link.source === link.target) continue
      try {
        g.addEdge(link.source, link.target, {
          size: edgeThickness(link.weight),
          color: '#cbd5e1',
          // Tooltip/label "why": the shared concept(s) behind the link.
          label: link.concepts.join(', '),
        })
      } catch {
        // duplicate edge — skip
      }
    }

    if (g.order > 1) {
      forceAtlas2.assign(g, {
        iterations: 200,
        settings: {
          ...forceAtlas2.inferSettings(g),
          barnesHutOptimize: true,
          gravity: 1.5,
          scalingRatio: 12,
          slowDown: 2,
        },
      })
    }

    const sigma = new Sigma(g, containerRef.current, {
      labelRenderedSizeThreshold: 6,
      defaultEdgeColor: '#cbd5e1',
      defaultEdgeType: 'line',
      renderEdgeLabels: true,
      labelColor: { color: '#1e293b' },
      edgeLabelColor: { color: '#64748b' },
      labelSize: 12,
      edgeLabelSize: 10,
      labelFont: 'system-ui, sans-serif',
    })

    sigma.on('clickNode', ({ node }) => onNodeClick?.(node))
    sigma.on('enterNode', () => {
      document.body.style.cursor = 'pointer'
    })
    sigma.on('leaveNode', () => {
      document.body.style.cursor = 'default'
    })

    sigmaRef.current = sigma
    return () => {
      sigma.kill()
      sigmaRef.current = null
      document.body.style.cursor = 'default'
    }
  }, [model, onNodeClick])

  const isLoading = graphLoading || sourcesLoading

  if (isLoading) {
    return (
      <div className="h-full w-full space-y-3" data-testid="document-graph-loading">
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-[calc(100%-3rem)] w-full" />
        <span className="sr-only">Loading the document graph…</span>
      </div>
    )
  }

  if (graphError) {
    return (
      <Alert variant="destructive" data-testid="document-graph-error">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Couldn’t load the document graph</AlertTitle>
        <AlertDescription className="flex items-center gap-3">
          <span>The document↔entity projection failed to load.</span>
          <Button size="sm" variant="outline" onClick={() => refetch()}>
            <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    )
  }

  const hasEdges = (graph?.edges.length ?? 0) > 0

  return (
    <div className="flex h-full flex-col gap-3">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4 rounded-md border bg-muted/30 p-3">
        <label className="flex items-center gap-2 text-sm" htmlFor="show-entities">
          <Switch
            id="show-entities"
            checked={showEntities}
            onCheckedChange={setShowEntities}
            data-testid="entity-layer-toggle"
            aria-label="Show shared-concept (entity) layer"
          />
          <span>Show shared concepts</span>
        </label>

        <div className="flex min-w-[14rem] items-center gap-2 text-sm">
          <label htmlFor="min-weight" className="whitespace-nowrap">
            Min weight: <span className="tabular-nums">{minWeight.toFixed(2)}</span>
          </label>
          <Slider
            id="min-weight"
            className="w-32"
            min={0}
            max={1.4}
            step={0.05}
            value={[minWeight]}
            onValueChange={([v]) => setMinWeight(v)}
            data-testid="min-weight-slider"
            aria-label="Minimum edge weight"
          />
        </div>

        <Button
          size="sm"
          variant={minWeight >= NAMED_ONLY_MIN_WEIGHT ? 'secondary' : 'outline'}
          onClick={() =>
            setMinWeight((w) => (w >= NAMED_ONLY_MIN_WEIGHT ? 0 : NAMED_ONLY_MIN_WEIGHT))
          }
          data-testid="named-only-preset"
          aria-pressed={minWeight >= NAMED_ONLY_MIN_WEIGHT}
        >
          Named-only
        </Button>

        <Button
          size="sm"
          variant="ghost"
          onClick={() => refetch()}
          disabled={isRefetching}
          aria-label="Reload the document graph"
        >
          <RefreshCw className={`mr-1.5 h-3.5 w-3.5 ${isRefetching ? 'animate-spin' : ''}`} />
          Reload
        </Button>
      </div>

      {/* Accessible summary — the WebGL canvas is not screen-reader / keyboard
          navigable, so this text alternative carries the same information and
          is the keyboard-reachable entry point to the graph. */}
      {summary && (
        <div
          className="rounded-md border bg-background p-3 text-sm"
          data-testid="document-graph-summary"
        >
          <p aria-live="polite" id="document-graph-caption">
            <Network className="mr-1.5 inline h-4 w-4 text-muted-foreground" aria-hidden />
            <strong>{summary.documentCount}</strong> document
            {summary.documentCount === 1 ? '' : 's'},{' '}
            <strong>{summary.conceptCount}</strong> shared concept
            {summary.conceptCount === 1 ? '' : 's'}, and{' '}
            <strong>{summary.linkCount}</strong> link
            {summary.linkCount === 1 ? '' : 's'}
            {showEntities
              ? ' (documents linked to the concepts they mention).'
              : ' (documents linked directly via the concepts they share).'}
          </p>
          {summary.isolatedDocuments.length > 0 && (
            <p className="mt-1 text-muted-foreground">
              {summary.isolatedDocuments.length} isolated document
              {summary.isolatedDocuments.length === 1 ? ' shares' : 's share'} no concept:{' '}
              {summary.isolatedDocuments.map((d) => d.label).join(', ')}.
            </p>
          )}
        </div>
      )}

      {!hasEdges ? (
        <div
          className="flex flex-1 flex-col items-center justify-center rounded-lg border bg-background p-8 text-center"
          data-testid="document-graph-empty"
        >
          <FileText className="mb-4 h-12 w-12 text-muted-foreground/60" />
          <h3 className="mb-2 text-lg font-medium">No shared-concept links yet</h3>
          <p className="mb-4 max-w-md text-muted-foreground">
            No <code>mentions</code> edges have been projected, so documents
            aren’t linked via shared concepts yet. Regenerate the document graph
            from the knowledge-graph projection to populate it.
          </p>
          <p className="text-xs text-muted-foreground">
            Run <code>POST /knowledge-graph/document-graph/regenerate</code> on the
            backend, then reload.
          </p>
        </div>
      ) : (
        <div
          className="relative flex-1 overflow-hidden rounded-lg border bg-background"
          role="figure"
          aria-labelledby="document-graph-caption"
          data-testid="document-graph-canvas"
        >
          <div ref={containerRef} className="h-full w-full" />

          {/* Legend — names both the colour AND the ◆ shape cue so the
              document/concept distinction is not colour-only. */}
          <div className="absolute bottom-3 left-3 space-y-1 rounded-md border bg-background/90 p-3 text-xs backdrop-blur">
            <div className="flex items-center gap-2">
              <span className="text-indigo-500" aria-hidden>◆</span>
              <span
                className="h-3 w-3 flex-shrink-0 rounded-full"
                style={{ backgroundColor: DOCUMENT_COLOR }}
                aria-hidden
              />
              <span className="text-muted-foreground">Document (◆)</span>
            </div>
            <div className="flex items-center gap-2">
              <span aria-hidden>●</span>
              <span
                className="h-3 w-3 flex-shrink-0 rounded-full"
                style={{ backgroundColor: CONCEPT_COLOR }}
                aria-hidden
              />
              <span className="text-muted-foreground">Shared concept</span>
            </div>
          </div>
        </div>
      )}

      {/* Text-list alternative to the canvas: every link with its "why". Keeps
          the graph's content reachable without WebGL. */}
      {model && model.links.length > 0 && (
        <details className="rounded-md border bg-background text-sm">
          <summary className="cursor-pointer px-3 py-2 font-medium">
            Links as a list ({model.links.length})
          </summary>
          <ul className="max-h-48 space-y-1 overflow-auto px-3 pb-3" data-testid="document-graph-link-list">
            {model.links.map((link) => {
              const src = model.nodes.find((n) => n.id === link.source)
              const dst = model.nodes.find((n) => n.id === link.target)
              return (
                <li key={link.id} className="text-muted-foreground">
                  <span className="text-foreground">{src?.label ?? link.source}</span>
                  {' ↔ '}
                  <span className="text-foreground">{dst?.label ?? link.target}</span>
                  {' — '}
                  <span>via {link.concepts.join(', ')}</span>
                </li>
              )
            })}
          </ul>
        </details>
      )}
    </div>
  )
}
