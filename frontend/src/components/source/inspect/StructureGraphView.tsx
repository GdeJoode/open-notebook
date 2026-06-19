'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import Graph from 'graphology'
import Sigma from 'sigma'
import { useQuery } from '@tanstack/react-query'
import { Network } from 'lucide-react'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { sourcesApi } from '@/lib/api/sources'
import { getElementColor } from '@/lib/constants/element-colors'

interface StructureGraphViewProps {
  sourceId: string
  /**
   * Called when a node mapped to a chunk is clicked. The inspect workspace wires
   * this to `setActiveChunkId`, which drives the PdfChunkViewer bbox highlight in
   * the middle pane (AC5). Section nodes carry no chunk and are ignored.
   */
  onSelectNode?: (chunkId: string) => void
  /** Currently-active chunk id, so the matching node renders highlighted. */
  selectedChunkId?: string | null
}

interface SelectedNodeInfo {
  elementType: string
  text: string | null
  page: number | null
  hasChunk: boolean
}

/**
 * Structure tab graph (Phase I.F): renders the document's structural graph
 * (section tree via `parent_of`, reading order via `next_node`) with Sigma.js +
 * graphology — the same stack as EntityGraphView, no Cytoscape dependency.
 *
 * Clicking a leaf node that maps to a chunk highlights that chunk's bbox in the
 * middle pane via `onSelectNode`.
 */
export function StructureGraphView({
  sourceId,
  onSelectNode,
  selectedChunkId,
}: StructureGraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sigmaRef = useRef<Sigma | null>(null)
  const graphRef = useRef<Graph | null>(null)
  const [selected, setSelected] = useState<SelectedNodeInfo | null>(null)

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['structure-graph', sourceId],
    queryFn: () => sourcesApi.getStructureGraph(sourceId),
    enabled: !!sourceId,
    staleTime: 60 * 1000,
  })

  // Map doc_node id -> its owning chunk id (leaf nodes only) so click → chunk.
  const chunkByNode = useMemo(() => {
    const map = new Map<string, string>()
    for (const node of data?.nodes ?? []) {
      if (node.chunk_id) map.set(node.id, node.chunk_id)
    }
    return map
  }, [data])

  useEffect(() => {
    if (!containerRef.current || !data || data.nodes.length === 0) {
      if (sigmaRef.current) {
        sigmaRef.current.kill()
        sigmaRef.current = null
      }
      return
    }

    if (sigmaRef.current) {
      sigmaRef.current.kill()
      sigmaRef.current = null
    }

    const graph = new Graph()

    // Tree layout: x by reading-order sequence, y by nesting level so sections
    // sit above their children. Deterministic (no random jitter) so the same
    // document always renders the same shape.
    const total = data.nodes.length
    data.nodes.forEach((node, i) => {
      if (graph.hasNode(node.id)) return
      const seq = node.sequence ?? i
      const level = node.level ?? 0
      // Base color only — the active-chunk highlight is applied in a separate
      // lightweight effect (setNodeAttribute + refresh) so selecting a node
      // never rebuilds the whole WebGL graph.
      graph.addNode(node.id, {
        x: (seq / Math.max(1, total)) * 100,
        y: -level * 8,
        size: node.element_type === 'section_header' ? 8 : 5,
        color: getElementColor(node.element_type ?? 'paragraph'),
        label: node.text ? node.text.slice(0, 40) : node.element_type ?? 'node',
        elementType: node.element_type ?? 'unknown',
        nodeText: node.text,
        nodePage: node.page,
      })
    })

    for (const edge of data.edges) {
      if (edge.type === 'derived_from') continue // chunk endpoint isn't a graph node
      if (
        graph.hasNode(edge.source) &&
        graph.hasNode(edge.target) &&
        edge.source !== edge.target
      ) {
        try {
          graph.addEdge(edge.source, edge.target, {
            size: edge.type === 'parent_of' ? 1.5 : 0.8,
            color: edge.type === 'parent_of' ? '#94a3b8' : '#cbd5e1',
            type: edge.type === 'next_node' ? 'arrow' : 'line',
          })
        } catch {
          // ignore duplicate edges
        }
      }
    }

    const sigma = new Sigma(graph, containerRef.current, {
      labelRenderedSizeThreshold: 7,
      defaultEdgeColor: '#94a3b8',
      labelColor: { color: '#e2e8f0' },
      labelSize: 11,
      labelFont: 'system-ui, sans-serif',
      renderEdgeLabels: false,
    })

    sigma.on('enterNode', () => {
      document.body.style.cursor = 'pointer'
    })
    sigma.on('leaveNode', () => {
      document.body.style.cursor = 'default'
    })
    sigma.on('clickNode', ({ node }) => {
      const attrs = graph.getNodeAttributes(node)
      const chunkId = chunkByNode.get(node)
      setSelected({
        elementType: attrs.elementType as string,
        text: (attrs.nodeText as string | null) ?? null,
        page: (attrs.nodePage as number | null) ?? null,
        hasChunk: !!chunkId,
      })
      if (chunkId && onSelectNode) onSelectNode(chunkId)
    })
    sigma.on('clickStage', () => setSelected(null))

    sigmaRef.current = sigma
    graphRef.current = graph

    return () => {
      sigma.kill()
      sigmaRef.current = null
      graphRef.current = null
      document.body.style.cursor = 'default'
    }
  }, [data, chunkByNode, onSelectNode])

  // Apply the active-chunk highlight in place — recolor nodes via
  // setNodeAttribute + refresh instead of rebuilding the Sigma instance, so a
  // click costs O(nodes) attribute writes, not a full WebGL teardown.
  useEffect(() => {
    const graph = graphRef.current
    const sigma = sigmaRef.current
    if (!graph || !sigma) return
    graph.forEachNode((nodeId, attrs) => {
      const chunkId = chunkByNode.get(nodeId)
      const highlighted = !!chunkId && chunkId === selectedChunkId
      graph.setNodeAttribute(
        nodeId,
        'color',
        highlighted
          ? '#f97316'
          : getElementColor((attrs.elementType as string) ?? 'paragraph')
      )
    })
    sigma.refresh()
  }, [selectedChunkId, data, chunkByNode])

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <LoadingSpinner />
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <Alert variant="destructive" className="max-w-md">
          <AlertTitle>Could not load structure graph</AlertTitle>
          <AlertDescription>
            {error instanceof Error ? error.message : 'Unexpected error.'}
          </AlertDescription>
        </Alert>
      </div>
    )
  }

  if (!data || data.nodes.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center">
        <Network className="h-8 w-8 text-muted-foreground/50" />
        <div>
          <p className="text-sm font-medium">No structure graph</p>
          <p className="mt-1 text-xs text-muted-foreground">
            This source has no structural data yet. Reprocess the document to
            build its structure graph.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="relative h-full w-full overflow-hidden rounded-lg border bg-background">
      <div ref={containerRef} className="h-full w-full" />

      {/* Stats */}
      <div className="absolute right-3 top-3 rounded-md border bg-background/90 px-2.5 py-1.5 text-xs text-muted-foreground backdrop-blur">
        {data.total_nodes} nodes
        {data.truncated && <span className="ml-1">(truncated)</span>}
      </div>

      {/* Selected node detail */}
      {selected && (
        <div className="absolute left-3 top-3 max-w-64 space-y-2 rounded-md border bg-background/95 p-3 text-xs backdrop-blur">
          <Badge variant="secondary" className="text-xs">
            {selected.elementType}
          </Badge>
          {selected.text && (
            <div className="text-muted-foreground">
              {selected.text.slice(0, 160)}
              {selected.text.length > 160 ? '…' : ''}
            </div>
          )}
          {selected.page != null && (
            <div className="text-muted-foreground">Page {selected.page + 1}</div>
          )}
          {selected.hasChunk && (
            <div className="text-[10px] text-primary">
              Highlighted in the PDF pane
            </div>
          )}
        </div>
      )}
    </div>
  )
}
