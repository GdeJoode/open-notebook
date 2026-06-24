'use client'

import { useEffect, useRef, useMemo } from 'react'
import Graph from 'graphology'
import forceAtlas2 from 'graphology-layout-forceatlas2'
import Sigma from 'sigma'
import { useGraphData } from '@/lib/hooks/use-knowledge-graph'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { EmptyState } from '@/components/common/EmptyState'
import { Share2 } from 'lucide-react'

// Color palette for entity types
const TYPE_COLORS = [
  '#6366f1', '#ec4899', '#f59e0b', '#10b981', '#3b82f6',
  '#8b5cf6', '#ef4444', '#14b8a6', '#f97316', '#06b6d4',
  '#84cc16', '#e879f9', '#22d3ee', '#a3e635', '#fb923c',
]

interface SigmaGraphViewProps {
  entityTypeFilter?: string
  onNodeClick?: (nodeId: string) => void
}

export default function SigmaGraphView({
  entityTypeFilter,
  onNodeClick,
}: SigmaGraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sigmaRef = useRef<Sigma | null>(null)

  const { data: graphData, isLoading, error } = useGraphData({
    entity_type: entityTypeFilter,
    limit: 5000,
  })

  // Build color map from unique entity types
  const typeColorMap = useMemo(() => {
    if (!graphData?.nodes) return new Map<string, string>()
    const types = [...new Set(graphData.nodes.map((n) => n.entity_type))]
    return new Map(types.map((t, i) => [t, TYPE_COLORS[i % TYPE_COLORS.length]]))
  }, [graphData])

  useEffect(() => {
    if (!containerRef.current || !graphData) return

    // Clean up previous instance
    if (sigmaRef.current) {
      sigmaRef.current.kill()
      sigmaRef.current = null
    }

    const { nodes, edges } = graphData
    if (nodes.length === 0) return

    // Build graphology graph
    const graph = new Graph()

    // Add nodes with random initial positions (Sigma will render them)
    const scale = Math.sqrt(nodes.length) * 10
    for (const node of nodes) {
      const x = (Math.random() - 0.5) * scale
      const y = (Math.random() - 0.5) * scale
      const color = typeColorMap.get(node.entity_type) ?? '#6366f1'
      const size = Math.max(3, Math.min(15, Math.log2((node.weight || 1) + 1) * 3))

      if (!graph.hasNode(node.id)) {
        graph.addNode(node.id, {
          x,
          y,
          size,
          color,
          label: node.label,
        })
      }
    }

    // Add edges
    for (const edge of edges) {
      if (
        graph.hasNode(edge.source) &&
        graph.hasNode(edge.target) &&
        edge.source !== edge.target
      ) {
        try {
          graph.addEdge(edge.source, edge.target, {
            label: edge.label,
            size: 1,
            color: '#94a3b8',
          })
        } catch {
          // Skip duplicate edges
        }
      }
    }

    // Force-directed layout so connected nodes cluster into a readable graph
    // instead of the random scatter. ForceAtlas2 derives its forces from the
    // edges added above (the random x/y are only the starting seed).
    if (graph.order > 1) {
      forceAtlas2.assign(graph, {
        iterations: graph.order > 400 ? 100 : 250,
        settings: {
          ...forceAtlas2.inferSettings(graph),
          barnesHutOptimize: true,
          gravity: 1,
          scalingRatio: 10,
          slowDown: 2,
        },
      })
    }

    // Create Sigma instance
    const sigma = new Sigma(graph, containerRef.current, {
      labelRenderedSizeThreshold: 8,
      defaultEdgeColor: '#94a3b8',
      defaultEdgeType: 'line',
      labelColor: { color: '#1e293b' },
      labelSize: 12,
      labelFont: 'system-ui, sans-serif',
    })

    // Handle node clicks
    sigma.on('clickNode', ({ node }) => {
      onNodeClick?.(node)
    })

    // Hover state
    sigma.on('enterNode', ({ node }) => {
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
  }, [graphData, typeColorMap, onNodeClick])

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <LoadingSpinner />
      </div>
    )
  }

  if (error) {
    return <p className="text-red-500">Failed to load graph data</p>
  }

  if (!graphData || graphData.nodes.length === 0) {
    return (
      <EmptyState
        icon={Share2}
        title="No graph data"
        description="No entities available for visualization."
      />
    )
  }

  return (
    <div className="h-full w-full relative rounded-lg border bg-background overflow-hidden">
      <div ref={containerRef} className="h-full w-full" />

      {/* Legend */}
      <div className="absolute bottom-3 left-3 bg-background/90 backdrop-blur rounded-md p-3 border text-xs space-y-1 max-h-48 overflow-y-auto">
        {Array.from(typeColorMap.entries()).map(([type, color]) => (
          <div key={type} className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full flex-shrink-0"
              style={{ backgroundColor: color }}
            />
            <span className="text-muted-foreground">{type}</span>
          </div>
        ))}
      </div>

      <div className="absolute top-3 right-3 text-xs text-muted-foreground bg-background/90 backdrop-blur rounded-md px-2 py-1 border">
        {graphData.nodes.length} nodes, {graphData.edges.length} edges
      </div>
    </div>
  )
}
