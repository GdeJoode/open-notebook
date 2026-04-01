'use client'

import { useEffect, useRef, useMemo } from 'react'
import Graph from 'graphology'
import Sigma from 'sigma'
import { Share2 } from 'lucide-react'
import type { ExtractedEntity, ExtractedRelation } from '@/lib/types/api'

// Color palette for entity types (same as SigmaGraphView)
const TYPE_COLORS = [
  '#6366f1', '#ec4899', '#f59e0b', '#10b981', '#3b82f6',
  '#8b5cf6', '#ef4444', '#14b8a6', '#f97316', '#06b6d4',
  '#84cc16', '#e879f9', '#22d3ee', '#a3e635', '#fb923c',
]

interface EntityGraphViewProps {
  entities: ExtractedEntity[]
  relations: ExtractedRelation[]
}

export function EntityGraphView({ entities, relations }: EntityGraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sigmaRef = useRef<Sigma | null>(null)

  // Build color map from unique entity labels
  const typeColorMap = useMemo(() => {
    const types = [...new Set(entities.map((e) => e.label))]
    return new Map(types.map((t, i) => [t, TYPE_COLORS[i % TYPE_COLORS.length]]))
  }, [entities])

  useEffect(() => {
    if (!containerRef.current || entities.length === 0) return

    // Clean up previous instance
    if (sigmaRef.current) {
      sigmaRef.current.kill()
      sigmaRef.current = null
    }

    const graph = new Graph()

    // Add entity nodes with random initial positions
    const scale = Math.sqrt(entities.length) * 10
    for (const entity of entities) {
      const nodeId = entity.text
      if (graph.hasNode(nodeId)) continue

      const x = (Math.random() - 0.5) * scale
      const y = (Math.random() - 0.5) * scale
      const color = typeColorMap.get(entity.label) ?? '#6366f1'

      graph.addNode(nodeId, {
        x,
        y,
        size: 6,
        color,
        label: entity.text,
      })
    }

    // Add relation edges
    for (const rel of relations) {
      const src = rel.source_entity
      const tgt = rel.target_entity
      if (
        graph.hasNode(src) &&
        graph.hasNode(tgt) &&
        src !== tgt
      ) {
        try {
          graph.addEdge(src, tgt, {
            label: rel.relation_type,
            size: 1,
            color: '#94a3b8',
          })
        } catch {
          // Skip duplicate edges
        }
      }
    }

    const sigma = new Sigma(graph, containerRef.current, {
      labelRenderedSizeThreshold: 8,
      defaultEdgeColor: '#94a3b8',
      defaultEdgeType: 'line',
      labelColor: { color: '#e2e8f0' },
      labelSize: 12,
      labelFont: 'system-ui, sans-serif',
    })

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
  }, [entities, relations, typeColorMap])

  if (entities.length === 0) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <div className="text-center space-y-2">
          <Share2 className="h-8 w-8 mx-auto text-muted-foreground/50" />
          <p className="text-sm text-muted-foreground">No entities to visualize.</p>
        </div>
      </div>
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
        {entities.length} nodes, {relations.length} edges
      </div>
    </div>
  )
}
