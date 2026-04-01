'use client'

import { useEffect, useRef, useMemo, useState, useCallback } from 'react'
import Graph from 'graphology'
import Sigma from 'sigma'
import forceAtlas2 from 'graphology-layout-forceatlas2'
import { Share2, Filter } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import type { ExtractedEntity, ExtractedRelation } from '@/lib/types/api'

const TYPE_COLORS = [
  '#6366f1', '#ec4899', '#f59e0b', '#10b981', '#3b82f6',
  '#8b5cf6', '#ef4444', '#14b8a6', '#f97316', '#06b6d4',
  '#84cc16', '#e879f9', '#22d3ee', '#a3e635', '#fb923c',
]

interface EntityGraphViewProps {
  entities: ExtractedEntity[]
  relations: ExtractedRelation[]
  mergeGroups?: string[][]
}

interface SelectedNode {
  text: string
  label: string
  confidence: number
  properties: Record<string, unknown>
  connections: number
}

export function EntityGraphView({ entities, relations, mergeGroups }: EntityGraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sigmaRef = useRef<Sigma | null>(null)
  const [selectedNode, setSelectedNode] = useState<SelectedNode | null>(null)
  const [hiddenTypes, setHiddenTypes] = useState<Set<string>>(new Set())

  // Build color map from unique entity labels
  const typeColorMap = useMemo(() => {
    const types = [...new Set(entities.map((e) => e.label))]
    return new Map(types.map((t, i) => [t, TYPE_COLORS[i % TYPE_COLORS.length]]))
  }, [entities])

  // Build merged entity lookup
  const mergedLookup = useMemo(() => {
    const lookup = new Set<string>()
    if (mergeGroups) {
      for (const group of mergeGroups) {
        for (const text of group) {
          lookup.add(text)
        }
      }
    }
    return lookup
  }, [mergeGroups])

  // Filter entities by hidden types
  const visibleEntities = useMemo(
    () => entities.filter((e) => !hiddenTypes.has(e.label)),
    [entities, hiddenTypes]
  )

  const visibleEntityTexts = useMemo(
    () => new Set(visibleEntities.map((e) => e.text)),
    [visibleEntities]
  )

  const visibleRelations = useMemo(
    () => relations.filter(
      (r) => visibleEntityTexts.has(r.source_entity) && visibleEntityTexts.has(r.target_entity)
    ),
    [relations, visibleEntityTexts]
  )

  const toggleType = useCallback((type: string) => {
    setHiddenTypes((prev) => {
      const next = new Set(prev)
      if (next.has(type)) {
        next.delete(type)
      } else {
        next.add(type)
      }
      return next
    })
  }, [])

  useEffect(() => {
    if (!containerRef.current || visibleEntities.length === 0) {
      if (sigmaRef.current) {
        sigmaRef.current.kill()
        sigmaRef.current = null
      }
      return
    }

    // Clean up previous instance
    if (sigmaRef.current) {
      sigmaRef.current.kill()
      sigmaRef.current = null
    }

    const graph = new Graph()

    // Add entity nodes with random initial positions
    const scale = Math.sqrt(visibleEntities.length) * 10
    for (const entity of visibleEntities) {
      const nodeId = entity.text
      if (graph.hasNode(nodeId)) continue

      const x = (Math.random() - 0.5) * scale
      const y = (Math.random() - 0.5) * scale
      const color = typeColorMap.get(entity.label) ?? '#6366f1'
      const isMerged = mergedLookup.has(entity.text)

      // Size based on confidence
      const size = Math.max(4, Math.min(14, entity.confidence * 12))

      graph.addNode(nodeId, {
        x,
        y,
        size,
        color,
        label: entity.text,
        // Store metadata for click handler
        entityLabel: entity.label,
        entityConfidence: entity.confidence,
        entityProperties: entity.properties,
        isMerged,
        borderColor: isMerged ? '#fbbf24' : undefined,
      })
    }

    // Add relation edges
    for (const rel of visibleRelations) {
      const src = rel.source_entity
      const tgt = rel.target_entity
      if (graph.hasNode(src) && graph.hasNode(tgt) && src !== tgt) {
        try {
          const edgeSize = Math.max(0.5, rel.confidence * 3)
          graph.addEdge(src, tgt, {
            label: rel.relation_type,
            size: edgeSize,
            color: rel.relation_type === 'RELATED' ? '#fbbf24' : '#94a3b8',
          })
        } catch {
          // Skip duplicate edges
        }
      }
    }

    // Run ForceAtlas2 layout
    if (graph.order > 1) {
      const settings = forceAtlas2.inferSettings(graph)
      forceAtlas2.assign(graph, {
        iterations: 100,
        settings: { ...settings, gravity: 1 },
      })
    }

    const sigma = new Sigma(graph, containerRef.current, {
      labelRenderedSizeThreshold: 6,
      defaultEdgeColor: '#94a3b8',
      defaultEdgeType: 'line',
      labelColor: { color: '#e2e8f0' },
      labelSize: 11,
      labelFont: 'system-ui, sans-serif',
      renderEdgeLabels: graph.order < 100,
    })

    sigma.on('enterNode', () => {
      document.body.style.cursor = 'pointer'
    })
    sigma.on('leaveNode', () => {
      document.body.style.cursor = 'default'
    })

    sigma.on('clickNode', ({ node }) => {
      const attrs = graph.getNodeAttributes(node)
      const connections = graph.degree(node)
      setSelectedNode({
        text: node,
        label: attrs.entityLabel || 'Unknown',
        confidence: attrs.entityConfidence || 0,
        properties: attrs.entityProperties || {},
        connections,
      })
    })

    sigma.on('clickStage', () => {
      setSelectedNode(null)
    })

    sigmaRef.current = sigma

    return () => {
      sigma.kill()
      sigmaRef.current = null
      document.body.style.cursor = 'default'
    }
  }, [visibleEntities, visibleRelations, typeColorMap, mergedLookup])

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

      {/* Type filter legend */}
      <div className="absolute bottom-3 left-3 bg-background/90 backdrop-blur rounded-md p-3 border text-xs space-y-1 max-h-48 overflow-y-auto">
        <div className="flex items-center gap-1 mb-2 text-muted-foreground">
          <Filter className="h-3 w-3" />
          <span className="font-medium">Entity Types</span>
        </div>
        {Array.from(typeColorMap.entries()).map(([type, color]) => (
          <button
            key={type}
            onClick={() => toggleType(type)}
            className={`flex items-center gap-2 w-full text-left hover:bg-muted/50 rounded px-1 py-0.5 transition-opacity ${
              hiddenTypes.has(type) ? 'opacity-30' : ''
            }`}
          >
            <div
              className="w-3 h-3 rounded-full flex-shrink-0"
              style={{ backgroundColor: color }}
            />
            <span className="text-muted-foreground">{type}</span>
          </button>
        ))}
      </div>

      {/* Stats */}
      <div className="absolute top-3 right-3 text-xs text-muted-foreground bg-background/90 backdrop-blur rounded-md px-2 py-1 border">
        {visibleEntities.length} nodes, {visibleRelations.length} edges
        {hiddenTypes.size > 0 && (
          <span className="ml-1">({entities.length - visibleEntities.length} hidden)</span>
        )}
      </div>

      {/* Selected node detail */}
      {selectedNode && (
        <div className="absolute top-3 left-3 bg-background/95 backdrop-blur rounded-md p-3 border text-xs space-y-2 max-w-64">
          <div className="font-medium text-sm">{selectedNode.text}</div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="text-xs">
              {selectedNode.label}
            </Badge>
            <span className="text-muted-foreground">
              {(selectedNode.confidence * 100).toFixed(0)}% confidence
            </span>
          </div>
          <div className="text-muted-foreground">
            {selectedNode.connections} connection{selectedNode.connections !== 1 ? 's' : ''}
          </div>
          {mergedLookup.has(selectedNode.text) && (
            <Badge variant="outline" className="text-xs border-yellow-500 text-yellow-500">
              Merged entity
            </Badge>
          )}
        </div>
      )}
    </div>
  )
}
