'use client'

import { StructureGraphView } from './StructureGraphView'

interface StructureViewerProps {
  sourceId: string
  /** Wired to the workspace's setActiveChunkId so a node click highlights the
   * matching chunk's bbox in the middle pane (Phase I.F, AC5). */
  onSelectNode?: (chunkId: string) => void
  selectedChunkId?: string | null
}

/**
 * Structure tab of the inspect workspace. The I.D-4 placeholder is replaced
 * (Phase I.F) with the real Sigma.js document-structure graph.
 */
export function StructureViewer({
  sourceId,
  onSelectNode,
  selectedChunkId,
}: StructureViewerProps) {
  return (
    <StructureGraphView
      sourceId={sourceId}
      onSelectNode={onSelectNode}
      selectedChunkId={selectedChunkId}
    />
  )
}
