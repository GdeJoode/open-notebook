'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import { Group, Panel, Separator, type Layout } from 'react-resizable-panels'
import { ArrowLeft } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { PdfChunkViewer } from '@/components/source/PdfChunkViewer'
import { ChunkListPanel, type InspectChunk } from './ChunkListPanel'
import { PropertiesPanel } from './PropertiesPanel'
import { useSourceChunks } from '@/lib/hooks/use-sources'
import { sourcesApi } from '@/lib/api/sources'
import {
  DEFAULT_PANEL_SIZES,
  MAX_PANEL_PCT,
  MIN_PANEL_PCT,
  useDocumentWorkspaceStore,
} from '@/lib/stores/document-workspace-store'

interface DocumentInspectWorkspaceProps {
  sourceId: string
  onBack?: () => void
}

/**
 * Three-pane resizable document inspection workspace (Phase I.B):
 *   left   — virtualized chunk list (ChunkListPanel)
 *   middle — PDF page + bbox overlay (PdfChunkViewer in fullscreen mode)
 *   right  — active-chunk properties + pipeline config (PropertiesPanel)
 *
 * Panel sizes are read from / written to the persisted document-workspace
 * store so a user's layout survives navigation and reload (AC3). Selection is
 * lifted here so the chunk list and the PDF overlay stay in sync.
 */
export function DocumentInspectWorkspace({
  sourceId,
  onBack,
}: DocumentInspectWorkspaceProps) {
  const { data: chunksData, isLoading } = useSourceChunks(sourceId)
  const panelSizes = useDocumentWorkspaceStore((s) => s.panelSizes)
  const setPanelSizes = useDocumentWorkspaceStore((s) => s.setPanelSizes)
  const activeChunkId = useDocumentWorkspaceStore((s) => s.activeChunkId)
  const setActiveChunkId = useDocumentWorkspaceStore((s) => s.setActiveChunkId)

  const [pageCount, setPageCount] = useState(0)

  const chunks = useMemo<InspectChunk[]>(
    () => (chunksData?.chunks ?? []) as InspectChunk[],
    [chunksData]
  )

  const activeChunk = useMemo(
    () => chunks.find((c) => c.id === activeChunkId) ?? null,
    [chunks, activeChunkId]
  )

  // Page count drives the Properties summary; loaded once per source.
  useEffect(() => {
    let cancelled = false
    sourcesApi
      .getPageCount(sourceId)
      .then((d) => {
        if (!cancelled) setPageCount(d.page_count)
      })
      .catch(() => {
        /* page count is non-critical; PdfChunkViewer surfaces PDF errors */
      })
    return () => {
      cancelled = true
    }
  }, [sourceId])

  // react-resizable-panels v4 keys layouts by panel id. Seed the default
  // layout from the persisted store once on mount (defaultLayout is
  // uncontrolled — subsequent live drags are owned by the Group and flow back
  // to the store via onLayoutChanged).
  const defaultLayout = useMemo<Layout>(
    () => ({
      'inspect-left': panelSizes.left,
      'inspect-middle': panelSizes.middle,
      'inspect-right': panelSizes.right,
    }),
    // Read once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  )

  const handleLayoutChanged = useCallback(
    (layout: Layout) => {
      const left = layout['inspect-left']
      const middle = layout['inspect-middle']
      const right = layout['inspect-right']
      if (left == null || middle == null || right == null) return
      setPanelSizes({ left, middle, right })
    },
    [setPanelSizes]
  )

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex flex-shrink-0 items-center gap-2 border-b px-4 py-2">
        {onBack && (
          <Button variant="ghost" size="sm" onClick={onBack}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
        )}
        <h1 className="text-sm font-semibold">Inspect document</h1>
      </div>

      {isLoading ? (
        <div className="flex flex-1 items-center justify-center">
          <LoadingSpinner />
        </div>
      ) : chunks.length === 0 ? (
        <div className="flex flex-1 items-center justify-center p-8">
          <Alert className="max-w-md">
            <AlertTitle>No chunks available</AlertTitle>
            <AlertDescription>
              This source has no extracted chunk data to inspect. Reprocess the
              document with Docling to enable the inspect workspace.
            </AlertDescription>
          </Alert>
        </div>
      ) : (
        <Group
          orientation="horizontal"
          defaultLayout={defaultLayout}
          onLayoutChanged={handleLayoutChanged}
          className="flex-1"
        >
          <Panel
            id="inspect-left"
            minSize={MIN_PANEL_PCT}
            maxSize={MAX_PANEL_PCT}
            defaultSize={DEFAULT_PANEL_SIZES.left}
          >
            <div
              role="region"
              aria-label="Chunk list"
              className="h-full overflow-hidden border-r"
            >
              <ChunkListPanel
                chunks={chunks}
                selectedChunkId={activeChunkId}
                onSelectChunk={setActiveChunkId}
              />
            </div>
          </Panel>

          <Separator
            className="w-1.5 bg-border transition-colors data-[separator=hover]:bg-primary/40 data-[separator=focus]:bg-primary/60 data-[separator=active]:bg-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label="Resize chunk list"
          />

          <Panel
            id="inspect-middle"
            minSize={MIN_PANEL_PCT}
            maxSize={MAX_PANEL_PCT}
            defaultSize={DEFAULT_PANEL_SIZES.middle}
          >
            <div
              role="region"
              aria-label="PDF preview"
              className="h-full overflow-hidden"
            >
              <PdfChunkViewer
                sourceId={sourceId}
                chunks={chunksData!.chunks}
                mode="fullscreen"
                selectedChunkId={activeChunkId}
                onSelectChunkId={setActiveChunkId}
              />
            </div>
          </Panel>

          <Separator
            className="w-1.5 bg-border transition-colors data-[separator=hover]:bg-primary/40 data-[separator=focus]:bg-primary/60 data-[separator=active]:bg-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label="Resize properties panel"
          />

          <Panel
            id="inspect-right"
            minSize={MIN_PANEL_PCT}
            maxSize={MAX_PANEL_PCT}
            defaultSize={DEFAULT_PANEL_SIZES.right}
          >
            <div
              role="region"
              aria-label="Properties"
              className="h-full overflow-hidden border-l"
            >
              <PropertiesPanel
                sourceId={sourceId}
                activeChunk={activeChunk}
                pageCount={pageCount}
              />
            </div>
          </Panel>
        </Group>
      )}
    </div>
  )
}
