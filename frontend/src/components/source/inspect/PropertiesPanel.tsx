'use client'

import { useState } from 'react'
import { toast } from 'sonner'
import { Badge } from '@/components/ui/badge'
import { PipelineConfigPanel } from '@/components/sources/pipeline/PipelineConfigPanel'
import {
  DEFAULT_PIPELINE_CONFIG,
  sourcesApi,
  type DoclingPipelineConfig,
} from '@/lib/api/sources'
import type { InspectChunk } from './ChunkListPanel'
import { ChunkActionsToolbar } from './ChunkActionsToolbar'

interface PropertiesPanelProps {
  sourceId: string
  activeChunk: InspectChunk | null
  chunks: InspectChunk[]
  pageCount: number
}

/**
 * Right pane of the inspect workspace: metadata for the currently selected
 * chunk plus the Docling pipeline-config reprocess panel. When no chunk is
 * selected it shows a hint rather than an empty void (zero-state, AC).
 */
export function PropertiesPanel({
  sourceId,
  activeChunk,
  chunks,
  pageCount,
}: PropertiesPanelProps) {
  const [reprocessConfig, setReprocessConfig] = useState<DoclingPipelineConfig>({
    ...DEFAULT_PIPELINE_CONFIG,
  })
  const [reprocessing, setReprocessing] = useState(false)

  const handleReprocess = async () => {
    try {
      setReprocessing(true)
      await sourcesApi.reprocess(sourceId, reprocessConfig)
      toast.success('Reprocessing started — document will be re-ingested')
    } catch {
      toast.error('Failed to start reprocessing')
    } finally {
      setReprocessing(false)
    }
  }

  const firstPos = activeChunk?.positions?.[0]

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-card">
      <div className="flex-shrink-0 border-b px-3 py-2">
        <h2 className="text-sm font-semibold">Properties</h2>
      </div>

      <div className="flex-1 space-y-4 p-3">
        {/* Active chunk metadata */}
        <section>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Active chunk
          </h3>
          {activeChunk ? (
            <dl className="space-y-2 text-xs">
              <div className="flex items-center justify-between gap-2">
                <dt className="text-muted-foreground">Type</dt>
                <dd>
                  <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
                    {activeChunk.element_type}
                  </Badge>
                </dd>
              </div>
              <div className="flex items-center justify-between gap-2">
                <dt className="text-muted-foreground">Content</dt>
                <dd>{activeChunk.is_content === false ? 'Noise' : 'Content'}</dd>
              </div>
              {firstPos && (
                <>
                  <div className="flex items-center justify-between gap-2">
                    <dt className="text-muted-foreground">Page</dt>
                    <dd className="mono-num">{firstPos[0]}</dd>
                  </div>
                  <div className="flex items-center justify-between gap-2">
                    <dt className="text-muted-foreground">Bbox (0–1)</dt>
                    <dd className="mono-num">
                      {firstPos
                        .slice(1)
                        .map((v) => v.toFixed(3))
                        .join(', ')}
                    </dd>
                  </div>
                </>
              )}
              <div>
                <dt className="mb-1 text-muted-foreground">Text</dt>
                <dd className="max-h-40 overflow-y-auto whitespace-pre-wrap rounded border bg-muted/30 p-2 text-muted-foreground">
                  {activeChunk.text}
                </dd>
              </div>
            </dl>
          ) : (
            <p className="text-xs text-muted-foreground">
              Select a chunk to see its metadata.
            </p>
          )}
          {activeChunk && (
            <div className="mt-3 border-t pt-3">
              <ChunkActionsToolbar
                sourceId={sourceId}
                activeChunk={activeChunk}
                chunks={chunks}
              />
            </div>
          )}
        </section>

        {/* Document summary */}
        <section className="border-t pt-3">
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Document
          </h3>
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Pages</span>
            <span className="mono-num">{pageCount || '—'}</span>
          </div>
        </section>

        {/* Pipeline config / reprocess */}
        <section className="border-t pt-3">
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Pipeline
          </h3>
          <PipelineConfigPanel
            config={reprocessConfig}
            onChange={setReprocessConfig}
            onReprocess={handleReprocess}
            reprocessing={reprocessing}
            collapsible
            defaultOpen={false}
          />
        </section>
      </div>
    </div>
  )
}
