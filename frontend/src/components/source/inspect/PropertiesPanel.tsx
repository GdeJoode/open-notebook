'use client'

import { Badge } from '@/components/ui/badge'
import type { InspectChunk } from './ChunkListPanel'
import { ChunkActionsToolbar } from './ChunkActionsToolbar'

interface PropertiesPanelProps {
  sourceId: string
  activeChunk: InspectChunk | null
  chunks: InspectChunk[]
  pageCount: number
}

/**
 * Properties tab of the inspect workspace right pane (I.B + I.D-4): metadata
 * for the currently selected chunk plus a document summary. When no chunk is
 * selected it shows a hint rather than an empty void (zero-state, AC).
 *
 * The Docling reprocess panel moved to its own "Config" tab in I.D-4; the tab
 * strip in DocumentInspectWorkspace now provides the section heading, so this
 * component no longer renders its own header.
 */
export function PropertiesPanel({
  sourceId,
  activeChunk,
  chunks,
  pageCount,
}: PropertiesPanelProps) {
  const firstPos = activeChunk?.positions?.[0]

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-card">
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
                  <div className="flex items-start justify-between gap-2">
                    <dt className="flex-shrink-0 text-muted-foreground">Bbox (0–1)</dt>
                    <dd className="mono-num min-w-0 break-words text-right">
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
      </div>
    </div>
  )
}
