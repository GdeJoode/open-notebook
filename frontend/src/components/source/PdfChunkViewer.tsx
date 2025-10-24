'use client'

import React, { useState, useMemo } from 'react'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Loader2, FileText, AlertCircle } from 'lucide-react'

interface Chunk {
  id: string
  text: string
  order: number
  physical_page: number
  printed_page: number | null
  chapter: string | null
  paragraph_number: number | null
  element_type: string
  positions: number[][]
  metadata: Record<string, any>
}

interface PdfChunkViewerProps {
  pdfUrl: string
  chunks: Chunk[]
  sourceId: string
}

export function PdfChunkViewer({ pdfUrl, chunks, sourceId }: PdfChunkViewerProps) {
  const [selectedChunkIndex, setSelectedChunkIndex] = useState<number>(0)
  const [pdfError, setPdfError] = useState<string | null>(null)

  // For now, we'll show a simplified view since react-pdf-highlighter requires complex setup
  // This is a placeholder that can be enhanced with the actual PDF highlighting library

  const selectedChunk = chunks[selectedChunkIndex]

  if (chunks.length === 0) {
    return (
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          No chunks available for this source. Chunks are extracted automatically when documents are processed with Docling.
        </AlertDescription>
      </Alert>
    )
  }

  const hasPositions = selectedChunk?.positions && selectedChunk.positions.length > 0

  return (
    <div className="flex h-[600px] border rounded-lg overflow-hidden">
      {/* Left Pane - Chunk List */}
      <div className="w-1/3 border-r bg-muted/30">
        <div className="p-4 border-b bg-background">
          <h3 className="font-semibold text-sm">
            Document Chunks ({chunks.length})
          </h3>
          <p className="text-xs text-muted-foreground mt-1">
            Click a chunk to view its location in the document
          </p>
        </div>
        <ScrollArea className="h-[calc(600px-80px)]">
          <div className="p-2 space-y-2">
            {chunks.map((chunk, idx) => (
              <div
                key={chunk.id || idx}
                onClick={() => setSelectedChunkIndex(idx)}
                className={`
                  p-3 rounded-md cursor-pointer border transition-all
                  ${
                    idx === selectedChunkIndex
                      ? 'border-primary bg-primary/10 shadow-sm'
                      : 'border-border hover:border-primary/50 hover:bg-muted/50'
                  }
                `}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-medium text-muted-foreground">
                    {chunk.element_type}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    #{idx + 1}
                  </span>
                </div>
                <div className="text-sm line-clamp-3 mb-2">
                  {chunk.text}
                </div>
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  {chunk.chapter && (
                    <span className="truncate max-w-[150px]" title={chunk.chapter}>
                      📖 {chunk.chapter}
                    </span>
                  )}
                  {chunk.positions && chunk.positions.length > 0 && (
                    <span>
                      📍 Page {chunk.physical_page + 1}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* Right Pane - PDF Viewer / Chunk Details */}
      <div className="flex-1 bg-background">
        <div className="p-4 border-b">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-sm">
              Chunk Details
            </h3>
            {selectedChunk && (
              <div className="text-xs text-muted-foreground">
                Page {selectedChunk.physical_page + 1}
                {selectedChunk.printed_page && selectedChunk.printed_page !== selectedChunk.physical_page + 1 && (
                  <span> (Printed: {selectedChunk.printed_page})</span>
                )}
              </div>
            )}
          </div>
        </div>

        <ScrollArea className="h-[calc(600px-80px)]">
          <div className="p-6">
            {selectedChunk ? (
              <div className="space-y-4">
                {/* Chunk Metadata */}
                <div className="grid grid-cols-2 gap-3 p-4 bg-muted/30 rounded-lg text-sm">
                  <div>
                    <span className="font-medium text-muted-foreground">Type:</span>
                    <p className="mt-1">{selectedChunk.element_type}</p>
                  </div>
                  <div>
                    <span className="font-medium text-muted-foreground">Position:</span>
                    <p className="mt-1">Chunk {selectedChunk.order + 1} of {chunks.length}</p>
                  </div>
                  {selectedChunk.chapter && (
                    <div className="col-span-2">
                      <span className="font-medium text-muted-foreground">Chapter:</span>
                      <p className="mt-1">{selectedChunk.chapter}</p>
                    </div>
                  )}
                  {hasPositions && (
                    <div className="col-span-2">
                      <span className="font-medium text-muted-foreground">Locations:</span>
                      <p className="mt-1 text-xs">
                        {selectedChunk.positions.length} bounding box{selectedChunk.positions.length > 1 ? 'es' : ''} on page {selectedChunk.physical_page + 1}
                      </p>
                    </div>
                  )}
                </div>

                {/* Chunk Text */}
                <div>
                  <h4 className="font-medium text-sm text-muted-foreground mb-2">Content:</h4>
                  <div className="p-4 bg-muted/30 rounded-lg whitespace-pre-wrap text-sm">
                    {selectedChunk.text}
                  </div>
                </div>

                {/* PDF Placeholder */}
                {hasPositions ? (
                  <div className="p-6 border-2 border-dashed rounded-lg text-center">
                    <FileText className="h-12 w-12 mx-auto mb-3 text-muted-foreground/50" />
                    <p className="text-sm text-muted-foreground mb-2">
                      PDF Visualization Coming Soon
                    </p>
                    <p className="text-xs text-muted-foreground max-w-md mx-auto">
                      Interactive PDF highlighting with bounding boxes will be enabled once react-pdf-highlighter is fully configured.
                      Bounding box data is available and ready for visualization.
                    </p>
                  </div>
                ) : (
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription className="text-xs">
                      No spatial position data available for this chunk.
                    </AlertDescription>
                  </Alert>
                )}

                {/* Debug Info (can be removed later) */}
                {process.env.NODE_ENV === 'development' && hasPositions && (
                  <details className="text-xs">
                    <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                      Debug: Bounding Box Data
                    </summary>
                    <pre className="mt-2 p-3 bg-muted rounded text-[10px] overflow-auto">
                      {JSON.stringify(selectedChunk.positions, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground">
                <p>Select a chunk to view details</p>
              </div>
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  )
}
