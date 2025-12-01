'use client'

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Loader2, AlertCircle } from 'lucide-react'
import dynamic from 'next/dynamic'
import { v4 as uuid } from 'uuid'

// Dynamically import react-pdf-highlighter components to prevent SSR issues
// The library uses pdfjs-dist which requires browser APIs
const PdfLoader = dynamic(
  () => import('react-pdf-highlighter').then((mod) => mod.PdfLoader),
  { ssr: false }
)
const PdfHighlighter = dynamic(
  () => import('react-pdf-highlighter').then((mod) => mod.PdfHighlighter),
  { ssr: false }
)
const Highlight = dynamic(
  () => import('react-pdf-highlighter').then((mod) => mod.Highlight),
  { ssr: false }
)
const AreaHighlight = dynamic(
  () => import('react-pdf-highlighter').then((mod) => mod.AreaHighlight),
  { ssr: false }
)
const Popup = dynamic(
  () => import('react-pdf-highlighter').then((mod) => mod.Popup),
  { ssr: false }
)

// Import the IHighlight type (types don't need dynamic import)
import type { IHighlight } from 'react-pdf-highlighter'

// Custom styles for PDF viewer and highlights
// Following ragflow's pattern for container and highlight styling
const pdfViewerStyles = `
  /* Container must properly constrain the PdfHighlighter */
  .pdf-viewer-container {
    width: 100%;
    height: 100%;
    position: relative;
    overflow: hidden;
    /* Ensure the container creates a proper stacking context */
    isolation: isolate;
  }

  /* The PdfLoader creates a wrapper div that needs sizing */
  .pdf-viewer-container > div {
    width: 100% !important;
    height: 100% !important;
    position: relative !important;
  }

  /* Fix PdfHighlighter positioning within container */
  .pdf-viewer-container .PdfHighlighter {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    width: 100% !important;
    height: 100% !important;
    overflow: auto !important;
    overflow-x: hidden !important;
  }

  /* Ensure pdfViewer container inside PdfHighlighter is properly sized */
  .pdf-viewer-container .pdfViewer {
    padding-top: 0 !important;
    padding-bottom: 10px !important;
  }

  /* Show clear page breaks between pages */
  .pdf-viewer-container .page {
    margin-bottom: 20px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
    border: 1px solid #ccc !important;
    position: relative !important;
  }

  /* Add page number indicator on each page */
  .pdf-viewer-container .page::after {
    content: "Page " attr(data-page-number);
    position: absolute;
    bottom: -18px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 11px;
    color: #666;
    background: #f5f5f5;
    padding: 2px 8px;
    border-radius: 3px;
    z-index: 10;
  }

  /* Ensure highlight layer is visible and properly positioned */
  .PdfHighlighter__highlight-layer {
    position: absolute !important;
    z-index: 3 !important;
    left: 0 !important;
    top: 0 !important;
    pointer-events: auto !important;
  }

  /* Highlight component - must be visible */
  .Highlight {
    position: absolute !important;
  }

  /* Highlight parts container */
  .Highlight__parts {
    opacity: 1 !important;
  }

  /* Highlight styling - make highlights very visible with solid yellow */
  .Highlight__part {
    cursor: pointer !important;
    position: absolute !important;
    background: rgba(255, 226, 143, 1) !important;
    opacity: 1 !important;
    mix-blend-mode: multiply !important;
    transition: background 0.3s !important;
    z-index: 3 !important;
  }

  /* Active/scrolled-to highlight - orange color */
  .Highlight--scrolledTo .Highlight__part {
    background: rgba(255, 140, 0, 1) !important;
  }

  /* Area highlight styling */
  .AreaHighlight {
    border: 2px solid #ff9800 !important;
    background-color: rgba(255, 226, 143, 0.8) !important;
    opacity: 1 !important;
    mix-blend-mode: multiply !important;
  }

  .AreaHighlight__part {
    cursor: pointer !important;
    position: absolute !important;
    background: rgba(255, 226, 143, 1) !important;
    transition: background 0.3s !important;
  }

  .AreaHighlight--scrolledTo .AreaHighlight__part {
    background: rgba(255, 140, 0, 1) !important;
  }

  /* Text layer should allow highlights to show through */
  .textLayer {
    z-index: 2 !important;
    opacity: 1 !important;
    mix-blend-mode: multiply !important;
  }
`

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
  metadata: Record<string, unknown>
}

interface PdfChunkViewerProps {
  pdfUrl: string
  chunks: Chunk[]
}

const HighlightPopup = ({
  comment,
}: {
  comment: { text: string; emoji: string }
}) =>
  comment.text ? (
    <div className="bg-popover p-2 rounded shadow-md border text-sm">
      {comment.emoji} {comment.text}
    </div>
  ) : null

/**
 * Build highlights from chunk positions.
 *
 * Positions format from backend: [page_num, x_left, x_right, y_top, y_bottom]
 * - page_num: 1-based page number from Docling (matches PDF.js 1-based indexing)
 * - All coordinates are NORMALIZED (0-1 range), origin is TOPLEFT.
 *
 * For react-pdf-highlighter (following ragflow's pattern):
 * - x1, y1, x2, y2 = absolute pixel coordinates at scale=1
 * - width/height = page dimensions (used as reference for scaling formula)
 * - Library formula: viewportX = (viewportWidth * x) / boundingRect.width
 */
function buildChunkHighlights(
  chunk: Chunk | null,
  pageSize: { width: number; height: number }
): IHighlight[] {
  if (!chunk?.positions || chunk.positions.length === 0) {
    return []
  }

  return chunk.positions.map((pos, index) => {
    const [pageNumber, xLeft, xRight, yTopRaw, yBottomRaw] = pos

    // Ensure yTop < yBottom (smaller y = higher on page in TOPLEFT coordinates)
    const yTop = Math.min(yTopRaw, yBottomRaw)
    const yBottom = Math.max(yTopRaw, yBottomRaw)

    // Convert normalized (0-1) coordinates to absolute pixel coordinates
    const x1 = xLeft * pageSize.width
    const y1 = yTop * pageSize.height
    const x2 = xRight * pageSize.width
    const y2 = yBottom * pageSize.height

    const boundingRect = {
      x1,
      y1,
      x2,
      y2,
      width: pageSize.width,
      height: pageSize.height,
    }

    return {
      id: `${chunk.id}-${index}`,
      comment: {
        text: '',
        emoji: '',
      },
      content: {
        text: chunk.text?.substring(0, 100) || '',
      },
      position: {
        boundingRect,
        rects: [boundingRect],
        pageNumber: pageNumber, // Docling already uses 1-based page numbers
      },
    }
  })
}

export function PdfChunkViewer({ pdfUrl, chunks }: PdfChunkViewerProps) {
  const [selectedChunkIndex, setSelectedChunkIndex] = useState<number>(0)
  const [pageSize, setPageSize] = useState({ width: 612, height: 792 }) // US Letter default
  const [pdfError, setPdfError] = useState<string | null>(null)
  const [isClient, setIsClient] = useState(false)
  const scrollToRef = useRef<(highlight: IHighlight) => void>(() => {})

  const selectedChunk = chunks[selectedChunkIndex] || null

  // Mark when we're on the client side and inject styles
  useEffect(() => {
    setIsClient(true)

    // Load react-pdf-highlighter CSS via link element (avoids TypeScript issues with CSS imports)
    const linkId = 'react-pdf-highlighter-css'
    if (!document.getElementById(linkId)) {
      const link = document.createElement('link')
      link.id = linkId
      link.rel = 'stylesheet'
      link.href = '/react-pdf-highlighter-style.css'
      document.head.appendChild(link)
    }

    // Inject custom PDF viewer and highlight styles
    const styleId = 'pdf-chunk-viewer-styles'
    if (!document.getElementById(styleId)) {
      const style = document.createElement('style')
      style.id = styleId
      style.textContent = pdfViewerStyles
      document.head.appendChild(style)
    }
  }, [])

  // Build highlights whenever selected chunk or page size changes
  // Following ragflow's pattern: compute highlights in parent, pass to child
  const highlights = useMemo(() => {
    const result = buildChunkHighlights(selectedChunk, pageSize)
    if (selectedChunk?.positions?.length) {
      const numPositions = selectedChunk.positions.length
      // pos format: [pageNumber, xLeft, xRight, yTop, yBottom] - all normalized 0-1
      // Log all positions to understand multi-position chunks
      const positionsSummary = selectedChunk.positions.map((pos, idx) => ({
        idx,
        page: pos[0],
        xLeft: `${(pos[1] * 100).toFixed(1)}%`,
        xRight: `${(pos[2] * 100).toFixed(1)}%`,
        yTop: `${(pos[3] * 100).toFixed(1)}%`,
        yBottom: `${(pos[4] * 100).toFixed(1)}%`,
        heightPct: `${((pos[4] - pos[3]) * 100).toFixed(1)}%`,
      }))

      // Debug: log raw positions for troubleshooting
      console.log(`Chunk ${selectedChunkIndex}: raw positions =`, selectedChunk.positions)
      console.log(`Chunk ${selectedChunkIndex}: positions summary =`, positionsSummary)

      // Find the topmost position (lowest yTop on lowest page)
      const sortedPositions = [...selectedChunk.positions].sort((a, b) => {
        if (a[0] !== b[0]) return a[0] - b[0] // page number
        return a[3] - b[3] // yTop
      })
      const topmostPos = sortedPositions[0]

      console.log(`[PdfChunkViewer] Chunk #${selectedChunkIndex + 1} (order: ${selectedChunk.order})`, {
        text: selectedChunk.text?.substring(0, 40) + '...',
        numPositions,
        positions: positionsSummary,
        // Y coordinates: yTop should be smaller for text higher on page (TOPLEFT)
        // If yTop is ~0.05, text is near page top
        // If yTop is ~0.90, text is near page bottom
        topmostPosition: {
          page: topmostPos[0],
          yTop: `${(topmostPos[3] * 100).toFixed(1)}%`,
          yBottom: `${(topmostPos[4] * 100).toFixed(1)}%`,
        },
        pageSize,
      })
    } else {
      console.log(`[PdfChunkViewer] Chunk #${selectedChunkIndex + 1} has no position data`)
    }
    return result
  }, [selectedChunk, pageSize, selectedChunkIndex])

  // Scroll to the topmost highlight when highlights change
  // For chunks with multiple positions, we want to scroll to the one highest on the page
  // (lowest y1 value since y increases downward in TOPLEFT coordinates)
  useEffect(() => {
    if (highlights.length > 0 && scrollToRef.current) {
      // Find the topmost highlight (lowest y1 on the lowest page number)
      const sortedHighlights = [...highlights].sort((a, b) => {
        // First sort by page number
        const pageA = a.position.pageNumber
        const pageB = b.position.pageNumber
        if (pageA !== pageB) return pageA - pageB
        // Then by y1 (top position) - smaller y1 = higher on page
        return a.position.boundingRect.y1 - b.position.boundingRect.y1
      })
      const topmostHighlight = sortedHighlights[0]

      // Small delay to ensure the PDF viewer has finished rendering
      const timeoutId = setTimeout(() => {
        if (scrollToRef.current) {
          scrollToRef.current(topmostHighlight)
        }
      }, 50)
      return () => clearTimeout(timeoutId)
    }
  }, [highlights])

  // Callback to update page size from PDF document
  const handleSetPageSize = useCallback((width: number, height: number) => {
    setPageSize((prev) => {
      if (prev.width !== width || prev.height !== height) {
        return { width, height }
      }
      return prev
    })
  }, [])

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

  // Check if ANY chunk has positions (to decide whether to show PDF viewer at all)
  const anyChunkHasPositions = chunks.some(c => c.positions && c.positions.length > 0)
  const selectedChunkHasPositions = selectedChunk?.positions && selectedChunk.positions.length > 0

  // Don't render PDF viewer until we're on the client side (SSR safety)
  if (!isClient) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="flex h-full border rounded-lg overflow-hidden bg-background">
      {/* Left Pane - Chunk List */}
      <div className="w-1/3 border-r bg-muted/30 flex flex-col">
        <div className="p-4 border-b bg-background flex-shrink-0">
          <h3 className="font-semibold text-sm">
            Document Chunks ({chunks.length})
          </h3>
          <p className="text-xs text-muted-foreground mt-1">
            Click a chunk to view its location in the PDF
          </p>
        </div>
        <ScrollArea className="flex-1 min-h-0">
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
                      {chunk.chapter}
                    </span>
                  )}
                  {chunk.positions && chunk.positions.length > 0 && (
                    <span>
                      Page {chunk.physical_page}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* Right Pane - PDF Viewer with Highlights */}
      <div className="flex-1 bg-background flex flex-col">
        <div className="p-4 border-b flex-shrink-0">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-sm">
              PDF Viewer
            </h3>
            {selectedChunk && (
              <div className="text-xs text-muted-foreground">
                Page {selectedChunk.physical_page}
                {selectedChunk.printed_page && selectedChunk.printed_page !== selectedChunk.physical_page && (
                  <span> (Printed: {selectedChunk.printed_page})</span>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="flex-1 min-h-0 relative">
          {anyChunkHasPositions ? (
            <div className="absolute inset-0 pdf-viewer-container">
              {/* Overlay message when selected chunk has no positions */}
              {!selectedChunkHasPositions && (
                <div className="absolute inset-0 z-10 flex items-center justify-center bg-background/80">
                  <Alert className="max-w-md">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      This chunk has no spatial position data. Select a chunk with position data to see it highlighted in the PDF.
                    </AlertDescription>
                  </Alert>
                </div>
              )}
              <PdfLoader
                url={pdfUrl}
                beforeLoad={
                  <div className="flex items-center justify-center h-full">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                }
                onError={(error) => {
                  console.error('PDF Loading Error:', error)
                  setPdfError(error?.message || 'Unknown error')
                }}
                errorMessage={
                  <div className="flex items-center justify-center h-full p-4">
                    <Alert variant="destructive" className="max-w-md">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>
                        Failed to load PDF. Please check the console for details.
                        {pdfError && (
                          <span className="text-xs mt-2 block font-mono break-all">Error: {pdfError}</span>
                        )}
                      </AlertDescription>
                    </Alert>
                  </div>
                }
                workerSrc="/pdfjs-dist/pdf.worker.min.js"
              >
                {(pdfDocument) => {
                  // Get page size from first page (following ragflow's pattern)
                  // Safety check: ensure pdfDocument and getPage exist
                  if (pdfDocument && typeof pdfDocument.getPage === 'function') {
                    pdfDocument.getPage(1).then((page) => {
                      const viewport = page.getViewport({ scale: 1 })
                      handleSetPageSize(viewport.width, viewport.height)
                    }).catch((err) => {
                      console.error('Error getting page:', err)
                    })
                  }

                  return (
                    <PdfHighlighter
                      pdfDocument={pdfDocument}
                      pdfScaleValue="auto"
                      enableAreaSelection={() => false}
                      onScrollChange={() => {}}
                      scrollRef={(scrollTo) => {
                        scrollToRef.current = scrollTo
                      }}
                      onSelectionFinished={() => null}
                      highlightTransform={(
                        highlight,
                        index,
                        setTip,
                        hideTip,
                        _viewportToScaled,
                        _screenshot,
                        isScrolledTo
                      ) => {
                        const isTextHighlight = !Boolean(
                          highlight.content && highlight.content.image
                        )

                        const component = isTextHighlight ? (
                          <Highlight
                            isScrolledTo={isScrolledTo}
                            position={highlight.position}
                            comment={highlight.comment}
                          />
                        ) : (
                          <AreaHighlight
                            isScrolledTo={isScrolledTo}
                            highlight={highlight}
                            onChange={() => {}}
                          />
                        )

                        return (
                          <Popup
                            popupContent={<HighlightPopup {...highlight} />}
                            onMouseOver={(popupContent) =>
                              setTip(highlight, () => popupContent)
                            }
                            onMouseOut={hideTip}
                            key={index}
                          >
                            {component}
                          </Popup>
                        )
                      }}
                      highlights={highlights}
                    />
                  )
                }}
              </PdfLoader>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full">
              <Alert className="max-w-md">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  No spatial position data available for this chunk. Position data is extracted
                  automatically when PDFs are processed with Docling.
                </AlertDescription>
              </Alert>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
