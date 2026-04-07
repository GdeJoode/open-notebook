'use client'

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Loader2,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Eye,
  EyeOff,
} from 'lucide-react'
import { sourcesApi } from '@/lib/api/sources'

// ---------------------------------------------------------------------------
// Element type color scheme (matches Docling Studio)
// ---------------------------------------------------------------------------

const ELEMENT_COLORS: Record<string, string> = {
  title: '#EF4444',
  section_header: '#F97316',
  text: '#3B82F6',
  paragraph: '#3B82F6',
  table: '#8B5CF6',
  picture: '#22C55E',
  list_item: '#06B6D4',
  list: '#06B6D4',
  formula: '#EC4899',
  code: '#14B8A6',
  caption: '#EAB308',
  heading: '#F97316',
  page_header: '#9CA3AF',
  page_footer: '#9CA3AF',
  footnote: '#9CA3AF',
}

const DEFAULT_COLOR = '#6B7280'

function getElementColor(elementType: string): string {
  const key = elementType.toLowerCase().replace(/\s+/g, '_')
  return ELEMENT_COLORS[key] ?? DEFAULT_COLOR
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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

interface PageInfo {
  page_number: number
  width: number   // PDF points
  height: number  // PDF points
}

interface BboxRect {
  x: number
  y: number
  w: number
  h: number
  chunkIndex: number
  elementType: string
  text: string
}

interface PdfChunkViewerProps {
  sourceId: string
  chunks: Chunk[]
}

// ---------------------------------------------------------------------------
// BboxOverlay — canvas overlay matching Docling Studio approach
// ---------------------------------------------------------------------------

interface BboxOverlayProps {
  imgRef: React.RefObject<HTMLImageElement | null>
  pageInfo: PageInfo
  rects: BboxRect[]
  highlightedChunkIndex: number | null
  onHoverChunk: (index: number | null) => void
  onClickChunk: (index: number) => void
  showOverlay: boolean
  hiddenTypes: Set<string>
}

function BboxOverlay({
  imgRef,
  pageInfo,
  rects,
  highlightedChunkIndex,
  onHoverChunk,
  onClickChunk,
  showOverlay,
  hiddenTypes,
}: BboxOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [tooltip, setTooltip] = useState<{
    x: number
    y: number
    type: string
    text: string
    color: string
  } | null>(null)

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img || !showOverlay) return

    const dpr = window.devicePixelRatio || 1
    const displayW = img.clientWidth
    const displayH = img.clientHeight

    canvas.width = displayW * dpr
    canvas.height = displayH * dpr
    canvas.style.width = `${displayW}px`
    canvas.style.height = `${displayH}px`

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, displayW, displayH)

    // Scale: PDF points → display pixels
    const sx = displayW / pageInfo.width
    const sy = displayH / pageInfo.height

    for (const rect of rects) {
      if (hiddenTypes.has(rect.elementType.toLowerCase().replace(/\s+/g, '_'))) continue

      const x = rect.x * sx
      const y = rect.y * sy
      const w = rect.w * sx
      const h = rect.h * sy
      const color = getElementColor(rect.elementType)
      const isHighlighted = rect.chunkIndex === highlightedChunkIndex
      const isDimmed = highlightedChunkIndex !== null && !isHighlighted

      ctx.strokeStyle = isDimmed ? color + '40' : color
      ctx.lineWidth = isHighlighted ? 3 : 2
      ctx.fillStyle = isHighlighted ? color + '40' : isDimmed ? color + '08' : color + '20'

      ctx.beginPath()
      ctx.rect(x, y, w, h)
      ctx.fill()
      ctx.stroke()
    }
  }, [imgRef, pageInfo, rects, highlightedChunkIndex, showOverlay, hiddenTypes])

  // Redraw on any state change, with a small delay to ensure img dimensions are available
  useEffect(() => {
    draw()
    // Retry after a frame in case img dimensions weren't ready
    const raf = requestAnimationFrame(() => draw())
    return () => cancelAnimationFrame(raf)
  }, [draw])

  // ResizeObserver for responsive redraw
  useEffect(() => {
    const img = imgRef.current
    if (!img) return
    const observer = new ResizeObserver(() => draw())
    observer.observe(img)
    return () => observer.disconnect()
  }, [imgRef, draw])

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current
      const img = imgRef.current
      if (!canvas || !img || !showOverlay) return

      const rect = canvas.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top

      const sx = img.clientWidth / pageInfo.width
      const sy = img.clientHeight / pageInfo.height

      // Hit-test against all rects (reverse order for topmost match)
      for (let i = rects.length - 1; i >= 0; i--) {
        const r = rects[i]
        if (hiddenTypes.has(r.elementType.toLowerCase().replace(/\s+/g, '_'))) continue
        const rx = r.x * sx
        const ry = r.y * sy
        const rw = r.w * sx
        const rh = r.h * sy
        if (mx >= rx && mx <= rx + rw && my >= ry && my <= ry + rh) {
          onHoverChunk(r.chunkIndex)
          setTooltip({
            x: Math.min(mx, img.clientWidth - 280),
            y: Math.max(my - 60, 0),
            type: r.elementType,
            text: r.text.substring(0, 150),
            color: getElementColor(r.elementType),
          })
          return
        }
      }
      onHoverChunk(null)
      setTooltip(null)
    },
    [imgRef, pageInfo, rects, showOverlay, hiddenTypes, onHoverChunk]
  )

  const handleMouseLeave = useCallback(() => {
    onHoverChunk(null)
    setTooltip(null)
  }, [onHoverChunk])

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current
      const img = imgRef.current
      if (!canvas || !img) return

      const rect = canvas.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top

      const sx = img.clientWidth / pageInfo.width
      const sy = img.clientHeight / pageInfo.height

      for (let i = rects.length - 1; i >= 0; i--) {
        const r = rects[i]
        if (hiddenTypes.has(r.elementType.toLowerCase().replace(/\s+/g, '_'))) continue
        const rx = r.x * sx
        const ry = r.y * sy
        const rw = r.w * sx
        const rh = r.h * sy
        if (mx >= rx && mx <= rx + rw && my >= ry && my <= ry + rh) {
          onClickChunk(r.chunkIndex)
          return
        }
      }
    },
    [imgRef, pageInfo, rects, hiddenTypes, onClickChunk]
  )

  if (!showOverlay) return null

  return (
    <>
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 cursor-crosshair"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      />
      {tooltip && (
        <div
          className="absolute z-20 max-w-[280px] rounded-md border bg-popover/95 backdrop-blur-sm p-2 shadow-lg pointer-events-none"
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          <div className="flex items-center gap-1.5 mb-1">
            <span
              className="inline-block w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: tooltip.color }}
            />
            <span className="text-xs font-semibold uppercase tracking-wide">
              {tooltip.type}
            </span>
          </div>
          <p className="text-xs text-muted-foreground line-clamp-3">{tooltip.text}</p>
        </div>
      )}
    </>
  )
}

// ---------------------------------------------------------------------------
// Legend bar
// ---------------------------------------------------------------------------

interface LegendBarProps {
  typeCounts: Record<string, number>
  hiddenTypes: Set<string>
  onToggle: (type: string) => void
}

function LegendBar({ typeCounts, hiddenTypes, onToggle }: LegendBarProps) {
  const types = Object.entries(typeCounts).sort((a, b) => b[1] - a[1])
  if (types.length === 0) return null

  return (
    <div className="absolute top-2 left-2 right-2 z-10 flex flex-wrap gap-1.5 rounded-lg bg-background/80 backdrop-blur-md border p-2 pointer-events-auto">
      {types.map(([type, count]) => {
        const key = type.toLowerCase().replace(/\s+/g, '_')
        const hidden = hiddenTypes.has(key)
        const color = getElementColor(type)
        return (
          <button
            key={type}
            onClick={() => onToggle(key)}
            className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-xs border transition-opacity ${
              hidden ? 'opacity-35' : 'opacity-100'
            }`}
          >
            <span
              className="inline-block w-2 h-2 rounded-full"
              style={{ backgroundColor: color }}
            />
            <span>{type}</span>
            <span className="text-muted-foreground">({count})</span>
          </button>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function PdfChunkViewer({ sourceId, chunks }: PdfChunkViewerProps) {
  const [currentPage, setCurrentPage] = useState(1)
  const [pageInput, setPageInput] = useState('1')
  const [pageCount, setPageCount] = useState(0)
  const [pagesInfo, setPagesInfo] = useState<PageInfo[]>([])
  const [previewUrl, setPreviewUrl] = useState<string>('')
  const [imgLoaded, setImgLoaded] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedChunkIndex, setSelectedChunkIndex] = useState<number | null>(null)
  const [hoveredChunkIndex, setHoveredChunkIndex] = useState<number | null>(null)
  const [showOverlay, setShowOverlay] = useState(true)
  const [hiddenTypes, setHiddenTypes] = useState<Set<string>>(new Set())
  const imgRef = useRef<HTMLImageElement>(null)
  const chunkListRef = useRef<HTMLDivElement>(null)

  const highlightedIndex = hoveredChunkIndex ?? selectedChunkIndex

  // Load page count and dimensions on mount
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const data = await sourcesApi.getPageCount(sourceId)
        if (cancelled) return
        setPageCount(data.page_count)
        setPagesInfo(data.pages)

        // Jump to first page that has chunks
        // Determine first page with chunks. Page numbers may be 0-based or 1-based.
        const firstChunkWithPos = chunks.find(c => c.positions?.length > 0)
        let firstPage = 1
        if (firstChunkWithPos?.positions?.length) {
          const posPage = firstChunkWithPos.positions[0][0]
          // If page number is 0, it's 0-indexed → convert to 1-based
          firstPage = posPage === 0 ? 1 : posPage
        }
        setCurrentPage(firstPage)
        setPageInput(String(firstPage))
      } catch (err) {
        if (!cancelled) setError('Failed to load PDF info')
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => { cancelled = true }
  }, [sourceId, chunks])

  // Update preview URL when page changes
  useEffect(() => {
    let cancelled = false
    setImgLoaded(false)
    ;(async () => {
      try {
        const url = await sourcesApi.getPagePreviewUrl(sourceId, currentPage, 150)
        if (!cancelled) setPreviewUrl(url)
      } catch {
        if (!cancelled) setError('Failed to load page preview')
      }
    })()
    return () => { cancelled = true }
  }, [sourceId, currentPage])

  // Get current page info
  const currentPageInfo = useMemo(
    () => pagesInfo.find(p => p.page_number === currentPage) ?? { page_number: currentPage, width: 595, height: 842 },
    [pagesInfo, currentPage]
  )

  // Detect coordinate format: if any position coordinate > 1.0, they're
  // raw PDF points; otherwise they're normalized 0-1.
  const isNormalized = useMemo(() => {
    for (const chunk of chunks) {
      if (!chunk.positions) continue
      for (const pos of chunk.positions) {
        const [, xLeft, xRight, yTop, yBottom] = pos
        if (xLeft > 1.0 || xRight > 1.0 || yTop > 1.0 || yBottom > 1.0) {
          return false
        }
      }
    }
    return true
  }, [chunks])

  // Build bounding box rects for current page (in PDF points, TOPLEFT)
  const pageRects = useMemo(() => {
    const rects: BboxRect[] = []
    const pw = currentPageInfo.width
    const ph = currentPageInfo.height

    for (let ci = 0; ci < chunks.length; ci++) {
      const chunk = chunks[ci]
      if (!chunk.positions) continue
      for (const pos of chunk.positions) {
        const [pageNum, xLeft, xRight, yTopRaw, yBottomRaw] = pos

        // Match page: positions may use 0-based or 1-based page numbers
        if (pageNum !== currentPage && pageNum !== currentPage - 1) continue
        // If 0-based matched, only accept if it didn't also match as 1-based
        if (pageNum === currentPage - 1 && pageNum === currentPage) continue

        const yTop = Math.min(yTopRaw, yBottomRaw)
        const yBottom = Math.max(yTopRaw, yBottomRaw)

        if (isNormalized) {
          // Normalized 0-1 → convert to PDF points
          rects.push({
            x: xLeft * pw,
            y: yTop * ph,
            w: (xRight - xLeft) * pw,
            h: (yBottom - yTop) * ph,
            chunkIndex: ci,
            elementType: chunk.element_type || 'unknown',
            text: chunk.text || '',
          })
        } else {
          // Already in PDF points (from source_processing_service)
          rects.push({
            x: xLeft,
            y: yTop,
            w: xRight - xLeft,
            h: yBottom - yTop,
            chunkIndex: ci,
            elementType: chunk.element_type || 'unknown',
            text: chunk.text || '',
          })
        }
      }
    }
    return rects
  }, [chunks, currentPage, currentPageInfo, isNormalized])

  // Element type counts for legend
  const typeCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const rect of pageRects) {
      counts[rect.elementType] = (counts[rect.elementType] || 0) + 1
    }
    return counts
  }, [pageRects])

  // Chunks visible on current page (for sidebar)
  const pageChunks = useMemo(() => {
    const indices: number[] = []
    const seen = new Set<number>()
    for (const rect of pageRects) {
      if (!seen.has(rect.chunkIndex)) {
        seen.add(rect.chunkIndex)
        indices.push(rect.chunkIndex)
      }
    }
    return indices
  }, [pageRects])

  // Scroll chunk list to highlighted chunk
  useEffect(() => {
    if (highlightedIndex === null || !chunkListRef.current) return
    const el = chunkListRef.current.querySelector(`[data-chunk-index="${highlightedIndex}"]`)
    if (el) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [highlightedIndex])

  // Navigate to page of selected chunk
  const handleSelectChunk = useCallback((index: number) => {
    setSelectedChunkIndex(index)
    const chunk = chunks[index]
    if (chunk?.positions?.length > 0) {
      const chunkPage = Math.max(1, chunk.positions[0][0])
      if (chunkPage !== currentPage) {
        setCurrentPage(chunkPage)
        setPageInput(String(chunkPage))
      }
    }
  }, [chunks, currentPage])

  const handlePageNav = useCallback((delta: number) => {
    const next = Math.max(1, Math.min(pageCount, currentPage + delta))
    setCurrentPage(next)
    setPageInput(String(next))
  }, [currentPage, pageCount])

  const handlePageInputSubmit = useCallback(() => {
    const p = parseInt(pageInput, 10)
    if (!isNaN(p) && p >= 1 && p <= pageCount) {
      setCurrentPage(p)
    } else {
      setPageInput(String(currentPage))
    }
  }, [pageInput, pageCount, currentPage])

  const toggleType = useCallback((type: string) => {
    setHiddenTypes(prev => {
      const next = new Set(prev)
      if (next.has(type)) next.delete(type)
      else next.add(type)
      return next
    })
  }, [])

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    )
  }

  if (chunks.length === 0) {
    return (
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          No chunks available. Chunks are extracted when documents are processed with Docling.
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <div className="flex h-full border rounded-lg overflow-hidden bg-background">
      {/* Left Pane — Chunk List */}
      <div className="w-80 min-w-[280px] border-r bg-muted/30 flex flex-col">
        <div className="p-3 border-b bg-background flex-shrink-0">
          <h3 className="font-semibold text-sm mb-1">
            Elements on Page {currentPage}
          </h3>
          <p className="text-xs text-muted-foreground">
            {pageChunks.length} elements · {chunks.length} total
          </p>
        </div>
        <ScrollArea className="flex-1 min-h-0">
          <div className="p-2 space-y-1.5" ref={chunkListRef}>
            {pageChunks.length === 0 ? (
              <p className="text-xs text-muted-foreground p-3">
                No elements with position data on this page.
              </p>
            ) : (
              pageChunks.map((ci) => {
                const chunk = chunks[ci]
                const color = getElementColor(chunk.element_type)
                const isSelected = ci === selectedChunkIndex
                const isHovered = ci === hoveredChunkIndex
                return (
                  <div
                    key={chunk.id || ci}
                    data-chunk-index={ci}
                    onClick={() => handleSelectChunk(ci)}
                    onMouseEnter={() => setHoveredChunkIndex(ci)}
                    onMouseLeave={() => setHoveredChunkIndex(null)}
                    className={`
                      p-2.5 rounded-md cursor-pointer border transition-all text-sm
                      ${isSelected
                        ? 'border-primary bg-primary/10 shadow-sm'
                        : isHovered
                          ? 'border-primary/50 bg-muted/50'
                          : 'border-transparent hover:border-border'
                      }
                    `}
                    style={isSelected || isHovered ? { borderLeftColor: color, borderLeftWidth: 3 } : undefined}
                  >
                    <div className="flex items-center gap-1.5 mb-1">
                      <span
                        className="inline-block w-2 h-2 rounded-full flex-shrink-0"
                        style={{ backgroundColor: color }}
                      />
                      <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                        {chunk.element_type}
                      </Badge>
                      {chunk.chapter && (
                        <span className="text-[10px] text-muted-foreground truncate ml-auto max-w-[120px]" title={chunk.chapter}>
                          {chunk.chapter}
                        </span>
                      )}
                    </div>
                    <p className="text-xs line-clamp-2 text-muted-foreground">
                      {chunk.text}
                    </p>
                  </div>
                )
              })
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Right Pane — PDF Page + Canvas Overlay */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Page navigation bar */}
        <div className="p-2 border-b flex items-center gap-2 flex-shrink-0 bg-background">
          <Button
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0"
            disabled={currentPage <= 1}
            onClick={() => handlePageNav(-1)}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <div className="flex items-center gap-1 text-sm">
            <Input
              className="w-12 h-7 text-center text-xs p-0"
              value={pageInput}
              onChange={(e) => setPageInput(e.target.value)}
              onBlur={handlePageInputSubmit}
              onKeyDown={(e) => e.key === 'Enter' && handlePageInputSubmit()}
            />
            <span className="text-muted-foreground text-xs">/ {pageCount}</span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0"
            disabled={currentPage >= pageCount}
            onClick={() => handlePageNav(1)}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>

          <div className="ml-auto">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 gap-1 text-xs"
              onClick={() => setShowOverlay(!showOverlay)}
            >
              {showOverlay ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}
              {showOverlay ? 'Hide' : 'Show'} boxes
            </Button>
          </div>
        </div>

        {/* Page viewer */}
        <div className="flex-1 min-h-0 overflow-auto bg-muted/20 flex items-start justify-center p-4">
          <div className="relative inline-block shadow-lg border rounded" style={{ pointerEvents: 'auto' }}>
            {/* Server-rendered page image */}
            {previewUrl && (
              <img
                ref={imgRef}
                src={previewUrl}
                alt={`Page ${currentPage}`}
                className="block max-w-full h-auto"
                style={{ maxHeight: 'calc(100vh - 200px)' }}
                onLoad={() => setImgLoaded(true)}
                onError={() => setError(`Failed to render page ${currentPage}`)}
              />
            )}

            {!imgLoaded && previewUrl && (
              <div className="flex items-center justify-center w-[595px] h-[842px]">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            )}

            {/* Legend */}
            {imgLoaded && showOverlay && (
              <LegendBar typeCounts={typeCounts} hiddenTypes={hiddenTypes} onToggle={toggleType} />
            )}

            {/* Canvas overlay */}
            {imgLoaded && (
              <BboxOverlay
                imgRef={imgRef}
                pageInfo={currentPageInfo}
                rects={pageRects}
                highlightedChunkIndex={highlightedIndex}
                onHoverChunk={setHoveredChunkIndex}
                onClickChunk={handleSelectChunk}
                showOverlay={showOverlay}
                hiddenTypes={hiddenTypes}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
