'use client'

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Loader2,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Eye,
  EyeOff,
  Pencil,
  Trash2,
  Check,
  X,
  Plus,
  Square,
} from 'lucide-react'
import { toast } from 'sonner'
import { sourcesApi } from '@/lib/api/sources'
import { useUpdateChunk, useDeleteChunk, useCreateChunk } from '@/lib/hooks/use-sources'
import {
  ELEMENT_TYPES,
  elementTypeKey,
  getElementColor,
} from '@/lib/constants/element-colors'

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
  is_content?: boolean
}

interface PageInfo {
  page_number: number
  width: number
  height: number
}

interface BboxRect {
  x: number
  y: number
  w: number
  h: number
  chunkIndex: number
  elementType: string
  text: string
  isContent: boolean
}

interface PdfChunkViewerProps {
  sourceId: string
  chunks: Chunk[]
  /**
   * Layout mode.
   * - `embed` (default): the self-contained two-pane viewer used by the
   *   Chunks tab — internal chunk list on the left, PDF + overlay on the right.
   * - `fullscreen`: PDF + overlay only, filling the available height. The
   *   surrounding inspect workspace (Phase I.B) supplies its own chunk list and
   *   properties panes, so the internal chunk list is suppressed.
   */
  mode?: 'embed' | 'fullscreen'
  /**
   * Controlled selection by chunk id (fullscreen mode). When provided, the
   * viewer mirrors this selection in the overlay and reports user clicks via
   * `onSelectChunkId` instead of holding selection state internally.
   */
  selectedChunkId?: string | null
  onSelectChunkId?: (chunkId: string | null) => void
  /**
   * Controlled element-type visibility (fullscreen mode). When provided, the
   * overlay hides bboxes whose normalized element-type key is in this set and
   * the internal `LegendBar` is suppressed — the inspect workspace renders its
   * own `LayersBar` driving this set via the document-workspace store. When
   * omitted (embed mode), the viewer keeps its own local hidden-types state and
   * its internal `LegendBar`, so the Chunks tab is unchanged.
   */
  hiddenTypes?: Set<string>
}

// ---------------------------------------------------------------------------
// Page indexing
// ---------------------------------------------------------------------------
//
// Coordinates in chunk.positions are always 0–1 normalized (TOPLEFT origin)
// — canonicalized at extraction time by BoundingBox.from_* (Track I.C). The
// frontend no longer sniffs the coordinate format.
//
// Page indexing is a separate concern: positions store a 1-indexed page, but
// some legacy rows used a 0-indexed page. detectZeroBasedPages() only inspects
// the page integer (never the coordinates) to keep navigation correct on such
// rows; it does not affect bbox scaling.

function detectZeroBasedPages(chunks: Chunk[], totalPages: number): boolean {
  if (totalPages <= 0) return false
  let maxPageNum = 0
  for (const chunk of chunks) {
    if (!chunk.positions) continue
    for (const pos of chunk.positions) {
      if (pos[0] > maxPageNum) maxPageNum = pos[0]
    }
  }
  return maxPageNum === totalPages - 1
}

function toViewerPage(positionPage: number, zeroBased: boolean): number {
  return zeroBased ? positionPage + 1 : Math.max(1, positionPage)
}

// ---------------------------------------------------------------------------
// BboxOverlay — canvas overlay with optional drawing mode
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
  drawingMode: boolean
  drawnRect: { x: number; y: number; w: number; h: number } | null
  onDrawComplete: (rect: { x: number; y: number; w: number; h: number }) => void
}

function BboxOverlay({
  imgRef, pageInfo, rects, highlightedChunkIndex, onHoverChunk, onClickChunk,
  showOverlay, hiddenTypes, drawingMode, drawnRect, onDrawComplete,
}: BboxOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const drawStartRef = useRef<{ x: number; y: number } | null>(null)
  const [tempRect, setTempRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null)
  const [tooltip, setTooltip] = useState<{
    x: number; y: number; type: string; text: string; color: string
  } | null>(null)

  const getScale = useCallback(() => {
    const img = imgRef.current
    if (!img) return { sx: 1, sy: 1 }
    return { sx: img.clientWidth / pageInfo.width, sy: img.clientHeight / pageInfo.height }
  }, [imgRef, pageInfo])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return

    const dpr = window.devicePixelRatio || 1
    const displayW = img.clientWidth
    const displayH = img.clientHeight
    if (displayW === 0 || displayH === 0) return

    canvas.width = displayW * dpr
    canvas.height = displayH * dpr
    canvas.style.width = `${displayW}px`
    canvas.style.height = `${displayH}px`

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, displayW, displayH)

    if (!showOverlay && !drawingMode) return

    const { sx, sy } = getScale()

    // Draw existing rects
    if (showOverlay) {
      for (const rect of rects) {
        if (hiddenTypes.has(elementTypeKey(rect.elementType))) continue
        const x = rect.x * sx, y = rect.y * sy, w = rect.w * sx, h = rect.h * sy
        const color = getElementColor(rect.elementType)
        const isHighlighted = rect.chunkIndex === highlightedChunkIndex
        const isDimmed = highlightedChunkIndex !== null && !isHighlighted
        const isNoise = !rect.isContent

        if (isNoise) {
          // Noise chunks: same style as normal but in gray
          ctx.strokeStyle = isDimmed ? '#9CA3AF40' : '#9CA3AF'
          ctx.lineWidth = isHighlighted ? 3 : 2
          ctx.fillStyle = isHighlighted ? '#9CA3AF40' : isDimmed ? '#9CA3AF08' : '#9CA3AF20'
        } else {
          ctx.strokeStyle = isDimmed ? color + '40' : color
          ctx.lineWidth = isHighlighted ? 3 : 2
          ctx.fillStyle = isHighlighted ? color + '40' : isDimmed ? color + '08' : color + '20'
        }
        ctx.beginPath()
        ctx.rect(x, y, w, h)
        ctx.fill()
        ctx.stroke()
      }
    }

    // Draw saved drawn rect (amber)
    const activeRect = tempRect || drawnRect
    if (activeRect) {
      const x = activeRect.x * sx, y = activeRect.y * sy
      const w = activeRect.w * sx, h = activeRect.h * sy
      ctx.strokeStyle = '#F59E0B'
      ctx.lineWidth = 2
      ctx.setLineDash([6, 3])
      ctx.fillStyle = 'rgba(245, 158, 11, 0.15)'
      ctx.beginPath()
      ctx.rect(x, y, w, h)
      ctx.fill()
      ctx.stroke()
      ctx.setLineDash([])
    }
  }, [imgRef, pageInfo, rects, highlightedChunkIndex, showOverlay, hiddenTypes, getScale, drawingMode, drawnRect, tempRect])

  useEffect(() => {
    draw()
    const raf = requestAnimationFrame(() => draw())
    return () => cancelAnimationFrame(raf)
  }, [draw])

  useEffect(() => {
    const img = imgRef.current
    if (!img) return
    const observer = new ResizeObserver(() => draw())
    observer.observe(img)
    return () => observer.disconnect()
  }, [imgRef, draw])

  // Mouse-to-PDF-points conversion
  const toPdfPoint = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }
    const r = canvas.getBoundingClientRect()
    const { sx, sy } = getScale()
    return { x: (e.clientX - r.left) / sx, y: (e.clientY - r.top) / sy }
  }, [getScale])

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawingMode) return
    const pt = toPdfPoint(e)
    drawStartRef.current = pt
    setTempRect(null)
  }, [drawingMode, toPdfPoint])

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    // Drawing mode: preview rectangle
    if (drawingMode && drawStartRef.current) {
      const pt = toPdfPoint(e)
      const s = drawStartRef.current
      setTempRect({
        x: Math.min(s.x, pt.x), y: Math.min(s.y, pt.y),
        w: Math.abs(pt.x - s.x), h: Math.abs(pt.y - s.y),
      })
      return
    }

    // Normal mode: hover detection
    if (!showOverlay) return
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return

    const cr = canvas.getBoundingClientRect()
    const mx = e.clientX - cr.left, my = e.clientY - cr.top
    const { sx, sy } = getScale()

    for (let i = rects.length - 1; i >= 0; i--) {
      const r = rects[i]
      if (hiddenTypes.has(elementTypeKey(r.elementType))) continue
      const rx = r.x * sx, ry = r.y * sy, rw = r.w * sx, rh = r.h * sy
      if (mx >= rx && mx <= rx + rw && my >= ry && my <= ry + rh) {
        onHoverChunk(r.chunkIndex)
        setTooltip({
          x: Math.min(mx, img.clientWidth - 280), y: Math.max(my - 60, 0),
          type: r.elementType, text: r.text.substring(0, 150), color: getElementColor(r.elementType),
        })
        return
      }
    }
    onHoverChunk(null)
    setTooltip(null)
  }, [imgRef, pageInfo, rects, showOverlay, hiddenTypes, onHoverChunk, drawingMode, toPdfPoint, getScale])

  const handleMouseUp = useCallback(() => {
    if (drawingMode && tempRect && tempRect.w > 5 && tempRect.h > 5) {
      onDrawComplete(tempRect)
      setTempRect(null)
    }
    drawStartRef.current = null
  }, [drawingMode, tempRect, onDrawComplete])

  const handleMouseLeave = useCallback(() => {
    if (!drawingMode) {
      onHoverChunk(null)
      setTooltip(null)
    }
    drawStartRef.current = null
    setTempRect(null)
  }, [drawingMode, onHoverChunk])

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (drawingMode) return
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return

    const cr = canvas.getBoundingClientRect()
    const mx = e.clientX - cr.left, my = e.clientY - cr.top
    const { sx, sy } = getScale()

    for (let i = rects.length - 1; i >= 0; i--) {
      const r = rects[i]
      if (hiddenTypes.has(elementTypeKey(r.elementType))) continue
      const rx = r.x * sx, ry = r.y * sy, rw = r.w * sx, rh = r.h * sy
      if (mx >= rx && mx <= rx + rw && my >= ry && my <= ry + rh) {
        onClickChunk(r.chunkIndex)
        return
      }
    }
  }, [imgRef, rects, hiddenTypes, onClickChunk, drawingMode, getScale])

  return (
    <>
      <canvas
        ref={canvasRef}
        className={`absolute top-0 left-0 ${drawingMode ? 'cursor-crosshair' : 'cursor-pointer'}`}
        style={{ zIndex: 5 }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      />
      {tooltip && !drawingMode && (
        <div
          className="absolute z-20 max-w-[280px] rounded-md border bg-popover/95 backdrop-blur-sm p-2 shadow-lg pointer-events-none"
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          <div className="flex items-center gap-1.5 mb-1">
            <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: tooltip.color }} />
            <span className="text-xs font-semibold uppercase tracking-wide">{tooltip.type}</span>
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

function LegendBar({ typeCounts, hiddenTypes, onToggle }: {
  typeCounts: Record<string, number>; hiddenTypes: Set<string>; onToggle: (type: string) => void
}) {
  const types = Object.entries(typeCounts).sort((a, b) => b[1] - a[1])
  if (types.length === 0) return null
  return (
    <div className="absolute top-2 left-2 right-2 z-10 flex flex-wrap gap-1.5 rounded-lg bg-background/80 backdrop-blur-md border p-2 pointer-events-auto">
      {types.map(([type, count]) => {
        const key = elementTypeKey(type)
        return (
          <button key={type} onClick={() => onToggle(key)}
            className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-xs border transition-opacity ${hiddenTypes.has(key) ? 'opacity-35' : 'opacity-100'}`}>
            <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: getElementColor(type) }} />
            <span>{type}</span>
            <span className="text-muted-foreground mono-num">({count})</span>
          </button>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function PdfChunkViewer({
  sourceId,
  chunks,
  mode = 'embed',
  selectedChunkId,
  onSelectChunkId,
  hiddenTypes: controlledHiddenTypes,
}: PdfChunkViewerProps) {
  const fullscreen = mode === 'fullscreen'
  const controlledSelection = selectedChunkId !== undefined
  const controlledLayers = controlledHiddenTypes !== undefined
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
  // Embed mode owns its hidden-types locally (drives the internal LegendBar).
  // Fullscreen mode is controlled: the workspace's LayersBar supplies the set.
  const [localHiddenTypes, setLocalHiddenTypes] = useState<Set<string>>(new Set())
  const hiddenTypes = controlledHiddenTypes ?? localHiddenTypes
  const [pagesAreZeroBased, setPagesAreZeroBased] = useState(false)

  // Editing state
  const [editingChunkId, setEditingChunkId] = useState<string | null>(null)
  const [editText, setEditText] = useState('')
  const [editType, setEditType] = useState('')
  const [drawingMode, setDrawingMode] = useState(false)
  const [drawnRect, setDrawnRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null)
  const [showAddForm, setShowAddForm] = useState(false)
  const [newChunkText, setNewChunkText] = useState('')
  const [newChunkType, setNewChunkType] = useState('paragraph')

  const imgRef = useRef<HTMLImageElement>(null)
  const chunkListRef = useRef<HTMLDivElement>(null)
  const initialPageSetRef = useRef(false)

  // In controlled (fullscreen) mode the selected chunk is driven by the parent
  // workspace via `selectedChunkId`; otherwise it's the internal index state.
  const controlledSelectedIndex = useMemo(() => {
    if (!controlledSelection) return null
    if (!selectedChunkId) return null
    const idx = chunks.findIndex((c) => c.id === selectedChunkId)
    return idx >= 0 ? idx : null
  }, [controlledSelection, selectedChunkId, chunks])

  const effectiveSelectedIndex = controlledSelection
    ? controlledSelectedIndex
    : selectedChunkIndex
  const highlightedIndex = hoveredChunkIndex ?? effectiveSelectedIndex

  // Mutations
  const updateChunk = useUpdateChunk(sourceId)
  const deleteChunk = useDeleteChunk(sourceId)
  const createChunk = useCreateChunk(sourceId)

  // Load page count on mount
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const data = await sourcesApi.getPageCount(sourceId)
        if (cancelled) return
        setPageCount(data.page_count)
        setPagesInfo(data.pages)
        const zeroBased = detectZeroBasedPages(chunks, data.page_count)
        setPagesAreZeroBased(zeroBased)

        // Only set initial page on first load, not after mutation refetch
        if (!initialPageSetRef.current) {
          initialPageSetRef.current = true
          const firstChunkWithPos = chunks.find(c => c.positions?.length > 0)
          if (firstChunkWithPos?.positions?.length) {
            const rawPage = firstChunkWithPos.positions[0][0]
            const viewerPage = toViewerPage(rawPage, zeroBased)
            setCurrentPage(viewerPage)
            setPageInput(String(viewerPage))
          }
        }
      } catch {
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

  const currentPageInfo = useMemo(
    () => pagesInfo.find(p => p.page_number === currentPage) ?? { page_number: currentPage, width: 595, height: 842 },
    [pagesInfo, currentPage]
  )

  // Build bounding box rects
  const pageRects = useMemo(() => {
    const rects: BboxRect[] = []
    const pw = currentPageInfo.width, ph = currentPageInfo.height

    for (let ci = 0; ci < chunks.length; ci++) {
      const chunk = chunks[ci]
      if (!chunk.positions) continue
      for (const pos of chunk.positions) {
        const [rawPageNum, xLeft, xRight, yTopRaw, yBottomRaw] = pos
        const viewerPage = toViewerPage(rawPageNum, pagesAreZeroBased)
        if (viewerPage !== currentPage) continue

        // Coordinates are always 0–1 normalized; scale into page pixels.
        const yTop = Math.min(yTopRaw, yBottomRaw)
        const yBottom = Math.max(yTopRaw, yBottomRaw)
        rects.push({
          x: xLeft * pw,
          y: yTop * ph,
          w: (xRight - xLeft) * pw,
          h: (yBottom - yTop) * ph,
          chunkIndex: ci,
          elementType: chunk.element_type || 'unknown',
          text: chunk.text || '',
          isContent: chunk.is_content !== false,
        })
      }
    }
    return rects
  }, [chunks, currentPage, currentPageInfo, pagesAreZeroBased])

  const typeCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const rect of pageRects) counts[rect.elementType] = (counts[rect.elementType] || 0) + 1
    return counts
  }, [pageRects])

  const pageChunks = useMemo(() => {
    const indices: number[] = [], seen = new Set<number>()
    for (const rect of pageRects) {
      if (!seen.has(rect.chunkIndex)) { seen.add(rect.chunkIndex); indices.push(rect.chunkIndex) }
    }
    return indices
  }, [pageRects])

  useEffect(() => {
    if (highlightedIndex === null || !chunkListRef.current) return
    const el = chunkListRef.current.querySelector(`[data-chunk-index="${highlightedIndex}"]`)
    if (el) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [highlightedIndex])

  // Controlled mode: when the parent workspace selects a chunk, follow it to
  // the page where its bbox lives so the overlay highlight is visible.
  useEffect(() => {
    if (!controlledSelection || controlledSelectedIndex === null) return
    const chunk = chunks[controlledSelectedIndex]
    if (!chunk?.positions?.length) return
    const viewerPage = toViewerPage(chunk.positions[0][0], pagesAreZeroBased)
    if (viewerPage !== currentPage) {
      setCurrentPage(viewerPage)
      setPageInput(String(viewerPage))
    }
    // currentPage intentionally omitted: we only react to selection changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [controlledSelection, controlledSelectedIndex, chunks, pagesAreZeroBased])

  const handleSelectChunk = useCallback((index: number) => {
    const chunk = chunks[index]
    if (controlledSelection) {
      onSelectChunkId?.(chunk?.id ?? null)
    } else {
      setSelectedChunkIndex(index)
    }
    if (chunk?.positions?.length > 0) {
      const rawPage = chunk.positions[0][0]
      const viewerPage = toViewerPage(rawPage, pagesAreZeroBased)
      if (viewerPage !== currentPage) { setCurrentPage(viewerPage); setPageInput(String(viewerPage)) }
    }
  }, [chunks, currentPage, pagesAreZeroBased, controlledSelection, onSelectChunkId])

  const handlePageNav = useCallback((delta: number) => {
    const next = Math.max(1, Math.min(pageCount, currentPage + delta))
    setCurrentPage(next); setPageInput(String(next))
  }, [currentPage, pageCount])

  const handlePageInputSubmit = useCallback(() => {
    const p = parseInt(pageInput, 10)
    if (!isNaN(p) && p >= 1 && p <= pageCount) setCurrentPage(p)
    else setPageInput(String(currentPage))
  }, [pageInput, pageCount, currentPage])

  const toggleType = useCallback((type: string) => {
    setLocalHiddenTypes(prev => { const next = new Set(prev); next.has(type) ? next.delete(type) : next.add(type); return next })
  }, [])

  // --- Editing handlers ---

  const startEdit = (chunk: Chunk) => {
    setEditingChunkId(chunk.id)
    setEditText(chunk.text)
    setEditType(chunk.element_type)
    setDrawnRect(null)
    setDrawingMode(false)
  }

  const cancelEdit = () => {
    setEditingChunkId(null)
    setEditText('')
    setEditType('')
    setDrawingMode(false)
    setDrawnRect(null)
  }

  const saveEdit = (chunk: Chunk) => {
    const data: Record<string, unknown> = {}
    if (editText !== chunk.text) data.text = editText
    if (editType !== chunk.element_type) data.element_type = editType

    // If bbox was drawn, convert to the canonical 0–1 position format.
    if (drawnRect) {
      const pw = currentPageInfo.width, ph = currentPageInfo.height
      const pageNum = pagesAreZeroBased ? currentPage - 1 : currentPage
      data.positions = [[pageNum, drawnRect.x / pw, (drawnRect.x + drawnRect.w) / pw, drawnRect.y / ph, (drawnRect.y + drawnRect.h) / ph]]
    }

    if (Object.keys(data).length === 0) { cancelEdit(); return }

    updateChunk.mutate({ chunkId: chunk.id, data }, {
      onSuccess: () => { toast.success('Chunk updated'); cancelEdit() },
      onError: () => toast.error('Failed to update chunk'),
    })
  }

  const handleDelete = (chunk: Chunk) => {
    if (!confirm(`Delete this chunk?\n\n"${chunk.text.substring(0, 100)}..."`)) return
    deleteChunk.mutate(chunk.id, {
      onSuccess: () => { toast.success('Chunk deleted'); setSelectedChunkIndex(null) },
      onError: () => toast.error('Failed to delete chunk'),
    })
  }

  const handleToggleContent = (chunk: Chunk) => {
    updateChunk.mutate({ chunkId: chunk.id, data: { is_content: !(chunk.is_content ?? true) } }, {
      onSuccess: () => toast.success(chunk.is_content ? 'Marked as noise' : 'Marked as content'),
    })
  }

  const handleCreateChunk = () => {
    if (!newChunkText.trim()) return
    const data: Parameters<typeof createChunk.mutate>[0] = {
      text: newChunkText,
      element_type: newChunkType,
      physical_page: currentPage - 1,
      is_content: true,
    }
    if (drawnRect) {
      const pw = currentPageInfo.width, ph = currentPageInfo.height
      const pageNum = pagesAreZeroBased ? currentPage - 1 : currentPage
      data.positions = [[pageNum, drawnRect.x / pw, (drawnRect.x + drawnRect.w) / pw, drawnRect.y / ph, (drawnRect.y + drawnRect.h) / ph]]
    }
    createChunk.mutate(data, {
      onSuccess: () => {
        toast.success('Chunk created')
        setShowAddForm(false); setNewChunkText(''); setNewChunkType('paragraph')
        setDrawingMode(false); setDrawnRect(null)
      },
      onError: () => toast.error('Failed to create chunk'),
    })
  }

  // --- Render ---

  if (loading) return <div className="flex h-full items-center justify-center"><Loader2 className="h-8 w-8 animate-spin text-muted-foreground" /></div>
  if (error) return <Alert variant="destructive"><AlertCircle className="h-4 w-4" /><AlertDescription>{error}</AlertDescription></Alert>
  if (chunks.length === 0 && !showAddForm) return <Alert><AlertCircle className="h-4 w-4" /><AlertDescription>No chunks available.</AlertDescription></Alert>

  return (
    <div className={`flex h-full overflow-hidden bg-background ${fullscreen ? '' : 'border rounded-lg'}`}>
      {/* Left Pane — Chunk List (embed mode only; fullscreen uses the
          workspace's ChunkListPanel instead). */}
      {!fullscreen && (
      <div className="w-80 min-w-[280px] border-r bg-muted/30 flex flex-col">
        <div className="p-3 border-b bg-background flex-shrink-0 flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-sm">Elements on Page <span className="mono-num">{currentPage}</span></h3>
            <p className="text-xs text-muted-foreground"><span className="mono-num">{pageChunks.length}</span> elements · <span className="mono-num">{chunks.length}</span> total</p>
          </div>
          <Button
            type="button" variant="ghost" size="sm" className="h-7 w-7 p-0"
            onClick={() => { setShowAddForm(!showAddForm); if (!showAddForm) { setDrawingMode(true); setDrawnRect(null) } else { setDrawingMode(false) } }}
            title="Add chunk"
          >
            <Plus className="h-4 w-4" />
          </Button>
        </div>

        <ScrollArea className="flex-1 min-h-0">
          <div className="p-2 space-y-1.5" ref={chunkListRef}>
            {/* Add chunk form */}
            {showAddForm && (
              <div className="p-2.5 rounded-md border border-amber-500 bg-amber-50/50 dark:bg-amber-950/20 space-y-2">
                <div className="flex items-center gap-1.5">
                  <Square className="h-3.5 w-3.5 text-amber-600" />
                  <span className="text-xs font-semibold text-amber-700 dark:text-amber-400">New Chunk</span>
                </div>
                <Textarea
                  value={newChunkText} onChange={(e) => setNewChunkText(e.target.value)}
                  placeholder="Enter chunk text..." className="text-xs min-h-[60px]" rows={3}
                />
                <Select value={newChunkType} onValueChange={setNewChunkType}>
                  <SelectTrigger className="h-7 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {ELEMENT_TYPES.map(t => <SelectItem key={t} value={t} className="text-xs">{t}</SelectItem>)}
                  </SelectContent>
                </Select>
                <p className="text-[10px] text-muted-foreground">
                  {drawnRect ? `Bbox: ${drawnRect.x.toFixed(0)},${drawnRect.y.toFixed(0)} ${drawnRect.w.toFixed(0)}x${drawnRect.h.toFixed(0)}` : 'Draw a bounding box on the PDF (optional)'}
                </p>
                <div className="flex gap-1.5">
                  <Button type="button" size="sm" className="h-7 flex-1 gap-1 text-xs" onClick={handleCreateChunk} disabled={!newChunkText.trim() || createChunk.isPending}>
                    {createChunk.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />} Save
                  </Button>
                  <Button type="button" size="sm" variant="ghost" className="h-7 text-xs" onClick={() => { setShowAddForm(false); setDrawingMode(false); setDrawnRect(null) }}>
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            )}

            {pageChunks.length === 0 && !showAddForm ? (
              <p className="text-xs text-muted-foreground p-3">No elements with position data on this page.</p>
            ) : (
              pageChunks.map((ci) => {
                const chunk = chunks[ci]
                const color = getElementColor(chunk.element_type)
                const isSelected = ci === selectedChunkIndex
                const isHovered = ci === hoveredChunkIndex
                const isEditing = editingChunkId === chunk.id
                const isNoise = chunk.is_content === false

                return (
                  <div
                    key={chunk.id || ci}
                    data-chunk-index={ci}
                    onClick={() => !isEditing && handleSelectChunk(ci)}
                    onMouseEnter={() => !isEditing && setHoveredChunkIndex(ci)}
                    onMouseLeave={() => !isEditing && setHoveredChunkIndex(null)}
                    className={`p-2.5 rounded-md border transition-all text-sm ${isNoise ? 'opacity-40' : ''} ${
                      isEditing ? 'border-amber-500 bg-amber-50/50 dark:bg-amber-950/20'
                        : isSelected ? 'border-primary bg-primary/10 shadow-sm cursor-pointer'
                        : isHovered ? 'border-primary/50 bg-muted/50 cursor-pointer'
                        : 'border-transparent hover:border-border cursor-pointer'
                    }`}
                    style={!isEditing && (isSelected || isHovered) ? { borderLeftColor: color, borderLeftWidth: 3 } : undefined}
                  >
                    {isEditing ? (
                      /* Editing mode */
                      <div className="space-y-2">
                        <Textarea value={editText} onChange={(e) => setEditText(e.target.value)} className="text-xs min-h-[80px]" rows={4} />
                        <Select value={editType} onValueChange={setEditType}>
                          <SelectTrigger className="h-7 text-xs"><SelectValue /></SelectTrigger>
                          <SelectContent>
                            {ELEMENT_TYPES.map(t => <SelectItem key={t} value={t} className="text-xs">{t}</SelectItem>)}
                          </SelectContent>
                        </Select>
                        <div className="flex items-center gap-1.5">
                          <Button type="button" size="sm" variant={drawingMode ? 'default' : 'outline'} className="h-7 gap-1 text-xs"
                            onClick={() => { setDrawingMode(!drawingMode); if (!drawingMode) setDrawnRect(null) }}>
                            <Square className="h-3 w-3" /> {drawingMode ? 'Drawing...' : 'Edit bbox'}
                          </Button>
                          {drawnRect && <span className="text-[10px] text-muted-foreground">New bbox set</span>}
                        </div>
                        <div className="flex gap-1.5">
                          <Button type="button" size="sm" className="h-7 flex-1 gap-1 text-xs" onClick={() => saveEdit(chunk)} disabled={updateChunk.isPending}>
                            {updateChunk.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />} Save
                          </Button>
                          <Button type="button" size="sm" variant="ghost" className="h-7 text-xs" onClick={cancelEdit}>
                            <X className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                    ) : (
                      /* Display mode */
                      <>
                        <div className="flex items-center gap-1.5 mb-1">
                          <span className="inline-block w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                          <Badge variant="outline" className="text-[10px] px-1.5 py-0">{chunk.element_type}</Badge>
                          {chunk.chapter && (
                            <span className="text-[10px] text-muted-foreground truncate ml-auto max-w-[80px]" title={chunk.chapter}>{chunk.chapter}</span>
                          )}
                          {/* Action buttons — visible when selected */}
                          {isSelected && (
                            <div className="flex items-center gap-0.5 ml-auto">
                              <Button type="button" variant="ghost" size="sm" className="h-5 w-5 p-0" onClick={(e) => { e.stopPropagation(); startEdit(chunk) }} title="Edit">
                                <Pencil className="h-3 w-3" />
                              </Button>
                              <Button type="button" variant="ghost" size="sm" className="h-5 w-5 p-0" onClick={(e) => { e.stopPropagation(); handleToggleContent(chunk) }}
                                title={isNoise ? 'Mark as content' : 'Mark as noise'}>
                                {isNoise ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
                              </Button>
                              <Button type="button" variant="ghost" size="sm" className="h-5 w-5 p-0 text-destructive" onClick={(e) => { e.stopPropagation(); handleDelete(chunk) }} title="Delete">
                                <Trash2 className="h-3 w-3" />
                              </Button>
                            </div>
                          )}
                        </div>
                        <p className="text-xs line-clamp-2 text-muted-foreground">{chunk.text}</p>
                      </>
                    )}
                  </div>
                )
              })
            )}
          </div>
        </ScrollArea>
      </div>
      )}

      {/* Right Pane — PDF Page + Canvas Overlay */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="p-2 border-b flex items-center gap-2 flex-shrink-0 bg-background">
          <Button type="button" variant="ghost" size="sm" className="h-7 w-7 p-0" disabled={currentPage <= 1} onClick={() => handlePageNav(-1)}>
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <div className="flex items-center gap-1 text-sm">
            <Input className="w-12 h-7 text-center text-xs p-0 mono-num" value={pageInput}
              aria-label="Current page"
              onChange={(e) => setPageInput(e.target.value)} onBlur={handlePageInputSubmit}
              onKeyDown={(e) => e.key === 'Enter' && handlePageInputSubmit()} />
            <span className="text-muted-foreground text-xs mono-num">/ {pageCount}</span>
          </div>
          <Button type="button" variant="ghost" size="sm" className="h-7 w-7 p-0" disabled={currentPage >= pageCount} onClick={() => handlePageNav(1)}>
            <ChevronRight className="h-4 w-4" />
          </Button>
          {drawingMode && (
            <Badge variant="default" className="bg-amber-600 text-xs ml-2">Drawing bbox — click and drag on PDF</Badge>
          )}
          <div className="ml-auto">
            <Button type="button" variant="ghost" size="sm" className="h-7 gap-1 text-xs" onClick={() => setShowOverlay(!showOverlay)}>
              {showOverlay ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}
              {showOverlay ? 'Hide' : 'Show'} boxes
            </Button>
          </div>
        </div>

        <div className="flex-1 min-h-0 overflow-auto bg-muted/20 flex items-start justify-center p-4">
          <div className="relative inline-block shadow-lg border rounded">
            {previewUrl && (
              <img ref={imgRef} src={previewUrl} alt={`Page ${currentPage}`}
                className="block max-w-full h-auto" style={{ maxHeight: 'calc(100vh - 200px)' }}
                onLoad={() => setImgLoaded(true)} onError={() => setError(`Failed to render page ${currentPage}`)} />
            )}
            {!imgLoaded && previewUrl && (
              <div className="flex items-center justify-center w-[595px] h-[842px]"><Loader2 className="h-8 w-8 animate-spin text-muted-foreground" /></div>
            )}
            {imgLoaded && showOverlay && !controlledLayers && <LegendBar typeCounts={typeCounts} hiddenTypes={hiddenTypes} onToggle={toggleType} />}
            {imgLoaded && (
              <BboxOverlay
                imgRef={imgRef} pageInfo={currentPageInfo} rects={pageRects}
                highlightedChunkIndex={highlightedIndex} onHoverChunk={setHoveredChunkIndex}
                onClickChunk={handleSelectChunk} showOverlay={showOverlay} hiddenTypes={hiddenTypes}
                drawingMode={drawingMode} drawnRect={drawnRect} onDrawComplete={setDrawnRect}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
