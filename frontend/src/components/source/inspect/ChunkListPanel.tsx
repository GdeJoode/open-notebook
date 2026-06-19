'use client'

import { useRef, useCallback, useEffect } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { Badge } from '@/components/ui/badge'
import { getElementColor } from '@/lib/constants/element-colors'

export interface InspectChunk {
  id: string
  text: string
  element_type: string
  is_content?: boolean
  positions: number[][]
}

interface ChunkListPanelProps {
  chunks: InspectChunk[]
  selectedChunkId: string | null
  onSelectChunk: (chunkId: string) => void
  loading?: boolean
}

const ROW_HEIGHT = 76

/**
 * Left pane of the inspect workspace: a virtualized, keyboard-navigable list
 * of every chunk in the document. Virtualization (TanStack Virtual) keeps the
 * DOM light for documents with thousands of chunks.
 *
 * Keyboard model: the list container is the single tab stop. ArrowUp/ArrowDown
 * move selection (and scroll the active row into view); Home/End jump to the
 * ends; Enter/Space confirm the focused selection. This is the listbox roving
 * pattern adapted to a single focusable container so Tab stays predictable
 * across the three workspace panels (AC4).
 */
export function ChunkListPanel({
  chunks,
  selectedChunkId,
  onSelectChunk,
  loading = false,
}: ChunkListPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  const selectedIndex = selectedChunkId
    ? chunks.findIndex((c) => c.id === selectedChunkId)
    : -1

  const virtualizer = useVirtualizer({
    count: chunks.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 8,
  })

  const moveSelection = useCallback(
    (nextIndex: number) => {
      if (chunks.length === 0) return
      const clamped = Math.min(chunks.length - 1, Math.max(0, nextIndex))
      onSelectChunk(chunks[clamped].id)
      virtualizer.scrollToIndex(clamped, { align: 'auto' })
    },
    [chunks, onSelectChunk, virtualizer]
  )

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault()
          moveSelection(selectedIndex < 0 ? 0 : selectedIndex + 1)
          break
        case 'ArrowUp':
          e.preventDefault()
          moveSelection(selectedIndex < 0 ? 0 : selectedIndex - 1)
          break
        case 'Home':
          e.preventDefault()
          moveSelection(0)
          break
        case 'End':
          e.preventDefault()
          moveSelection(chunks.length - 1)
          break
        default:
          break
      }
    },
    [moveSelection, selectedIndex, chunks.length]
  )

  // Keep the active row visible when selection is driven from elsewhere
  // (e.g. clicking a bbox in the middle pane).
  useEffect(() => {
    if (selectedIndex >= 0) {
      virtualizer.scrollToIndex(selectedIndex, { align: 'auto' })
    }
    // virtualizer identity is stable; only react to index changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedIndex])

  return (
    <div className="flex h-full flex-col bg-card">
      <div className="flex-shrink-0 border-b px-3 py-2">
        <h2 className="text-sm font-semibold">Chunks</h2>
        <p className="text-xs text-muted-foreground mono-num">
          {chunks.length} total
        </p>
      </div>

      {loading ? (
        <div className="flex flex-1 items-center justify-center p-4 text-sm text-muted-foreground">
          Loading chunks…
        </div>
      ) : chunks.length === 0 ? (
        <div className="flex flex-1 items-center justify-center p-4 text-center text-sm text-muted-foreground">
          No chunks available for this source.
        </div>
      ) : (
        <div
          ref={scrollRef}
          role="listbox"
          aria-label="Document chunks"
          aria-activedescendant={
            selectedIndex >= 0 ? `chunk-row-${chunks[selectedIndex].id}` : undefined
          }
          tabIndex={0}
          onKeyDown={handleKeyDown}
          className="flex-1 overflow-auto outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset"
        >
          <div
            style={{ height: virtualizer.getTotalSize(), position: 'relative' }}
          >
            {virtualizer.getVirtualItems().map((item) => {
              const chunk = chunks[item.index]
              const isSelected = chunk.id === selectedChunkId
              const isNoise = chunk.is_content === false
              return (
                <div
                  key={chunk.id}
                  id={`chunk-row-${chunk.id}`}
                  role="option"
                  aria-selected={isSelected}
                  onClick={() => onSelectChunk(chunk.id)}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: item.size,
                    transform: `translateY(${item.start}px)`,
                  }}
                  className={`cursor-pointer border-b px-3 py-2 transition-colors ${
                    isSelected
                      ? 'bg-accent text-accent-foreground'
                      : 'hover:bg-muted/50'
                  } ${isNoise ? 'opacity-50' : ''}`}
                >
                  <div className="mb-1 flex items-center gap-1.5">
                    <span
                      className="inline-block h-2 w-2 flex-shrink-0 rounded-full"
                      style={{ backgroundColor: getElementColor(chunk.element_type) }}
                    />
                    <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
                      {chunk.element_type}
                    </Badge>
                  </div>
                  <p className="line-clamp-2 text-xs text-muted-foreground">
                    {chunk.text}
                  </p>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
