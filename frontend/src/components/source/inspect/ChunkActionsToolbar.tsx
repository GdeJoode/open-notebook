'use client'

import { useEffect, useState } from 'react'
import { Merge, Split } from 'lucide-react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useMergeChunk, useSplitChunk } from '@/lib/hooks/use-sources'
import type { InspectChunk } from './ChunkListPanel'

interface ChunkActionsToolbarProps {
  sourceId: string
  /** The currently selected chunk, or null when nothing is selected. */
  activeChunk: InspectChunk | null
  /** All chunks in document order — used to find the merge target. */
  chunks: InspectChunk[]
}

/**
 * Merge/split controls for the active chunk (Track I.D-3). Renders inside the
 * active-chunk section of the inspect workspace's properties pane.
 *
 * Merge folds the active chunk into the next chunk in document order; it is
 * disabled when the active chunk is last. Split breaks the active chunk at a
 * character offset (1..len-1) typed into the offset field. Both actions show
 * pending state on their button and surface failures via toast.
 */
export function ChunkActionsToolbar({
  sourceId,
  activeChunk,
  chunks,
}: ChunkActionsToolbarProps) {
  const merge = useMergeChunk(sourceId)
  const split = useSplitChunk(sourceId)

  const textLen = activeChunk?.text.length ?? 0
  const [offset, setOffset] = useState<number>(Math.floor(textLen / 2))

  // Reset the proposed split point to mid-text whenever the selection changes.
  useEffect(() => {
    setOffset(Math.floor(textLen / 2))
  }, [activeChunk?.id, textLen])

  if (!activeChunk) {
    return null
  }

  const activeIndex = chunks.findIndex((c) => c.id === activeChunk.id)
  const hasNext = activeIndex >= 0 && activeIndex < chunks.length - 1
  const offsetValid = offset >= 1 && offset <= textLen - 1

  const handleMerge = async () => {
    try {
      await merge.mutateAsync({ chunkId: activeChunk.id })
      toast.success('Chunks merged')
    } catch {
      toast.error('Failed to merge chunks')
    }
  }

  const handleSplit = async () => {
    if (!offsetValid) {
      toast.error(`Split offset must be between 1 and ${textLen - 1}`)
      return
    }
    try {
      await split.mutateAsync({ chunkId: activeChunk.id, cursorOffset: offset })
      toast.success('Chunk split')
    } catch {
      toast.error('Failed to split chunk')
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="justify-start"
        disabled={!hasNext || merge.isPending}
        onClick={handleMerge}
        aria-label="Merge this chunk with the next chunk"
      >
        <Merge aria-hidden="true" />
        {merge.isPending ? 'Merging…' : 'Merge with next'}
      </Button>

      <div className="flex items-center gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="justify-start"
          disabled={!offsetValid || split.isPending}
          onClick={handleSplit}
          aria-label="Split this chunk at the offset"
        >
          <Split aria-hidden="true" />
          {split.isPending ? 'Splitting…' : 'Split at'}
        </Button>
        <Input
          type="number"
          min={1}
          max={Math.max(1, textLen - 1)}
          value={offset}
          onChange={(e) => setOffset(Number(e.target.value))}
          className="mono-num h-8 w-20 text-xs"
          aria-label="Character offset to split the chunk at"
        />
        <span className="text-xs text-muted-foreground">/ {textLen}</span>
      </div>
    </div>
  )
}
