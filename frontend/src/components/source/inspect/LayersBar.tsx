'use client'

import { useMemo } from 'react'
import {
  ELEMENT_COLORS,
  getElementColor,
} from '@/lib/constants/element-colors'
import { useDocumentWorkspaceStore } from '@/lib/stores/document-workspace-store'

/**
 * LayersBar (Phase I.D-1): a chip-row of element-type visibility toggles for
 * the inspect workspace's middle pane. Each chip toggles whether the matching
 * bbox overlays are drawn in the PDF viewer; the hidden-types set lives in the
 * document-workspace store, so the PdfChunkViewer (controlled `hiddenTypes`)
 * and this bar stay in sync.
 *
 * Each chip is a native toggle `<button aria-pressed>` — so it is focusable and
 * Space/Enter toggle the focused chip for free (AC: keyboard accessible). The
 * row is wrapped in a `role="group"` so assistive tech announces it as one set.
 */
export function LayersBar() {
  const hiddenTypes = useDocumentWorkspaceStore((s) => s.hiddenTypes)
  const toggleType = useDocumentWorkspaceStore((s) => s.toggleType)
  const showAllTypes = useDocumentWorkspaceStore((s) => s.showAllTypes)

  // ELEMENT_COLORS has aliases that map to the same color (e.g. heading /
  // section_header); keep every distinct key as its own chip so users can hide
  // each label independently, matching the overlay's per-key skip logic.
  const types = useMemo(() => Object.keys(ELEMENT_COLORS), [])
  const hidden = useMemo(() => new Set(hiddenTypes), [hiddenTypes])
  const anyHidden = hiddenTypes.length > 0

  return (
    <div
      role="group"
      aria-label="Element layer visibility"
      className="flex flex-wrap items-center gap-1.5 border-b bg-card px-2 py-1.5"
    >
      {types.map((type) => {
        const isHidden = hidden.has(type)
        const label = type.replace(/_/g, ' ')
        return (
          <button
            key={type}
            type="button"
            aria-pressed={!isHidden}
            aria-label={`${label} layer ${isHidden ? 'hidden' : 'visible'}`}
            onClick={() => toggleType(type)}
            className={`flex items-center gap-1 rounded-full border border-border px-2 py-0.5 text-xs transition-opacity focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
              isHidden ? 'opacity-40' : 'opacity-100'
            }`}
          >
            <span
              aria-hidden="true"
              className="inline-block h-2 w-2 flex-shrink-0 rounded-full"
              style={{ backgroundColor: getElementColor(type) }}
            />
            <span className="capitalize">{label}</span>
          </button>
        )
      })}

      {anyHidden && (
        <button
          type="button"
          onClick={showAllTypes}
          className="ml-auto rounded-full px-2 py-0.5 text-xs text-muted-foreground underline-offset-2 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          Show all
        </button>
      )}
    </div>
  )
}
