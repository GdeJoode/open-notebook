'use client'

/**
 * JsonlExportButton — Track D Phase D.2.
 *
 * Single button + lightweight Popover (NOT a Dialog) exposing the JSONL
 * export surface. The popover stays in the page flow rather than
 * trapping focus, matching the plan's "minimal popover" directive: the
 * filter set is the SAME three sliders + two switches as the Obsidian
 * dialog (D.1c), so users get a consistent knob layout while keeping
 * JSONL a lighter-weight interaction.
 *
 * Why a popover instead of reusing ObsidianExportDialog?
 * - The JSONL surface has no mode toggle (zip vs vault_path).
 * - The JSONL surface has no entity-types input (V1 -- the plan keeps
 *   the popover minimal).
 * - A popover keeps the user in the page; a dialog would steal focus
 *   for a one-click download.
 *
 * Filter pipeline reuses ``ExportPreviewCounts`` from D.1c so the live
 * count widget is identical to the Obsidian dialog -- one source of
 * truth for the preview rendering.
 */

import { useEffect, useId, useMemo, useState } from 'react'
import { AlertTriangle, Download, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { useExportPreview } from '@/lib/hooks/use-export-preview'
import { useJsonlExport } from '@/lib/hooks/use-jsonl-export'
import type { ExportFilter } from '@/lib/types/exports'

import { ExportPreviewCounts } from './ExportPreviewCounts'

interface JsonlExportButtonProps {
  notebookId: string
}

/** Match the Pydantic ExportFilter defaults exactly (mirrors D.1c). */
const DEFAULT_FILTER: ExportFilter = {
  min_connections: 5,
  min_confidence: 0.9,
  min_relation_confidence: 0.9,
  entity_types: undefined,
  include_orphans: false,
  include_archived: false,
}

const SLIDER_RANGES = {
  min_connections: { min: 0, max: 20, step: 1 },
  min_confidence: { min: 0, max: 1, step: 0.05 },
  min_relation_confidence: { min: 0, max: 1, step: 0.05 },
} as const

export function JsonlExportButton({ notebookId }: JsonlExportButtonProps) {
  const [open, setOpen] = useState(false)
  const [filter, setFilter] = useState<ExportFilter>(DEFAULT_FILTER)

  const { mutate, isPending, error } = useJsonlExport(notebookId)
  const previewQuery = useExportPreview(notebookId, filter)

  // Reset filter state when popover re-opens. Matches the dialog's
  // "clean state on open" idiom (D.1c) so a re-open after a download
  // doesn't show stale slider values.
  useEffect(() => {
    if (open) {
      setFilter(DEFAULT_FILTER)
    }
  }, [open])

  const minConnectionsId = useId()
  const minConfidenceId = useId()
  const minRelationConfidenceId = useId()
  const orphansId = useId()
  const archivedId = useId()

  const downloadDisabled = useMemo(() => {
    if (isPending) return true
    if (previewQuery.isLoading) return true
    return false
  }, [isPending, previewQuery.isLoading])

  const handleDownload = () => {
    if (downloadDisabled) return
    mutate(
      { filter },
      {
        onSuccess: () => {
          // Close the popover after the browser download fires; the
          // download itself is the user-visible confirmation.
          setOpen(false)
        },
      },
    )
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          data-testid="open-jsonl-export-popover"
          aria-label="Export this notebook as JSONL"
        >
          <Download className="h-4 w-4 mr-2" aria-hidden="true" />
          Export JSONL
        </Button>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        className="w-96"
        data-testid="jsonl-export-popover"
      >
        <div className="space-y-4">
          <div className="space-y-1">
            <h3 className="text-sm font-semibold">Export to JSONL</h3>
            <p className="text-xs text-muted-foreground">
              Streams a zip of <code>entities.jsonl</code> +{' '}
              <code>relations.jsonl</code> shaped for Neo4j and LangChain
              RAG loaders.
            </p>
          </div>

          <SliderRow
            id={minConnectionsId}
            label="Min connections"
            value={filter.min_connections}
            onChange={(v) => setFilter((f) => ({ ...f, min_connections: v }))}
            {...SLIDER_RANGES.min_connections}
            format={(v) => String(v)}
            testId="jsonl-min-connections"
          />
          <SliderRow
            id={minConfidenceId}
            label="Min entity confidence"
            value={filter.min_confidence}
            onChange={(v) => setFilter((f) => ({ ...f, min_confidence: v }))}
            {...SLIDER_RANGES.min_confidence}
            format={(v) => v.toFixed(2)}
            testId="jsonl-min-confidence"
          />
          <SliderRow
            id={minRelationConfidenceId}
            label="Min relation confidence"
            value={filter.min_relation_confidence ?? filter.min_confidence}
            onChange={(v) =>
              setFilter((f) => ({ ...f, min_relation_confidence: v }))
            }
            {...SLIDER_RANGES.min_relation_confidence}
            format={(v) => v.toFixed(2)}
            testId="jsonl-min-relation-confidence"
          />

          <div className="flex items-center gap-3">
            <Switch
              id={orphansId}
              checked={filter.include_orphans}
              onCheckedChange={(checked) =>
                setFilter((f) => ({ ...f, include_orphans: checked }))
              }
              aria-label="Include orphan entities"
              data-testid="jsonl-include-orphans-switch"
            />
            <Label htmlFor={orphansId} className="text-sm cursor-pointer">
              Include orphaned entities
            </Label>
          </div>

          <div className="flex items-center gap-3">
            <Switch
              id={archivedId}
              checked={filter.include_archived}
              onCheckedChange={(checked) =>
                setFilter((f) => ({ ...f, include_archived: checked }))
              }
              aria-label="Include archived entities"
              data-testid="jsonl-include-archived-switch"
            />
            <Label htmlFor={archivedId} className="text-sm cursor-pointer">
              Include archived entities
            </Label>
          </div>

          <ExportPreviewCounts notebookId={notebookId} filter={filter} />

          {error && (
            <div
              role="alert"
              className="flex items-start gap-2 rounded-md border border-destructive/60 bg-destructive/10 p-3 text-sm text-destructive"
              data-testid="jsonl-export-error"
            >
              <AlertTriangle
                className="h-4 w-4 mt-0.5 shrink-0"
                aria-hidden="true"
              />
              <span>
                {error.message || 'Export failed. Please try again.'}
              </span>
            </div>
          )}

          <div className="flex justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setOpen(false)}
              disabled={isPending}
              data-testid="jsonl-export-cancel"
            >
              Cancel
            </Button>
            <Button
              type="button"
              size="sm"
              onClick={handleDownload}
              disabled={downloadDisabled}
              data-testid="jsonl-export-submit"
              aria-label="Download notebook as JSONL"
            >
              {isPending && (
                <Loader2
                  className="h-4 w-4 mr-2 animate-spin"
                  aria-hidden="true"
                />
              )}
              {isPending ? 'Exporting…' : 'Download'}
            </Button>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}

interface SliderRowProps {
  id: string
  label: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
  format: (value: number) => string
  testId: string
}

function SliderRow({
  id,
  label,
  value,
  onChange,
  min,
  max,
  step,
  format,
  testId,
}: SliderRowProps) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <Label htmlFor={id} className="text-sm">
          {label}
        </Label>
        <span
          className="text-sm font-mono tabular-nums text-muted-foreground"
          data-testid={`${testId}-value`}
        >
          {format(value)}
        </span>
      </div>
      <Slider
        id={id}
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([next]) => onChange(next)}
        aria-label={label}
        data-testid={`${testId}-slider`}
      />
    </div>
  )
}
