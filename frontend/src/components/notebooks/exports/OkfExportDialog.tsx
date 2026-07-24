'use client'

/**
 * OkfExportDialog — Track OKF.5.
 *
 * Exports a notebook's curated knowledge-graph projection as an **Open
 * Knowledge Format** (OKF v0.1) bundle — a vendor-neutral tree of Markdown
 * concepts + an index, portable to arbitrary AI agents. It mirrors
 * ``ObsidianExportDialog`` (same shared ``ExportFilter`` knobs, same Radix
 * scaffolding, same preview widget) with three deliberate differences:
 *
 *   * **Zip-only.** OKF has no vault_path write mode, so there is no delivery-
 *     mode toggle — the export always streams a downloadable zip.
 *   * **Lossy-by-design banner.** OKF is an interchange adapter at the edges
 *     of the system; embeddings, chunk-level provenance, and verdict /
 *     contradiction edges have no OKF representation and are dropped. That
 *     loss is stated up-front (banner) and, after export, itemised from the
 *     server's omitted-field ledger (the ``X-OKF-Export-Report`` header) so
 *     it is explicit, never silent.
 *   * **Success = ledger.** The post-export success state renders the
 *     omitted-field ledger the server returned, not just the counts.
 *
 * Accessibility notes
 * -------------------
 *   * Radix ``Dialog`` traps focus, restores it on close, binds Esc.
 *   * Sliders report ``aria-valuenow`` via ``@radix-ui/react-slider``.
 *   * The lossy banner carries ``role="note"`` and the ledger a labelled
 *     ``<details>`` disclosure that is keyboard-operable by default.
 */

import { useEffect, useId, useMemo, useState } from 'react'
import { AlertTriangle, Info, Loader2 } from 'lucide-react'

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { useOkfExport } from '@/lib/hooks/use-okf-export'
import type { ExportFilter, OkfExportReport } from '@/lib/types/exports'

import { ExportPreviewCounts } from './ExportPreviewCounts'

interface OkfExportDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  notebookId: string
}

/** Match the Pydantic ExportFilter defaults exactly (shared with D.1c). */
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

export function OkfExportDialog({
  open,
  onOpenChange,
  notebookId,
}: OkfExportDialogProps) {
  const [filter, setFilter] = useState<ExportFilter>(DEFAULT_FILTER)
  const [entityTypesRaw, setEntityTypesRaw] = useState('')
  // Keeps the ledger success state up after a zip export until the user
  // dismisses it (unlike the Obsidian zip branch, which closes immediately —
  // OKF's loss ledger is worth pausing on).
  const [successReport, setSuccessReport] = useState<OkfExportReport | null>(
    null,
  )

  const { mutate, isPending, error } = useOkfExport(notebookId)

  // Reset to a clean state each time the dialog opens.
  useEffect(() => {
    if (open) {
      setFilter(DEFAULT_FILTER)
      setEntityTypesRaw('')
      setSuccessReport(null)
    }
  }, [open])

  const titleId = useId()
  const descriptionId = useId()
  const minConnectionsId = useId()
  const minConfidenceId = useId()
  const minRelationConfidenceId = useId()
  const entityTypesId = useId()
  const orphansId = useId()
  const archivedId = useId()

  const exportDisabled = useMemo(() => isPending, [isPending])

  const handleExport = () => {
    if (exportDisabled) return
    const parsedFilter = withParsedEntityTypes(filter, entityTypesRaw)
    mutate(parsedFilter, {
      onSuccess: (result) => {
        if (result.kind === 'zip') {
          // Show the ledger; keep the dialog open so the user sees the loss.
          setSuccessReport(result.report ?? EMPTY_REPORT)
        } else {
          // Deferred to a job — nothing to download inline; close and let the
          // hook's toast carry the "queued" confirmation.
          onOpenChange(false)
        }
      },
    })
  }

  const close = () => onOpenChange(false)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="sm:max-w-[560px]"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        data-testid="okf-export-dialog"
      >
        <DialogHeader>
          <DialogTitle id={titleId}>Export to Open Knowledge Format</DialogTitle>
          <DialogDescription id={descriptionId}>
            Generate a vendor-neutral OKF v0.1 bundle (Markdown concepts +
            index) from this notebook&rsquo;s knowledge graph — portable to any
            OKF-aware agent. Tune the filter below; the count updates as you
            adjust.
          </DialogDescription>
        </DialogHeader>

        {!successReport && (
          <div className="space-y-4">
            <LossyBanner />

            <SliderRow
              id={minConnectionsId}
              label="Min connections"
              value={filter.min_connections}
              onChange={(v) =>
                setFilter((f) => ({ ...f, min_connections: v }))
              }
              {...SLIDER_RANGES.min_connections}
              format={(v) => String(v)}
              testId="okf-min-connections"
            />
            <SliderRow
              id={minConfidenceId}
              label="Min entity confidence"
              value={filter.min_confidence}
              onChange={(v) =>
                setFilter((f) => ({ ...f, min_confidence: v }))
              }
              {...SLIDER_RANGES.min_confidence}
              format={(v) => v.toFixed(2)}
              testId="okf-min-confidence"
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
              testId="okf-min-relation-confidence"
            />

            <div className="flex items-center gap-3">
              <Switch
                id={orphansId}
                checked={filter.include_orphans}
                onCheckedChange={(checked) =>
                  setFilter((f) => ({ ...f, include_orphans: checked }))
                }
                aria-label="Include orphan entities"
                data-testid="okf-include-orphans-switch"
              />
              <Label htmlFor={orphansId} className="text-sm cursor-pointer">
                Include orphaned entities (pending reconnect)
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
                data-testid="okf-include-archived-switch"
              />
              <Label htmlFor={archivedId} className="text-sm cursor-pointer">
                Include archived entities
              </Label>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor={entityTypesId} className="text-sm">
                Entity types (optional)
              </Label>
              <Input
                id={entityTypesId}
                value={entityTypesRaw}
                placeholder="e.g. Person, Organization"
                onChange={(e) => setEntityTypesRaw(e.target.value)}
                data-testid="okf-entity-types-input"
                aria-label="Entity types filter (comma-separated)"
              />
              <p className="text-xs text-muted-foreground">
                Leave blank for all types. Separate multiple with commas.
              </p>
            </div>

            <ExportPreviewCounts
              notebookId={notebookId}
              filter={withParsedEntityTypes(filter, entityTypesRaw)}
            />

            {error && (
              <div
                role="alert"
                className="flex items-start gap-2 rounded-md border border-destructive/60 bg-destructive/10 p-3 text-sm text-destructive"
                data-testid="okf-export-error"
              >
                <AlertTriangle
                  className="h-4 w-4 mt-0.5 shrink-0"
                  aria-hidden="true"
                />
                <span>{error.message || 'Export failed. Please try again.'}</span>
              </div>
            )}
          </div>
        )}

        {successReport && (
          <OkfSuccess report={successReport} />
        )}

        <DialogFooter className="gap-2 sm:gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={close}
            disabled={isPending}
            data-testid="okf-export-cancel"
          >
            {successReport ? 'Done' : 'Cancel'}
          </Button>
          {!successReport && (
            <Button
              type="button"
              onClick={handleExport}
              disabled={exportDisabled}
              data-testid="okf-export-submit"
              aria-label="Download Open Knowledge Format bundle as zip"
            >
              {isPending && (
                <Loader2
                  className="h-4 w-4 mr-2 animate-spin"
                  aria-hidden="true"
                />
              )}
              {isPending ? 'Exporting…' : 'Export'}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

/** Fallback report so the success state can still render (with an empty
 *  ledger) even when the ``X-OKF-Export-Report`` header was unreadable. */
const EMPTY_REPORT: OkfExportReport = {
  entities_written: 0,
  relations_written: 0,
  files_written: 0,
  bytes_written: 0,
  duration_ms: 0,
  filter_applied: DEFAULT_FILTER,
  metadata: {},
}

/** Up-front lossy-by-design notice. Rendered as ``role="note"`` so screen
 *  readers announce it as advisory context rather than an alert. */
function LossyBanner() {
  return (
    <div
      role="note"
      aria-label="OKF export is lossy by design"
      className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-800 dark:text-amber-200"
      data-testid="okf-lossy-banner"
    >
      <Info className="h-4 w-4 mt-0.5 shrink-0" aria-hidden="true" />
      <span>
        <strong>Lossy by design.</strong> OKF is a portable interchange format:
        embeddings, chunk-level provenance, and verdict / contradiction edges
        are <strong>not included</strong> in the bundle. The full loss ledger
        is shown after export.
      </span>
    </div>
  )
}

interface OkfSuccessProps {
  report: OkfExportReport
}

/** Post-export success state: counts + the itemised omitted-field ledger. */
function OkfSuccess({ report }: OkfSuccessProps) {
  const omitted = report.metadata?.omitted_fields ?? {}
  const omittedEntries = Object.entries(omitted)
  const notesWritten = report.metadata?.notes_written ?? 0
  const sourcesWritten = report.metadata?.sources_written ?? 0
  const okfVersion = report.metadata?.okf_version

  return (
    <div
      className="space-y-3 rounded-md border border-border bg-card p-3"
      data-testid="okf-export-success"
    >
      <div className="text-sm font-medium">
        OKF bundle downloaded
        {okfVersion ? (
          <span className="ml-1 text-xs font-normal text-muted-foreground">
            (OKF {okfVersion})
          </span>
        ) : null}
      </div>
      <div className="text-sm text-muted-foreground">
        Wrote <strong>{report.files_written}</strong> files —{' '}
        <strong>{report.entities_written}</strong> entities,{' '}
        <strong>{report.relations_written}</strong> relations,{' '}
        <strong>{notesWritten}</strong> notes, <strong>{sourcesWritten}</strong>{' '}
        sources.
      </div>
      {report.bytes_written > 0 && (
        <div className="text-xs text-muted-foreground">
          Total {Math.round(report.bytes_written / 1024)} KB in{' '}
          {report.duration_ms} ms.
        </div>
      )}

      {omittedEntries.length > 0 && (
        <details className="group" data-testid="okf-omitted-ledger">
          <summary className="cursor-pointer text-sm font-medium text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-sm">
            Omitted from this bundle ({omittedEntries.length} fields)
          </summary>
          <ul className="mt-2 space-y-1.5 pl-1">
            {omittedEntries.map(([field, reason]) => (
              <li
                key={field}
                className="text-xs text-muted-foreground"
                data-testid="okf-omitted-row"
              >
                <code className="font-mono text-foreground">{field}</code>
                {' — '}
                {reason}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}

/**
 * Parse the comma-separated entity-types input into the array shape the API
 * expects. Empty input → ``undefined`` so the filter omits the param entirely
 * (matches the FastAPI default). Shared idiom with ``ObsidianExportDialog``.
 */
function withParsedEntityTypes(
  filter: ExportFilter,
  raw: string,
): ExportFilter {
  const trimmed = raw.trim()
  if (!trimmed) {
    return { ...filter, entity_types: undefined }
  }
  const types = trimmed
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean)
  return {
    ...filter,
    entity_types: types.length > 0 ? types : undefined,
  }
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
