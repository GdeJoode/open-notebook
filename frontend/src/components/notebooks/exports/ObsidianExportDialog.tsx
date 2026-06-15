'use client'

/**
 * ObsidianExportDialog — Track D Phase D.1c.
 *
 * Single dialog exposing the full Obsidian export surface:
 *
 *   * Mode toggle: "Download ZIP" vs "Write to Vault Path".
 *   * Three sliders: ``min_connections``, ``min_confidence``,
 *     ``min_relation_confidence``.
 *   * Two switches: ``include_orphans``, ``include_archived``.
 *   * Entity-types: comma-separated input (V1 simplification per
 *     plan §D.1c -- the multi-select can land later without breaking
 *     the persisted filter shape).
 *   * Live preview-counts widget hooked to the debounced
 *     ``GET /export-preview`` endpoint.
 *
 * On submit the dialog calls ``useObsidianExport().mutate(...)``.
 * The hook branches on response Content-Type so this dialog doesn't
 * need to know whether the user picked zip vs vault_path:
 *
 *   * Zip mode -> browser download fires automatically; the dialog
 *     closes once the mutation resolves.
 *   * Vault-path mode -> a success state with the ``lastReport`` body
 *     stays open until the user clicks "Done".
 *
 * Accessibility notes
 * -------------------
 *
 *   * Radix ``Dialog`` traps focus, restores it on close, and binds
 *     Esc by default.
 *   * Sliders report ``aria-valuenow`` automatically through
 *     ``@radix-ui/react-slider``.
 *   * The ``vault_path`` Tab is wrapped in a tooltip explaining the
 *     "configure in Settings" requirement when the path is empty;
 *     the disabled state is `disabled` on the Tab trigger so the
 *     Export button can't fire the mutation without a configured
 *     path (regression-test inversion: button must be DISABLED, not
 *     just hidden).
 */

import { useEffect, useId, useMemo, useState } from 'react'
import { AlertTriangle, Loader2 } from 'lucide-react'

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
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { useObsidianExport } from '@/lib/hooks/use-obsidian-export'
import { useExportPreview } from '@/lib/hooks/use-export-preview'
import { useSettings } from '@/lib/hooks/use-settings'
import type {
  ExportFilter,
  ObsidianExportMode,
} from '@/lib/types/exports'

import { ExportPreviewCounts } from './ExportPreviewCounts'

interface ObsidianExportDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  notebookId: string
}

/** Match the Pydantic ExportFilter defaults exactly. */
const DEFAULT_FILTER: ExportFilter = {
  min_connections: 5,
  min_confidence: 0.9,
  min_relation_confidence: 0.9,
  entity_types: undefined,
  include_orphans: false,
  include_archived: false,
}

const SLIDER_RANGES = {
  // 0..20 mirrors the typical "minimum degree" knob the user would
  // tune for a connected subgraph. Beyond ~20 the export becomes
  // unrepresentative for typical notebooks (D.3 default fixtures
  // top out at degree-12 entities).
  min_connections: { min: 0, max: 20, step: 1 },
  min_confidence: { min: 0, max: 1, step: 0.05 },
  min_relation_confidence: { min: 0, max: 1, step: 0.05 },
} as const

export function ObsidianExportDialog({
  open,
  onOpenChange,
  notebookId,
}: ObsidianExportDialogProps) {
  const [mode, setMode] = useState<ObsidianExportMode>('zip')
  const [filter, setFilter] = useState<ExportFilter>(DEFAULT_FILTER)
  // ``successVisible`` keeps the post-export success banner up after a
  // vault_path run. Zip-mode runs close the dialog instead so this
  // state isn't read in that branch.
  const [successVisible, setSuccessVisible] = useState(false)
  // Entity-types are stored as a comma-separated string in the UI for
  // V1 (per plan §D.1c) and parsed to an array right before submit.
  // Keeping the source-of-truth as the raw string lets the user type
  // without us aggressively trimming/parsing on every keystroke.
  const [entityTypesRaw, setEntityTypesRaw] = useState('')

  const { data: settings } = useSettings()
  const vaultPathConfigured = Boolean(settings?.vault_path?.trim())
  const vaultPath = settings?.vault_path?.trim() ?? ''

  const { mutate, isPending, error, lastReport } = useObsidianExport(notebookId)
  const previewQuery = useExportPreview(notebookId, withParsedEntityTypes(filter, entityTypesRaw))

  // Reset state on open so a re-open after a prior export starts
  // clean. Same idiom as MergeNotebookDialog.
  useEffect(() => {
    if (open) {
      setMode('zip')
      setFilter(DEFAULT_FILTER)
      setEntityTypesRaw('')
      setSuccessVisible(false)
    }
  }, [open])

  // Vault-path mode flips to disabled when the operator hasn't
  // configured it in Settings -- defer to zip in that case so the
  // dialog can't enter an unrecoverable state through state churn.
  useEffect(() => {
    if (mode === 'vault_path' && !vaultPathConfigured) {
      setMode('zip')
    }
  }, [mode, vaultPathConfigured])

  const titleId = useId()
  const descriptionId = useId()
  const minConnectionsId = useId()
  const minConfidenceId = useId()
  const minRelationConfidenceId = useId()
  const entityTypesId = useId()
  const orphansId = useId()
  const archivedId = useId()

  // Derive what state to disable the Export button in. The vault_path
  // + unset-vault check is the inversion-test target: even if the
  // operator clicks a disabled tab, the actual disabled state must
  // come from this expression so the mutation cannot fire.
  const exportDisabled = useMemo(() => {
    if (isPending) return true
    if (previewQuery.isLoading) return true
    if (mode === 'vault_path' && !vaultPathConfigured) return true
    return false
  }, [isPending, previewQuery.isLoading, mode, vaultPathConfigured])

  const handleExport = () => {
    if (exportDisabled) return
    const parsedFilter = withParsedEntityTypes(filter, entityTypesRaw)
    mutate(
      { mode, filter: parsedFilter },
      {
        onSuccess: () => {
          if (mode === 'zip') {
            // Browser download is the visible confirmation; close
            // the dialog so the user can resume work.
            onOpenChange(false)
          } else {
            // Vault-path: show the success state with counts.
            setSuccessVisible(true)
          }
        },
      },
    )
  }

  const close = () => onOpenChange(false)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="sm:max-w-[560px]"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        data-testid="obsidian-export-dialog"
      >
        <DialogHeader>
          <DialogTitle id={titleId}>Export to Obsidian</DialogTitle>
          <DialogDescription id={descriptionId}>
            Generate a flat Obsidian vault from this notebook&rsquo;s
            knowledge graph. Tune the filter below; the count updates
            as you adjust.
          </DialogDescription>
        </DialogHeader>

        {!successVisible && (
          <div className="space-y-4">
            <ModeToggle
              mode={mode}
              onChange={setMode}
              vaultPath={vaultPath}
              vaultPathConfigured={vaultPathConfigured}
            />

            <SliderRow
              id={minConnectionsId}
              label="Min connections"
              value={filter.min_connections}
              onChange={(v) =>
                setFilter((f) => ({ ...f, min_connections: v }))
              }
              {...SLIDER_RANGES.min_connections}
              format={(v) => String(v)}
              testId="min-connections"
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
              testId="min-confidence"
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
              testId="min-relation-confidence"
            />

            <div className="flex items-center gap-3">
              <Switch
                id={orphansId}
                checked={filter.include_orphans}
                onCheckedChange={(checked) =>
                  setFilter((f) => ({ ...f, include_orphans: checked }))
                }
                aria-label="Include orphan entities"
                data-testid="include-orphans-switch"
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
                data-testid="include-archived-switch"
              />
              <Label htmlFor={archivedId} className="text-sm cursor-pointer">
                Include archived entities
              </Label>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor={entityTypesId} className="text-sm">
                Entity types (optional)
              </Label>
              {/* v1: comma-separated input. The multi-select can land
                  in a follow-up without breaking the persisted shape
                  since ExportFilter.entity_types stays an array. */}
              <Input
                id={entityTypesId}
                value={entityTypesRaw}
                placeholder="e.g. Person, Organization"
                onChange={(e) => setEntityTypesRaw(e.target.value)}
                data-testid="entity-types-input"
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
                data-testid="obsidian-export-error"
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

        {successVisible && lastReport && (
          <div
            className="space-y-2 rounded-md border border-border bg-card p-3"
            data-testid="obsidian-export-success"
          >
            <div className="text-sm font-medium">Export complete</div>
            <div className="text-sm text-muted-foreground">
              Wrote <strong>{lastReport.files_written}</strong> files —{' '}
              <strong>{lastReport.entities_written}</strong> entities,{' '}
              <strong>{lastReport.relations_written}</strong> relations.
            </div>
            <div className="text-xs text-muted-foreground">
              Total {Math.round(lastReport.bytes_written / 1024)} KB in{' '}
              {lastReport.duration_ms} ms.
            </div>
          </div>
        )}

        <DialogFooter className="gap-2 sm:gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={close}
            disabled={isPending}
            data-testid="obsidian-export-cancel"
          >
            {successVisible ? 'Done' : 'Cancel'}
          </Button>
          {!successVisible && (
            <Button
              type="button"
              onClick={handleExport}
              disabled={exportDisabled}
              data-testid="obsidian-export-submit"
              aria-label={
                mode === 'zip'
                  ? 'Download Obsidian vault as zip'
                  : 'Write Obsidian vault to configured path'
              }
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

/**
 * Parse the comma-separated entity-types input into the array shape
 * the API + filter type expect. Empty input -> ``undefined`` so the
 * filter omits the param entirely (matches the FastAPI default).
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

interface ModeToggleProps {
  mode: ObsidianExportMode
  onChange: (mode: ObsidianExportMode) => void
  vaultPath: string
  vaultPathConfigured: boolean
}

function ModeToggle({
  mode,
  onChange,
  vaultPath,
  vaultPathConfigured,
}: ModeToggleProps) {
  return (
    <div className="space-y-2">
      <Label>Delivery mode</Label>
      <Tabs
        value={mode}
        onValueChange={(v) => onChange(v as ObsidianExportMode)}
        data-testid="mode-toggle"
      >
        <TabsList className="w-full">
          <TabsTrigger
            value="zip"
            data-testid="mode-zip"
            aria-label="Download as zip archive"
          >
            Download ZIP
          </TabsTrigger>
          {vaultPathConfigured ? (
            <TabsTrigger
              value="vault_path"
              data-testid="mode-vault-path"
              aria-label="Write directly to configured vault path"
            >
              Write to Vault
            </TabsTrigger>
          ) : (
            // Render the disabled trigger inside a tooltip so the user
            // understands WHY they can't pick it. Radix won't fire
            // ``onValueChange`` for a disabled trigger, so the
            // controlled state can't flip to ``vault_path``.
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="flex-1">
                    <TabsTrigger
                      value="vault_path"
                      disabled
                      data-testid="mode-vault-path"
                      aria-label="Vault path mode (disabled until configured)"
                      className="w-full"
                    >
                      Write to Vault
                    </TabsTrigger>
                  </span>
                </TooltipTrigger>
                <TooltipContent>
                  Configure vault_path in Settings first.
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </TabsList>
      </Tabs>
      {mode === 'vault_path' && vaultPathConfigured && (
        <p
          className="text-xs text-muted-foreground font-mono break-all"
          data-testid="vault-path-display"
        >
          Vault: {vaultPath}
        </p>
      )}
    </div>
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
