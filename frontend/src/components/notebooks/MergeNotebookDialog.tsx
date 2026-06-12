'use client'

/**
 * MergeNotebookDialog — modal for the B.6 cross-notebook graph merge.
 *
 * The dialog is opened from the per-notebook overflow menu with that
 * notebook as the TARGET — operators pick one-or-more SOURCE notebooks
 * to merge INTO the current one.
 *
 * Two-stage UX (matches the AC's "dry-run preview" requirement):
 *
 *   1. User picks sources → clicks "Preview" → fires a `dry_run=true`
 *      mutation → counts + conflicts render under the source list.
 *   2. User clicks "Merge" → fires the same call with `dry_run=false`
 *      → success toast + dialog closes.
 *
 * Conflict UX:
 *
 *   When `type_match_required` is on (default), the dry-run can return
 *   a non-empty conflicts list (e.g. "Smith" appears as both
 *   Organization and Person). We render them as warning rows so the
 *   operator sees them BEFORE committing. The user can flip the toggle
 *   to merge anyway, then re-Preview.
 *
 * Design decisions:
 *
 *   1. **Sources panel uses `CheckboxList`** (the existing primitive
 *      shared with the search/save-to-notebooks dialog). Keeps the
 *      visual language consistent with other multi-select dialogs.
 *   2. **No react-hook-form** — two pieces of state (selected ids,
 *      type-match flag) and one derived (preview result). RHF would
 *      add ceremony without ergonomic gain.
 *   3. **The preview is held in component state**, not a separate
 *      query. The mutation already returns the report; storing it
 *      locally avoids a second roundtrip when the user clicks
 *      "Merge" after a successful preview.
 */

import { useEffect, useId, useMemo, useState } from 'react'
import { AlertTriangle } from 'lucide-react'

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { CheckboxList } from '@/components/ui/checkbox-list'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { useMergeNotebooks, useNotebooks } from '@/lib/hooks/use-notebooks'
import type {
  NotebookMergeConflict,
  NotebookMergeReport,
} from '@/lib/api/notebooks'

interface MergeNotebookDialogProps {
  /** Target notebook — receives the merged entities + relations. */
  targetNotebookId: string
  /** Display name for the target — used in the dialog title. */
  targetNotebookName: string
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function MergeNotebookDialog({
  targetNotebookId,
  targetNotebookName,
  open,
  onOpenChange,
}: MergeNotebookDialogProps) {
  const [selectedSourceIds, setSelectedSourceIds] = useState<string[]>([])
  const [typeMatchRequired, setTypeMatchRequired] = useState(true)
  const [preview, setPreview] = useState<NotebookMergeReport | null>(null)

  // Fetch the active notebook list to populate the source picker.
  // Archived notebooks are EXCLUDED — merging into / from archived
  // notebooks is a separate workflow and would clutter the picker.
  const { data: notebooks, isLoading: notebooksLoading } = useNotebooks(false)

  const mergeMutation = useMergeNotebooks()

  // Reset state when the dialog opens so a re-open after a previous
  // run starts clean. We DON'T clear on close — the cleanup happens
  // on open instead, so the dialog can be re-opened with the same
  // selection if the user cancels accidentally.
  useEffect(() => {
    if (open) {
      setSelectedSourceIds([])
      setTypeMatchRequired(true)
      setPreview(null)
    }
  }, [open])

  // Exclude the target itself from the picker — merging a notebook
  // into itself is meaningless.
  const sourceCandidates = useMemo(() => {
    if (!notebooks) return []
    return notebooks
      .filter((nb) => nb.id !== targetNotebookId)
      .map((nb) => ({
        id: nb.id,
        title: nb.name,
        description: nb.description || undefined,
      }))
  }, [notebooks, targetNotebookId])

  const handleToggle = (id: string) => {
    setSelectedSourceIds((current) =>
      current.includes(id)
        ? current.filter((x) => x !== id)
        : [...current, id],
    )
    // Invalidate the preview when the selection changes — the counts
    // no longer reflect the new selection.
    setPreview(null)
  }

  const handlePreview = async () => {
    if (selectedSourceIds.length === 0) return
    try {
      const report = await mergeMutation.mutateAsync({
        source_ids: selectedSourceIds,
        target_id: targetNotebookId,
        type_match_required: typeMatchRequired,
        dry_run: true,
      })
      setPreview(report)
    } catch {
      // Toast surfaced by the hook; keep the dialog open.
    }
  }

  const handleMerge = async () => {
    if (selectedSourceIds.length === 0) return
    try {
      await mergeMutation.mutateAsync({
        source_ids: selectedSourceIds,
        target_id: targetNotebookId,
        type_match_required: typeMatchRequired,
        dry_run: false,
      })
      onOpenChange(false)
    } catch {
      // Toast surfaced by the hook; keep the dialog open.
    }
  }

  const close = () => onOpenChange(false)

  const titleId = useId()
  const descriptionId = useId()

  const pending = mergeMutation.isPending
  const canPreview = selectedSourceIds.length > 0 && !pending
  const canMerge = canPreview // Same prerequisites; we don't gate on preview.

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="sm:max-w-[560px]"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
      >
        <DialogHeader>
          <DialogTitle id={titleId}>
            Merge into &ldquo;{targetNotebookName}&rdquo;
          </DialogTitle>
          <DialogDescription id={descriptionId}>
            Aggregate entities + relations from one or more source notebooks
            into this notebook. Re-running the same merge is a no-op.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label>Source notebooks</Label>
            <CheckboxList
              items={sourceCandidates}
              selectedIds={selectedSourceIds}
              onToggle={handleToggle}
              loading={notebooksLoading}
              emptyMessage="No other notebooks available to merge from."
            />
          </div>

          <div className="flex items-center gap-3">
            <Switch
              id="merge-type-match"
              checked={typeMatchRequired}
              onCheckedChange={(checked) => {
                setTypeMatchRequired(checked)
                // Flag changes the merge semantics — reset preview.
                setPreview(null)
              }}
              aria-label="Require matching entity types"
              data-testid="merge-type-match-switch"
            />
            <Label
              htmlFor="merge-type-match"
              className="text-sm cursor-pointer"
            >
              Require matching entity types (otherwise: union conflicting tags)
            </Label>
          </div>

          {preview && (
            <MergePreviewPanel report={preview} />
          )}
        </div>

        <DialogFooter className="gap-2 sm:gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={close}
            disabled={pending}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="secondary"
            onClick={handlePreview}
            disabled={!canPreview}
            data-testid="merge-dialog-preview"
          >
            {pending && mergeMutation.variables?.dry_run
              ? 'Previewing…'
              : 'Preview'}
          </Button>
          <Button
            type="button"
            onClick={handleMerge}
            disabled={!canMerge}
            data-testid="merge-dialog-confirm"
          >
            {pending && mergeMutation.variables?.dry_run === false
              ? 'Merging…'
              : 'Merge'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

interface MergePreviewPanelProps {
  report: NotebookMergeReport
}

/**
 * Renders the dry-run preview's counters + conflict list.
 *
 * Extracted as a sub-component so the main dialog stays scannable.
 * Conflict rows are styled as warnings (amber border) so they stand
 * out — the operator should reconcile them before committing.
 */
function MergePreviewPanel({ report }: MergePreviewPanelProps) {
  return (
    <div
      data-testid="merge-dialog-preview-panel"
      className="rounded-md border border-border bg-card p-3 space-y-2"
    >
      <div className="text-sm font-medium">Preview</div>
      <div className="text-sm text-muted-foreground">
        <div data-testid="merge-preview-entities">
          {report.entities_merged} entities would be merged
        </div>
        <div data-testid="merge-preview-relations">
          {report.relations_created} relations would be created
        </div>
      </div>

      {report.conflicts.length > 0 && (
        <div className="mt-3 space-y-2">
          <div className="text-sm font-medium flex items-center gap-1.5 text-amber-600 dark:text-amber-400">
            <AlertTriangle className="h-4 w-4" aria-hidden />
            <span>{report.conflicts.length} type conflict{report.conflicts.length === 1 ? '' : 's'}</span>
          </div>
          <ul
            data-testid="merge-preview-conflicts"
            className="text-xs space-y-1 max-h-32 overflow-y-auto"
          >
            {report.conflicts.map((conflict, idx) => (
              <ConflictRow key={`${conflict.normalized_name}-${idx}`} conflict={conflict} />
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function ConflictRow({ conflict }: { conflict: NotebookMergeConflict }) {
  return (
    <li className="rounded-sm border border-amber-300/40 bg-amber-50/40 dark:bg-amber-900/10 px-2 py-1">
      <span className="font-mono">{conflict.normalized_name}</span>
      <span className="text-muted-foreground">
        {' '}
        — tags{' '}
        {conflict.conflicting_type_tags.join(', ')}
      </span>
    </li>
  )
}
