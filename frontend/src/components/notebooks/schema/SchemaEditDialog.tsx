'use client'

/**
 * SchemaEditDialog — modal for the four schema-edit ops introduced in B.3b.
 *
 * One dialog, four shapes, switched via the `mode` prop:
 *
 *   mode="rename"  → single old → new field pair.
 *   mode="merge"   → comma-separated source types + a target name.
 *   mode="split"   → source type + comma-separated targets + criterion.
 *   mode="delete"  → confirmation only (soft-delete).
 *
 * Each mode calls the matching mutation hook from
 * `lib/hooks/use-notebook-schema.ts`. The dialog stays open until the
 * mutation settles to keep the loading state visible — `onOpenChange`
 * fires on cancel + on successful submit.
 *
 * # Design decisions
 *
 *  1. **One component, multiple modes** matches the plan instruction
 *     (`mode` prop). Keeps the API surface in `SchemaBrowser` tight —
 *     a single `<SchemaEditDialog>` import per edit op.
 *  2. **No `react-hook-form` for this dialog** — each form has at most
 *     three fields and bespoke validation rules (e.g. `into` must have
 *     2+ distinct items). Hand-rolled state is simpler than mapping
 *     RHF's primitives onto comma-separated lists.
 *  3. **`comma-separated → string[]` parser is centralised** in
 *     `parseTypeList`. Strips whitespace, drops empties, dedupes.
 *  4. **Keyboard accessibility** is inherited from Radix Dialog (focus
 *     trap, Escape to close, focus restoration). The submit button is
 *     focusable + activatable via Enter; Cancel via Esc.
 */

import { useEffect, useId, useState } from 'react'
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
import { Textarea } from '@/components/ui/textarea'
import {
  useDeleteType,
  useMergeTypes,
  useRenameType,
  useSplitType,
} from '@/lib/hooks/use-notebook-schema'

export type SchemaEditMode = 'rename' | 'merge' | 'split' | 'delete'

interface SchemaEditDialogProps {
  notebookId: string
  mode: SchemaEditMode
  /** Pre-populated type name (the row the user invoked the menu on). */
  typeName: string
  open: boolean
  onOpenChange: (open: boolean) => void
}

/**
 * Parse a comma-separated list of type names into a deduped string[].
 *
 * Defensive of:
 *  - leading / trailing whitespace per entry,
 *  - empty entries (`"A,,B"` → `["A","B"]`),
 *  - duplicates (case-sensitive — keep "User" and "user" distinct since
 *    the backend treats them as distinct identifiers).
 */
export function parseTypeList(raw: string): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const part of raw.split(',')) {
    const trimmed = part.trim()
    if (trimmed.length === 0) continue
    if (seen.has(trimmed)) continue
    seen.add(trimmed)
    out.push(trimmed)
  }
  return out
}

export function SchemaEditDialog({
  notebookId,
  mode,
  typeName,
  open,
  onOpenChange,
}: SchemaEditDialogProps) {
  // One state slot per field. Reset on open/close transitions so a
  // re-open doesn't carry stale values.
  const [newName, setNewName] = useState(typeName)
  const [mergeSources, setMergeSources] = useState('')
  const [mergedName, setMergedName] = useState('')
  const [splitInto, setSplitInto] = useState('')
  const [splitCriterion, setSplitCriterion] = useState('')

  // Reset state when the dialog opens with a new target type so the
  // previous edit's input doesn't bleed across invocations.
  useEffect(() => {
    if (open) {
      setNewName(typeName)
      setMergeSources(typeName)
      setMergedName('')
      setSplitInto('')
      setSplitCriterion('')
    }
  }, [open, typeName])

  const renameMutation = useRenameType(notebookId)
  const mergeMutation = useMergeTypes(notebookId)
  const splitMutation = useSplitType(notebookId)
  const deleteMutation = useDeleteType(notebookId)

  const pending =
    renameMutation.isPending ||
    mergeMutation.isPending ||
    splitMutation.isPending ||
    deleteMutation.isPending

  const titleId = useId()
  const descriptionId = useId()

  const close = () => onOpenChange(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (pending) return

    try {
      if (mode === 'rename') {
        if (!newName.trim() || newName.trim() === typeName) {
          // No-op; the backend would short-circuit but we skip the call
          // entirely to keep the toast noise down.
          close()
          return
        }
        await renameMutation.mutateAsync({
          old_name: typeName,
          new_name: newName.trim(),
        })
      } else if (mode === 'merge') {
        const sources = parseTypeList(mergeSources)
        if (sources.length < 2 || !mergedName.trim()) return
        await mergeMutation.mutateAsync({
          type_names: sources,
          merged_name: mergedName.trim(),
        })
      } else if (mode === 'split') {
        const into = parseTypeList(splitInto)
        if (into.length < 2 || !splitCriterion.trim()) return
        await splitMutation.mutateAsync({
          type_name: typeName,
          into,
          criterion: splitCriterion.trim(),
        })
      } else if (mode === 'delete') {
        await deleteMutation.mutateAsync(typeName)
      }
      close()
    } catch {
      // The hook surfaces a toast; we leave the dialog open so the user
      // can correct + retry without re-opening from the overflow menu.
    }
  }

  const { title, description } = describeMode(mode, typeName)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="sm:max-w-[480px]"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
      >
        <DialogHeader>
          <DialogTitle id={titleId}>{title}</DialogTitle>
          <DialogDescription id={descriptionId}>{description}</DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          {mode === 'rename' && (
            <div className="space-y-2">
              <Label htmlFor="rename-new-name">New name</Label>
              <Input
                id="rename-new-name"
                data-testid="schema-edit-rename-input"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                autoFocus
                disabled={pending}
              />
            </div>
          )}

          {mode === 'merge' && (
            <>
              <div className="space-y-2">
                <Label htmlFor="merge-sources">
                  Source types (comma-separated)
                </Label>
                <Input
                  id="merge-sources"
                  data-testid="schema-edit-merge-sources"
                  value={mergeSources}
                  onChange={(e) => setMergeSources(e.target.value)}
                  placeholder="Author, Editor"
                  autoFocus
                  disabled={pending}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="merge-target">Merged name</Label>
                <Input
                  id="merge-target"
                  data-testid="schema-edit-merge-target"
                  value={mergedName}
                  onChange={(e) => setMergedName(e.target.value)}
                  placeholder="Contributor"
                  disabled={pending}
                />
              </div>
            </>
          )}

          {mode === 'split' && (
            <>
              <div className="space-y-2">
                <Label htmlFor="split-into">
                  New types (comma-separated, at least two)
                </Label>
                <Input
                  id="split-into"
                  data-testid="schema-edit-split-into"
                  value={splitInto}
                  onChange={(e) => setSplitInto(e.target.value)}
                  placeholder="StudyCohort, ControlCohort"
                  autoFocus
                  disabled={pending}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="split-criterion">Split criterion</Label>
                <Textarea
                  id="split-criterion"
                  data-testid="schema-edit-split-criterion"
                  value={splitCriterion}
                  onChange={(e) => setSplitCriterion(e.target.value)}
                  placeholder="How should Pass 2 decide which one applies?"
                  rows={3}
                  disabled={pending}
                />
              </div>
            </>
          )}

          {mode === 'delete' && (
            <p className="text-sm text-muted-foreground">
              The base ontology will not be modified — this type is hidden
              from the effective schema only. You can restore it by
              accepting an extension proposal with the same name.
            </p>
          )}

          <DialogFooter className="gap-2 sm:gap-0">
            <Button
              type="button"
              variant="outline"
              onClick={close}
              disabled={pending}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              data-testid="schema-edit-submit"
              variant={mode === 'delete' ? 'destructive' : 'default'}
              disabled={pending || !isFormValid(mode, {
                newName,
                mergeSources,
                mergedName,
                splitInto,
                splitCriterion,
              })}
            >
              {pending ? 'Saving…' : confirmLabel(mode)}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function describeMode(mode: SchemaEditMode, typeName: string) {
  switch (mode) {
    case 'rename':
      return {
        title: `Rename "${typeName}"`,
        description:
          'A rename is recorded as a synonym; the base ontology is not mutated.',
      }
    case 'merge':
      return {
        title: 'Merge types',
        description:
          'Future extractions will treat the source types as a single merged type.',
      }
    case 'split':
      return {
        title: `Split "${typeName}"`,
        description:
          'Future Pass 2 prompts will receive your criterion to disambiguate instances.',
      }
    case 'delete':
      return {
        title: `Hide "${typeName}"?`,
        description:
          'Soft-delete: hidden from the effective schema until restored. No data is lost.',
      }
  }
}

function confirmLabel(mode: SchemaEditMode): string {
  switch (mode) {
    case 'rename':
      return 'Rename'
    case 'merge':
      return 'Merge'
    case 'split':
      return 'Split'
    case 'delete':
      return 'Hide type'
  }
}

interface FormState {
  newName: string
  mergeSources: string
  mergedName: string
  splitInto: string
  splitCriterion: string
}

function isFormValid(mode: SchemaEditMode, state: FormState): boolean {
  switch (mode) {
    case 'rename':
      return state.newName.trim().length > 0
    case 'merge':
      return (
        parseTypeList(state.mergeSources).length >= 2 &&
        state.mergedName.trim().length > 0
      )
    case 'split':
      return (
        parseTypeList(state.splitInto).length >= 2 &&
        state.splitCriterion.trim().length > 0
      )
    case 'delete':
      return true
  }
}
