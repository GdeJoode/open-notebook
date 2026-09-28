'use client'

/**
 * OE.5: accept a Pass-1 proposal into a SHARED ontology.
 *
 * The ontologies are shared by every notebook, so accepting a proposal adds
 * the type to one of them, through the ontology editor's gate: the curator
 * picks the ontology (default: the one that introduces the proposed parent,
 * else `schema_core`), may change the parent, previews what the publish would
 * do, and publishes. A blocking finding refuses it and is shown here; nothing
 * is stored then. An open draft of that ontology in the editor also refuses it
 * (the publish would leave the draft stale).
 */

import Link from 'next/link'
import { useEffect, useId, useState } from 'react'

import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import type { Finding } from '@/lib/api/ontologies'
import { errorDetail } from '@/lib/hooks/use-ontology-editor'
import {
  useAcceptExtension,
  useAcceptPreview,
  useAcceptTarget,
} from '@/lib/hooks/use-notebook-schema'
import type { ExtensionView } from '@/lib/types/notebook_schema'

interface AcceptExtensionDialogProps {
  notebookId: string
  extension: ExtensionView | null
  onOpenChange: (open: boolean) => void
}

/** The gate's findings from a refused publish (422 `{message, findings}`). */
function refusedFindings(err: unknown): Finding[] {
  const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail
  if (detail && typeof detail === 'object' && 'findings' in detail) {
    return ((detail as { findings?: Finding[] }).findings ?? [])
  }
  return []
}

export function AcceptExtensionDialog({
  notebookId,
  extension,
  onOpenChange,
}: AcceptExtensionDialogProps) {
  const typeName = extension?.type_name ?? null
  const target = useAcceptTarget(notebookId, typeName)
  const preview = useAcceptPreview(notebookId)
  const accept = useAcceptExtension(notebookId)
  const [ontology, setOntology] = useState('')
  const [parent, setParent] = useState('')
  const ontologyId = useId()
  const parentId = useId()

  // Start from the server's default each time a proposal is opened.
  useEffect(() => {
    if (target.data) {
      setOntology(target.data.default_ontology)
      setParent(target.data.parent_type ?? '')
    }
  }, [target.data])
  useEffect(() => {
    preview.reset()
    accept.reset()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [typeName, ontology, parent])

  if (!extension || !typeName) return null

  const vars = { typeName, ontology, parentType: parent.trim() }
  const shown = preview.data
  const blocking = (shown?.findings ?? []).filter((f) => f.severity === 'block')
  const refused = accept.error ? refusedFindings(accept.error) : []
  const canPublish =
    !!shown && !shown.blocking && !shown.draft_open && !accept.isPending && !preview.isPending

  return (
    <Dialog open onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl" data-testid="accept-extension-dialog">
        <DialogHeader>
          <DialogTitle>Accept {typeName} into a shared ontology</DialogTitle>
          <DialogDescription>
            The ontologies are shared by every notebook: the type is added to the
            ontology you pick and types entities in every notebook from the next
            extraction on.
          </DialogDescription>
        </DialogHeader>

        {target.isLoading ? (
          <div role="status" aria-live="polite" className="flex justify-center py-6">
            <LoadingSpinner />
          </div>
        ) : target.isError || !target.data ? (
          <p role="alert" className="text-sm text-destructive">
            Could not load the proposal: {errorDetail(target.error)}
          </p>
        ) : (
          <div className="space-y-3 text-sm">
            <div className="space-y-1">
              <Label htmlFor={ontologyId}>Ontology</Label>
              <select
                id={ontologyId}
                data-testid="accept-ontology"
                value={ontology}
                onChange={(e) => setOntology(e.target.value)}
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                {target.data.ontologies.map((name) => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
              <p className="text-xs text-muted-foreground" data-testid="accept-default-note">
                Default: <code>{target.data.default_ontology}</code>, the ontology that
                introduces the proposed parent (<code>schema_core</code> when none does).
              </p>
            </div>
            <div className="space-y-1">
              <Label htmlFor={parentId}>Parent type</Label>
              <Input
                id={parentId}
                data-testid="accept-parent"
                value={parent}
                placeholder="none"
                onChange={(e) => setParent(e.target.value)}
              />
            </div>

            {shown && (
              <div className="space-y-1 rounded-md border p-3" data-testid="accept-preview-result">
                {shown.draft_open && (
                  <p role="alert" className="text-destructive" data-testid="accept-draft-open">
                    {ontology} has an unpublished draft in the{' '}
                    <Link className="underline" href={`/ontologies?name=${encodeURIComponent(ontology)}&tab=edit`}>
                      ontology editor
                    </Link>
                    ; publish or discard it first.
                  </p>
                )}
                {blocking.length > 0 ? (
                  <FindingLines findings={blocking} testId="accept-preview-blocking" />
                ) : (
                  <p data-testid="accept-preview-clean">No blocking findings.</p>
                )}
                <p className="text-muted-foreground">
                  {shown.label_changes.length} label(s) change type;{' '}
                  {shown.retype.caused_by_this_edit} existing entit(y/ies) would be retyped.
                </p>
              </div>
            )}

            {accept.isError && (
              <div role="alert" className="text-destructive" data-testid="accept-refused">
                <p>Not published: {errorDetail(accept.error)}</p>
                {refused.length > 0 && <FindingLines findings={refused.filter((f) => f.severity === 'block')} />}
              </div>
            )}
            {preview.isError && (
              <p role="alert" className="text-destructive">
                Preview failed: {errorDetail(preview.error)}
              </p>
            )}
          </div>
        )}

        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            type="button"
            variant="outline"
            data-testid="accept-preview"
            disabled={!ontology || preview.isPending || !target.data}
            onClick={() => preview.mutate(vars)}
          >
            {preview.isPending ? 'Previewing…' : 'Preview'}
          </Button>
          <Button
            type="button"
            data-testid="accept-publish"
            disabled={!canPublish}
            onClick={() => accept.mutate(vars, { onSuccess: () => onOpenChange(false) })}
          >
            {accept.isPending ? 'Publishing…' : `Publish to ${ontology || '…'}`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function FindingLines({ findings, testId }: { findings: Finding[]; testId?: string }) {
  return (
    <ul className="space-y-0.5" data-testid={testId}>
      {findings.map((f, i) => (
        <li key={i} className="text-destructive">
          <strong>{f.rule}</strong>: {f.detail}
        </li>
      ))}
    </ul>
  )
}
