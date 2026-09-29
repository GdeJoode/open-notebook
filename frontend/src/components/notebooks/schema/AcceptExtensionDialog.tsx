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
 *
 * A name the shared ontologies already define (by name or alias) is not
 * published again: the dialog says where it is defined and offers "Mark as
 * already covered", which takes it out of the queue without a rejection.
 * After a publish that moves existing entities, it links to the retype.
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
  useMarkCovered,
} from '@/lib/hooks/use-notebook-schema'
import type { AcceptExtensionResponse, ExtensionView } from '@/lib/types/notebook_schema'
import { ontologyEditorHref } from '@/components/notebooks/schema/SchemaBrowser'

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
  const covered = useMarkCovered(notebookId)
  // A publish that left existing entities on their old type: offer the retype.
  const [published, setPublished] = useState<NonNullable<AcceptExtensionResponse['published']> | null>(null)
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
  const definedIn = target.data?.defined_in ?? []
  const canPublish =
    !!shown && !shown.blocking && !shown.draft_open && !accept.isPending && !preview.isPending
  const needsRetype = (r: NonNullable<AcceptExtensionResponse['published']>['retype']) =>
    r.changes + r.flag_only > 0 || r.collisions_caused_by_this_edit > 0

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

        {published ? (
          <div className="space-y-2 text-sm" data-testid="accept-retype-offer">
            <p>
              {typeName} is now a type of {published.name} (version {published.version}).
              {' '}{published.retype.changes} existing entit(y/ies) still carry their old type.
            </p>
            <Link className="underline" data-testid="accept-retype-link"
              href={`${ontologyEditorHref(published.name)}&retype=1`}>
              Review the retype in the ontology editor
            </Link>
          </div>
        ) : target.isLoading ? (
          <div role="status" aria-live="polite" className="flex justify-center py-6">
            <LoadingSpinner />
          </div>
        ) : target.isError || !target.data ? (
          <p role="alert" className="text-sm text-destructive">
            Could not load the proposal: {errorDetail(target.error)}
          </p>
        ) : definedIn.length > 0 ? (
          <div className="space-y-2 text-sm" data-testid="accept-already-defined">
            <p>
              {typeName} is already a shared type, so it is not published again:
            </p>
            <ul className="list-disc pl-5">
              {definedIn.map((d) => (
                <li key={`${d.ontology}:${d.type}`}>
                  <Link className="underline" href={ontologyEditorHref(d.ontology, d.type)}>
                    {d.type}
                  </Link>{' '}in {d.ontology}
                </li>
              ))}
            </ul>
            <p className="text-muted-foreground">
              Mark it as already covered to take it out of the queue. That is not a
              rejection: nothing is recorded against the name.
            </p>
            {covered.isError && (
              <p role="alert" className="text-destructive">
                Could not mark it: {errorDetail(covered.error)}
              </p>
            )}
          </div>
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
            {published ? 'Close' : 'Cancel'}
          </Button>
          {!published && definedIn.length > 0 && (
            <Button
              type="button"
              data-testid="accept-mark-covered"
              disabled={covered.isPending}
              onClick={() => covered.mutate(typeName, { onSuccess: () => onOpenChange(false) })}
            >
              {covered.isPending ? 'Marking…' : 'Mark as already covered'}
            </Button>
          )}
          {!published && definedIn.length === 0 && (<>
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
            onClick={() => accept.mutate(vars, {
              onSuccess: (data) => {
                if (data.published && needsRetype(data.published.retype)) setPublished(data.published)
                else onOpenChange(false)
              },
            })}
          >
            {accept.isPending ? 'Publishing…' : `Publish to ${ontology || '…'}`}
          </Button>
          </>)}
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
