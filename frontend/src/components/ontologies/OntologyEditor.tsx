'use client'

/**
 * OE.3: the shared-ontology editor. Edits go to a DRAFT (nothing changes for
 * extraction until a publish); the OE.1 gate re-validates the draft after each
 * edit, the preview shows what a publish would do to the graph, and a publish
 * is refused while a finding blocks or while the draft is based on a version
 * that is no longer active. Versions can be rolled back. After a publish that
 * moves entities, the retype of the existing rows is offered (OE.4).
 */

import { useQueries } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'

import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { ConfirmButton } from '@/components/ontologies/ConfirmButton'
import { RelationshipTypesEditor } from '@/components/ontologies/RelationshipTypesEditor'
import { RetypePanel } from '@/components/ontologies/RetypePanel'
import { TypeForm } from '@/components/ontologies/TypeForm'
import { TypeTree } from '@/components/ontologies/TypeTree'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  ontologiesApi,
  ontologyEditApi,
  type Finding,
  type OntologyPreview,
  type StoredEntityType,
} from '@/lib/api/ontologies'
import { QUERY_KEYS } from '@/lib/api/query-client'
import { useOntology } from '@/lib/hooks/use-ontologies'
import {
  errorDetail,
  useDiscardDraft,
  useDraftEdit,
  useDraftFindings,
  useDraftPreview,
  useOntologyDraft,
  useOntologyVersions,
  usePublishDraft,
  useRollback,
} from '@/lib/hooks/use-ontology-editor'
import { useToast } from '@/lib/hooks/use-toast'
import {
  buildTypeTree,
  inheritanceClosure,
  inheritedTypes,
  nameTaken,
  parentsOf,
  type InheritedType,
  type MergedOntology,
} from '@/lib/utils/ontology-tree'

export function OntologyEditor({ name, initialType }: { name: string; initialType?: string }) {
  const draftQuery = useOntologyDraft(name)
  const active = useOntology(name)
  const findings = useDraftFindings(name)
  const edit = useDraftEdit(name)
  const discard = useDiscardDraft(name)
  const previewMutation = useDraftPreview(name)
  const publish = usePublishDraft(name)
  const [selected, setSelected] = useState<string | null>(initialType ?? null)
  const [note, setNote] = useState('')
  // Bumped when the draft is replaced wholesale (discard, publish, rollback):
  // the type form starts over then, and only then. Keying it on the draft's
  // existence remounted it on the FIRST edit, which creates the draft, and
  // lost what was being typed (review OE.3 round 3).
  const [formEpoch, setFormEpoch] = useState(0)
  const restart = () => setFormEpoch((e) => e + 1)
  const [newType, setNewType] = useState('')
  const [retypeOpen, setRetypeOpen] = useState(false)

  // Every ontology this one inherits from, through the whole chain (`extends`
  // then `includes`, at every level), so each inherited type can be labelled
  // with the ontology that DEFINES it. The chain is discovered as it loads.
  const metadata = draftQuery.data?.ontology.metadata
  const direct = useMemo(() => (metadata ? parentsOf(metadata) : []), [metadata])
  const [chain, setChain] = useState<string[]>([])
  const chainQueries = useQueries({
    queries: chain.map((p) => ({
      queryKey: QUERY_KEYS.ontology(p),
      queryFn: () => ontologiesApi.get(p),
    })),
  })
  const loaded = useMemo(() => {
    const out: Record<string, MergedOntology | undefined> = {}
    chain.forEach((p, i) => { out[p] = chainQueries[i]?.data as MergedOntology | undefined })
    return out
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chain, ...chainQueries.map((q) => q.data)])
  const { missing } = inheritanceClosure(direct, loaded)
  const toLoad = missing.filter((m) => !chain.includes(m))
  useEffect(() => {
    if (toLoad.length) setChain((prev) => [...prev, ...toLoad.filter((m) => !prev.includes(m))])
  }, [toLoad.join('|')]) // eslint-disable-line react-hooks/exhaustive-deps
  const parentsFailed = chainQueries.some((q) => q.isError)
  const parentsLoading = !parentsFailed && (missing.length > 0 || chainQueries.some((q) => q.isLoading))
  const inherited = useMemo<InheritedType[]>(() => inheritedTypes(direct, loaded), [direct, loaded])
  const own = useMemo(() => draftQuery.data?.ontology.entity_types ?? {}, [draftQuery.data])
  const tree = useMemo(() => buildTypeTree(own, inherited), [own, inherited])

  // The preview describes the draft it was made from: drop it when the draft changes.
  const draftStamp = draftQuery.dataUpdatedAt
  useEffect(() => {
    previewMutation.reset()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draftStamp])

  if (draftQuery.isLoading) return <LoadingSpinner />
  if (!draftQuery.data) return <p className="text-sm text-destructive">Could not load the draft.</p>

  const draft = draftQuery.data
  const activeVersion = active.data?.version
  const stale = draft.exists && activeVersion !== undefined && activeVersion !== draft.base_version
  const blocking = (findings.data ?? []).filter((f) => f.severity === 'block')
  const findingsByType = new Map<string, number>()
  for (const f of blocking) {
    if (f.subject) findingsByType.set(f.subject.toLowerCase(), (findingsByType.get(f.subject.toLowerCase()) ?? 0) + 1)
  }
  const busy = edit.isPending || publish.isPending || discard.isPending
  const selectedOwn = selected ? Object.values(own).find((t) => t.name === selected) : undefined
  const put = (type: StoredEntityType) => edit.mutate(() => ontologyEditApi.putEntityType(name, type))
  const newTypeTaken = newType.trim() !== '' && nameTaken(own, newType)

  return (
    <div className="space-y-4" data-testid="ontology-editor">
      <div className="flex flex-wrap items-center gap-2 rounded-md border p-3">
        <Badge variant={draft.exists ? 'default' : 'outline'} data-testid="ontology-draft-state">
          {draft.exists ? 'Draft' : 'No draft'}
        </Badge>
        <span className="text-xs text-muted-foreground">
          based on {draft.base_version}; active {activeVersion ?? '…'}
        </span>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <Button size="sm" variant="outline" disabled={!draft.exists || previewMutation.isPending}
            onClick={() => previewMutation.mutate(true)} data-testid="ontology-preview">
            {previewMutation.isPending ? 'Previewing…' : 'Preview'}
          </Button>
          <Input className="h-8 w-56" placeholder="Note for this version" aria-label="Publish note"
            value={note} onChange={(e) => setNote(e.target.value)} />
          <Button size="sm" disabled={!draft.exists || blocking.length > 0 || stale || busy}
            onClick={() => publish.mutate(note.trim() || null, {
              onSuccess: ({ retype }) => {
                setNote('')
                restart()
                // Offer the retype when this publish left rows on their old type.
                if (retype.changes + retype.flag_only > 0 || retype.collisions_caused_by_this_edit > 0) {
                  setRetypeOpen(true)
                }
              },
            })}
            data-testid="ontology-publish">
            Publish
          </Button>
          <ConfirmButton size="sm" variant="ghost" disabled={!draft.exists || busy}
            data-testid="ontology-discard" title="Discard the draft?"
            description="Every unpublished edit of this ontology is lost."
            confirmLabel="Discard" onConfirm={() => discard.mutate(undefined, { onSuccess: restart })}>
            Discard draft
          </ConfirmButton>
          <Button size="sm" variant="outline" onClick={() => setRetypeOpen(true)}
            disabled={retypeOpen} data-testid="ontology-retype-open">
            Retype entities
          </Button>
        </div>
      </div>

      {stale && (
        <p role="alert" className="text-sm text-destructive" data-testid="ontology-draft-stale">
          This draft was started from version {draft.base_version}, but {activeVersion} is active
          now (a publish or rollback happened since). It cannot be published; discard it and
          start again from the active version.
        </p>
      )}

      {retypeOpen && <RetypePanel onClose={() => setRetypeOpen(false)} />}

      <FindingList findings={findings.data ?? []} />

      <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Input className="h-8" placeholder="New type name" aria-label="New type name"
              value={newType} onChange={(e) => setNewType(e.target.value)} />
            <Button size="sm" variant="outline" disabled={!newType.trim() || newTypeTaken || busy}
              data-testid="ontology-add-type"
              onClick={() => {
                const typeName = newType.trim()
                edit.mutate(() => ontologyEditApi.putEntityType(name, { name: typeName }), {
                  onSuccess: () => { setSelected(typeName); setNewType('') },
                })
              }}>
              Add type
            </Button>
          </div>
          {newTypeTaken && (
            <p className="text-xs text-destructive" data-testid="ontology-add-type-taken">
              {newType.trim()} exists already; select it to edit it.
            </p>
          )}
          {parentsLoading ? (
            <p className="text-sm text-muted-foreground">Loading the inherited types…</p>
          ) : (
            <>
              {parentsFailed && (
                <p className="text-xs text-destructive">
                  Could not load every inherited ontology; types they define show as not defined.
                </p>
              )}
              <TypeTree tree={tree} own={own} selected={selected} onSelect={setSelected}
                onReparent={put} findingsByType={findingsByType} disabled={busy} />
            </>
          )}
        </div>
        <div>
          {selectedOwn ? (
            <TypeForm
              key={`${selectedOwn.name}:${formEpoch}`}
              type={selectedOwn}
              parentOptions={[...tree.nodes.values()].map((n) => n.name).sort()}
              disabled={busy}
              onSave={put}
              onRename={(newName) =>
                edit.mutate(() => ontologyEditApi.renameEntityType(name, selectedOwn.name, newName),
                  { onSuccess: () => setSelected(newName) })}
              onDelete={() =>
                edit.mutate(() => ontologyEditApi.deleteEntityType(name, selectedOwn.name),
                  { onSuccess: () => setSelected(null) })}
            />
          ) : (
            <p className="text-sm text-muted-foreground">
              {selected
                ? `${selected} is inherited or not defined; edit it in the ontology that defines it.`
                : 'Select a type to edit it, or drag a type onto its new parent (or cut and paste it with Ctrl+X / Ctrl+V).'}
            </p>
          )}
        </div>
      </div>

      <RelationshipTypesEditor
        relationships={draft.ontology.relationship_types ?? {}}
        disabled={busy}
        onSave={(rel) => edit.mutate(() => ontologyEditApi.putRelationshipType(name, rel))}
        onDelete={(rel) => edit.mutate(() => ontologyEditApi.deleteRelationshipType(name, rel))}
      />

      {previewMutation.data && <PreviewPanel preview={previewMutation.data} />}
      <Versions name={name} draftOpen={draft.exists} onRolledBack={restart} />
    </div>
  )
}

function FindingList({ findings }: { findings: Finding[] }) {
  if (findings.length === 0) {
    return <p className="text-xs text-muted-foreground" data-testid="ontology-findings-clean">
      No findings.</p>
  }
  const blocks = findings.filter((f) => f.severity === 'block')
  const warns = findings.filter((f) => f.severity === 'warn')
  return (
    <div className="space-y-1" data-testid="ontology-findings">
      {blocks.map((f, i) => (
        <p key={`b${i}`} role="alert" className="text-sm text-destructive">
          <strong>{f.rule}</strong>: {f.detail}
        </p>
      ))}
      {warns.length > 0 && (
        <details className="text-xs text-muted-foreground">
          <summary>{warns.length} warning(s), already in the active set</summary>
          {warns.map((f, i) => <p key={`w${i}`}>{f.rule}: {f.detail}</p>)}
        </details>
      )}
    </div>
  )
}

function PreviewPanel({ preview }: { preview: OntologyPreview }) {
  const r = preview.retype
  return (
    <div className="space-y-2 rounded-md border p-3 text-sm" data-testid="ontology-preview-result">
      <h3 className="font-semibold">What publishing this draft would do</h3>
      {preview.blocking && (
        <p className="text-destructive">A finding blocks this draft; it cannot be published.</p>
      )}
      {preview.label_changes.length === 0 ? (
        <p className="text-muted-foreground">No label changes type.</p>
      ) : (
        <ul className="list-disc pl-5" data-testid="ontology-preview-labels">
          {preview.label_changes.map((c) => (
            <li key={c.label}>
              <code>{c.label}</code>: {c.before} → {c.after} ({c.entities} entities)
            </li>
          ))}
        </ul>
      )}
      <p>
        Retype after publishing: {r.changes} change(s), {r.caused_by_this_edit} caused by this
        draft; {r.collisions} collision(s) left for a merge, {r.collisions_caused_by_this_edit} caused
        by this draft.
      </p>
      {preview.detection.length > 0 && (
        <p>{preview.detection.length} document(s) would select different ontologies.</p>
      )}
      {preview.detection_error && (
        <p className="text-destructive">Detection replay failed: {preview.detection_error}</p>
      )}
    </div>
  )
}

function Versions({ name, draftOpen, onRolledBack }: {
  name: string
  draftOpen: boolean
  onRolledBack: () => void
}) {
  const versions = useOntologyVersions(name)
  const rollback = useRollback(name)
  const { toast } = useToast()
  const download = async () => {
    try {
      const text = await ontologyEditApi.exportYaml(name)
      const url = URL.createObjectURL(new Blob([text], { type: 'application/x-yaml' }))
      const a = document.createElement('a')
      a.href = url
      a.download = `${name}.yaml`
      a.click()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (err) {
      toast({ title: 'Export failed', description: errorDetail(err), variant: 'destructive' })
    }
  }
  return (
    <div className="space-y-2 rounded-md border p-3" data-testid="ontology-versions">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-sm">Versions</h3>
        <Button size="sm" variant="outline" onClick={download}>Export YAML</Button>
      </div>
      {(versions.data ?? []).length === 0 ? (
        <p className="text-xs text-muted-foreground">
          Only the shipped YAML so far; the first publish stores it as a version.</p>
      ) : (
        <ul className="space-y-1 text-sm">
          {(versions.data ?? []).map((v) => (
            <li key={v.version} className="flex items-center gap-2">
              <code>{v.version}</code>
              {v.is_active && <Badge>active</Badge>}
              <span className="text-xs text-muted-foreground">{v.note ?? ''}</span>
              {!v.is_active && (
                <ConfirmButton size="sm" variant="ghost" className="ml-auto"
                  disabled={rollback.isPending}
                  data-testid={`ontology-rollback-${v.version}`}
                  title={`Roll ${name} back to ${v.version}?`}
                  description={
                    `Every notebook types with this version from the next extraction on.` +
                    (draftOpen ? ' The open draft was started from the current version and ' +
                      'can no longer be published afterwards.' : '')}
                  confirmLabel="Roll back" onConfirm={() => rollback.mutate(v.version, { onSuccess: onRolledBack })}>
                  Roll back to this
                </ConfirmButton>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
