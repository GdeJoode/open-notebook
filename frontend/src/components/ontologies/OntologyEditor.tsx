'use client'

/**
 * OE.3: the shared-ontology editor. Edits go to a DRAFT (nothing changes for
 * extraction until a publish); the OE.1 gate re-validates the draft after each
 * edit, the preview shows what a publish would do to the graph, and a publish
 * is refused while a finding blocks. Versions can be rolled back.
 */

import { useMemo, useState } from 'react'

import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { RelationshipTypesEditor } from '@/components/ontologies/RelationshipTypesEditor'
import { TypeForm } from '@/components/ontologies/TypeForm'
import { TypeTree } from '@/components/ontologies/TypeTree'
import {
  ontologyEditApi,
  type Finding,
  type OntologyPreview,
  type StoredEntityType,
} from '@/lib/api/ontologies'
import { useOntology } from '@/lib/hooks/use-ontologies'
import {
  useDiscardDraft,
  useDraftEdit,
  useDraftFindings,
  useDraftPreview,
  useOntologyDraft,
  useOntologyVersions,
  usePublishDraft,
  useRollback,
} from '@/lib/hooks/use-ontology-editor'
import { buildTypeTree } from '@/lib/utils/ontology-tree'

export function OntologyEditor({ name }: { name: string }) {
  const draftQuery = useOntologyDraft(name)
  const findings = useDraftFindings(name)
  const extendsName = draftQuery.data?.ontology.metadata.extends ?? ''
  const parent = useOntology(extendsName)
  const edit = useDraftEdit(name)
  const discard = useDiscardDraft(name)
  const previewMutation = useDraftPreview(name)
  const publish = usePublishDraft(name)
  const [selected, setSelected] = useState<string | null>(null)
  const [note, setNote] = useState('')
  const [newType, setNewType] = useState('')

  const own = useMemo(() => draftQuery.data?.ontology.entity_types ?? {}, [draftQuery.data])
  const tree = useMemo(
    () => buildTypeTree(own, extendsName ? (parent.data?.entity_types ?? []) : []),
    [own, extendsName, parent.data],
  )

  if (draftQuery.isLoading) return <LoadingSpinner />
  if (!draftQuery.data) return <p className="text-sm text-destructive">Could not load the draft.</p>

  const draft = draftQuery.data
  const blocking = (findings.data ?? []).filter((f) => f.severity === 'block')
  const busy = edit.isPending || publish.isPending || discard.isPending
  const selectedOwn = selected ? Object.values(own).find((t) => t.name === selected) : undefined
  const put = (type: StoredEntityType) => edit.mutate(() => ontologyEditApi.putEntityType(name, type))

  return (
    <div className="space-y-4" data-testid="ontology-editor">
      <div className="flex flex-wrap items-center gap-2 rounded-md border p-3">
        <Badge variant={draft.exists ? 'default' : 'outline'} data-testid="ontology-draft-state">
          {draft.exists ? 'Draft' : 'No draft'}
        </Badge>
        <span className="text-xs text-muted-foreground">
          based on version {draft.base_version}
        </span>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <Button size="sm" variant="outline" disabled={!draft.exists || previewMutation.isPending}
            onClick={() => previewMutation.mutate(true)} data-testid="ontology-preview">
            {previewMutation.isPending ? 'Previewing…' : 'Preview'}
          </Button>
          <Input className="h-8 w-56" placeholder="Note for this version" aria-label="Publish note"
            value={note} onChange={(e) => setNote(e.target.value)} />
          <Button size="sm" disabled={!draft.exists || blocking.length > 0 || busy}
            onClick={() => publish.mutate(note.trim() || null, { onSuccess: () => setNote('') })}
            data-testid="ontology-publish">
            Publish
          </Button>
          <Button size="sm" variant="ghost" disabled={!draft.exists || busy}
            onClick={() => discard.mutate()} data-testid="ontology-discard">
            Discard draft
          </Button>
        </div>
      </div>

      <FindingList findings={findings.data ?? []} />

      <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="space-y-2">
          <div className="flex gap-2">
            <Input className="h-8" placeholder="New type name" aria-label="New type name"
              value={newType} onChange={(e) => setNewType(e.target.value)} />
            <Button size="sm" variant="outline" disabled={!newType.trim() || busy}
              data-testid="ontology-add-type"
              onClick={() => {
                const typeName = newType.trim()
                put({ name: typeName })
                setSelected(typeName)
                setNewType('')
              }}>
              Add type
            </Button>
          </div>
          <TypeTree tree={tree} own={own} selected={selected} onSelect={setSelected}
            onReparent={put} disabled={busy} />
        </div>
        <div>
          {selectedOwn ? (
            <TypeForm
              type={selectedOwn}
              parentOptions={[...tree.nodes.values()].map((n) => n.name).sort()}
              disabled={busy}
              onSave={put}
              onRename={(newName) => {
                edit.mutate(() => ontologyEditApi.renameEntityType(name, selectedOwn.name, newName))
                setSelected(newName)
              }}
              onDelete={() => {
                edit.mutate(() => ontologyEditApi.deleteEntityType(name, selectedOwn.name))
                setSelected(null)
              }}
            />
          ) : (
            <p className="text-sm text-muted-foreground">
              {selected
                ? `${selected} is inherited or a schema.org base; edit it in the ontology that defines it.`
                : 'Select a type to edit it, or drag a type onto its new parent.'}
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
      <Versions name={name} />
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
      <h3 className="font-semibold">What publishing would do</h3>
      {preview.label_changes.length === 0 ? (
        <p className="text-muted-foreground">No label changes type.</p>
      ) : (
        <ul className="list-disc pl-5">
          {preview.label_changes.map((c) => (
            <li key={c.label}>
              <code>{c.label}</code>: {c.before} → {c.after} ({c.entities} entities)
            </li>
          ))}
        </ul>
      )}
      <p>
        Retype after publishing: {r.changes} change(s), of which {r.caused_by_this_edit} caused
        by this draft; {r.collisions} collision(s) left for a merge.
      </p>
      {preview.detection.length > 0 && (
        <p>{preview.detection.length} document(s) would select different ontologies.</p>
      )}
      {preview.detection_error && (
        <p className="text-destructive">Detection replay failed: {preview.detection_error}</p>
      )}
      {preview.notebook_edits ? (
        <p className="text-muted-foreground">
          {preview.notebook_edits} notebook-level schema edit(s) are not included.</p>
      ) : null}
    </div>
  )
}

function Versions({ name }: { name: string }) {
  const versions = useOntologyVersions(name)
  const rollback = useRollback(name)
  const download = async () => {
    const text = await ontologyEditApi.exportYaml(name)
    const url = URL.createObjectURL(new Blob([text], { type: 'application/x-yaml' }))
    const a = document.createElement('a')
    a.href = url
    a.download = `${name}.yaml`
    a.click()
    URL.revokeObjectURL(url)
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
                <Button size="sm" variant="ghost" className="ml-auto" disabled={rollback.isPending}
                  onClick={() => rollback.mutate(v.version)}
                  data-testid={`ontology-rollback-${v.version}`}>
                  Roll back to this
                </Button>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
