'use client'

/**
 * OE.3: edit one OWN entity type of the draft. Saving sends the WHOLE type —
 * the edit API replaces a type, so every key the form does not show (e.g.
 * `graph_relevance`) is carried over from the stored type. Rename keeps the
 * old name as an alias (server side), so what uses it keeps resolving.
 */

import { useEffect, useState } from 'react'

import { ConfirmButton } from '@/components/ontologies/ConfirmButton'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import type { StoredEntityType } from '@/lib/api/ontologies'

const list = (text: string, sep: RegExp) =>
  text.split(sep).map((s) => s.trim()).filter(Boolean)

export interface TypeFormProps {
  type: StoredEntityType
  parentOptions: string[]
  onSave: (type: StoredEntityType) => void
  onRename: (newName: string) => void
  onDelete: () => void
  disabled?: boolean
}

type Field = 'description' | 'parent_type' | 'schema_org_type' | 'aliases' | 'extraction_hints'

/** The form's text for a field of the stored type. */
function shown(type: StoredEntityType, field: Field): string {
  const value = type[field]
  if (Array.isArray(value)) return value.join(field === 'aliases' ? ', ' : '\n')
  return typeof value === 'string' ? value : ''
}

/** A field's text back into what the stored type holds (`undefined` = remove). */
function parsed(field: Field, text: string): unknown {
  if (field === 'aliases' || field === 'extraction_hints') {
    const items = list(text, field === 'aliases' ? /,/ : /\n/)
    return items.length ? items : undefined
  }
  return text.trim() || undefined
}

const FIELDS: Field[] = ['description', 'parent_type', 'schema_org_type', 'aliases', 'extraction_hints']

export function TypeForm({ type, parentOptions, onSave, onRename, onDelete, disabled }: TypeFormProps) {
  // Only what the user EDITED is kept as form state; every other field shows
  // the stored type as it is now. Review OE.3 round 2: a form holding a copy
  // of every field put back a parent a drag or a paste had changed meanwhile.
  const [edited, setEdited] = useState<Partial<Record<Field, string>>>({})
  const [newName, setNewName] = useState('')

  useEffect(() => {
    setEdited({})
    setNewName('')
  }, [type.name])

  const value = (field: Field) => edited[field] ?? shown(type, field)
  const change = (field: Field) => (text: string) => setEdited((prev) => ({ ...prev, [field]: text }))
  const setDescription = change('description')
  const setParent = change('parent_type')
  const setBase = change('schema_org_type')
  const setAliases = change('aliases')
  const setHints = change('extraction_hints')
  const description = value('description')
  const parent = value('parent_type')
  const base = value('schema_org_type')
  const aliases = value('aliases')
  const hints = value('extraction_hints')

  const save = () => {
    // The latest stored type, with only the edited fields replaced.
    const next: Record<string, unknown> = { ...type }
    for (const field of FIELDS) {
      if (edited[field] === undefined) continue
      const v = parsed(field, edited[field] as string)
      if (v === undefined) delete next[field]
      else next[field] = v
    }
    setEdited({})
    onSave(next as StoredEntityType) // `name` is kept from the stored type
  }

  const listId = `parents-${type.name}`
  return (
    <div className="space-y-3" data-testid="ontology-type-form">
      <h3 className="font-semibold">{type.name}</h3>
      <div className="space-y-1">
        <Label htmlFor="oe-description">Description</Label>
        <Textarea id="oe-description" value={description} rows={2}
          onChange={(e) => setDescription(e.target.value)} />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <Label htmlFor="oe-parent">Parent type</Label>
          <Input id="oe-parent" list={listId} value={parent}
            onChange={(e) => setParent(e.target.value)} />
          <datalist id={listId}>
            {parentOptions.map((p) => <option key={p} value={p} />)}
          </datalist>
        </div>
        <div className="space-y-1">
          <Label htmlFor="oe-base">schema.org base</Label>
          <Input id="oe-base" placeholder="schema:Organization" value={base}
            onChange={(e) => setBase(e.target.value)} />
        </div>
      </div>
      <div className="space-y-1">
        <Label htmlFor="oe-aliases">Aliases (comma-separated)</Label>
        <Input id="oe-aliases" value={aliases} onChange={(e) => setAliases(e.target.value)} />
      </div>
      <div className="space-y-1">
        <Label htmlFor="oe-hints">Extraction hints (one per line)</Label>
        <Textarea id="oe-hints" value={hints} rows={3} onChange={(e) => setHints(e.target.value)} />
      </div>
      <div className="flex flex-wrap items-end gap-2">
        <Button size="sm" onClick={save} disabled={disabled} data-testid="ontology-type-save">
          Save to draft
        </Button>
        <Input className="h-8 w-40" placeholder="New name" value={newName} aria-label="New name"
          onChange={(e) => setNewName(e.target.value)} />
        <Button size="sm" variant="outline" disabled={disabled || !newName.trim()}
          onClick={() => onRename(newName.trim())} data-testid="ontology-type-rename">
          Rename
        </Button>
        <ConfirmButton size="sm" variant="outline" className="text-destructive" disabled={disabled}
          data-testid="ontology-type-delete" title={`Delete ${type.name} from the draft?`}
          description="Only the draft changes; discard the draft to undo it."
          confirmLabel="Delete" onConfirm={onDelete}>
          Delete
        </ConfirmButton>
      </div>
    </div>
  )
}
