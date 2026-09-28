'use client'

/**
 * OE.3: edit one OWN entity type of the draft. Saving sends the WHOLE type —
 * the edit API replaces a type, so every key the form does not show (e.g.
 * `graph_relevance`) is carried over from the stored type. Rename keeps the
 * old name as an alias (server side), so what uses it keeps resolving.
 */

import { useEffect, useState } from 'react'

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

export function TypeForm({ type, parentOptions, onSave, onRename, onDelete, disabled }: TypeFormProps) {
  const [description, setDescription] = useState('')
  const [parent, setParent] = useState('')
  const [base, setBase] = useState('')
  const [aliases, setAliases] = useState('')
  const [hints, setHints] = useState('')
  const [newName, setNewName] = useState('')

  useEffect(() => {
    setDescription(type.description ?? '')
    setParent(type.parent_type ?? '')
    setBase(type.schema_org_type ?? '')
    setAliases((type.aliases ?? []).join(', '))
    setHints((type.extraction_hints ?? []).join('\n'))
    setNewName('')
  }, [type])

  const save = () => {
    const next: StoredEntityType = { ...type }
    const set = (k: string, v: unknown, empty: boolean) => {
      if (empty) delete next[k]
      else next[k] = v
    }
    set('description', description.trim(), !description.trim())
    set('parent_type', parent.trim(), !parent.trim())
    set('schema_org_type', base.trim(), !base.trim())
    const aliasList = list(aliases, /,/)
    set('aliases', aliasList, aliasList.length === 0)
    const hintList = list(hints, /\n/)
    set('extraction_hints', hintList, hintList.length === 0)
    onSave(next)
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
        <Button size="sm" variant="outline" className="text-destructive" disabled={disabled}
          onClick={onDelete} data-testid="ontology-type-delete">
          Delete
        </Button>
      </div>
    </div>
  )
}
