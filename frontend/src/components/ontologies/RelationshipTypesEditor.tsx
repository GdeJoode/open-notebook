'use client'

/**
 * OE.3: the draft's OWN relationship types — name, description, domain and
 * range. Like entity types, a save sends the whole type (unshown keys kept).
 * The gate reports a domain/range that names no type
 * (`relationship_endpoint_unknown`).
 */

import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { StoredRelationshipType } from '@/lib/api/ontologies'

const csv = (items?: string[]) => (items ?? []).join(', ')
const parse = (text: string) => text.split(',').map((s) => s.trim()).filter(Boolean)

interface Props {
  relationships: Record<string, StoredRelationshipType>
  onSave: (rel: StoredRelationshipType) => void
  onDelete: (name: string) => void
  disabled?: boolean
}

export function RelationshipTypesEditor({ relationships, onSave, onDelete, disabled }: Props) {
  const [newName, setNewName] = useState('')
  const rels = Object.values(relationships).sort((a, b) => a.name.localeCompare(b.name))
  return (
    <div className="space-y-2 rounded-md border p-3" data-testid="ontology-relationships">
      <h3 className="font-semibold text-sm">Relationship types</h3>
      {rels.map((rel) => (
        <RelationshipRow key={rel.name} rel={rel} onSave={onSave} onDelete={onDelete}
          disabled={disabled} />
      ))}
      <div className="flex gap-2">
        <Input className="h-8" placeholder="NEW_RELATION" aria-label="New relationship name"
          value={newName} onChange={(e) => setNewName(e.target.value)} />
        <Button size="sm" variant="outline" disabled={!newName.trim() || disabled}
          onClick={() => { onSave({ name: newName.trim(), domain: [], range: [] }); setNewName('') }}>
          Add relationship
        </Button>
      </div>
    </div>
  )
}

function RelationshipRow({ rel, onSave, onDelete, disabled }: {
  rel: StoredRelationshipType
  onSave: Props['onSave']
  onDelete: Props['onDelete']
  disabled?: boolean
}) {
  const [description, setDescription] = useState(rel.description ?? '')
  const [domain, setDomain] = useState(csv(rel.domain))
  const [range, setRange] = useState(csv(rel.range))
  return (
    <div className="grid grid-cols-[8rem_1fr_1fr_1fr_auto_auto] items-center gap-2 text-sm">
      <code className="truncate" title={rel.name}>{rel.name}</code>
      <Input className="h-8" aria-label={`${rel.name} description`} value={description}
        onChange={(e) => setDescription(e.target.value)} />
      <Input className="h-8" aria-label={`${rel.name} domain`} placeholder="domain types"
        value={domain} onChange={(e) => setDomain(e.target.value)} />
      <Input className="h-8" aria-label={`${rel.name} range`} placeholder="range types"
        value={range} onChange={(e) => setRange(e.target.value)} />
      <Button size="sm" variant="outline" disabled={disabled}
        onClick={() => {
          const next: StoredRelationshipType = { ...rel, domain: parse(domain), range: parse(range) }
          // The source may spell them source_types/target_types; one spelling wins.
          delete next.source_types
          delete next.target_types
          if (description.trim()) next.description = description.trim()
          else delete next.description
          onSave(next)
        }}>
        Save
      </Button>
      <Button size="sm" variant="ghost" className="text-destructive" disabled={disabled}
        onClick={() => onDelete(rel.name)}>
        Delete
      </Button>
    </div>
  )
}

