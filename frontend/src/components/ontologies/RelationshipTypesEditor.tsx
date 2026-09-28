'use client'

/**
 * OE.3: the draft's OWN relationship types — name, description, domain and
 * range. A save sends the whole relationship (unshown keys kept) with its
 * endpoints in the spelling it already uses (`domain`/`range` or
 * `source_types`/`target_types`; see `relationship-endpoints`). The gate
 * reports a domain/range that names no type (`relationship_endpoint_unknown`).
 */

import { useState } from 'react'

import { ConfirmButton } from '@/components/ontologies/ConfirmButton'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { StoredRelationshipType } from '@/lib/api/ontologies'
import { nameTaken } from '@/lib/utils/ontology-tree'
import { endpointsOf, withEndpoints } from '@/lib/utils/relationship-endpoints'

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
  const taken = newName.trim() !== '' && nameTaken(relationships, newName)
  return (
    <div className="space-y-2 rounded-md border p-3" data-testid="ontology-relationships">
      <h3 className="font-semibold text-sm">Relationship types</h3>
      {rels.map((rel) => (
        // Keyed on the stored content: a draft change (a rename moving the
        // endpoints, a discard) resets the row instead of saving stale values.
        <RelationshipRow key={`${rel.name}:${JSON.stringify(rel)}`} rel={rel} onSave={onSave}
          onDelete={onDelete} disabled={disabled} />
      ))}
      <div className="flex items-center gap-2">
        <Input className="h-8" placeholder="NEW_RELATION" aria-label="New relationship name"
          value={newName} onChange={(e) => setNewName(e.target.value)} />
        <Button size="sm" variant="outline" disabled={!newName.trim() || taken || disabled}
          onClick={() => { onSave({ name: newName.trim(), domain: [], range: [] }); setNewName('') }}>
          Add relationship
        </Button>
        {taken && <span className="text-xs text-destructive">That relationship exists already.</span>}
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
  const stored = endpointsOf(rel)
  const [description, setDescription] = useState(rel.description ?? '')
  const [domain, setDomain] = useState(stored.domain.join(', '))
  const [range, setRange] = useState(stored.range.join(', '))
  return (
    <div className="grid grid-cols-[8rem_1fr_1fr_1fr_auto_auto] items-center gap-2 text-sm"
      data-testid={`ontology-relationship-${rel.name}`}>
      <code className="truncate" title={rel.name}>{rel.name}</code>
      <Input className="h-8" aria-label={`${rel.name} description`} value={description}
        onChange={(e) => setDescription(e.target.value)} />
      <Input className="h-8" aria-label={`${rel.name} domain`} placeholder="domain types"
        value={domain} onChange={(e) => setDomain(e.target.value)} />
      <Input className="h-8" aria-label={`${rel.name} range`} placeholder="range types"
        value={range} onChange={(e) => setRange(e.target.value)} />
      <Button size="sm" variant="outline" disabled={disabled}
        onClick={() => {
          const next = withEndpoints(rel, parse(domain), parse(range))
          if (description.trim()) next.description = description.trim()
          else delete next.description
          onSave(next)
        }}>
        Save
      </Button>
      <ConfirmButton size="sm" variant="ghost" className="text-destructive" disabled={disabled}
        title={`Delete ${rel.name} from the draft?`}
        description="Only the draft changes; discard the draft to undo it."
        confirmLabel="Delete" onConfirm={() => onDelete(rel.name)}>
        Delete
      </ConfirmButton>
    </div>
  )
}
