'use client'

import { useState } from 'react'

import type { OverlayCreateInput } from '@/lib/api/entity-resolution'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Plus } from 'lucide-react'

interface AliasManagerProps {
  /** The entity's canonical name — one side of every alias equivalence. */
  canonicalName: string
  entityType?: string
  /** Existing aliases (read) from `entity.aliases` / the entity_alias table. */
  aliases: string[]
  /**
   * Add an alias. Backed by a global force-merge overlay (`canonical ⇔ alias`)
   * which the next dedup scan materializes into an `entity_alias` edge — the one
   * alias-writing path available over the merged K.5 overlay endpoints.
   */
  onAddAlias: (rule: OverlayCreateInput) => void
  /** Remove an alias equivalence (records a force-split so it is not re-merged). */
  onRemoveAlias: (rule: OverlayCreateInput) => void
  disabled?: boolean
}

/**
 * Per-entity alias management. Lists the entity's known surface forms and lets
 * the user add a new one (force-merge equivalence) or drop one (force-split).
 * Both flow through the K.5 `alias_overlay` — the human escape hatch over the
 * matcher — so no new backend surface is needed.
 */
export function AliasManager({
  canonicalName,
  entityType = '',
  aliases,
  onAddAlias,
  onRemoveAlias,
  disabled = false,
}: AliasManagerProps) {
  const [draft, setDraft] = useState('')

  const submit = () => {
    const value = draft.trim()
    if (!value) return
    onAddAlias({
      kind: 'merge',
      name_a: canonicalName,
      name_b: value,
      entity_type: entityType,
      scope: 'global',
      notebook_id: null,
    })
    setDraft('')
  }

  return (
    <div className="space-y-3" data-testid="alias-manager">
      <div>
        <Label className="text-sm font-medium">Aliases</Label>
        {aliases.length > 0 ? (
          <div className="flex flex-wrap gap-1.5 mt-1.5">
            {aliases.map((alias) => (
              <Badge
                key={alias}
                variant="secondary"
                className="gap-1 text-[11px]"
              >
                {alias}
                <button
                  type="button"
                  onClick={() =>
                    onRemoveAlias({
                      kind: 'split',
                      name_a: canonicalName,
                      name_b: alias,
                      entity_type: entityType,
                      scope: 'global',
                      notebook_id: null,
                    })
                  }
                  disabled={disabled}
                  aria-label={`Remove alias ${alias}`}
                  className="ml-0.5 rounded-sm hover:text-destructive focus:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                >
                  ×
                </button>
              </Badge>
            ))}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground mt-1.5">
            No aliases recorded.
          </p>
        )}
      </div>

      <div className="flex items-center gap-2">
        <Input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              submit()
            }
          }}
          disabled={disabled}
          placeholder="Add an alias…"
          aria-label={`Add an alias for ${canonicalName}`}
          className="h-8 text-sm"
        />
        <Button
          size="sm"
          variant="outline"
          className="gap-1 shrink-0"
          disabled={disabled || !draft.trim()}
          onClick={submit}
        >
          <Plus className="h-3.5 w-3.5" />
          Add
        </Button>
      </div>
    </div>
  )
}
