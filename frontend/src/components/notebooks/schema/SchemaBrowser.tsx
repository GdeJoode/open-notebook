'use client'

import { useMemo, useState } from 'react'
import { ChevronRight } from 'lucide-react'

import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import type {
  EntityTypeNode,
  ExtensionView,
  NotebookSchemaResponse,
} from '@/lib/types/notebook_schema'

interface SchemaBrowserProps {
  schema: NotebookSchemaResponse
}

/**
 * Effective-schema browser.
 *
 * Renders a two-column layout:
 *  - left: a flat (single-level for now) collapsible list of entity
 *    types in the effective schema = base ontology + accepted
 *    extensions. Hierarchy is shown via the `parent_type` badge on
 *    each row rather than nested tree depth, because (a) the base
 *    YAML rarely defines deep hierarchies and (b) a flat list scans
 *    much faster keyboard-only than a true tree.
 *  - right: a side panel for the selected type, showing its
 *    description + properties.
 *
 * Keyboard accessibility:
 *  - Items are `<button>` rows; native Tab order works out of the box.
 *  - Enter / Space selects an item (default button activation).
 *  - The selected item gets `aria-current="true"` for screen readers.
 *
 * Tooltips: each row's name has a tooltip with the description when
 * present — keeps the row label short while preserving discoverability.
 */
export function SchemaBrowser({ schema }: SchemaBrowserProps) {
  // Merge base + accepted extensions into a single tree. Accepted
  // extensions are flagged so the UI can render an "Extension" badge.
  const items: TreeItem[] = useMemo(() => {
    const base: TreeItem[] = schema.base_ontology_types.map((et) => ({
      kind: 'base',
      key: `base:${et.name}`,
      name: et.name,
      description: et.description ?? null,
      parent_type: et.parent_type ?? null,
      properties: et.properties,
    }))
    const accepted: TreeItem[] = schema.accepted_extensions.map((ext) => ({
      kind: 'extension',
      key: `ext:${ext.extension_id ?? ext.type_name}`,
      name: ext.type_name,
      description: ext.description ?? null,
      parent_type: ext.parent_type ?? null,
      properties: ext.properties,
    }))
    // Stable order: base types first (preserves YAML order), then
    // accepted extensions in their stored order. Both groups are
    // alphabetised within their group for predictability.
    base.sort((a, b) => a.name.localeCompare(b.name))
    accepted.sort((a, b) => a.name.localeCompare(b.name))
    return [...base, ...accepted]
  }, [schema])

  const [selectedKey, setSelectedKey] = useState<string | null>(
    items.length > 0 ? items[0].key : null,
  )
  const selected = items.find((it) => it.key === selectedKey) ?? null

  if (items.length === 0) {
    return (
      <div
        role="status"
        className="rounded-md border border-dashed p-6 text-sm text-muted-foreground"
      >
        No entity types in the effective schema yet. Try running pass-1
        extraction on a source to populate the base ontology.
      </div>
    )
  }

  return (
    <TooltipProvider>
      <div
        className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]"
        data-testid="schema-browser"
      >
        <ul
          role="listbox"
          aria-label="Entity types"
          className="flex max-h-[60vh] flex-col gap-1 overflow-y-auto rounded-md border p-2"
        >
          {items.map((item) => {
            const isSelected = item.key === selectedKey
            return (
              <li key={item.key}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      role="option"
                      aria-selected={isSelected}
                      aria-current={isSelected ? 'true' : undefined}
                      data-testid={`schema-tree-item-${item.name}`}
                      onClick={() => setSelectedKey(item.key)}
                      className={cn(
                        'flex w-full items-center justify-between gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors',
                        'hover:bg-accent hover:text-accent-foreground',
                        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                        isSelected && 'bg-accent text-accent-foreground',
                      )}
                    >
                      <span className="flex min-w-0 items-center gap-1.5">
                        <ChevronRight
                          aria-hidden="true"
                          className={cn(
                            'h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform',
                            isSelected && 'rotate-90',
                          )}
                        />
                        <span className="truncate font-medium">
                          {item.name}
                        </span>
                      </span>
                      {item.kind === 'extension' && (
                        <Badge variant="secondary" className="text-[10px]">
                          Extension
                        </Badge>
                      )}
                    </button>
                  </TooltipTrigger>
                  {item.description && (
                    <TooltipContent side="right" className="max-w-sm">
                      {item.description}
                    </TooltipContent>
                  )}
                </Tooltip>
              </li>
            )
          })}
        </ul>

        <aside
          data-testid="schema-side-panel"
          aria-label="Selected entity type details"
          className="rounded-md border p-4"
        >
          {selected ? (
            <SelectedTypePanel item={selected} />
          ) : (
            <p className="text-sm text-muted-foreground">
              Select an entity type to see its properties.
            </p>
          )}
        </aside>
      </div>
    </TooltipProvider>
  )
}

type TreeItem = {
  kind: 'base' | 'extension'
  key: string
  name: string
  description: string | null
  parent_type: string | null
  properties: EntityTypeNode['properties']
}

function SelectedTypePanel({ item }: { item: TreeItem }) {
  return (
    <div className="space-y-3">
      <header className="space-y-1">
        <div className="flex items-center gap-2">
          <h3 className="text-base font-semibold">{item.name}</h3>
          {item.kind === 'extension' && (
            <Badge variant="secondary">Extension</Badge>
          )}
          {item.parent_type && (
            <span className="text-xs text-muted-foreground">
              extends <code className="font-mono">{item.parent_type}</code>
            </span>
          )}
        </div>
        {item.description && (
          <p className="text-sm text-muted-foreground">{item.description}</p>
        )}
      </header>

      <section>
        <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Properties
        </h4>
        {item.properties.length === 0 ? (
          <p className="text-sm italic text-muted-foreground">
            No properties defined.
          </p>
        ) : (
          <ul
            className="space-y-1"
            data-testid={`schema-properties-${item.name}`}
          >
            {item.properties.map((p) => (
              <li
                key={p.name}
                className="flex items-baseline gap-2 text-sm"
              >
                <code className="font-mono text-xs">{p.name}</code>
                <span className="text-xs text-muted-foreground">
                  {p.data_type}
                </span>
                {p.required && (
                  <Badge variant="outline" className="text-[10px]">
                    required
                  </Badge>
                )}
                {p.description && (
                  <span className="text-xs text-muted-foreground">
                    — {p.description}
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}

// Re-export ExtensionView for the consumers that import it alongside.
export type { ExtensionView }
