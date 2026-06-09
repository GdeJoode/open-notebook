'use client'

import { useId, useMemo, useState } from 'react'
import { ChevronRight, MoreHorizontal } from 'lucide-react'

import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import {
  SchemaEditDialog,
  type SchemaEditMode,
} from '@/components/notebooks/schema/SchemaEditDialog'
import type {
  EntityTypeNode,
  ExtensionView,
  NotebookSchemaResponse,
} from '@/lib/types/notebook_schema'

interface SchemaBrowserProps {
  schema: NotebookSchemaResponse
  /** When set, the per-row ⋯ menu fires edit ops scoped to this notebook. */
  notebookId?: string
}

/**
 * Effective-schema browser.
 *
 * Renders a two-column layout:
 *  - left: a flat single-select `role="listbox"` of entity types in
 *    the effective schema = base ontology + accepted extensions. Each
 *    row shows the type name and (when present) its `parent_type` in
 *    the side panel — the list itself is flat.
 *  - right: a side panel for the selected type, showing its
 *    description + properties.
 *
 * # Listbox vs tree — design decision (B.3a attempt-2)
 *
 * The original plan (`docs/tracks/B-kg-quality/plan.md` §Phase B.3a
 * AC #2) asked for a "collapsible tree". We pivoted to a flat listbox
 * after the attempt-1 review. The reasoning:
 *
 *  1. **Shallow hierarchy in practice.** The in-scope base ontologies
 *     (scholarly, general) use a single-level `parent_type` at most —
 *     no nested type-of-type-of-type chains. A flat list with a
 *     "extends X" annotation on the side panel conveys the same info
 *     without forcing the user to expand nodes to see leaf classes.
 *  2. **Listbox is the correct ARIA semantic for single-select.**
 *     `role="tree"` carries `aria-expanded` semantics + arrow-key
 *     traversal that imply child nodes; using it for a flat list
 *     mis-signals expandability to screen readers.
 *  3. **No B.3b dependency.** The rename/merge/split edit ops shipped
 *     in B.3b operate on individual types, not on tree edges — they
 *     read the same flat `items[]` list. Pivoting later would be cheap
 *     if a deeply-nested ontology lands in scope.
 *  4. **Keyboard scan speed.** A flat list is faster to skim with Tab
 *     + Enter than a tree where each parent has to be expanded first.
 *
 * Plan AC #2 + AC #6 were softened to match in the same review cycle.
 *
 * # Keyboard accessibility
 *
 *  - Items are `<button>` rows inside `<li>` inside `<ul role="listbox">`.
 *    Native Tab order steps through them; Space / Enter activate
 *    selection via default button semantics.
 *  - The listbox carries `aria-activedescendant` pointing to the
 *    currently-selected option's id, which lets screen readers
 *    announce selection changes consistently across browsers (some
 *    don't read `aria-selected` reliably on a `<button role="option">`).
 *  - Each option additionally has `aria-current="true"` so navigation
 *    patterns that prefer the WAI-ARIA "current" hint also work.
 *
 * Tooltips: each row's name has a tooltip with the description when
 * present — keeps the row label short while preserving discoverability.
 */
export function SchemaBrowser({ schema, notebookId }: SchemaBrowserProps) {
  // Merge base + accepted extensions into a single tree. Accepted
  // extensions are flagged so the UI can render an "Extension" badge.
  // Types in `excluded_types` (B.3b soft-delete) are filtered out so
  // the browser reflects the effective schema, not the raw lists.
  const items: TreeItem[] = useMemo(() => {
    const excluded = new Set(schema.excluded_types ?? [])
    const base: TreeItem[] = schema.base_ontology_types
      .filter((et) => !excluded.has(et.name))
      .map((et) => ({
        kind: 'base',
        key: `base:${et.name}`,
        name: et.name,
        description: et.description ?? null,
        parent_type: et.parent_type ?? null,
        properties: et.properties,
      }))
    const accepted: TreeItem[] = schema.accepted_extensions
      .filter((ext) => !excluded.has(ext.type_name))
      .map((ext) => ({
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

  // Edit-dialog state — `target` carries the row name the user invoked
  // the menu on. `mode === null` means dialog is closed.
  const [editMode, setEditMode] = useState<SchemaEditMode | null>(null)
  const [editTarget, setEditTarget] = useState<string>('')

  const openEditDialog = (mode: SchemaEditMode, name: string) => {
    setEditTarget(name)
    setEditMode(mode)
  }

  // Stable id prefix for option ids. Required so `aria-activedescendant`
  // on the listbox can point at the option DOM node. `useId` keeps the
  // prefix unique across multiple SchemaBrowser instances on a page.
  const optionIdPrefix = useId()
  const optionId = (key: string) =>
    `${optionIdPrefix}-${key.replace(/[^a-zA-Z0-9_-]/g, '_')}`
  const activeOptionId = selectedKey ? optionId(selectedKey) : undefined

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
          aria-activedescendant={activeOptionId}
          className="flex max-h-[60vh] flex-col gap-1 overflow-y-auto rounded-md border p-2"
        >
          {items.map((item) => {
            const isSelected = item.key === selectedKey
            return (
              <li key={item.key} className="flex items-center gap-1">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      role="option"
                      id={optionId(item.key)}
                      aria-selected={isSelected}
                      aria-current={isSelected ? 'true' : undefined}
                      data-testid={`schema-tree-item-${item.name}`}
                      onClick={() => setSelectedKey(item.key)}
                      className={cn(
                        'flex flex-1 items-center justify-between gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors',
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
                {notebookId && (
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="h-7 w-7 shrink-0 p-0"
                        aria-label={`Actions for ${item.name}`}
                        data-testid={`schema-row-actions-${item.name}`}
                      >
                        <MoreHorizontal
                          aria-hidden="true"
                          className="h-4 w-4"
                        />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-48">
                      <DropdownMenuItem
                        data-testid={`schema-row-rename-${item.name}`}
                        onSelect={() => openEditDialog('rename', item.name)}
                      >
                        Rename…
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        data-testid={`schema-row-merge-${item.name}`}
                        onSelect={() => openEditDialog('merge', item.name)}
                      >
                        Merge into…
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        data-testid={`schema-row-split-${item.name}`}
                        onSelect={() => openEditDialog('split', item.name)}
                      >
                        Split…
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        data-testid={`schema-row-delete-${item.name}`}
                        className="text-destructive focus:text-destructive"
                        onSelect={() => openEditDialog('delete', item.name)}
                      >
                        Hide type
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                )}
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
      {notebookId && editMode && (
        <SchemaEditDialog
          notebookId={notebookId}
          mode={editMode}
          typeName={editTarget}
          open={editMode !== null}
          onOpenChange={(open) => {
            if (!open) setEditMode(null)
          }}
        />
      )}
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
