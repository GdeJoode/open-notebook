'use client'

import Link from 'next/link'
import { useId, useMemo, useState } from 'react'
import { ChevronRight, Pencil } from 'lucide-react'

import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { useToggleReviewRequired } from '@/lib/hooks/use-notebook-schema'
import type {
  EntityTypeNode,
  NotebookSchemaResponse,
} from '@/lib/types/notebook_schema'

interface SchemaBrowserProps {
  schema: NotebookSchemaResponse
}

/** Where a type of a shared ontology is edited: the ontology editor's Edit tab. */
export function ontologyEditorHref(ontology: string, typeName?: string): string {
  const params = new URLSearchParams({ name: ontology, tab: 'edit' })
  if (typeName) params.set('type', typeName)
  return `/ontologies?${params.toString()}`
}

/**
 * Schema browser: the notebook's declared base ontology, read-only.
 *
 * OE.5: the ontologies are shared by every notebook, so nothing here edits
 * them. Each row links to the type in the ontology editor, where it is
 * renamed, merged, split, re-parented or deleted for every notebook at once.
 *
 * Renders a two-column layout:
 *  - left: a flat single-select `role="listbox"` of the base ontology's own
 *    entity types. Each row shows the type name and (when present) its
 *    `parent_type` in the side panel — the list itself is flat.
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
 *  3. **The tree lives in the editor.** The ontology editor shows the full
 *     type tree across inheritance; this list is a per-notebook entry point.
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
export function SchemaBrowser({ schema }: SchemaBrowserProps) {
  const base = schema.base_ontology
  const items: TreeItem[] = useMemo(() => {
    const rows: TreeItem[] = schema.base_ontology_types.map((et) => ({
      key: `base:${et.name}`,
      name: et.name,
      description: et.description ?? null,
      parent_type: et.parent_type ?? null,
      properties: et.properties,
    }))
    rows.sort((a, b) => a.name.localeCompare(b.name))
    return rows
  }, [schema])

  const [selectedKey, setSelectedKey] = useState<string | null>(
    items.length > 0 ? items[0].key : null,
  )
  const selected = items.find((it) => it.key === selectedKey) ?? null

  // Stable id prefix for option ids. Required so `aria-activedescendant`
  // on the listbox can point at the option DOM node. `useId` keeps the
  // prefix unique across multiple SchemaBrowser instances on a page.
  const optionIdPrefix = useId()
  const optionId = (key: string) =>
    `${optionIdPrefix}-${key.replace(/[^a-zA-Z0-9_-]/g, '_')}`
  const activeOptionId = selectedKey ? optionId(selectedKey) : undefined

  const sharedNote = (
    <p className="text-xs text-muted-foreground" data-testid="schema-shared-note">
      The ontologies are shared by every notebook. Rename, merge, split,
      re-parent or delete a type in the{' '}
      <Link
        href={base ? ontologyEditorHref(base) : '/ontologies'}
        className="underline"
        data-testid="schema-open-editor"
      >
        ontology editor
      </Link>
      .
    </p>
  )

  if (items.length === 0) {
    return (
      <div className="space-y-4">
        <ReviewRequiredToggle
          notebookId={schema.notebook_id}
          reviewRequired={schema.review_required}
        />
        <div
          role="status"
          className="rounded-md border border-dashed p-6 text-sm text-muted-foreground"
        >
          This notebook declares no base ontology; each document selects its
          ontologies itself.
        </div>
        {sharedNote}
      </div>
    )
  }

  return (
    <TooltipProvider>
      <div className="space-y-4">
        <ReviewRequiredToggle
          notebookId={schema.notebook_id}
          reviewRequired={schema.review_required}
        />
        {sharedNote}
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
                    </button>
                  </TooltipTrigger>
                  {item.description && (
                    <TooltipContent side="right" className="max-w-sm">
                      {item.description}
                    </TooltipContent>
                  )}
                </Tooltip>
                <Link
                  href={ontologyEditorHref(base, item.name)}
                  aria-label={`Edit ${item.name} in the ontology editor`}
                  title="Edit in the ontology editor"
                  data-testid={`schema-row-edit-${item.name}`}
                  className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <Pencil aria-hidden="true" className="h-3.5 w-3.5" />
                </Link>
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
      </div>
    </TooltipProvider>
  )
}

/**
 * B.3c — Switch at the top of the SchemaBrowser that toggles
 * `notebook_schema.review_required`. When enabled, the next entity
 * extraction for any source in this notebook will halt at the Pass-1
 * boundary until the user resumes via the workspace banner.
 *
 * We render this in the SchemaBrowser specifically (rather than the
 * notebook header) because the impact is schema-scoped: pause until
 * the schema has been reviewed.
 */
function ReviewRequiredToggle({
  notebookId,
  reviewRequired,
}: {
  notebookId: string
  reviewRequired: boolean
}) {
  const id = useId()
  const toggle = useToggleReviewRequired(notebookId)
  // The switch is optimistic — Radix's controlled value reflects the
  // pending toggle while the mutation is in flight. On error we
  // re-render with the latest `reviewRequired` from the schema query
  // (parent will refetch via onSuccess invalidation).
  return (
    <div
      className="flex flex-wrap items-center justify-between gap-3 rounded-md border bg-muted/30 p-3"
      data-testid="review-required-toggle"
    >
      <div className="min-w-0 space-y-0.5">
        <Label htmlFor={id} className="text-sm font-medium">
          Require review before extraction
        </Label>
        <p className="text-xs text-muted-foreground">
          When enabled, entity extraction pauses for newly-added sources
          until you accept or reject schema extensions.
        </p>
      </div>
      <Switch
        id={id}
        checked={reviewRequired}
        disabled={toggle.isPending}
        onCheckedChange={(checked) => toggle.mutate(checked)}
        aria-label="Require review before extraction"
        data-testid="review-required-switch"
      />
    </div>
  )
}

type TreeItem = {
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
                  <Badge variant="outline" className="text-xs">
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
