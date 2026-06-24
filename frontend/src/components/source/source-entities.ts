import type { Entity } from '@/lib/api/knowledge-graph'

/** A single row of the source Entities tab, pre-projected for display. */
export interface SourceEntityRow {
  id: string
  /** Resolved display name: canonical form preferred over raw surface name. */
  name: string
  /** Display type: rich ontology type preferred over the raw label. */
  type: string
  status?: string
  manualOverride?: boolean
  /** Effective structural degree, or null when un-triaged / absent. */
  degree: number | null
  /** Distinct cross-document count, or null when absent. */
  docCount: number | null
}

/** Render-state discriminant for the source Entities tab. */
export type SourceEntitiesView = 'loading' | 'error' | 'empty' | 'list'

/**
 * Project a KG `Entity` into the source-tab display row. Pure so the
 * field-selection logic (canonical-over-surface name, rich-over-raw type,
 * numeric-or-null) is unit-testable without rendering.
 */
export function toSourceEntityRow(entity: Entity): SourceEntityRow {
  return {
    id: entity.id,
    name: entity.canonical_name || entity.name,
    type: entity.primary_type || entity.entity_type,
    status: entity.status,
    manualOverride: entity.manual_override,
    degree:
      typeof entity.structural_degree === 'number'
        ? entity.structural_degree
        : null,
    docCount: typeof entity.doc_count === 'number' ? entity.doc_count : null,
  }
}

/**
 * Decide which view the tab should render. Loading and error take precedence
 * over data; an empty (or absent) item list is the empty state; otherwise the
 * table renders.
 */
export function sourceEntitiesView(args: {
  isLoading: boolean
  isError: boolean
  items: Entity[] | undefined
}): SourceEntitiesView {
  if (args.isLoading) return 'loading'
  if (args.isError) return 'error'
  if (!args.items || args.items.length === 0) return 'empty'
  return 'list'
}
