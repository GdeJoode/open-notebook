/**
 * Pure helpers for the triage UI surface (Track Q.5).
 *
 * The project's node-environment vitest cannot render Radix components, so the
 * load-bearing logic the status filter / badge / queue / backbone components
 * rely on lives here as framework-free functions the `.test.ts` suite asserts
 * directly. DOM / a11y / render behaviour is covered by the Playwright E2E.
 */

import type {
  BackboneWarning,
  DecisionInput,
  QueueItem,
  TriageStatus,
} from '@/lib/api/triage'
import { displaySourceTitle } from '@/lib/utils/source-title'

/** The status values the KG status filter offers (plus an "all" sentinel). */
export const TRIAGE_STATUSES: readonly TriageStatus[] = [
  'active',
  'reference',
  'archived',
] as const

/** Visual config for a status badge: label + the `Badge` variant to use. */
export interface StatusDisplay {
  label: string
  variant: 'default' | 'secondary' | 'destructive' | 'outline'
}

const STATUS_DISPLAY: Record<TriageStatus, StatusDisplay> = {
  // active = in the active KG → the primary/affirmative variant.
  active: { label: 'Active', variant: 'default' },
  // reference = kept but outside the active KG → muted secondary.
  reference: { label: 'Reference', variant: 'secondary' },
  // archived = removed from view → outline (least emphasis).
  archived: { label: 'Archived', variant: 'outline' },
}

/**
 * Resolve a raw status string to its badge display. Unknown / missing values
 * fall back to an outline "Unknown" badge so a legacy row never crashes the
 * table.
 */
export function statusDisplay(status: string | undefined | null): StatusDisplay {
  if (status && status in STATUS_DISPLAY) {
    return STATUS_DISPLAY[status as TriageStatus]
  }
  return { label: status ? String(status) : 'Unknown', variant: 'outline' }
}

/** Is `status` one of the known triage statuses (filter validity guard)? */
export function isTriageStatus(status: string): status is TriageStatus {
  return (TRIAGE_STATUSES as readonly string[]).includes(status)
}

/**
 * Build the decision payload for a queue-row decision control. The three
 * operator actions map to:
 *   - promote  → pin `active`
 *   - keep     → pin `reference`
 *   - override → pin the current status by hand (operator wants it to stick)
 * `override` keeps whatever status the row already implies; we treat it as a
 * `reference` pin unless a status is supplied, because an unsure_review row is a
 * reference row until promoted.
 */
export type QueueAction = 'promote' | 'keep' | 'override'

export function buildDecision(
  action: QueueAction,
  currentStatus?: string,
): DecisionInput {
  switch (action) {
    case 'promote':
      return { status: 'active', decision: 'promote → active' }
    case 'keep':
      return { status: 'reference', decision: 'keep reference' }
    case 'override': {
      // Pin whatever the entity already is; default to reference.
      const status =
        currentStatus === 'active' ? 'active' : 'reference'
      return { status, decision: 'manual override' }
    }
  }
}

/**
 * Group queue rows by their (first) reason clause so the queue view can render
 * one section per reason ("unsure — operator decides", "isolated core actor",
 * "candidate active", "match conflict"). A row whose reason carries several
 * `"; "`-joined clauses is grouped under its FIRST clause (the status-flag
 * reason that opened it); the full reason is still shown on the row. Order is
 * stable: groups appear in first-seen order, rows keep their input order.
 */
export function groupByReason(items: QueueItem[]): Array<{
  reason: string
  items: QueueItem[]
}> {
  const order: string[] = []
  const byReason = new Map<string, QueueItem[]>()
  for (const item of items) {
    // TOOI rows form one section of their own: their reason line lists every
    // candidate, so grouping on it would give each row its own section.
    const key = isTooiRow(item) ? TOOI_SECTION : primaryReason(item.reason)
    if (!byReason.has(key)) {
      byReason.set(key, [])
      order.push(key)
    }
    byReason.get(key)!.push(item)
  }
  return order.map((reason) => ({ reason, items: byReason.get(reason)! }))
}

/** The section heading for TOOI link candidates (D2). */
export const TOOI_SECTION = 'TOOI link candidate — pick the organisation or dismiss'

/** Is this a TOOI link row (decided with link/dismiss, never a status pin)? */
export function isTooiRow(item: Pick<QueueItem, 'kind'>): boolean {
  return item.kind === 'tooi'
}

/** A candidate's display: its label, the TOOI code, and its end date if ended. */
export function tooiCandidateLabel(c: {
  uri: string
  label: string
  valid_until?: string
}): string {
  const code = c.uri.split('/').slice(-2).join('/')
  return `${c.label} (${code})${c.valid_until ? ` — ended ${c.valid_until}` : ''}`
}

/**
 * A source title for display: the upload pipeline prefixes stored titles with a
 * 12-hex content hash (`49b9320d0e1e__Convenant …`), which means nothing to a
 * curator. An untitled source shows its id.
 */
export function sourceDisplayTitle(source: { id: string; title: string }): string {
  const title = displaySourceTitle(source.title).trim()
  return title || source.id
}

/**
 * The hover on a queue row's *Docs* number (Q-B1): one line per document, and a
 * note when the live list no longer matches the triage run's `doc_count`.
 */
export function docsTooltipLines(
  item: Pick<QueueItem, 'doc_count' | 'sources'>,
): string[] {
  const sources = item.sources ?? []
  const lines = sources.length
    ? sources.map(sourceDisplayTitle)
    : ['No source document recorded']
  if (sources.length !== item.doc_count) {
    lines.push(`(triage counted ${item.doc_count}; ${sources.length} now)`)
  }
  return lines
}

/** The first `"; "`-joined clause of a (possibly multi-) reason string. */
export function primaryReason(reason: string): string {
  const first = (reason || '').split(';')[0]?.trim()
  return first || 'other'
}

/** Is the queue row still awaiting an operator (rendered in the open queue)? */
export function isOpen(item: Pick<QueueItem, 'decision'>): boolean {
  return (item.decision || 'open') === 'open'
}

/**
 * Total weak-edge count across a set of backbone warnings — the headline number
 * the `BackboneWarnings` summary shows ("N weak edges on M backbone entities").
 */
export function totalWeakEdges(warnings: BackboneWarning[]): number {
  return warnings.reduce((sum, w) => sum + w.weak_edges.length, 0)
}

/** Backbone warnings only matter for degree >= 3 entities (the plan's gate). */
export function isBackboneRisk(warning: BackboneWarning): boolean {
  return warning.structural_degree >= 3 && warning.weak_edges.length > 0
}
