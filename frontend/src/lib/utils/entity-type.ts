/**
 * Entity types as shown (WB.3): the canonical `entity_type` (`organization`,
 * `administrative_area`) first, everywhere, and the ontology's display label
 * (`primary_type`, e.g. `Gemeente`; a triage row's stored `type`) only as
 * secondary text, and only when it says something the canonical type does not.
 */
export interface EntityTypeLabels {
  /** The canonical type; '' when unknown. */
  canonical: string
  /** The display label, or null when absent or the same as the canonical type. */
  label: string | null
}

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '')

export function entityTypeLabels(
  canonical: string | null | undefined,
  display: string | null | undefined,
): EntityTypeLabels {
  const c = (canonical ?? '').trim()
  const d = (display ?? '').trim()
  return { canonical: c, label: d && norm(d) !== norm(c) ? d : null }
}
