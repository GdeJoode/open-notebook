/**
 * OE.3 review B1: a relationship's endpoints are stored under ONE of two
 * spellings — `domain`/`range`, or `source_types`/`target_types` (about 80
 * shipped relationships). The backend derives `domain`/`range` from the latter
 * and does not store them twice, so the editor must read either and write back
 * the spelling the relationship already uses. Reading `domain` alone showed
 * those relationships with no endpoints, and saving wiped them.
 */

import type { StoredRelationshipType } from '@/lib/api/ontologies'

const asList = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((v): v is string => typeof v === 'string') : []

export function endpointsOf(rel: StoredRelationshipType): { domain: string[]; range: string[] } {
  const domain = asList(rel.domain)
  const range = asList(rel.range)
  return {
    domain: domain.length ? domain : asList(rel.source_types),
    range: range.length ? range : asList(rel.target_types),
  }
}

/** The relationship with new endpoints, in the spelling it already uses; every
 *  other key kept (the edit API replaces a relationship whole). */
export function withEndpoints(
  rel: StoredRelationshipType,
  domain: string[],
  range: string[],
): StoredRelationshipType {
  const next: StoredRelationshipType = { ...rel }
  const legacy = !asList(rel.domain).length && ('source_types' in rel || 'target_types' in rel)
  if (legacy) {
    next.source_types = domain
    next.target_types = range
    delete next.domain
    delete next.range
  } else {
    next.domain = domain
    next.range = range
  }
  return next
}
