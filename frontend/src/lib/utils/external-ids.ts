/**
 * Pure helpers for rendering `entity.external_ids` as provider badges (K.6, AC6).
 *
 * Kept framework-free (no JSX) so the node-environment vitest suite can assert
 * the classification/linkification logic directly — the badge render itself is
 * covered by the Playwright E2E.
 */

/** Classify a stable URI into a short provider label for the badge. */
export function providerLabel(uri: string): string {
  const u = uri.toLowerCase()
  // A DOI is either an explicit doi.org URL / doi: scheme, or a bare `10.xxxx`.
  if (u.includes('doi.org') || u.startsWith('doi:') || /(^|\/)10\.\d{4,}/.test(u))
    return 'DOI'
  if (u.includes('tooi') || u.includes('standaarden.overheid.nl')) return 'TOOI'
  if (u.includes('orcid')) return 'ORCID'
  if (u.includes('wikidata')) return 'Wikidata'
  if (u.includes('arxiv')) return 'arXiv'
  return 'ID'
}

/** Turn a bare DOI / scheme-less id into something linkable. */
export function toHref(uri: string): string {
  if (/^https?:\/\//i.test(uri)) return uri
  if (/^10\.\d{4,}/.test(uri)) return `https://doi.org/${uri}`
  if (/^doi:/i.test(uri)) return `https://doi.org/${uri.slice(4)}`
  return uri
}
