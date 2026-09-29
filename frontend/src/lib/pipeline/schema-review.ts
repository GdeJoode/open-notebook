/**
 * Where a source gated at `awaiting_schema_review` is released (WB.2).
 *
 * Its extraction anchored on the first notebook linked to it
 * (`SourceRepository.get_notebook_id`); `GET /sources/{id}` lists `notebooks`
 * from the same edge in the same order, so the first is that notebook. Its
 * Schema tab holds the proposal queue (accepted into the shared ontology) and
 * the paused extraction's Resume. A source in no notebook has no Schema tab:
 * The Tray lists it.
 */
export function schemaReviewHref(notebooks?: string[] | null): string {
  const anchor = notebooks?.[0]
  return anchor ? `/notebooks/${encodeURIComponent(anchor)}/schema` : '/'
}
