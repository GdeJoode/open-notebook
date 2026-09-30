/**
 * The upload pipeline stages files as `<12 hex>__<name>` and the title is taken
 * from that name (`410a65bba3f8__Convenant Regio Deal …`). The prefix means
 * nothing to the owner and makes copies look like near-duplicates.
 */
const UPLOAD_PREFIX = /^[0-9a-f]{12}__/

/**
 * A source title as shown (WB.3). Every place that renders a source title goes
 * through this; `source-title.test.ts` finds them by scanning `src/`. The stored
 * title is unchanged, and the source's Details tab keeps the uploaded file name.
 * Text the server writes around a title uses its twin, `source_title.py`.
 */
export function displaySourceTitle(title: string | null | undefined): string {
  return (title ?? '').replace(UPLOAD_PREFIX, '')
}
