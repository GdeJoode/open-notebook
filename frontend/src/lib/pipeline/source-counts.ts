/**
 * Adapter from a list/detail source payload to the {@link PipelineCounts} the
 * pipeline state machine consumes.
 *
 * The crux (Track UX.3 AC0 — carry-forward from the UX.2 review): the list
 * endpoint types `embedded_chunks`/`entity_count`/`insights_count` as required
 * `number` with a default of `0`. Feeding those raw `0`s into
 * `derivePipelineNodes` is safe on the success paths (a count of `0` never moves
 * a node there), but it is NOT safe on the terminal `failed` path.
 *
 * On the `failed` path the backend overwrites `processing_stage = 'failed'`,
 * discarding the reached stage, so `derivePipelineNodes` locates the failure at
 * the FIRST spine stage whose output signal is absent. It treats a count that is
 * merely `undefined` as "stage not reached" and a count of `0` as "stage reached
 * but empty". A genuine parse failure therefore arrives as
 * `{embedded_chunks: 0, entity_count: 0}` — and if those `0`s are passed
 * verbatim, the raw `0` (being `!== undefined`) is read as "Ingest reached",
 * mislabelling the failure as "Ingest done / Embed failed" instead of the true
 * "Ingest failed".
 *
 * The fix is to collapse `0 ⇒ undefined` for the count-gated stages at this
 * wiring seam, so "reached-but-empty" and "never-reached" are only ever
 * distinguished by real evidence (a positive count), never by the list default.
 * A positive count passes through unchanged.
 *
 * Note (UX.4 nit): the Graph node's "linked" badge is keyed off the STAGE, not
 * off `relation_count`, so no graph flag is derived here. `relation_count`
 * counts entity↔entity relations, which is NOT the same as document-graph
 * `mentions` membership; a `graphed` source with zero relations is still linked.
 */

import type { SourceListResponse } from '@/lib/types/api'
import type { PipelineCounts } from './pipeline-stages'

/** Collapse the list-endpoint default `0` to `undefined`; pass `> 0` through. */
function positiveOrUndefined(value: number | undefined): number | undefined {
  return value && value > 0 ? value : undefined
}

/**
 * Build {@link PipelineCounts} from a source row, mapping `0 ⇒ undefined` for
 * the count-gated stages (see the module doc for why).
 */
export function toPipelineCounts(
  source: Pick<
    SourceListResponse,
    'embedded_chunks' | 'entity_count' | 'insights_count'
  >
): PipelineCounts {
  return {
    embedded_chunks: positiveOrUndefined(source.embedded_chunks),
    entity_count: positiveOrUndefined(source.entity_count),
    insights_count: positiveOrUndefined(source.insights_count),
  }
}
