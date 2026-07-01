/**
 * Pure (React-free) state machine that turns a source's `processing_stage`
 * plus the job-status axis and output counts into a renderable pipeline model.
 *
 * This is the successor to `derivePipelineStatuses` in `CreateSourcePipeline.tsx`.
 * That function inferred the stage from output-count presence, in the wrong
 * order (Extract before Embed), and conflated the job axis with the spine. The
 * rules here follow Track UX's locked decisions:
 *
 *   - `processing_stage` is the single source of truth for which spine node a
 *     source has reached; the ordered spine is Ingest → Embed → Extract →
 *     Graph → Complete (Embed BEFORE Extract), with Insights as a PARALLEL node
 *     branching off Embed (never inline in the spine).
 *   - The job status is a second axis: it only decides whether the CURRENT node
 *     shows a spinner (`active`) versus a check (`done`). It never determines
 *     which stage a source is in.
 *   - Output counts (`embedded_chunks`, `entity_count`, `insights_count`, and a
 *     graph-present flag) are ENRICHMENT on `done` nodes only. A count of `0`
 *     must never change a node's state (e.g. a `graphed` source with
 *     `entity_count === 0` still renders Extract as `done`).
 *
 * The single, deliberate exception to that last rule is the terminal `failed`
 * path: the backend writes `processing_stage = 'failed'` at three points (parse,
 * embed, entity-extract), OVERWRITING the prior good stage, so the string loses
 * its position. There — and ONLY there — the output counts SET state, because a
 * positive output is the only honest evidence of which stage was actually
 * reached before the break. On every non-`failed` path `processing_stage` stays
 * authoritative and counts remain enrichment-only.
 */

import type { ProcessingStage } from './processing-stage'

/** Visual state of a single pipeline node. */
export type PipelineNodeState = 'pending' | 'active' | 'done' | 'gated' | 'failed'

/** Stable identity for each node (spine + the parallel Insights branch). */
export type PipelineNodeKey =
  | 'ingest'
  | 'embed'
  | 'extract'
  | 'graph'
  | 'complete'
  | 'insights'

/**
 * Tab a node deep-links to when clicked. Kept as string constants so the
 * consuming surfaces (card / detail / create) can map them to their own tab
 * keys. `graph` currently shares the entities tab (no dedicated graph tab).
 */
export type PipelineDeepLinkTab = 'chunks' | 'entities' | 'insights'

/** Recovery affordance a node exposes, if any. */
export type PipelineNodeAction = 'review-schema' | 'retry'

/**
 * The job-status axis (GET `/sources/{id}/status`). Only the "still working"
 * subset drives the current node's spinner; everything else resolves the
 * current node to `done` (failure is carried by the spine, not this axis —
 * except the pre-overwrite window handled below).
 */
export type PipelineJobStatus =
  | 'new'
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'paused_for_review'

/** Output counts used purely to enrich `done` nodes (never to set state). */
export interface PipelineCounts {
  embedded_chunks?: number
  entity_count?: number
  insights_count?: number
  /** Whether the source participates in the document graph (mentions edges). */
  graph_present?: boolean
}

export interface PipelineNode {
  key: PipelineNodeKey
  label: string
  state: PipelineNodeState
  /** Raw enrichment count (present on `done` nodes only). */
  count?: number
  /** Human label for the count, e.g. "24 chunks" (present when count > 0). */
  countLabel?: string
  deepLinkTab?: PipelineDeepLinkTab
  action?: PipelineNodeAction
  /** True for the Insights node — rendered branching off Embed, not inline. */
  parallel?: boolean
}

export interface DerivePipelineNodesInput {
  processingStage: ProcessingStage | undefined
  jobStatus?: PipelineJobStatus | string
  counts?: PipelineCounts
}

/**
 * The ordered spine in RUNTIME order. Insights is intentionally absent — it is
 * a parallel branch appended separately by {@link derivePipelineNodes}.
 */
export const SPINE_NODES = [
  { key: 'ingest', label: 'Ingest', deepLinkTab: 'chunks' },
  { key: 'embed', label: 'Embed', deepLinkTab: 'chunks' },
  { key: 'extract', label: 'Extract', deepLinkTab: 'entities' },
  { key: 'graph', label: 'Graph', deepLinkTab: 'entities' },
  { key: 'complete', label: 'Complete', deepLinkTab: undefined },
] as const satisfies ReadonlyArray<{
  key: PipelineNodeKey
  label: string
  deepLinkTab?: PipelineDeepLinkTab
}>

/** The parallel Insights node descriptor (branches off Embed). */
export const INSIGHTS_NODE = {
  key: 'insights',
  label: 'Insights',
  deepLinkTab: 'insights',
} as const satisfies { key: PipelineNodeKey; label: string; deepLinkTab: PipelineDeepLinkTab }

/**
 * Spine index that a `processing_stage` value maps to. `awaiting_schema_review`
 * parks the Extract node, so it shares Extract's index. `complete` maps to the
 * terminal Complete node. `failed` has no position (see derive logic) and is
 * intentionally excluded here.
 */
const STAGE_TO_SPINE_INDEX: Record<
  Exclude<ProcessingStage, 'failed'>,
  number
> = {
  ingested: 0,
  embedded: 1,
  extracted: 2,
  awaiting_schema_review: 2,
  graphed: 3,
  complete: 4,
}

const EMBED_INDEX = 1

/** Job statuses that keep the current node spinning rather than resolving it. */
const ACTIVE_JOB_STATUSES: readonly string[] = ['new', 'queued', 'running']

function isActiveJob(jobStatus?: string): boolean {
  return jobStatus !== undefined && ACTIVE_JOB_STATUSES.includes(jobStatus)
}

/**
 * Build the count enrichment for a `done` spine node. Returns `undefined` when
 * the node carries no count. A count of `0` yields a raw `count` (so callers
 * can assert it) but no `countLabel` (avoids a misleading "0 chunks" chip) —
 * crucially it never affects the node's `state`.
 */
function enrichCount(
  key: PipelineNodeKey,
  counts: PipelineCounts | undefined
): Pick<PipelineNode, 'count' | 'countLabel'> {
  if (!counts) return {}
  switch (key) {
    case 'embed': {
      const n = counts.embedded_chunks
      if (n === undefined) return {}
      // Omit the label key entirely on 0 (avoid a misleading "0 chunks" chip and
      // a stray `countLabel: undefined` on the node).
      return n > 0 ? { count: n, countLabel: `${n} chunks` } : { count: n }
    }
    case 'extract': {
      const n = counts.entity_count
      if (n === undefined) return {}
      return n > 0 ? { count: n, countLabel: `${n} entities` } : { count: n }
    }
    case 'graph': {
      // Graph carries no numeric count; the graph-present flag is a soft badge.
      return counts.graph_present ? { countLabel: 'linked' } : {}
    }
    default:
      return {}
  }
}

/**
 * Derive the Insights parallel node. Insights never affects the spine.
 *
 *   - `done` once at least one insight exists (`insights_count > 0`).
 *   - `active` while the insights branch is plausibly generating: the source
 *     has passed Embed, a job is still running, and no insight has landed yet.
 *     (There is no dedicated insights job-status axis, so this is derived from
 *     the shared job status + stage position.)
 *   - `pending` otherwise.
 */
function deriveInsightsNode(input: DerivePipelineNodesInput): PipelineNode {
  const insightsCount = input.counts?.insights_count ?? 0
  const stage = input.processingStage
  const base: PipelineNode = {
    key: INSIGHTS_NODE.key,
    label: INSIGHTS_NODE.label,
    deepLinkTab: INSIGHTS_NODE.deepLinkTab,
    parallel: true,
    state: 'pending',
  }

  if (insightsCount > 0) {
    return {
      ...base,
      state: 'done',
      count: insightsCount,
      countLabel: `${insightsCount} insights`,
    }
  }

  const stageIndex =
    stage !== undefined && stage !== 'failed'
      ? STAGE_TO_SPINE_INDEX[stage]
      : -1
  const pastEmbed = stageIndex >= EMBED_INDEX && stage !== 'complete'
  if (pastEmbed && isActiveJob(input.jobStatus)) {
    return { ...base, state: 'active' }
  }
  return base
}

/**
 * Derive the full pipeline node model for a source.
 *
 * Returns the five spine nodes in runtime order followed by the parallel
 * Insights node (flagged `parallel: true`). Consumers that render only the
 * spine can filter on `!node.parallel`.
 */
export function derivePipelineNodes(
  input: DerivePipelineNodesInput
): PipelineNode[] {
  const { processingStage: stage, jobStatus, counts } = input

  const spine: PipelineNode[] = SPINE_NODES.map((n) => ({
    key: n.key,
    label: n.label,
    deepLinkTab: n.deepLinkTab,
    state: 'pending' as PipelineNodeState,
  }))

  const setState = (index: number, state: PipelineNodeState) => {
    spine[index].state = state
  }

  const hydrateDone = () => {
    // Enrich every currently-`done` spine node with its output count.
    for (const node of spine) {
      if (node.state !== 'done') continue
      Object.assign(node, enrichCount(node.key, counts))
    }
  }

  if (stage === undefined) {
    // Graceful pre-migration / list-only fallback: nothing is known, so the
    // whole spine is pending. Insights still follows its own count rule.
    return [...spine, deriveInsightsNode(input)]
  }

  if (stage === 'failed') {
    // TERMINAL FAILED PATH — the ONE place counts SET state.
    //
    // The backend writes `processing_stage = 'failed'` at three distinct points
    // (parse `handlers.py:183`, entity-extract `handlers.py:361`, embed
    // `handlers.py:403`), OVERWRITING the prior good stage, so the string
    // carries NO position. Pinning the failure at the entry node would regress
    // stages that demonstrably completed (e.g. a source that ingested + embedded
    // fine but broke at extraction) and misdirect Retry at Ingest.
    //
    // Positive output counts are reliable truth even when the stage string lost
    // its position, so we locate the failure at the FIRST spine stage whose
    // output signal is absent. This is the deliberate, documented exception to
    // the "counts never set state" rule that governs the (unchanged) success
    // paths, where a count of 0 must never move a node.
    const embeddedChunks = counts?.embedded_chunks
    const entityCount = counts?.entity_count
    const embedReached = (embeddedChunks ?? 0) > 0
    const extractReached = (entityCount ?? 0) > 0
    // PipelineCounts has no dedicated parse/chunk count, so Ingest is treated as
    // reached when ANY downstream count signal is present (even a value of 0): a
    // source that exposes an embed/extract count at all necessarily got past
    // parse and row-creation. With no count signal whatsoever, the only honest
    // conclusion is that parse itself failed.
    const ingestReached =
      embedReached ||
      extractReached ||
      embeddedChunks !== undefined ||
      entityCount !== undefined

    let failedIndex: number
    if (!ingestReached) {
      failedIndex = 0 // Parse failed — no evidence the source got past Ingest.
    } else if (!embedReached) {
      failedIndex = EMBED_INDEX // Embed produced no chunks.
    } else if (!extractReached) {
      failedIndex = 2 // Extract produced no entities.
    } else {
      // All three spine signals are present. The backend does not write `failed`
      // at graph/complete, so this is unexpected; rather than false-claim an
      // earlier node completed-then-failed, deterministically pin the failure at
      // the furthest spine node (Graph).
      failedIndex = 3
    }

    for (let i = 0; i < failedIndex; i++) setState(i, 'done')
    setState(failedIndex, 'failed')
    spine[failedIndex].action = 'retry'
    hydrateDone()
    return [...spine, deriveInsightsNode(input)]
  }

  if (stage === 'complete') {
    for (let i = 0; i < spine.length; i++) setState(i, 'done')
    hydrateDone()
    return [...spine, deriveInsightsNode(input)]
  }

  const currentIndex = STAGE_TO_SPINE_INDEX[stage]

  // Earlier spine nodes are unconditionally done.
  for (let i = 0; i < currentIndex; i++) setState(i, 'done')

  // The current node's state is driven by the second axis (job status), except
  // the schema-review gate which parks Extract regardless of the job.
  if (stage === 'awaiting_schema_review') {
    setState(currentIndex, 'gated')
    spine[currentIndex].action = 'review-schema'
  } else if (jobStatus === 'failed') {
    // Pre-overwrite window: the job axis reports failure before the handler has
    // committed `processing_stage = 'failed'`. Localise the failure at the
    // current node (earlier stages remain done).
    setState(currentIndex, 'failed')
    spine[currentIndex].action = 'retry'
  } else if (isActiveJob(jobStatus)) {
    setState(currentIndex, 'active')
  } else {
    setState(currentIndex, 'done')
  }

  hydrateDone()
  return [...spine, deriveInsightsNode(input)]
}
