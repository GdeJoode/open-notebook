/**
 * The per-source pipeline stage, mirroring the backend enum
 * `packages/shared/src/shared/types/enums.py` → `class ProcessingStage`.
 *
 * This is the single source of truth the frontend reads to render per-document
 * progress (Track UX). The happy path is `ingested → embedded → extracted →
 * graphed → complete`; `awaiting_schema_review` parks the EXTRACT gate and
 * `failed` records a hard error.
 */
export type ProcessingStage =
  | 'ingested'
  | 'embedded'
  | 'extracted'
  | 'awaiting_schema_review'
  | 'graphed'
  | 'complete'
  | 'failed'

/**
 * Stages the pipeline never leaves on its own — polling/spinners should stop
 * here. `complete` is the success terminus, `failed` the error terminus.
 */
export const TERMINAL_STAGES = ['complete', 'failed'] as const

/**
 * Stages parked awaiting a human decision. Not terminal in the pipeline sense
 * (the source can resume once reviewed), but polling should stop because no
 * further automatic progress happens until the user acts.
 */
export const GATED_STAGES = ['awaiting_schema_review'] as const

export function isTerminalStage(stage: ProcessingStage | undefined): boolean {
  return stage !== undefined && (TERMINAL_STAGES as readonly string[]).includes(stage)
}

export function isGatedStage(stage: ProcessingStage | undefined): boolean {
  return stage !== undefined && (GATED_STAGES as readonly string[]).includes(stage)
}

/**
 * Whether a stage should keep polling the source read API. Polling continues
 * while the pipeline can still advance on its own; it stops once the stage is
 * terminal (`complete`/`failed`) or gated (`awaiting_schema_review`).
 *
 * `undefined` (stage not yet known — e.g. list-only data before the backfill,
 * or a fresh source) keeps polling: we cannot conclude the source is done.
 */
export function isPollableStage(stage: ProcessingStage | undefined): boolean {
  return !isTerminalStage(stage) && !isGatedStage(stage)
}
