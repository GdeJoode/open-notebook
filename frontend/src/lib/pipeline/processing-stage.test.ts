import { describe, it, expect } from 'vitest'
import {
  ProcessingStage,
  TERMINAL_STAGES,
  GATED_STAGES,
  isTerminalStage,
  isGatedStage,
  isPollableStage,
} from './processing-stage'

const ALL_STAGES: ProcessingStage[] = [
  'ingested',
  'embedded',
  'extracted',
  'awaiting_schema_review',
  'graphed',
  'complete',
  'failed',
]

describe('ProcessingStage constants', () => {
  it('mirrors the backend enum: exactly 7 values', () => {
    expect(ALL_STAGES).toHaveLength(7)
  })

  it('TERMINAL_STAGES is complete + failed', () => {
    expect([...TERMINAL_STAGES]).toEqual(['complete', 'failed'])
  })

  it('GATED_STAGES is awaiting_schema_review', () => {
    expect([...GATED_STAGES]).toEqual(['awaiting_schema_review'])
  })
})

describe('isTerminalStage', () => {
  it.each(ALL_STAGES)('resolves %s against TERMINAL_STAGES', (stage) => {
    const expected = stage === 'complete' || stage === 'failed'
    expect(isTerminalStage(stage)).toBe(expected)
  })

  it('is false for undefined', () => {
    expect(isTerminalStage(undefined)).toBe(false)
  })
})

describe('isGatedStage', () => {
  it.each(ALL_STAGES)('resolves %s against GATED_STAGES', (stage) => {
    const expected = stage === 'awaiting_schema_review'
    expect(isGatedStage(stage)).toBe(expected)
  })

  it('is false for undefined', () => {
    expect(isGatedStage(undefined)).toBe(false)
  })
})

describe('isPollableStage', () => {
  it.each(ALL_STAGES)('polls %s unless terminal or gated', (stage) => {
    const shouldStop =
      stage === 'complete' ||
      stage === 'failed' ||
      stage === 'awaiting_schema_review'
    expect(isPollableStage(stage)).toBe(!shouldStop)
  })

  it('keeps polling when the stage is unknown (undefined)', () => {
    expect(isPollableStage(undefined)).toBe(true)
  })
})
