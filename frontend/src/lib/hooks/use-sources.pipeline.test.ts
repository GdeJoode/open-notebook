import { describe, it, expect } from 'vitest'
import { pipelineRefetchInterval } from './use-sources'
import type { ProcessingStage } from '@/lib/pipeline/processing-stage'
import type { SourceResponse } from '@/lib/types/api'

/** Build the minimal react-query `query` shape the callback reads. */
function query(stage: ProcessingStage | undefined) {
  const data =
    stage === undefined
      ? ({} as SourceResponse)
      : ({ processing_stage: stage } as SourceResponse)
  return { state: { data } }
}

describe('pipelineRefetchInterval', () => {
  const CONTINUE: (ProcessingStage | undefined)[] = [
    'ingested',
    'embedded',
    'extracted',
    'graphed',
    undefined,
  ]
  const STOP: ProcessingStage[] = ['complete', 'failed', 'awaiting_schema_review']

  it.each(CONTINUE)('polls at 2000ms while stage is %s', (stage) => {
    expect(pipelineRefetchInterval(query(stage))).toBe(2000)
  })

  it.each(STOP)('stops polling (false) at terminal/gated stage %s', (stage) => {
    expect(pipelineRefetchInterval(query(stage))).toBe(false)
  })

  it('keeps polling when no data has arrived yet', () => {
    expect(pipelineRefetchInterval({ state: { data: undefined } })).toBe(2000)
  })
})
