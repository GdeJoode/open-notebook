/**
 * Unit tests for the overlay-rule logic behind `OverlayEditor` (Track K.6).
 *
 * Node-environment vitest cannot render the Radix RadioGroup scope-toggle, so we
 * test the pure validation + candidate-promotion + reject-overlay logic the
 * editor and candidate rows rely on. Keyboard/a11y of the RadioGroup is covered
 * by the Playwright E2E (AC5).
 */

import { describe, expect, it } from 'vitest'

import type { MergeCandidate, OverlayCreateInput } from '@/lib/api/entity-resolution'
import {
  buildCandidateRejectOverlay,
  candidateToApplyCluster,
  isValidOverlay,
} from '@/lib/utils/entity-resolution'

const base: OverlayCreateInput = {
  kind: 'split',
  name_a: 'BZK',
  name_b: 'EZK',
  entity_type: 'organization',
  scope: 'global',
  notebook_id: null,
}

describe('isValidOverlay', () => {
  it('accepts a well-formed global rule', () => {
    expect(isValidOverlay(base)).toBe(true)
  })

  it('rejects an unknown kind', () => {
    expect(
      isValidOverlay({ ...base, kind: 'bogus' as OverlayCreateInput['kind'] }),
    ).toBe(false)
  })

  it('rejects empty/whitespace surface forms', () => {
    expect(isValidOverlay({ ...base, name_a: '' })).toBe(false)
    expect(isValidOverlay({ ...base, name_b: '   ' })).toBe(false)
  })

  it('requires a notebook id when scope is notebook', () => {
    expect(
      isValidOverlay({ ...base, scope: 'notebook', notebook_id: null }),
    ).toBe(false)
    expect(
      isValidOverlay({ ...base, scope: 'notebook', notebook_id: 'notebook:x' }),
    ).toBe(true)
  })

  it('accepts a force-merge rule', () => {
    expect(isValidOverlay({ ...base, kind: 'merge' })).toBe(true)
  })
})

const candidate: MergeCandidate = {
  id_a: 'entity:a',
  id_b: 'entity:b',
  name_a: 'Koninkrijksrelaties',
  name_b: 'Koninkrijksreiaties',
  entity_type: 'organization',
  score: 0.93,
  band: 'auto_merge',
  method: 'fuzzy',
  winner_id: 'entity:a',
  loser_id: 'entity:b',
}

describe('candidateToApplyCluster', () => {
  it('maps a fuzzy candidate to an apply cluster with one loser', () => {
    const out = candidateToApplyCluster(candidate)
    expect(out.winner_id).toBe('entity:a')
    expect(out.loser_ids).toEqual(['entity:b'])
    expect(out.new_canonical).toBe('Koninkrijksrelaties')
    expect(out.member_surface_forms).toEqual([
      'Koninkrijksrelaties',
      'Koninkrijksreiaties',
    ])
  })
})

describe('buildCandidateRejectOverlay', () => {
  it('records a force-split for the candidate pair (global by default)', () => {
    const rule = buildCandidateRejectOverlay(candidate)
    expect(rule.kind).toBe('split')
    expect(rule.name_a).toBe('Koninkrijksrelaties')
    expect(rule.name_b).toBe('Koninkrijksreiaties')
    expect(rule.scope).toBe('global')
  })

  it('scopes to a notebook when supplied', () => {
    const rule = buildCandidateRejectOverlay(candidate, 'notebook:z')
    expect(rule.scope).toBe('notebook')
    expect(rule.notebook_id).toBe('notebook:z')
  })
})
