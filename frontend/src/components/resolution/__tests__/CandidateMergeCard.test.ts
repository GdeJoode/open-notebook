/**
 * Unit tests for the fuzzy-candidate merge logic behind `CandidateMergeCard`
 * (Track K.6). The card renders the shared destructive-merge confirmation gate
 * (Radix AlertDialog), which the node-environment vitest cannot mount — the
 * dialog render + click flow is covered by the Playwright E2E. Here we assert
 * the pure apply-payload the card hands to that gate: a candidate promotes to a
 * single-loser cluster with the winner/loser ids and surface forms preserved.
 */

import { describe, expect, it } from 'vitest'

import type { MergeCandidate } from '@/lib/api/entity-resolution'
import { candidateToApplyCluster } from '@/lib/utils/entity-resolution'

const CANDIDATE: MergeCandidate = {
  id_a: 'entity:vws1',
  id_b: 'entity:vws2',
  name_a: 'VWS',
  name_b: 'Volksgezondheid',
  entity_type: 'organization',
  score: 0.82,
  band: 'review',
  method: 'embedding',
  winner_id: 'entity:vws1',
  loser_id: 'entity:vws2',
}

describe('candidateToApplyCluster (the gated apply payload — AC2)', () => {
  it('promotes a candidate pair to a single-loser apply cluster', () => {
    const out = candidateToApplyCluster(CANDIDATE)
    expect(out.winner_id).toBe('entity:vws1')
    expect(out.loser_ids).toEqual(['entity:vws2'])
    expect(out.entity_type).toBe('organization')
  })

  it('carries both surface forms so the confirm gate can label them', () => {
    const out = candidateToApplyCluster(CANDIDATE)
    expect(out.member_surface_forms).toEqual(['VWS', 'Volksgezondheid'])
    expect(out.new_canonical).toBe('VWS')
  })
})
