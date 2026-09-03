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
import {
  candidateToApplyCluster,
  candidateTypeLabel,
  isCrossTypeCandidate,
} from '@/lib/utils/entity-resolution'

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

describe('the survivor is labelled by the winner, not by position (PC.2)', () => {
  it('uses b\'s name when b wins on confidence', () => {
    const out = candidateToApplyCluster({
      ...CANDIDATE,
      winner_id: 'entity:vws2',
      loser_id: 'entity:vws1',
    })
    // The apply repoints relations onto the winner; labelling the survivor
    // `name_a` would rename it to the entity that was just absorbed.
    expect(out.new_canonical).toBe('Volksgezondheid')
    expect(out.loser_ids).toEqual(['entity:vws1'])
  })

  it('uses b\'s type for a cross-type pair when b wins', () => {
    const crossType: MergeCandidate = {
      ...CANDIDATE,
      name_a: 'Regio Deal',
      name_b: 'Regio Deal',
      entity_type: 'programme',
      entity_type_b: 'topic',
      method: 'fold_equal_cross_type',
      band: 'review',
      winner_id: 'entity:vws2',
      loser_id: 'entity:vws1',
    }
    expect(candidateToApplyCluster(crossType).entity_type).toBe('topic')
    expect(
      candidateToApplyCluster({
        ...crossType,
        winner_id: 'entity:vws1',
        loser_id: 'entity:vws2',
      }).entity_type,
    ).toBe('programme')
  })

  it('leaves a same-type pair on its single type', () => {
    // `entity_type_b` is "" for same-type pairs; it must never blank the type.
    expect(
      candidateToApplyCluster({
        ...CANDIDATE,
        entity_type_b: '',
        winner_id: 'entity:vws2',
        loser_id: 'entity:vws1',
      }).entity_type,
    ).toBe('organization')
  })
})

describe('candidateTypeLabel (what the card shows for a cross-type pair)', () => {
  const CROSS: MergeCandidate = {
    ...CANDIDATE,
    name_a: 'Regio Deal',
    name_b: 'Regio Deal',
    entity_type: 'programme',
    entity_type_b: 'topic',
    method: 'fold_equal_cross_type',
  }

  it('shows both types when they differ', () => {
    // Without this the card reads "Regio Deal ↔ Regio Deal · programme", which
    // gives a curator no reason the two rows exist.
    expect(isCrossTypeCandidate(CROSS)).toBe(true)
    expect(candidateTypeLabel(CROSS)).toBe('programme ↔ topic')
  })

  it('shows one type for a same-type pair', () => {
    expect(isCrossTypeCandidate(CANDIDATE)).toBe(false)
    expect(candidateTypeLabel(CANDIDATE)).toBe('organization')
  })

  it('treats an equal entity_type_b as same-type', () => {
    const same = { ...CROSS, entity_type_b: 'programme' }
    expect(isCrossTypeCandidate(same)).toBe(false)
    expect(candidateTypeLabel(same)).toBe('programme')
  })
})
