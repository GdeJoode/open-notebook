/**
 * Unit tests for the merge-cluster logic behind `MergeClusterCard` (Track K.6).
 *
 * The project's node-environment vitest cannot render the Radix AlertDialog, so
 * we test the pure apply-payload + confirmation-summary + reject-overlay logic
 * the card renders — the behaviour behind AC2 (confirm shows exactly what
 * merges) and AC3 (reject = reversible force-split). The dialog render,
 * keyboard, and click flow are covered by the Playwright E2E.
 */

import { describe, expect, it } from 'vitest'

import type { MergeCluster } from '@/lib/api/entity-resolution'
import {
  buildApplyCluster,
  buildRejectOverlay,
  describeMerge,
} from '@/lib/utils/entity-resolution'

const CLUSTER: MergeCluster = {
  new_canonical: 'binnenlandse zaken en koninkrijksrelaties',
  entity_type: 'organization',
  winner_id: 'entity:bzk1',
  loser_ids: ['entity:bzk2', 'entity:bzk3'],
  member_surface_forms: ['BZK', 'Ministerie van BZK', 'Binnenlandse Zaken'],
  total_source_docs: 4,
  size: 3,
}

describe('buildApplyCluster', () => {
  it('keeps the planned winner and all others as losers', () => {
    const out = buildApplyCluster(CLUSTER, 'entity:bzk1')
    expect(out.winner_id).toBe('entity:bzk1')
    expect(out.loser_ids).toEqual(['entity:bzk2', 'entity:bzk3'])
    expect(out.new_canonical).toBe(CLUSTER.new_canonical)
  })

  it('swaps the winner when the user picks a former loser (membership invariant)', () => {
    const out = buildApplyCluster(CLUSTER, 'entity:bzk2')
    expect(out.winner_id).toBe('entity:bzk2')
    // The old winner is now a loser; exactly one survivor remains.
    expect(out.loser_ids.sort()).toEqual(['entity:bzk1', 'entity:bzk3'])
    expect([out.winner_id, ...out.loser_ids].sort()).toEqual(
      ['entity:bzk1', 'entity:bzk2', 'entity:bzk3'].sort(),
    )
  })

  it('falls back to the planned winner for an unknown winner id', () => {
    const out = buildApplyCluster(CLUSTER, 'entity:nope')
    expect(out.winner_id).toBe('entity:bzk1')
    expect(out.loser_ids).toEqual(['entity:bzk2', 'entity:bzk3'])
  })
})

describe('describeMerge (the confirmation summary — AC2)', () => {
  it('reports the surviving label and the merged-away count', () => {
    const apply = buildApplyCluster(CLUSTER, 'entity:bzk1')
    const summary = describeMerge(apply)
    expect(summary.winnerLabel).toBe(CLUSTER.new_canonical)
    expect(summary.loserCount).toBe(2)
    expect(summary.loserIds).toEqual(['entity:bzk2', 'entity:bzk3'])
  })
})

describe('buildRejectOverlay (reject = reversible force-split — AC3)', () => {
  it('records a global force-split for the first two surface forms', () => {
    const rule = buildRejectOverlay(CLUSTER)
    expect(rule).not.toBeNull()
    expect(rule!.kind).toBe('split')
    expect(rule!.name_a).toBe('BZK')
    expect(rule!.name_b).toBe('Ministerie van BZK')
    expect(rule!.scope).toBe('global')
    expect(rule!.entity_type).toBe('organization')
  })

  it('scopes the split to a notebook when given one', () => {
    const rule = buildRejectOverlay(CLUSTER, 'notebook:abc')
    expect(rule!.scope).toBe('notebook')
    expect(rule!.notebook_id).toBe('notebook:abc')
  })

  it('returns null when there are not two surface forms to split', () => {
    const single = { ...CLUSTER, member_surface_forms: ['BZK'] }
    expect(buildRejectOverlay(single)).toBeNull()
  })
})
