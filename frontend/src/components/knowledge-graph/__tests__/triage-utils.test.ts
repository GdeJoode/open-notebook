/**
 * Unit tests for the pure triage-UI helpers (Track Q.5).
 *
 * The project's node-environment vitest cannot render Radix components, so these
 * assert the load-bearing logic behind the StatusBadge / StatusFilter / triage
 * queue / BackboneWarnings components — the status→display mapping (the badge),
 * the filter validity guard, the decision-payload builder (the queue decision
 * control), reason grouping (the queue sections), and the backbone summary. The
 * render / a11y / click flow is covered by the Playwright E2E.
 */

import { describe, expect, it } from 'vitest'

import type { BackboneWarning, QueueItem } from '@/lib/api/triage'
import {
  TRIAGE_STATUSES,
  buildDecision,
  groupByReason,
  isBackboneRisk,
  isOpen,
  isTriageStatus,
  primaryReason,
  statusDisplay,
  totalWeakEdges,
} from '@/lib/utils/triage'

function queueItem(over: Partial<QueueItem> = {}): QueueItem {
  return {
    id: 'triage_queue:1',
    entity: 'entity:a',
    name: 'Org A',
    type: 'Organisatie',
    structural_degree: 0,
    doc_count: 2,
    reason: 'unsure — operator decides',
    decision: 'open',
    batch_id: 'b1',
    ...over,
  }
}

describe('statusDisplay (StatusBadge)', () => {
  it('maps each known status to a label + variant', () => {
    expect(statusDisplay('active')).toEqual({ label: 'Active', variant: 'default' })
    expect(statusDisplay('reference')).toEqual({
      label: 'Reference',
      variant: 'secondary',
    })
    expect(statusDisplay('archived')).toEqual({
      label: 'Archived',
      variant: 'outline',
    })
  })

  it('degrades an unknown / missing status to an outline Unknown badge', () => {
    expect(statusDisplay('weird')).toEqual({ label: 'weird', variant: 'outline' })
    expect(statusDisplay(undefined)).toEqual({
      label: 'Unknown',
      variant: 'outline',
    })
    expect(statusDisplay(null)).toEqual({ label: 'Unknown', variant: 'outline' })
  })
})

describe('isTriageStatus / TRIAGE_STATUSES (StatusFilter)', () => {
  it('offers exactly the three triage statuses', () => {
    expect(TRIAGE_STATUSES).toEqual(['active', 'reference', 'archived'])
  })

  it('guards filter validity', () => {
    expect(isTriageStatus('reference')).toBe(true)
    expect(isTriageStatus('all')).toBe(false)
    expect(isTriageStatus('bogus')).toBe(false)
  })
})

describe('buildDecision (queue decision control)', () => {
  it('promote pins active', () => {
    expect(buildDecision('promote')).toEqual({
      status: 'active',
      decision: 'promote → active',
    })
  })

  it('keep pins reference', () => {
    expect(buildDecision('keep')).toEqual({
      status: 'reference',
      decision: 'keep reference',
    })
  })

  it('override pins the current status (active stays active, else reference)', () => {
    expect(buildDecision('override', 'active').status).toBe('active')
    expect(buildDecision('override', 'reference').status).toBe('reference')
    expect(buildDecision('override').status).toBe('reference')
    expect(buildDecision('override', 'active').decision).toBe('manual override')
  })
})

describe('groupByReason / primaryReason (queue sections)', () => {
  it('groups rows by their first reason clause in first-seen order', () => {
    const items = [
      queueItem({ id: 'q:1', reason: 'unsure — operator decides' }),
      queueItem({ id: 'q:2', reason: 'isolated core actor' }),
      queueItem({ id: 'q:3', reason: 'unsure — operator decides' }),
    ]
    const groups = groupByReason(items)
    expect(groups.map((g) => g.reason)).toEqual([
      'unsure — operator decides',
      'isolated core actor',
    ])
    expect(groups[0].items.map((i) => i.id)).toEqual(['q:1', 'q:3'])
    expect(groups[1].items.map((i) => i.id)).toEqual(['q:2'])
  })

  it('keys a multi-clause reason on its first clause (the status-flag reason)', () => {
    expect(
      primaryReason('candidate active; match conflict: A ~ A.'),
    ).toBe('candidate active')
    const items = [
      queueItem({ id: 'q:1', reason: 'candidate active; match conflict' }),
      queueItem({ id: 'q:2', reason: 'candidate active' }),
    ]
    const groups = groupByReason(items)
    expect(groups).toHaveLength(1)
    expect(groups[0].reason).toBe('candidate active')
    expect(groups[0].items).toHaveLength(2)
  })

  it('falls back to "other" for an empty reason', () => {
    expect(primaryReason('')).toBe('other')
  })
})

describe('isOpen (open-queue rendering)', () => {
  it('treats the default/open decision as open and any other as closed', () => {
    expect(isOpen({ decision: 'open' })).toBe(true)
    expect(isOpen({ decision: '' })).toBe(true)
    expect(isOpen({ decision: 'active' })).toBe(false)
  })
})

describe('backbone helpers (BackboneWarnings)', () => {
  const warnings: BackboneWarning[] = [
    {
      entity_id: 'entity:a',
      structural_degree: 4,
      weak_edges: [
        { other: 'entity:b', relation_type: 'RELATED', doc_count: 1 },
        { other: 'entity:c', relation_type: 'RELATED', doc_count: 1 },
      ],
    },
    {
      entity_id: 'entity:d',
      structural_degree: 3,
      weak_edges: [{ other: 'entity:e', relation_type: 'RELATED', doc_count: 2 }],
    },
  ]

  it('sums weak edges across warnings for the headline count', () => {
    expect(totalWeakEdges(warnings)).toBe(3)
    expect(totalWeakEdges([])).toBe(0)
  })

  it('flags backbone risk only for degree >= 3 with weak edges', () => {
    expect(isBackboneRisk(warnings[0])).toBe(true)
    expect(isBackboneRisk(warnings[1])).toBe(true)
    expect(
      isBackboneRisk({
        entity_id: 'x',
        structural_degree: 2,
        weak_edges: [{ other: 'y', relation_type: 'RELATED', doc_count: 1 }],
      }),
    ).toBe(false)
    expect(
      isBackboneRisk({ entity_id: 'x', structural_degree: 5, weak_edges: [] }),
    ).toBe(false)
  })
})
