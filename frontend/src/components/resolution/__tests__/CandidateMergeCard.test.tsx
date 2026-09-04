// @vitest-environment jsdom
/**
 * The candidate card RENDERS the deciding information (PC.2).
 *
 * These tests exist because of a specific failure: `candidateTypeLabel` and
 * `isCrossTypeCandidate` were written, unit-tested, imported by this card — and
 * never called. The suite stayed green, `tsc` stayed clean, and a cross-type
 * candidate rendered as `Regio Deal ↔ Regio Deal · programme`, one click from a
 * destructive merge with the only distinguishing fact hidden.
 *
 * Testing the pure function was not enough; nothing asserted the caller used it.
 * So these tests mount the real component. The sibling `.test.ts` file covers the
 * apply payload in the fast node environment; this file covers what a curator
 * actually sees.
 */
import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'

import { CandidateMergeCard } from '../CandidateMergeCard'
import type { MergeCandidate } from '@/lib/api/entity-resolution'

afterEach(cleanup)

const SAME_TYPE: MergeCandidate = {
  id_a: 'entity:a',
  id_b: 'entity:b',
  name_a: 'VWS',
  name_b: 'Volksgezondheid',
  entity_type: 'organization',
  score: 0.82,
  band: 'review',
  method: 'embedding',
  winner_id: 'entity:a',
  loser_id: 'entity:b',
}

const CROSS_TYPE: MergeCandidate = {
  ...SAME_TYPE,
  name_a: 'Regio Deal',
  name_b: 'Regio Deal',
  entity_type: 'programme',
  entity_type_b: 'topic',
  method: 'fold_equal_cross_type',
  score: 1.0,
}

describe('CandidateMergeCard — the curator can see what differs', () => {
  it('shows both types when a cross-type pair has identical names', () => {
    render(<CandidateMergeCard candidate={CROSS_TYPE} onApprove={() => {}} />)
    const card = screen.getByTestId('merge-candidate-entity:a:entity:b')
    const text = card.textContent ?? ''
    // The two types are the entire content of this decision: the names are
    // byte-identical, so without them the card asks a question with no data.
    expect(text).toContain('programme')
    expect(text).toContain('topic')
  })

  it('shows one type for a same-type pair', () => {
    render(<CandidateMergeCard candidate={SAME_TYPE} onApprove={() => {}} />)
    const text = screen.getByTestId('merge-candidate-entity:a:entity:b').textContent ?? ''
    expect(text).toContain('organization')
    // No spurious second type, and no bare "↔" between two identical labels.
    expect(text).not.toContain('organization ↔ organization')
  })

  it('does not treat an equal entity_type_b as cross-type', () => {
    render(
      <CandidateMergeCard
        candidate={{ ...CROSS_TYPE, entity_type_b: 'programme' }}
        onApprove={() => {}}
      />,
    )
    const text = screen.getByTestId('merge-candidate-entity:a:entity:b').textContent ?? ''
    expect(text).not.toContain('programme ↔ programme')
  })
})
