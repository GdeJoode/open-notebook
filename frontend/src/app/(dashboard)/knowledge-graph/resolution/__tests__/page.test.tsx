// @vitest-environment jsdom
/**
 * PC.5b: the auto band has a reader.
 *
 * An OKF import no longer merges the auto band unattended; it proposes, and the
 * dialog says the proposals wait on this page. This page rendered only the
 * review band, so the auto band was proposed to nobody. These tests mount the
 * page with both bands and check a curator sees, and can act on, each.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import type { ReactNode } from 'react'

import type { MergeCandidate } from '@/lib/api/entity-resolution'

const apply = vi.fn()

vi.mock('next/link', () => ({
  default: ({ children, href }: { children: ReactNode; href: string }) => <a href={href}>{children}</a>,
}))

const CAND: MergeCandidate = {
  id_a: 'entity:a',
  id_b: 'entity:b',
  name_a: 'Economische Zaken',
  name_b: 'economische zaken',
  entity_type: 'government_organization',
  score: 0.97,
  band: 'auto_merge',
  method: 'fold_equal',
  winner_id: 'entity:a',
  loser_id: 'entity:b',
}
const REVIEW: MergeCandidate = {
  ...CAND,
  id_a: 'entity:c',
  id_b: 'entity:d',
  winner_id: 'entity:c',
  loser_id: 'entity:d',
  band: 'review',
  score: 0.82,
  method: 'embedding',
}

vi.mock('@/lib/hooks/use-entity-resolution', () => ({
  useMergePlan: () => ({ data: { clusters: [] }, isLoading: false, isError: false }),
  useMergeCandidates: () => ({
    data: { auto_merge: [CAND], review: [REVIEW], auto_merge_count: 1, review_count: 1 },
    isLoading: false,
    isError: false,
  }),
  useApplyMerge: () => ({ mutate: apply, isPending: false }),
  useAddOverlay: () => ({ mutate: vi.fn(), isPending: false }),
}))

import EntityResolutionPage from '../page'

afterEach(() => {
  cleanup()
  apply.mockReset()
})

describe('Entity Resolution page — both candidate bands', () => {
  it('renders the auto band next to the review band', () => {
    render(<EntityResolutionPage />)
    const auto = screen.getByTestId('resolution-auto-band')
    expect(within(auto).getByTestId('merge-candidate-entity:a:entity:b')).toBeTruthy()
    expect(within(auto).queryByTestId('merge-candidate-entity:c:entity:d')).toBeNull()
    expect(screen.getByTestId('merge-candidate-entity:c:entity:d')).toBeTruthy()
  })

  it('approving an auto-band pair applies it as a curator decision', () => {
    render(<EntityResolutionPage />)
    const card = screen.getByTestId('merge-candidate-entity:a:entity:b')
    fireEvent.click(within(card).getByRole('button', { name: 'Merge' }))
    fireEvent.click(screen.getByTestId('merge-confirm-apply'))
    expect(apply).toHaveBeenCalledTimes(1)
    const [[clusters]] = apply.mock.calls
    expect(clusters[0]).toMatchObject({ origin: 'curator_approved', winner_id: 'entity:a', loser_ids: ['entity:b'] })
  })
})
