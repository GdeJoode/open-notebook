// @vitest-environment jsdom
/**
 * PC.5b: the auto band has a reader.
 *
 * An OKF import no longer merges the auto band unattended; it proposes, and the
 * dialog says the proposals wait on this page. This page rendered only the
 * review band, so the auto band was proposed to nobody. These tests mount the
 * page with both bands and check a curator sees, and can act on, each.
 *
 * WB.2: the page is the one Resolution entry. It holds the resolution log (the
 * Knowledge Graph tab is gone), links on to The Tray, and says so plainly once
 * nothing is left to merge.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import type { ReactNode } from 'react'

import type { MergeCandidate } from '@/lib/api/entity-resolution'

const apply = vi.fn()
const queue = vi.hoisted(() => ({ empty: false, planFails: false, candidatesFail: false }))

vi.mock('next/link', () => ({
  default: ({ children, href, ...rest }: { children: ReactNode; href: string }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}))
vi.mock('@/components/resolution/ResolutionLog', () => ({
  ResolutionLog: () => <div data-testid="resolution-log" />,
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
  useMergePlan: () => (queue.planFails
    ? { data: undefined, isLoading: false, isError: true, refetch: vi.fn() }
    : { data: { clusters: [] }, isLoading: false, isError: false, refetch: vi.fn() }),
  useMergeCandidates: () => (queue.candidatesFail
    ? { data: undefined, isLoading: false, isError: true, refetch: vi.fn() }
    : {
      data: queue.empty
        ? { auto_merge: [], review: [], auto_merge_count: 0, review_count: 0 }
        : { auto_merge: [CAND], review: [REVIEW], auto_merge_count: 1, review_count: 1 },
      isLoading: false,
      isError: false,
      refetch: vi.fn(),
    }),
  useApplyMerge: () => ({ mutate: apply, isPending: false }),
  useAddOverlay: () => ({ mutate: vi.fn(), isPending: false }),
}))

import EntityResolutionPage from '../page'

afterEach(() => {
  cleanup()
  apply.mockReset()
  queue.empty = false
  queue.planFails = false
  queue.candidatesFail = false
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

describe('Entity Resolution page — the one Resolution entry', () => {
  it('holds the resolution log and links on to The Tray', () => {
    render(<EntityResolutionPage />)
    expect(screen.getByRole('heading', { name: 'Resolution log' })).toBeTruthy()
    expect(screen.getByTestId('resolution-log')).toBeTruthy()
    expect(screen.getByTestId('resolution-next').getAttribute('href')).toBe('/')
    // Candidates are waiting: the empty state offers no way out yet.
    expect(screen.queryByTestId('resolution-empty-next')).toBeNull()
  })

  it('with nothing left to merge, the next step is The Tray', () => {
    queue.empty = true
    render(<EntityResolutionPage />)
    expect(screen.getByText('No duplicates pending')).toBeTruthy()
    expect(screen.getByTestId('resolution-empty-next').getAttribute('href')).toBe('/')
  })
})

describe('Entity Resolution page — a read that failed (WB.3, m4)', () => {
  it('when only the candidates failed: the error, not "no duplicates" and The Tray', () => {
    queue.candidatesFail = true
    render(<EntityResolutionPage />)
    expect(screen.getByTestId('resolution-candidates-error')).toBeTruthy()
    expect(screen.queryByText('No duplicates pending')).toBeNull()
    expect(screen.queryByTestId('resolution-empty-next')).toBeNull()
    expect(screen.getByTestId('resolution-no-merges')).toBeTruthy()
  })

  it('when only the plan failed: its error, and still no way out', () => {
    queue.planFails = true
    queue.empty = true
    render(<EntityResolutionPage />)
    expect(screen.getByTestId('resolution-plan-error')).toBeTruthy()
    expect(screen.queryByText('No duplicates pending')).toBeNull()
    expect(screen.queryByTestId('resolution-empty-next')).toBeNull()
  })

  it('when both failed: one error for the page', () => {
    queue.planFails = true
    queue.candidatesFail = true
    render(<EntityResolutionPage />)
    expect(screen.getByTestId('resolution-error')).toBeTruthy()
    expect(screen.queryByTestId('resolution-candidates-error')).toBeNull()
  })
})
