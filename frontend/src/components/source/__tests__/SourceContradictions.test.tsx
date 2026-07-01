// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'

import { SourceContradictions } from '../SourceContradictions'
import type { ContradictionVerdict } from '../SourceContradictions'

afterEach(cleanup)

describe('SourceContradictions — deferred, data-absent-safe stub (UX.6 AC3)', () => {
  it('renders nothing when no verdicts fetcher is wired (today)', () => {
    const { container } = render(<SourceContradictions sourceId="source:1" />)
    expect(container.innerHTML).toBe('')
    expect(screen.queryByTestId('source-contradictions')).toBeNull()
  })

  it('shows the quiet empty state when the fetcher resolves no verdicts', async () => {
    const fetchVerdicts = vi.fn(async () => [] as ContradictionVerdict[])
    render(<SourceContradictions sourceId="source:1" fetchVerdicts={fetchVerdicts} />)
    expect(await screen.findByText('No contradictions detected')).toBeTruthy()
  })

  it('renders verdicts when the fetcher returns them', async () => {
    const fetchVerdicts = vi.fn(async () => [
      { id: 'v1', claim: 'The sky is green', verdict: 'contradicted', rationale: 'It is blue' },
    ])
    render(<SourceContradictions sourceId="source:1" fetchVerdicts={fetchVerdicts} />)
    expect(await screen.findByText('The sky is green')).toBeTruthy()
    expect(screen.getByText('contradicted')).toBeTruthy()
  })

  describe('absent-data safety', () => {
    let rejectionHandler: (e: PromiseRejectionEvent) => void
    const rejections: unknown[] = []

    beforeEach(() => {
      rejections.length = 0
      rejectionHandler = (e: PromiseRejectionEvent) => rejections.push(e.reason)
      window.addEventListener('unhandledrejection', rejectionHandler)
    })
    afterEach(() => {
      window.removeEventListener('unhandledrejection', rejectionHandler)
    })

    it('does not throw or leak a rejection when the endpoint is absent (rejects)', async () => {
      const fetchVerdicts = vi.fn(async () => {
        throw new Error('404: no verdicts endpoint')
      })
      expect(() =>
        render(<SourceContradictions sourceId="source:1" fetchVerdicts={fetchVerdicts} />)
      ).not.toThrow()
      // Degrades to the quiet empty state.
      expect(await screen.findByText('No contradictions detected')).toBeTruthy()
      // Give the microtask queue a beat, then assert nothing leaked.
      await waitFor(() => expect(fetchVerdicts).toHaveBeenCalled())
      expect(rejections).toEqual([])
    })
  })
})
