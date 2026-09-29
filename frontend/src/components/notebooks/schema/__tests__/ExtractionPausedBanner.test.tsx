// @vitest-environment jsdom
/**
 * WB.2: the banner also sits on the Schema tab, where its "Open schema
 * editor" link would point at the page itself, so the tab turns it off.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'

const resume = vi.hoisted(() => vi.fn())
vi.mock('next/link', () => ({
  default: ({ children, href, ...rest }: { children: ReactNode; href: string }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}))
vi.mock('@/lib/hooks/use-notebook-schema', () => ({
  usePausedExtraction: () => ({
    isError: false,
    data: { notebook_id: 'notebook:nb1', paused_count: 1, paused_source_ids: ['source:g1'] },
  }),
  useResumeExtraction: () => ({ mutate: resume, isPending: false }),
}))

import { ExtractionPausedBanner } from '../ExtractionPausedBanner'

afterEach(() => cleanup())

describe('ExtractionPausedBanner', () => {
  it('links to the Schema tab by default', () => {
    render(<ExtractionPausedBanner notebookId="notebook:nb1" />)
    expect(screen.getByTestId('extraction-paused-open-schema').getAttribute('href'))
      .toBe('/notebooks/notebook%3Anb1/schema')
  })

  it('on the Schema tab: Resume only', () => {
    render(<ExtractionPausedBanner notebookId="notebook:nb1" showSchemaLink={false} />)
    expect(screen.getByTestId('extraction-paused-resume')).toBeTruthy()
    expect(screen.queryByTestId('extraction-paused-open-schema')).toBeNull()
  })
})
