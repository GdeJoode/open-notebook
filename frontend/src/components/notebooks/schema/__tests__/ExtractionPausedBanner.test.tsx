// @vitest-environment jsdom
/**
 * WB.2: the banner also sits on the Schema tab, where its "Open schema
 * editor" link would point at the page itself, so the tab turns it off.
 *
 * WB.3 (review minor m1): Resume is the secondary action. While nothing is
 * accepted (`review_gate_active`), or the schema is not read, it asks first and
 * says plainly that it turns the notebook's review gate off.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'

const resume = vi.hoisted(() => vi.fn())
const schema = vi.hoisted(() => ({ data: undefined as { review_gate_active: boolean } | undefined }))
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
  useNotebookSchema: () => ({ data: schema.data }),
}))

import { ExtractionPausedBanner } from '../ExtractionPausedBanner'

beforeEach(() => {
  resume.mockReset()
  schema.data = { review_gate_active: true }
})
afterEach(() => cleanup())

const GATE_OFF = /turns off the review gate for this notebook/

describe('ExtractionPausedBanner', () => {
  it('links to the Schema tab by default, as the primary action', () => {
    render(<ExtractionPausedBanner notebookId="notebook:nb1" />)
    const open = screen.getByTestId('extraction-paused-open-schema')
    const resumeBtn = screen.getByTestId('extraction-paused-resume')
    expect(open.getAttribute('href')).toBe('/notebooks/notebook%3Anb1/schema')
    // The editor link comes first; Resume follows it.
    expect(open.compareDocumentPosition(resumeBtn) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('on the Schema tab: Resume only', () => {
    render(<ExtractionPausedBanner notebookId="notebook:nb1" showSchemaLink={false} />)
    expect(screen.getByTestId('extraction-paused-resume')).toBeTruthy()
    expect(screen.queryByTestId('extraction-paused-open-schema')).toBeNull()
  })

  it('with nothing accepted, Resume asks first and says it turns the gate off', () => {
    render(<ExtractionPausedBanner notebookId="notebook:nb1" showSchemaLink={false} />)
    fireEvent.click(screen.getByTestId('extraction-paused-resume'))
    expect(resume).not.toHaveBeenCalled()
    expect(screen.getByText(GATE_OFF)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Turn off review and resume' }))
    expect(resume).toHaveBeenCalledTimes(1)
  })

  it('cancelling leaves the gate alone', () => {
    render(<ExtractionPausedBanner notebookId="notebook:nb1" showSchemaLink={false} />)
    fireEvent.click(screen.getByTestId('extraction-paused-resume'))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(resume).not.toHaveBeenCalled()
  })

  it('with the schema not read, it asks too', () => {
    schema.data = undefined
    render(<ExtractionPausedBanner notebookId="notebook:nb1" showSchemaLink={false} />)
    fireEvent.click(screen.getByTestId('extraction-paused-resume'))
    expect(resume).not.toHaveBeenCalled()
    expect(screen.getByText(GATE_OFF)).toBeTruthy()
  })

  it('once something is accepted, Resume resumes at once', () => {
    schema.data = { review_gate_active: false }
    render(<ExtractionPausedBanner notebookId="notebook:nb1" showSchemaLink={false} />)
    fireEvent.click(screen.getByTestId('extraction-paused-resume'))
    expect(resume).toHaveBeenCalledTimes(1)
    expect(screen.queryByText(GATE_OFF)).toBeNull()
  })
})
