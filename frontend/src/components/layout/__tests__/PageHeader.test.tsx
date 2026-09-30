// @vitest-environment jsdom
/**
 * WB.4: the one page header and page body. The header carries the page's only
 * h1, its actions on the right, and rarely used actions behind a "⋯" menu with
 * the destructive ones set apart; the body has one width scale.
 */
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { Trash2 } from 'lucide-react'

import { PageBody } from '../PageBody'
import { PageHeader } from '../PageHeader'

beforeAll(() => {
  // Radix menus read pointer capture, which jsdom lacks.
  Element.prototype.hasPointerCapture = vi.fn(() => false) as unknown as typeof Element.prototype.hasPointerCapture
  Element.prototype.releasePointerCapture = vi.fn() as unknown as typeof Element.prototype.releasePointerCapture
  Element.prototype.scrollIntoView = vi.fn()
})
afterEach(cleanup)

function openMenu(name: string) {
  const trigger = screen.getByRole('button', { name })
  fireEvent.keyDown(trigger, { key: 'Enter' })
  return screen.findByRole('menu')
}

describe('PageHeader', () => {
  it('renders one h1 with the subtitle, the back row, the actions and extra rows', () => {
    render(
      <PageHeader
        title="Sources"
        subtitle="The pipeline board"
        back={<a href="/">Back</a>}
        actions={<button type="button">Add source</button>}
      >
        <p>Showing only: failed</p>
      </PageHeader>,
    )
    expect(screen.getAllByRole('heading', { level: 1 }).map((h) => h.textContent)).toEqual(['Sources'])
    expect(screen.getByText('The pipeline board')).toBeTruthy()
    expect(screen.getByRole('link', { name: 'Back' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Add source' })).toBeTruthy()
    expect(screen.getByText('Showing only: failed')).toBeTruthy()
    expect(screen.queryByTestId('page-overflow-trigger')).toBeNull()
  })

  it('puts overflow actions in a labelled menu, destructive ones last behind a separator', async () => {
    const archive = vi.fn()
    const remove = vi.fn()
    render(
      <PageHeader
        title="Alpha"
        overflowLabel="More notebook actions"
        overflow={[
          { label: 'Archive', onSelect: archive },
          { label: 'Delete…', icon: Trash2, destructive: true, onSelect: remove, testId: 'del' },
        ]}
      />,
    )
    const menu = await openMenu('More notebook actions')
    const items = [...menu.querySelectorAll('[role="menuitem"], [role="separator"]')]
    expect(items.map((el) => el.getAttribute('role'))).toEqual(['menuitem', 'separator', 'menuitem'])
    const del = screen.getByTestId('del')
    expect(del.getAttribute('data-variant')).toBe('destructive')
    fireEvent.click(del)
    expect(remove).toHaveBeenCalledTimes(1)
    expect(archive).not.toHaveBeenCalled()
  })
})

describe('PageBody', () => {
  it('scrolls, with the wide width by default and the form width on request', () => {
    const { container, rerender } = render(<PageBody>x</PageBody>)
    const outer = () => container.firstElementChild as HTMLElement
    expect(outer().getAttribute('data-page-body')).toBe('wide')
    expect(outer().className).toContain('overflow-y-auto')
    expect((outer().firstElementChild as HTMLElement).className).toContain('max-w-7xl')

    rerender(<PageBody width="form">x</PageBody>)
    expect(outer().getAttribute('data-page-body')).toBe('form')
    expect((outer().firstElementChild as HTMLElement).className).toContain('max-w-3xl')
  })

  it('fills the height and leaves scrolling to the page when asked', () => {
    const { container } = render(<PageBody fill>x</PageBody>)
    const outer = container.firstElementChild as HTMLElement
    expect(outer.className).not.toContain('overflow-y-auto')
    expect((outer.firstElementChild as HTMLElement).className).toContain('flex-1')
  })
})
