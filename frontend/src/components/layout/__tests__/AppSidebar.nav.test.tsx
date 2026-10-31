// @vitest-environment jsdom
/**
 * WB.2: the sidebar in pipeline order, one highlighted entry per page, and the
 * out-of-scope features under a "More" group that stays folded until opened
 * (remembered per browser; a browser without storage still toggles).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import type { ReactNode } from 'react'

const nav = vi.hoisted(() => ({ pathname: '/' }))
vi.mock('@/lib/hooks/use-attention', () => ({ useAttention: () => ({ data: undefined }) }))
vi.mock('next/navigation', () => ({
  usePathname: () => nav.pathname,
  useRouter: () => ({ push: vi.fn() }),
}))
vi.mock('next/image', () => ({ default: () => null }))
vi.mock('next/link', () => ({
  default: ({ children, href, ...rest }: { children: ReactNode; href: string }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}))
vi.mock('@/lib/hooks/use-auth', () => ({ useAuth: () => ({ logout: vi.fn() }) }))
vi.mock('@/components/common/ThemeToggle', () => ({ ThemeToggle: () => null }))
vi.mock('@/components/notebooks/CreateNotebookDialog', () => ({
  CreateNotebookDialog: () => null,
}))
vi.mock('@/components/podcasts/GeneratePodcastDialog', () => ({
  GeneratePodcastDialog: () => null,
}))

import { AppSidebar } from '../AppSidebar'
import { NAV_SECTIONS, activeNavHref } from '../nav-items'

const KEY = 'sidebar.moreOpen'

beforeEach(() => {
  nav.pathname = '/'
  window.localStorage.clear()
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

const links = () =>
  within(screen.getByRole('navigation'))
    .queryAllByRole('link')
    .map((a) => [a.textContent, a.getAttribute('href')])

describe('sidebar navigation', () => {
  it('lists the entries in pipeline order, with More folded', () => {
    render(<AppSidebar />)
    expect(links()).toEqual([
      ['The Tray', '/'],
      ['Sources', '/sources'],
      ['Schema proposals', '/#schema_proposals'],
      ['Triage', '/knowledge-graph/triage'],
      ['Duplicates', '/knowledge-graph/resolution'],
      ['Retype', '/ontologies?name=schema_core&tab=edit&retype=1'],
      ['Graph', '/knowledge-graph'],
      ['Ontologies', '/ontologies'],
      ['Search', '/search'],
      ['Notebooks', '/notebooks'],
      ['Models', '/models'],
      ['Settings', '/settings'],
    ])
    const headings = screen.getAllByRole('heading').map((h) => h.textContent)
    expect(headings).toEqual(['Review', 'Knowledge', 'Collections', 'Settings'])
    expect(screen.getByTestId('nav-more-toggle').getAttribute('aria-expanded')).toBe('false')
  })

  it('More opens, is remembered, and closes again', () => {
    const first = render(<AppSidebar />)
    fireEvent.click(screen.getByTestId('nav-more-toggle'))
    expect(screen.getByTestId('nav-more-toggle').getAttribute('aria-expanded')).toBe('true')
    expect(links().slice(-3)).toEqual([
      ['Podcasts', '/podcasts'],
      ['Transformations', '/transformations'],
      ['Summaries', '/summaries'],
    ])
    expect(window.localStorage.getItem(KEY)).toBe('true')

    first.unmount()
    render(<AppSidebar />)
    expect(screen.getByRole('link', { name: 'Podcasts' })).toBeTruthy()
    fireEvent.click(screen.getByTestId('nav-more-toggle'))
    expect(screen.queryByRole('link', { name: 'Podcasts' })).toBeNull()
    expect(window.localStorage.getItem(KEY)).toBe('false')
  })

  it('More still toggles when storage throws', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('denied')
    })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('denied')
    })
    render(<AppSidebar />)
    fireEvent.click(screen.getByTestId('nav-more-toggle'))
    expect(screen.getByRole('link', { name: 'Summaries' })).toBeTruthy()
  })

  it('highlights exactly one entry: the most specific one', () => {
    const cases: [string, string | undefined][] = [
      ['/', '/'],
      ['/sources', '/sources'],
      ['/sources/source%3Aa', '/sources'],
      ['/sources/new', '/sources'],
      ['/knowledge-graph', '/knowledge-graph'],
      ['/knowledge-graph/triage', '/knowledge-graph/triage'],
      ['/knowledge-graph/resolution', '/knowledge-graph/resolution'],
      ['/ontologies', '/ontologies'],
      ['/notebooks/notebook%3Aa/schema', '/notebooks'],
      ['/settings', '/settings'],
      ['/podcasts', '/podcasts'],
      ['/sourcesx', undefined],
      ['/login', undefined],
    ]
    for (const [pathname, href] of cases) expect([pathname, activeNavHref(pathname)]).toEqual([pathname, href])

    nav.pathname = '/knowledge-graph/triage'
    render(<AppSidebar />)
    const active = screen.getAllByRole('link').filter((a) => a.getAttribute('aria-current') === 'page')
    expect(active.map((a) => a.textContent)).toEqual(['Triage'])
  })

  it('every entry href is unique', () => {
    const hrefs = NAV_SECTIONS.flatMap((s) => s.items.map((i) => i.href))
    expect(new Set(hrefs).size).toBe(hrefs.length)
  })
})
