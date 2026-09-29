// @vitest-environment jsdom
/**
 * WB.1/WB.2: the sidebar's The Tray entry carries the report's total, and each
 * entry the sum of `counted` of the groups whose `nav` it is.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'

import fixture from '../../../../e2e/track-wb/attention.fixture.json'
import type { AttentionReport } from '@/lib/api/attention'

const report = vi.hoisted(() => ({ data: undefined as unknown }))
vi.mock('@/lib/hooks/use-attention', () => ({ useAttention: () => ({ data: report.data }) }))
vi.mock('next/navigation', () => ({
  usePathname: () => '/',
  useRouter: () => ({ push: vi.fn() }),
}))
vi.mock('next/image', () => ({ default: () => null }))
vi.mock('next/link', () => ({
  default: ({ children, href }: { children: ReactNode; href: string }) => (
    <a href={href}>{children}</a>
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
import { NAV_SECTIONS } from '../nav-items'

const REPORT = fixture as AttentionReport

afterEach(() => cleanup())

const badge = (name: string) => screen.queryByTestId(`nav-badge-${name}`)?.textContent ?? null

describe('sidebar badges', () => {
  it('The Tray shows the total; each entry shows its groups', () => {
    report.data = REPORT
    render(<AppSidebar />)
    expect(screen.getByRole('link', { name: /The Tray/ }).getAttribute('href')).toBe('/')
    // The fixture's merge candidates timed out: the total and the Duplicates
    // badge miss them, and say so.
    expect(badge('The Tray')).toBe('1257+ need you, some groups did not load')
    // gated 1 + stopped 2 (failed is 0; "new sources" asks nothing)
    expect(badge('Sources')).toBe('3 need you')
    expect(badge('Schema proposals')).toBe('3 need you')
    expect(badge('Triage')).toBe('1238 need you')
    expect(badge('Duplicates')).toBe('0+ need you, some groups did not load')
    expect(badge('Retype')).toBe('12 need you')
    // The paused extraction is its gated source's (counted 0).
    expect(badge('Notebooks')).toBeNull()
    expect(badge('Graph')).toBeNull()
    expect(badge('Ontologies')).toBeNull()
  })

  it('every group that names an entry finds it, and adds to no other', () => {
    // A producer names its consumer: each `nav` the report carries is an
    // entry's href, so no badge is lost to a stale name.
    const hrefs = NAV_SECTIONS.flatMap((s) => s.items.map((i) => i.href))
    const navs = [...REPORT.groups, ...REPORT.arrived].map((g) => g.nav).filter(Boolean)
    for (const nav of navs) expect(hrefs).toContain(nav)
  })

  it('no partial marker when every group loaded', () => {
    report.data = {
      ...REPORT,
      partial: false,
      groups: REPORT.groups.map((g) => ({ ...g, error: null })),
    }
    render(<AppSidebar />)
    expect(badge('The Tray')).toBe('1257 need you')
    expect(badge('Duplicates')).toBeNull()
  })

  it('no badges before the report arrives', () => {
    report.data = undefined
    render(<AppSidebar />)
    expect(badge('The Tray')).toBeNull()
    expect(badge('Sources')).toBeNull()
  })
})
