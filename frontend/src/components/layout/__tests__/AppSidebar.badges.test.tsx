// @vitest-environment jsdom
/**
 * WB.1: the sidebar's Werkbak entry carries the report's total, and each entry
 * the sum of `counted` of the groups whose `nav` it is.
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

const REPORT = fixture as AttentionReport

afterEach(() => cleanup())

const badge = (name: string) => screen.queryByTestId(`nav-badge-${name}`)?.textContent ?? null

describe('sidebar badges', () => {
  it('Werkbak shows the total; entries show their groups', () => {
    report.data = REPORT
    render(<AppSidebar />)
    expect(screen.getByRole('link', { name: /The Tray/ }).getAttribute('href')).toBe('/')
    // The fixture's merge candidates timed out: the total and the Knowledge
    // Graph badge miss them, and say so.
    expect(badge('The Tray')).toBe('1257+ need you, some groups did not load')
    // gated 1 + stopped 2 (failed is 0; "new sources" asks nothing)
    expect(badge('Sources')).toBe('3 need you')
    // proposals 3; the paused extraction is its gated source's (counted 0)
    expect(badge('Notebooks')).toBe('3 need you')
    // triage 1238; merge candidates failed (count 0), so partial
    expect(badge('Knowledge Graph')).toBe('1238+ need you, some groups did not load')
    expect(badge('Ontologies')).toBe('12 need you')
    expect(badge('Podcasts')).toBeNull()
  })

  it('no partial marker when every group loaded', () => {
    report.data = {
      ...REPORT,
      partial: false,
      groups: REPORT.groups.map((g) => ({ ...g, error: null })),
    }
    render(<AppSidebar />)
    expect(badge('The Tray')).toBe('1257 need you')
    expect(badge('Knowledge Graph')).toBe('1238 need you')
  })

  it('no badges before the report arrives', () => {
    report.data = undefined
    render(<AppSidebar />)
    expect(badge('The Tray')).toBeNull()
    expect(badge('Sources')).toBeNull()
  })
})
