// @vitest-environment jsdom
/**
 * WB.5: the sidebar is icons below xl (a badge becomes a dot), a drawer below md
 * opened from a top bar. The drawer traps focus, closes on navigation, and
 * gives focus back to its button.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import type { ReactNode } from 'react'

import fixture from '../../../../e2e/track-wb/attention.fixture.json'
import { installViewport, removeViewport, setViewport } from '@/lib/hooks/__tests__/viewport'
import { useSidebarStore } from '@/lib/stores/sidebar-store'

const nav = vi.hoisted(() => ({ pathname: '/sources' }))
vi.mock('@/lib/hooks/use-attention', () => ({ useAttention: () => ({ data: fixture }) }))
vi.mock('next/navigation', () => ({
  usePathname: () => nav.pathname,
  useRouter: () => ({ push: vi.fn() }),
}))
vi.mock('next/image', () => ({ default: () => null }))
vi.mock('next/link', () => ({
  default: ({ children, href, ...rest }: { children: ReactNode; href: string }) => (
    <a
      href={href}
      {...rest}
      onClick={(e) => {
        e.preventDefault()
        ;(rest as { onClick?: (e: unknown) => void }).onClick?.(e)
      }}
    >
      {children}
    </a>
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

import { AppShell } from '../AppShell'

beforeEach(() => {
  nav.pathname = '/sources'
  useSidebarStore.setState({ isCollapsed: false })
})
afterEach(() => {
  cleanup()
  removeViewport()
})

const sidebar = () => document.querySelector('.app-sidebar') as HTMLElement | null
const shell = () => <AppShell><h1>Page</h1><button>In the page</button></AppShell>

describe('sidebar by width', () => {
  it('from xl it follows the owner: expanded, with the collapse toggle', () => {
    installViewport(1440)
    render(shell())
    expect(sidebar()!.className).toContain('w-64')
    expect(screen.getByRole('link', { name: /^Sources/ }).textContent).toContain('Sources')
    fireEvent.click(screen.getByRole('button', { name: 'Collapse sidebar' }))
    expect(sidebar()!.className).toContain('w-16')
  })

  it('below xl it is icons whatever was stored, with no toggle; badges stay', () => {
    installViewport(1024)
    render(shell())
    expect(sidebar()!.className).toContain('w-16')
    expect(screen.queryByRole('button', { name: /(Collapse|Expand) sidebar/ })).toBeNull()
    // Icon mode: no labels, but each entry that needs you keeps a dot on its
    // icon, with the count for screen readers (and in the tooltip).
    const triage = screen.getByTestId('nav-badge-Triage')
    expect(triage.getAttribute('data-badge')).toBe('dot')
    expect(triage.textContent).toBe('1238 need you')
    expect(screen.getByTestId('nav-badge-Sources').textContent).toBe('3 need you')
    expect(screen.getByTestId('nav-badge-The Tray').textContent)
      .toBe('1257+ need you, some groups did not load')
    expect(screen.queryByTestId('nav-badge-Graph')).toBeNull()
    // Wider again: the stored choice (expanded) is back.
    act(() => setViewport(1280))
    expect(sidebar()!.className).toContain('w-64')
    expect(screen.getByTestId('nav-badge-Triage').textContent).toBe('1238 need you')
  })

  it('from md up there is no top bar', () => {
    installViewport(768)
    render(shell())
    expect(screen.queryByTestId('mobile-top-bar')).toBeNull()
    expect(sidebar()).not.toBeNull()
  })
})

describe('phone drawer', () => {
  it('below md: a top bar with the menu button and the page title, no rail', () => {
    installViewport(390)
    render(shell())
    expect(sidebar()).toBeNull()
    expect(screen.getByTestId('mobile-top-bar-title').textContent).toBe('Sources')
    expect(screen.getByRole('button', { name: 'Open navigation' }).getAttribute('aria-expanded'))
      .toBe('false')
  })

  it('opens expanded, traps focus, and closes on a link with focus back on its button', async () => {
    installViewport(390)
    render(shell())
    const trigger = screen.getByRole('button', { name: 'Open navigation' })
    trigger.focus()
    fireEvent.click(trigger)
    const drawer = await screen.findByRole('dialog')
    expect(sidebar()!.className).toContain('w-full')
    // Labels and badges, as the full sidebar has them.
    expect(within(drawer).getByTestId('nav-badge-Triage').textContent).toBe('1238 need you')
    // Modal: the page behind is hidden from assistive tech, and focus is inside.
    expect(drawer.contains(document.activeElement)).toBe(true)
    expect(screen.getByText('In the page').closest('[aria-hidden="true"]')).not.toBeNull()

    fireEvent.click(within(drawer).getByRole('link', { name: /^Sources/ }))
    expect(screen.queryByRole('dialog')).toBeNull()
    await vi.waitFor(() => expect(document.activeElement).toBe(trigger))
  })

  it('closes when the path changes', async () => {
    installViewport(390)
    const view = render(shell())
    fireEvent.click(screen.getByRole('button', { name: 'Open navigation' }))
    await screen.findByRole('dialog')
    nav.pathname = '/knowledge-graph/triage'
    view.rerender(shell())
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(screen.getByTestId('mobile-top-bar-title').textContent).toBe('Triage')
  })
})

describe('drawer across md (WB.5 review m5)', () => {
  it('open, widened past md and narrowed again: it stays closed', async () => {
    installViewport(390)
    render(shell())
    fireEvent.click(screen.getByRole('button', { name: 'Open navigation' }))
    await screen.findByRole('dialog')
    act(() => setViewport(800))
    expect(screen.queryByRole('dialog')).toBeNull()
    act(() => setViewport(390))
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(screen.getByRole('button', { name: 'Open navigation' }).getAttribute('aria-expanded')).toBe('false')
  })
})
