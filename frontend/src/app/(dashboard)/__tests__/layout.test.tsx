// @vitest-environment jsdom
/**
 * WB.2: the dashboard layout renders the one shell (sidebar and all), so a
 * navigation swaps the page and keeps the sidebar mounted: its attention
 * query and refresh interval run on, and no page renders a second shell.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { useEffect } from 'react'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import path from 'node:path'

const mounts = vi.hoisted(() => ({ sidebar: 0 }))
vi.mock('@/components/layout/AppSidebar', () => ({
  AppSidebar: () => {
    useEffect(() => {
      mounts.sidebar += 1
    }, [])
    return <nav data-testid="sidebar" />
  },
}))
vi.mock('@/lib/hooks/use-auth', () => ({
  useAuth: () => ({ isAuthenticated: true, isLoading: false }),
}))
vi.mock('@/lib/hooks/use-models', () => ({
  useModelStatus: () => ({ data: { valid: true }, isLoading: false }),
}))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => '/',
}))
vi.mock('@/components/providers/ModalProvider', () => ({ ModalProvider: () => null }))

import DashboardLayout from '../layout'

afterEach(() => cleanup())

describe('dashboard layout', () => {
  it('keeps one sidebar mounted while the page changes', () => {
    mounts.sidebar = 0
    const view = render(<DashboardLayout><p>The Tray</p></DashboardLayout>)
    expect(screen.getByTestId('sidebar')).toBeTruthy()
    view.rerender(<DashboardLayout><p>Triage</p></DashboardLayout>)
    view.rerender(<DashboardLayout><p>Source</p></DashboardLayout>)
    expect(screen.getByText('Source')).toBeTruthy()
    expect(screen.getAllByTestId('sidebar')).toHaveLength(1)
    expect(mounts.sidebar).toBe(1)
  })

  it('no page renders a shell of its own', () => {
    const root = path.resolve(__dirname, '..')
    const pages: string[] = []
    const walk = (dir: string) => {
      for (const name of readdirSync(dir)) {
        const full = path.join(dir, name)
        if (statSync(full).isDirectory()) walk(full)
        else if (name === 'page.tsx') pages.push(full)
      }
    }
    walk(root)
    expect(pages.length).toBeGreaterThan(15)
    const withShell = pages.filter((p) => readFileSync(p, 'utf8').includes('AppShell'))
    expect(withShell.map((p) => path.relative(root, p))).toEqual([])
  })
})
