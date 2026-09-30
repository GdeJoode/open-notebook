// @vitest-environment jsdom
/**
 * WB.4: `/advanced` is gone; the embedding-model change dialog sends the owner
 * to `/settings?tab=advanced`, which opens on System Info and Rebuild
 * Embeddings. Without the parameter (or with an unknown one) the page opens on
 * General.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'

const nav = vi.hoisted(() => ({ params: new URLSearchParams() }))
vi.mock('next/navigation', () => ({ useSearchParams: () => nav.params }))
vi.mock('@/lib/hooks/use-settings', () => ({ useSettings: () => ({ refetch: vi.fn() }) }))
const stub = vi.hoisted(() => (id: string) => {
  function Stub() {
    return <div data-testid={id} />
  }
  return Stub
})
vi.mock('../components/SettingsForm', () => ({ SettingsForm: stub('general') }))
vi.mock('@/components/settings/SystemInfo', () => ({ SystemInfo: stub('system-info') }))
vi.mock('@/components/settings/RebuildEmbeddings', () => ({ RebuildEmbeddings: stub('rebuild') }))
vi.mock('@/components/settings/VaultSync', () => ({ VaultSync: stub('vault') }))
vi.mock('@/components/settings/ZoteroSettings', () => ({ ZoteroSettings: stub('zotero') }))
vi.mock('@/components/settings/ApiKeys', () => ({ ApiKeys: stub('api-keys') }))
vi.mock('@/components/settings/FileWatcher', () => ({ FileWatcher: stub('watcher') }))

import SettingsPage from '../page'

afterEach(cleanup)

describe('Settings', () => {
  it('opens on the Advanced tab from ?tab=advanced', () => {
    nav.params = new URLSearchParams('tab=advanced')
    render(<SettingsPage />)
    expect(screen.getByRole('tab', { name: 'Advanced' }).getAttribute('aria-selected')).toBe('true')
    expect(screen.getByTestId('system-info')).toBeTruthy()
    expect(screen.getByTestId('rebuild')).toBeTruthy()
  })

  it('opens on General otherwise', () => {
    for (const query of ['', 'tab=nonsense']) {
      nav.params = new URLSearchParams(query)
      render(<SettingsPage />)
      expect(screen.getByRole('tab', { name: 'General' }).getAttribute('aria-selected')).toBe('true')
      expect(screen.queryByTestId('rebuild')).toBeNull()
      cleanup()
    }
  })
})
