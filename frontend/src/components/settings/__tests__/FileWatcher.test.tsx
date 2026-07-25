// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

import { FileWatcher } from '../FileWatcher'
import type { WatcherStatus } from '@/lib/api/watcher'

let state: { data?: WatcherStatus; isLoading: boolean; isError: boolean }

vi.mock('@/lib/hooks/use-watcher', () => ({
  useWatcherStatus: () => state,
}))

const STATUS: WatcherStatus = {
  enabled: true,
  paths: ['/data/inbox'],
  default_notebook_id: 'notebook:d',
  debounce_seconds: 3,
  recent: [
    { name: 'done.pdf', status: 'processed', when: '2026-07-25T10:00:00Z' },
    { name: 'bad.mp3', status: 'errored', when: '2026-07-25T09:00:00Z' },
  ],
}

afterEach(cleanup)
beforeEach(() => {
  state = { data: STATUS, isLoading: false, isError: false }
})

describe('FileWatcher', () => {
  it('renders the loading state', () => {
    state = { data: undefined, isLoading: true, isError: false }
    render(<FileWatcher />)
    expect(screen.getByRole('status')).toBeTruthy()
  })

  it('renders the error state', () => {
    state = { data: undefined, isLoading: false, isError: true }
    render(<FileWatcher />)
    expect(screen.getByTestId('watcher-error')).toBeTruthy()
  })

  it('shows enabled + paths + default notebook + activity', () => {
    render(<FileWatcher />)
    expect(screen.getByTestId('watcher-enabled')).toBeTruthy()
    expect(screen.getByTestId('watcher-paths').textContent).toContain('/data/inbox')
    expect(screen.getByText('notebook:d')).toBeTruthy()
    const activity = screen.getByTestId('watcher-activity')
    expect(activity.textContent).toContain('done.pdf')
    expect(activity.textContent).toContain('processed')
    expect(activity.textContent).toContain('bad.mp3')
    expect(activity.textContent).toContain('errored')
  })

  it('shows the disabled state with an enable hint and empty activity', () => {
    state = {
      data: { ...STATUS, enabled: false, recent: [] },
      isLoading: false,
      isError: false,
    }
    render(<FileWatcher />)
    expect(screen.getByTestId('watcher-disabled')).toBeTruthy()
    expect(screen.getByText(/FILE_WATCHER_ENABLED/)).toBeTruthy()
    expect(screen.getByTestId('watcher-activity-empty')).toBeTruthy()
  })
})
