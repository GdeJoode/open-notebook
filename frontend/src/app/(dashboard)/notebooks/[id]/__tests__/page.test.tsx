// @vitest-environment jsdom
/**
 * WB.5: the notebook page puts Sources first. From lg up Sources sits beside
 * Notes and Chat as tabs; below lg the three are tabs, Sources open. A tab
 * switch keeps the others mounted (a draft note or a chat survives it).
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'

import { installViewport, removeViewport } from '@/lib/hooks/__tests__/viewport'

const mounts = vi.hoisted(() => ({ chat: 0 }))
vi.mock('next/navigation', () => ({ useParams: () => ({ id: 'notebook%3Aoe5' }) }))
vi.mock('@/lib/hooks/use-notebooks', () => ({
  useNotebook: () => ({ data: { id: 'notebook:oe5', name: 'Regio Deals' }, isLoading: false }),
}))
vi.mock('@/lib/hooks/use-sources', () => ({
  useSources: () => ({ data: [], isLoading: false, refetch: vi.fn() }),
}))
vi.mock('@/lib/hooks/use-notes', () => ({ useNotes: () => ({ data: [], isLoading: false }) }))
vi.mock('../../components/NotebookHeader', () => ({ NotebookHeader: () => <h1>Regio Deals</h1> }))
vi.mock('../../components/SourcesColumn', () => ({
  SourcesColumn: () => <section data-testid="sources-column">Sources column</section>,
}))
vi.mock('../../components/NotesColumn', () => ({
  NotesColumn: () => <section data-testid="notes-column">Notes column</section>,
}))
vi.mock('../../components/ChatColumn', async () => {
  const { useEffect } = await import('react')
  return {
    ChatColumn: () => {
      useEffect(() => {
        mounts.chat += 1
      }, [])
      return <section data-testid="chat-column">Chat column</section>
    },
  }
})
vi.mock('@/components/notebooks/schema/SchemaSoftNudge', () => ({ SchemaSoftNudge: () => null }))
vi.mock('@/components/notebooks/schema/ExtractionPausedBanner', () => ({
  ExtractionPausedBanner: () => null,
}))
vi.mock('@/components/notebooks/schema/ReextractPromptBanner', () => ({
  ReextractPromptBanner: () => null,
}))

import NotebookPage from '../page'

afterEach(() => {
  cleanup()
  removeViewport()
  mounts.chat = 0
})

const visible = (id: string) => !screen.getByTestId(id).closest('[hidden], .hidden, [data-state="inactive"]')

/** Radix tabs switch on mouse down. */
const pick = (tab: HTMLElement) => act(() => {
  fireEvent.mouseDown(tab, { button: 0, ctrlKey: false })
})

describe('notebook page by width', () => {
  it('from lg: Sources beside a Notes | Chat tab set', () => {
    installViewport(1024)
    render(<NotebookPage />)
    expect(screen.getByTestId('notebook-columns')).toBeTruthy()
    const tabs = within(screen.getByRole('tablist', { name: 'Notes and chat' })).getAllByRole('tab')
    expect(tabs.map((t) => t.textContent)).toEqual(['Notes', 'Chat'])
    expect(visible('sources-column')).toBe(true)
    expect(visible('notes-column')).toBe(true)
    expect(visible('chat-column')).toBe(false)
    pick(tabs[1])
    expect(visible('chat-column')).toBe(true)
    expect(visible('notes-column')).toBe(false)
    expect(visible('sources-column')).toBe(true)
  })

  it('below lg: Sources, Notes, Chat as tabs, Sources open; the others stay mounted', () => {
    installViewport(390)
    render(<NotebookPage />)
    const tabs = within(screen.getByRole('tablist', { name: 'Notebook sections' })).getAllByRole('tab')
    expect(tabs.map((t) => t.textContent)).toEqual(['Sources', 'Notes', 'Chat'])
    expect(tabs[0].getAttribute('aria-selected')).toBe('true')
    expect(visible('sources-column')).toBe(true)
    expect(visible('chat-column')).toBe(false)
    pick(tabs[2])
    expect(visible('chat-column')).toBe(true)
    expect(visible('sources-column')).toBe(false)
    pick(tabs[0])
    expect(visible('sources-column')).toBe(true)
    pick(tabs[2])
    // Chat mounted once, and not again for the switches.
    expect(mounts.chat).toBe(1)
  })
})
