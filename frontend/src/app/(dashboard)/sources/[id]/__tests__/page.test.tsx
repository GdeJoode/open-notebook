// @vitest-environment jsdom
/**
 * WB.5: the source page's chat is a column from xl up; below it the header's
 * "Chat with source" opens it in a sheet, and closing gives focus back.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'

import { installViewport, removeViewport, setViewport } from '@/lib/hooks/__tests__/viewport'

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
  useParams: () => ({ id: 'source%3Awb4' }),
}))
vi.mock('@/lib/hooks/use-navigation', () => ({
  useNavigation: () => ({
    getReturnPath: () => '/sources',
    getReturnLabel: () => 'Back to sources',
    clearReturnTo: vi.fn(),
  }),
}))
vi.mock('@/lib/hooks/useSourceChat', () => ({
  useSourceChat: () => ({
    messages: [], isStreaming: false, contextIndicators: null, sendMessage: vi.fn(),
    currentSession: null, currentSessionId: null, updateSession: vi.fn(), sessions: [],
    createSession: vi.fn(), switchSession: vi.fn(), deleteSession: vi.fn(), loadingSessions: false,
  }),
}))
// The detail's header shows its chat button when asked, as the real one does.
vi.mock('@/components/source/SourceDetailContent', () => ({
  SourceDetailContent: ({ showChatButton, onChatClick }: { showChatButton?: boolean; onChatClick?: () => void }) => (
    <header>
      <h1>Source</h1>
      {showChatButton && onChatClick && <button onClick={onChatClick}>Chat with source</button>}
    </header>
  ),
}))
vi.mock('@/components/source/ChatPanel', () => ({
  ChatPanel: () => <section data-testid="chat-panel">Chat</section>,
}))

import SourceDetailPage from '../page'

afterEach(() => {
  cleanup()
  removeViewport()
})

describe('source page chat by width', () => {
  it('from xl the chat is a column, with no button for it', () => {
    installViewport(1280)
    render(<SourceDetailPage />)
    expect(screen.getByTestId('chat-panel')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Chat with source' })).toBeNull()
  })

  it('below xl the chat opens in a sheet from the header and gives focus back', async () => {
    installViewport(1024)
    render(<SourceDetailPage />)
    expect(screen.queryByTestId('chat-panel')).toBeNull()
    const open = screen.getByRole('button', { name: 'Chat with source' })
    open.focus()
    fireEvent.click(open)
    const sheet = await screen.findByRole('dialog', { name: 'Chat with source' })
    expect(sheet.querySelector('[data-testid="chat-panel"]')).not.toBeNull()
    fireEvent.keyDown(sheet, { key: 'Escape' })
    expect(screen.queryByRole('dialog')).toBeNull()
    await vi.waitFor(() => expect(document.activeElement).toBe(open))
  })
})

describe('source chat across xl (WB.5 review m5)', () => {
  it('an open chat sheet does not come back after a wider window', async () => {
    installViewport(1024)
    render(<SourceDetailPage />)
    fireEvent.click(screen.getByRole('button', { name: 'Chat with source' }))
    await screen.findByRole('dialog', { name: 'Chat with source' })
    act(() => setViewport(1440))
    expect(screen.queryByRole('dialog')).toBeNull()
    act(() => setViewport(1024))
    expect(screen.queryByRole('dialog')).toBeNull()
  })
})
