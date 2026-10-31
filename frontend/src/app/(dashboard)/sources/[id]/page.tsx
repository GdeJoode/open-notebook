'use client'

import { useRouter, useParams } from 'next/navigation'
import { useCallback, useState } from 'react'
import { Button } from '@/components/ui/button'
import { ArrowLeft } from 'lucide-react'
import { useSourceChat } from '@/lib/hooks/useSourceChat'
import { ChatPanel } from '@/components/source/ChatPanel'
import { useNavigation } from '@/lib/hooks/use-navigation'
import { SourceDetailContent } from '@/components/source/SourceDetailContent'
import { PageBody } from '@/components/layout/PageBody'
import { Sheet, SheetContent, SheetDescription, SheetTitle } from '@/components/ui/sheet'
import { useMinWidth } from '@/lib/hooks/use-media-query'
import { cn } from '@/lib/utils'

export default function SourceDetailPage() {
  const router = useRouter()
  const params = useParams()
  const sourceId = decodeURIComponent(params.id as string)
  const navigation = useNavigation()
  // WB.5: the chat is a column from xl up; below it (a 200 px column at 1024)
  // it opens in a sheet from "Chat with source" in the header.
  const isXl = useMinWidth('xl')
  const [chatOpen, setChatOpen] = useState(false)
  // Crossing xl closes the sheet, so it never reopens by itself on a resize.
  const [chatAtXl, setChatAtXl] = useState(isXl)
  if (chatAtXl !== isXl) {
    setChatAtXl(isXl)
    setChatOpen(false)
  }

  // Initialize source chat
  const chat = useSourceChat(sourceId)

  const handleBack = useCallback(() => {
    const returnPath = navigation.getReturnPath()
    router.push(returnPath)
    navigation.clearReturnTo()
  }, [navigation, router])

  const chatPanel = (
    <ChatPanel
      messages={chat.messages}
      isStreaming={chat.isStreaming}
      contextIndicators={chat.contextIndicators}
      onSendMessage={(message) => chat.sendMessage(message)}
      sessions={chat.sessions}
      currentSessionId={chat.currentSessionId}
      onCreateSession={(title) => chat.createSession({ title })}
      onSelectSession={chat.switchSession}
      onUpdateSession={(sessionId, title) => chat.updateSession(sessionId, { title })}
      onDeleteSession={chat.deleteSession}
      loadingSessions={chat.loadingSessions}
    />
  )

  return (
    <PageBody fill>
      {/* Main content: Source detail + Chat */}
      <div className={cn('flex-1 grid gap-6 overflow-hidden min-h-0', isXl && 'grid-cols-[2fr_1fr]')}>
        {/* Left column - Source detail, with the page header */}
        <div className="overflow-y-auto px-4 pb-6">
          <SourceDetailContent
            sourceId={sourceId}
            showChatButton={!isXl}
            onChatClick={() => setChatOpen(true)}
            onClose={handleBack}
            back={
              <Button variant="ghost" size="sm" className="-ml-3" onClick={handleBack}>
                <ArrowLeft className="mr-2 h-4 w-4" />
                {navigation.getReturnLabel()}
              </Button>
            }
          />
        </div>

        {/* Right column - Chat */}
        {isXl && <div className="overflow-y-auto px-4 pb-6">{chatPanel}</div>}
      </div>

      {!isXl && (
        <Sheet open={chatOpen} onOpenChange={setChatOpen}>
          <SheetContent
            side="right"
            className="w-full gap-0 p-0 pt-14 sm:max-w-md"
            data-testid="source-chat-sheet"
          >
            <SheetTitle className="sr-only">Chat with source</SheetTitle>
            <SheetDescription className="sr-only">Ask questions about this source</SheetDescription>
            {chatPanel}
          </SheetContent>
        </Sheet>
      )}
    </PageBody>
  )
}
