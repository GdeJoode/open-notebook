'use client'

import { useRouter, useParams } from 'next/navigation'
import { useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { ArrowLeft } from 'lucide-react'
import { useSourceChat } from '@/lib/hooks/useSourceChat'
import { ChatPanel } from '@/components/source/ChatPanel'
import { useNavigation } from '@/lib/hooks/use-navigation'
import { SourceDetailContent } from '@/components/source/SourceDetailContent'
import { PageBody } from '@/components/layout/PageBody'

export default function SourceDetailPage() {
  const router = useRouter()
  const params = useParams()
  const sourceId = decodeURIComponent(params.id as string)
  const navigation = useNavigation()

  // Initialize source chat
  const chat = useSourceChat(sourceId)

  const handleBack = useCallback(() => {
    const returnPath = navigation.getReturnPath()
    router.push(returnPath)
    navigation.clearReturnTo()
  }, [navigation, router])

  return (
    <PageBody fill>
      {/* Main content: Source detail + Chat */}
      <div className="flex-1 grid gap-6 lg:grid-cols-[2fr_1fr] overflow-hidden min-h-0">
        {/* Left column - Source detail, with the page header */}
        <div className="overflow-y-auto px-4 pb-6">
          <SourceDetailContent
            sourceId={sourceId}
            showChatButton={false}
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
        <div className="overflow-y-auto px-4 pb-6">
          <ChatPanel
            messages={chat.messages}
            isStreaming={chat.isStreaming}
            contextIndicators={chat.contextIndicators}
            onSendMessage={(message, model) => chat.sendMessage(message, model)}
            modelOverride={chat.currentSession?.model_override}
            onModelChange={(model) => {
              if (chat.currentSessionId) {
                chat.updateSession(chat.currentSessionId, { model_override: model })
              }
            }}
            sessions={chat.sessions}
            currentSessionId={chat.currentSessionId}
            onCreateSession={(title) => chat.createSession({ title })}
            onSelectSession={chat.switchSession}
            onUpdateSession={(sessionId, title) => chat.updateSession(sessionId, { title })}
            onDeleteSession={chat.deleteSession}
            loadingSessions={chat.loadingSessions}
          />
        </div>
      </div>
    </PageBody>
  )
}
