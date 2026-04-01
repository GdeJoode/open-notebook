'use client'

import { useState, useCallback, useRef, useMemo } from 'react'
import { toast } from 'sonner'
import { sourceChatApi } from '@/lib/api/source-chat'
import { QUERY_KEYS } from '@/lib/api/query-client'
import {
  SourceChatSession,
  SourceChatSessionWithMessages,
  SourceChatMessage,
  SourceChatContextIndicator,
  CreateSourceChatSessionRequest,
  UpdateSourceChatSessionRequest,
} from '@/lib/types/api'
import { useBaseChat, ChatApiAdapter } from './useBaseChat'

export function useSourceChat(sourceId: string) {
  const [isStreaming, setIsStreaming] = useState(false)
  const [contextIndicators, setContextIndicators] = useState<SourceChatContextIndicator | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  type CreateData = Omit<CreateSourceChatSessionRequest, 'source_id'>

  const adapter: ChatApiAdapter<
    SourceChatSession,
    SourceChatSessionWithMessages,
    CreateData,
    UpdateSourceChatSessionRequest
  > = useMemo(() => ({
    listSessions: () => sourceChatApi.listSessions(sourceId),
    getSession: (sessionId: string) => sourceChatApi.getSession(sourceId, sessionId),
    createSession: (data: CreateData) =>
      sourceChatApi.createSession(sourceId, data),
    updateSession: (sessionId: string, data: UpdateSourceChatSessionRequest) =>
      sourceChatApi.updateSession(sourceId, sessionId, data),
    deleteSession: (sessionId: string) => sourceChatApi.deleteSession(sourceId, sessionId),
  }), [sourceId])

  const queryKeys = useMemo(() => ({
    sessions: QUERY_KEYS.sourceChatSessions(sourceId),
    session: (sessionId: string) => QUERY_KEYS.sourceChatSession(sourceId, sessionId),
  }), [sourceId])

  const base = useBaseChat({
    adapter,
    queryKeys,
    enabled: !!sourceId,
  })

  // Destructure for stable references in dependency arrays
  const {
    currentSessionId, sessions, loadingSessions,
    setMessages, setCurrentSessionId, refetchCurrentSession, queryClient,
    createSession: baseCreateSession, updateSession, deleteSession,
    switchSession: baseSwitchSession, refetchSessions,
  } = base

  // Send message with streaming
  const sendMessage = useCallback(async (message: string, modelOverride?: string) => {
    let sessionId = currentSessionId

    // Auto-create session if none exists
    if (!sessionId) {
      try {
        const defaultTitle = message.length > 30 ? `${message.substring(0, 30)}...` : message
        const newSession = await sourceChatApi.createSession(sourceId, { title: defaultTitle })
        sessionId = newSession.id
        setCurrentSessionId(sessionId)
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sourceChatSessions(sourceId) })
      } catch (error) {
        console.error('Failed to create chat session:', error)
        toast.error('Failed to create chat session')
        return
      }
    }

    // Add user message optimistically
    const userMessage: SourceChatMessage = {
      id: `temp-${Date.now()}`,
      type: 'human',
      content: message,
      timestamp: new Date().toISOString()
    }
    setMessages(prev => [...prev, userMessage])
    setIsStreaming(true)

    try {
      const response = await sourceChatApi.sendMessage(sourceId, sessionId, {
        message,
        model_override: modelOverride
      })

      if (!response) {
        throw new Error('No response body')
      }

      const reader = response.getReader()
      const decoder = new TextDecoder()
      let aiMessage: SourceChatMessage | null = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const text = decoder.decode(value)
        const lines = text.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))

              if (data.type === 'ai_message') {
                if (!aiMessage) {
                  aiMessage = {
                    id: `ai-${Date.now()}`,
                    type: 'ai',
                    content: data.content || '',
                    timestamp: new Date().toISOString()
                  }
                  setMessages(prev => [...prev, aiMessage!])
                } else {
                  aiMessage.content += data.content || ''
                  setMessages(prev =>
                    prev.map(msg => msg.id === aiMessage!.id
                      ? { ...msg, content: aiMessage!.content }
                      : msg
                    )
                  )
                }
              } else if (data.type === 'context_indicators') {
                setContextIndicators(data.data)
              } else if (data.type === 'error') {
                throw new Error(data.message || 'Stream error')
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e)
            }
          }
        }
      }
    } catch (error) {
      console.error('Error sending message:', error)
      toast.error('Failed to send message')
      setMessages(prev => prev.filter(msg => !msg.id.startsWith('temp-')))
    } finally {
      setIsStreaming(false)
      refetchCurrentSession()
    }
  }, [sourceId, currentSessionId, refetchCurrentSession, queryClient, setCurrentSessionId, setMessages])

  // Cancel streaming
  const cancelStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      setIsStreaming(false)
    }
  }, [])

  // Override switchSession to also clear context indicators
  const switchSession = useCallback((sessionId: string) => {
    baseSwitchSession(sessionId)
    setContextIndicators(null)
  }, [baseSwitchSession])

  return {
    // State
    sessions,
    currentSession: sessions.find(s => s.id === currentSessionId),
    currentSessionId,
    messages: base.messages,
    isStreaming,
    contextIndicators,
    loadingSessions,

    // Actions
    createSession: (data: Omit<CreateSourceChatSessionRequest, 'source_id'>) =>
      baseCreateSession(data),
    updateSession,
    deleteSession,
    switchSession,
    sendMessage,
    cancelStreaming,
    refetchSessions,
  }
}
