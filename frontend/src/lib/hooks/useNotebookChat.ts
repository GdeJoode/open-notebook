'use client'

import { useState, useCallback, useEffect, useMemo } from 'react'
import { toast } from 'sonner'
import { chatApi } from '@/lib/api/chat'
import { QUERY_KEYS } from '@/lib/api/query-client'
import {
  NotebookChatMessage,
  UpdateNotebookChatSessionRequest,
  SourceListResponse,
  NoteResponse
} from '@/lib/types/api'
import { ContextSelections } from '@/app/(dashboard)/notebooks/[id]/page'
import { useBaseChat, ChatApiAdapter } from './useBaseChat'

interface UseNotebookChatParams {
  notebookId: string
  sources: SourceListResponse[]
  notes: NoteResponse[]
  contextSelections: ContextSelections
}

export function useNotebookChat({ notebookId, sources, notes, contextSelections }: UseNotebookChatParams) {
  const [isSending, setIsSending] = useState(false)
  const [tokenCount, setTokenCount] = useState<number>(0)
  const [charCount, setCharCount] = useState<number>(0)

  // Adapter bridges chatApi to the generic interface
  const adapter: ChatApiAdapter<any, any> = useMemo(() => ({
    listSessions: () => chatApi.listSessions(notebookId),
    getSession: (sessionId: string) => chatApi.getSession(sessionId),
    createSession: (data: Record<string, unknown>) =>
      chatApi.createSession(data as any),
    updateSession: (sessionId: string, data: Record<string, unknown>) =>
      chatApi.updateSession(sessionId, data as UpdateNotebookChatSessionRequest),
    deleteSession: (sessionId: string) => chatApi.deleteSession(sessionId),
  }), [notebookId])

  const queryKeys = useMemo(() => ({
    sessions: QUERY_KEYS.notebookChatSessions(notebookId),
    session: (sessionId: string) => QUERY_KEYS.notebookChatSession(sessionId),
  }), [notebookId])

  const base = useBaseChat({
    adapter,
    queryKeys,
    enabled: !!notebookId,
  })

  // Build context from sources and notes based on user selections
  const buildContext = useCallback(async () => {
    const context_config: { sources: Record<string, string>, notes: Record<string, string> } = {
      sources: {},
      notes: {}
    }

    sources.forEach(source => {
      const mode = contextSelections.sources[source.id]
      if (mode === 'insights') {
        context_config.sources[source.id] = 'insights'
      } else if (mode === 'full') {
        context_config.sources[source.id] = 'full content'
      } else {
        context_config.sources[source.id] = 'not in'
      }
    })

    notes.forEach(note => {
      const mode = contextSelections.notes[note.id]
      if (mode === 'full') {
        context_config.notes[note.id] = 'full content'
      } else {
        context_config.notes[note.id] = 'not in'
      }
    })

    const response = await chatApi.buildContext({
      notebook_id: notebookId,
      context_config
    })

    setTokenCount(response.token_count)
    setCharCount(response.char_count)

    return response.context
  }, [notebookId, sources, notes, contextSelections])

  // Send message (synchronous, no streaming)
  const sendMessage = useCallback(async (message: string, modelOverride?: string) => {
    let sessionId = base.currentSessionId

    // Auto-create session if none exists
    if (!sessionId) {
      try {
        const defaultTitle = message.length > 30
          ? `${message.substring(0, 30)}...`
          : message
        const newSession = await chatApi.createSession({
          notebook_id: notebookId,
          title: defaultTitle
        })
        sessionId = newSession.id
        base.setCurrentSessionId(sessionId)
        base.queryClient.invalidateQueries({
          queryKey: QUERY_KEYS.notebookChatSessions(notebookId)
        })
      } catch {
        toast.error('Failed to create chat session')
        return
      }
    }

    // Add user message optimistically
    const userMessage: NotebookChatMessage = {
      id: `temp-${Date.now()}`,
      type: 'human',
      content: message,
      timestamp: new Date().toISOString()
    }
    base.setMessages(prev => [...prev, userMessage])
    setIsSending(true)

    try {
      const context = await buildContext()
      const response = await chatApi.sendMessage({
        session_id: sessionId,
        message,
        context,
        model_override: modelOverride ?? (base.currentSession?.model_override ?? undefined)
      })

      base.setMessages(response.messages)
      await base.refetchCurrentSession()
    } catch (error) {
      console.error('Error sending message:', error)
      toast.error('Failed to send message')
      base.setMessages(prev => prev.filter(msg => !msg.id.startsWith('temp-')))
    } finally {
      setIsSending(false)
    }
  }, [
    notebookId,
    base.currentSessionId,
    base.currentSession,
    buildContext,
    base.refetchCurrentSession,
    base.queryClient,
    base.setCurrentSessionId,
    base.setMessages,
  ])

  // Update token/char counts when context selections change
  useEffect(() => {
    const updateContextCounts = async () => {
      try {
        await buildContext()
      } catch (error) {
        console.error('Error updating context counts:', error)
      }
    }
    updateContextCounts()
  }, [buildContext])

  return {
    // State
    sessions: base.sessions,
    currentSession: base.currentSession || base.sessions.find((s: any) => s.id === base.currentSessionId),
    currentSessionId: base.currentSessionId,
    messages: base.messages,
    isSending,
    loadingSessions: base.loadingSessions,
    tokenCount,
    charCount,

    // Actions
    createSession: (title?: string) => base.createSession({ notebook_id: notebookId, title }),
    updateSession: base.updateSession,
    deleteSession: base.deleteSession,
    switchSession: base.switchSession,
    sendMessage,
    refetchSessions: base.refetchSessions,
  }
}
