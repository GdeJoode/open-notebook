'use client'

import { useState, useCallback, useEffect, useMemo } from 'react'
import { toast } from 'sonner'
import { chatApi } from '@/lib/api/chat'
import { QUERY_KEYS } from '@/lib/api/query-client'
import {
  NotebookChatMessage,
  NotebookChatSession,
  NotebookChatSessionWithMessages,
  CreateNotebookChatSessionRequest,
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
  const adapter: ChatApiAdapter<
    NotebookChatSession,
    NotebookChatSessionWithMessages,
    CreateNotebookChatSessionRequest,
    UpdateNotebookChatSessionRequest
  > = useMemo(() => ({
    listSessions: () => chatApi.listSessions(notebookId),
    getSession: (sessionId: string) => chatApi.getSession(sessionId),
    createSession: (data: CreateNotebookChatSessionRequest) =>
      chatApi.createSession(data),
    updateSession: (sessionId: string, data: UpdateNotebookChatSessionRequest) =>
      chatApi.updateSession(sessionId, data),
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

  // Destructure for stable references in dependency arrays
  const {
    currentSessionId, currentSession, sessions, messages, loadingSessions,
    setMessages, setCurrentSessionId, refetchCurrentSession, queryClient,
    createSession: baseCreateSession, updateSession, deleteSession, switchSession,
    refetchSessions,
  } = base

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
    let sessionId = currentSessionId

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
        setCurrentSessionId(sessionId)
        queryClient.invalidateQueries({
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
    setMessages(prev => [...prev, userMessage])
    setIsSending(true)

    try {
      const context = await buildContext()
      const response = await chatApi.sendMessage({
        session_id: sessionId,
        message,
        context,
        model_override: modelOverride ?? (currentSession?.model_override ?? undefined)
      })

      setMessages(response.messages)
      await refetchCurrentSession()
    } catch (error) {
      console.error('Error sending message:', error)
      toast.error('Failed to send message')
      setMessages(prev => prev.filter(msg => !msg.id.startsWith('temp-')))
    } finally {
      setIsSending(false)
    }
  }, [
    notebookId,
    currentSessionId,
    currentSession,
    buildContext,
    refetchCurrentSession,
    queryClient,
    setCurrentSessionId,
    setMessages,
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
    sessions,
    currentSession: currentSession || sessions.find(s => s.id === currentSessionId),
    currentSessionId,
    messages,
    isSending,
    loadingSessions,
    tokenCount,
    charCount,

    // Actions
    createSession: (title?: string) => baseCreateSession({ notebook_id: notebookId, title } as CreateNotebookChatSessionRequest),
    updateSession,
    deleteSession,
    switchSession,
    sendMessage,
    refetchSessions,
  }
}
