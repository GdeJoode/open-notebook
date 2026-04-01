'use client'

import { useState, useCallback, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

/**
 * Adapter interface for plugging in different chat API backends.
 * Both notebook chat and source chat implement this shape.
 */
export interface ChatApiAdapter<TSession, TSessionWithMessages> {
  listSessions: () => Promise<TSession[]>
  getSession: (sessionId: string) => Promise<TSessionWithMessages>
  createSession: (data: Record<string, unknown>) => Promise<TSession>
  updateSession: (sessionId: string, data: Record<string, unknown>) => Promise<TSession>
  deleteSession: (sessionId: string) => Promise<void>
}

export interface BaseChatQueryKeys {
  sessions: readonly unknown[]
  session: (sessionId: string) => readonly unknown[]
}

export interface ChatMessage {
  id: string
  type: 'human' | 'ai'
  content: string
  timestamp?: string
}

export interface UseBaseChatOptions<TSession, TSessionWithMessages> {
  adapter: ChatApiAdapter<TSession, TSessionWithMessages>
  queryKeys: BaseChatQueryKeys
  enabled: boolean
}

export function useBaseChat<
  TSession extends { id: string },
  TSessionWithMessages extends { messages?: ChatMessage[] },
>({
  adapter,
  queryKeys,
  enabled,
}: UseBaseChatOptions<TSession, TSessionWithMessages>) {
  const queryClient = useQueryClient()
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])

  // Fetch sessions
  const {
    data: sessions = [],
    isLoading: loadingSessions,
    refetch: refetchSessions,
  } = useQuery<TSession[]>({
    queryKey: queryKeys.sessions,
    queryFn: () => adapter.listSessions(),
    enabled,
  })

  // Fetch current session with messages
  const {
    data: currentSession,
    refetch: refetchCurrentSession,
  } = useQuery({
    queryKey: queryKeys.session(currentSessionId!),
    queryFn: () => adapter.getSession(currentSessionId!),
    enabled: enabled && !!currentSessionId,
  })

  // Update messages when session changes
  useEffect(() => {
    if (currentSession?.messages) {
      setMessages(currentSession.messages)
    }
  }, [currentSession])

  // Auto-select most recent session
  useEffect(() => {
    if (sessions.length > 0 && !currentSessionId) {
      setCurrentSessionId(sessions[0].id)
    }
  }, [sessions, currentSessionId])

  // Create session mutation
  const createSessionMutation = useMutation({
    mutationFn: (data: Record<string, unknown>) => adapter.createSession(data),
    onSuccess: (newSession) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.sessions })
      setCurrentSessionId(newSession.id)
      toast.success('Chat session created')
    },
    onError: () => {
      toast.error('Failed to create chat session')
    },
  })

  // Update session mutation
  const updateSessionMutation = useMutation({
    mutationFn: ({ sessionId, data }: { sessionId: string; data: Record<string, unknown> }) =>
      adapter.updateSession(sessionId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.sessions })
      if (currentSessionId) {
        queryClient.invalidateQueries({ queryKey: queryKeys.session(currentSessionId) })
      }
      toast.success('Session updated')
    },
    onError: () => {
      toast.error('Failed to update session')
    },
  })

  // Delete session mutation
  const deleteSessionMutation = useMutation({
    mutationFn: (sessionId: string) => adapter.deleteSession(sessionId),
    onSuccess: (_, deletedId) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.sessions })
      if (currentSessionId === deletedId) {
        setCurrentSessionId(null)
        setMessages([])
      }
      toast.success('Session deleted')
    },
    onError: () => {
      toast.error('Failed to delete session')
    },
  })

  const switchSession = useCallback((sessionId: string) => {
    setCurrentSessionId(sessionId)
  }, [])

  const createSession = useCallback(
    (data: Record<string, unknown>) => createSessionMutation.mutate(data),
    [createSessionMutation]
  )

  const updateSession = useCallback(
    (sessionId: string, data: Record<string, unknown>) =>
      updateSessionMutation.mutate({ sessionId, data }),
    [updateSessionMutation]
  )

  const deleteSession = useCallback(
    (sessionId: string) => deleteSessionMutation.mutate(sessionId),
    [deleteSessionMutation]
  )

  return {
    // State
    sessions,
    currentSession,
    currentSessionId,
    messages,
    setMessages,
    loadingSessions,

    // Actions
    createSession,
    updateSession,
    deleteSession,
    switchSession,
    setCurrentSessionId,
    refetchSessions,
    refetchCurrentSession,
    queryClient,
  }
}
