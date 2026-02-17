'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { getApiUrl } from '@/lib/config'
import { sourcesApi } from '@/lib/api/sources'

export interface LogEntry {
  level: string
  message: string
}

/**
 * Fetch historical processing logs for a source from the backend.
 */
export async function fetchProcessingLogs(sourceId: string): Promise<LogEntry[]> {
  const data = await sourcesApi.getProcessingLogs(sourceId)
  return data.map(e => ({ level: e.level, message: e.message }))
}

export function useProcessingLogs(sourceId: string | undefined, enabled: boolean = true) {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [connected, setConnected] = useState(false)
  const eventSourceRef = useRef<EventSource | null>(null)
  // Track how many messages we've received so reconnections can skip replay
  const messageCountRef = useRef(0)

  const clear = useCallback(() => {
    setLogs([])
    messageCountRef.current = 0
  }, [])

  useEffect(() => {
    if (!sourceId || !enabled) return

    let es: EventSource | null = null

    // On (re)connect, tell the backend how many entries we already have
    // so it skips the replay for those. Also clear local state to avoid
    // appending duplicates if strict-mode double-mounts the effect.
    const alreadyHave = messageCountRef.current
    setLogs(prev => prev.slice(0, alreadyHave))

    const connect = async () => {
      const apiUrl = await getApiUrl()
      const afterParam = alreadyHave > 0 ? `?after=${alreadyHave}` : ''
      const url = `${apiUrl}/api/sources/${sourceId}/logs${afterParam}`

      es = new EventSource(url)
      eventSourceRef.current = es

      es.onopen = () => setConnected(true)

      es.onmessage = (event) => {
        const data = event.data as string
        const pipeIdx = data.indexOf('|')
        if (pipeIdx === -1) return

        const level = data.slice(0, pipeIdx)
        const message = data.slice(pipeIdx + 1)
        messageCountRef.current++
        setLogs(prev => [...prev, { level, message }])
      }

      es.onerror = () => {
        setConnected(false)
        es?.close()
      }
    }

    connect()

    return () => {
      es?.close()
      eventSourceRef.current = null
      setConnected(false)
    }
  }, [sourceId, enabled])

  return { logs, connected, clear }
}
