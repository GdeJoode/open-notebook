'use client'

import { useEffect, useRef, useState } from 'react'
import { Loader2 } from 'lucide-react'

import {
  useProcessingLogs,
  fetchProcessingLogs,
  type LogEntry,
} from '@/lib/hooks/use-processing-logs'

/**
 * The streaming-log console kept beneath the live `PipelineStatus` tracker in
 * the create flow (Track UX.5). It shows the source's live processing output
 * (SSE via {@link useProcessingLogs}) while the pipeline is in flight and the
 * persisted logs once it settles, so the console stays populated after the
 * spine reaches a terminal stage.
 */
export interface ProcessingLogConsoleProps {
  sourceId: string | undefined
  /** Whether the source is still processing (drives the live SSE stream). */
  active: boolean
}

function levelColor(level: string): string {
  switch (level.toUpperCase()) {
    case 'ERROR':
      return 'text-red-400'
    case 'WARNING':
      return 'text-yellow-400'
    case 'INFO':
      return 'text-green-400'
    default:
      return 'text-zinc-400'
  }
}

export function ProcessingLogConsole({ sourceId, active }: ProcessingLogConsoleProps) {
  const endRef = useRef<HTMLDivElement>(null)
  const [persistedLogs, setPersistedLogs] = useState<LogEntry[]>([])

  const { logs: liveLogs } = useProcessingLogs(active ? sourceId : undefined, active)

  // Once the source settles (no longer active), fetch the persisted logs so the
  // console keeps its content instead of clearing when the SSE stream closes.
  useEffect(() => {
    if (active || !sourceId) return
    let cancelled = false
    fetchProcessingLogs(sourceId)
      .then((logs) => {
        if (!cancelled) setPersistedLogs(logs)
      })
      .catch(() => {
        if (!cancelled) setPersistedLogs([])
      })
    return () => {
      cancelled = true
    }
  }, [active, sourceId])

  const logs = active ? liveLogs : persistedLogs.length > 0 ? persistedLogs : liveLogs

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs.length])

  if (!sourceId) return null

  return (
    <div
      className="overflow-hidden rounded-md border"
      role="log"
      aria-label="Processing output"
    >
      <div className="shrink-0 border-b bg-muted/50 px-3 py-2">
        <p className="text-xs font-medium text-muted-foreground">Processing output</p>
      </div>
      <div className="max-h-64 min-h-24 overflow-y-auto bg-zinc-950 p-3 font-mono text-xs">
        {logs.length === 0 && active && (
          <div className="flex items-center gap-2 text-zinc-500">
            <Loader2 className="h-3 w-3 animate-spin" aria-hidden />
            <span>Waiting for output...</span>
          </div>
        )}
        {logs.length === 0 && !active && (
          <div className="text-zinc-500">No processing output.</div>
        )}
        {logs.map((log, i) => (
          <div key={i} className="leading-5">
            <span className={levelColor(log.level)}>[{log.level}]</span>{' '}
            <span className="text-zinc-300">{log.message}</span>
          </div>
        ))}
        <div ref={endRef} />
      </div>
    </div>
  )
}

export default ProcessingLogConsole
