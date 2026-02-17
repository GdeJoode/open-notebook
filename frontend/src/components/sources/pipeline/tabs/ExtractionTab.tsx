'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { FileText, Loader2, CheckCircle2, XCircle, RotateCw, Clock, ArrowRight, Image as ImageIcon, Table2 as TableIcon } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { cn } from '@/lib/utils'
import { useProcessingLogs, fetchProcessingLogs, type LogEntry } from '@/lib/hooks/use-processing-logs'
import { sourcesApi } from '@/lib/api/sources'
import type { StepStatus } from '../PipelineStepper'

interface ChunkEntry {
  id: string
  text: string
  order: number
  physical_page: number
  printed_page: number | null
  element_type: string
  metadata: Record<string, unknown>
}

export interface FileEntry {
  name: string
  sourceId?: string
  status: 'pending' | 'creating' | 'processing' | 'completed' | 'failed'
}

interface ExtractionTabProps {
  files: FileEntry[]
  overallStatus: StepStatus
  onContinue?: () => void
}

export function ExtractionTab({
  files,
  overallStatus,
  onContinue,
}: ExtractionTabProps) {
  const isActive = overallStatus === 'running' || overallStatus === 'completed' || overallStatus === 'failed'

  // Selected file index
  const [selectedIndex, setSelectedIndex] = useState(0)

  // Log cache: Map<sourceId, LogEntry[]> — persists across file switches
  const logCacheRef = useRef<Map<string, LogEntry[]>>(new Map())

  // Selected file's full text
  const [selectedFullText, setSelectedFullText] = useState<string | null>(null)
  const [loadingText, setLoadingText] = useState(false)

  // Chunks for selected file (tables + images)
  const [selectedChunks, setSelectedChunks] = useState<ChunkEntry[]>([])
  const [loadingChunks, setLoadingChunks] = useState(false)

  // Base URL for image serving (resolved async from config)
  const [imageBaseUrl, setImageBaseUrl] = useState<string | null>(null)

  // Cached logs for non-live display
  const [cachedLogs, setCachedLogs] = useState<LogEntry[]>([])

  const selectedFile = files[selectedIndex] || null
  const selectedSourceId = selectedFile?.sourceId
  const isSelectedProcessing = selectedFile?.status === 'processing' || selectedFile?.status === 'creating'

  // Live SSE logs for the currently selected file (only when processing)
  const { logs: liveLogs, clear: clearLiveLogs } = useProcessingLogs(
    isSelectedProcessing ? selectedSourceId : undefined,
    isSelectedProcessing
  )

  // Cache live logs as they come in (always cache, even if status changes)
  useEffect(() => {
    if (selectedSourceId && liveLogs.length > 0) {
      logCacheRef.current.set(selectedSourceId, [...liveLogs])
    }
  }, [liveLogs, selectedSourceId])

  // Auto-select the first currently processing file
  const fileStatuses = files.map(f => f.status).join(',')
  useEffect(() => {
    const processingIdx = files.findIndex(f => f.status === 'processing' || f.status === 'creating')
    if (processingIdx !== -1) {
      setSelectedIndex(processingIdx)
    }
  }, [fileStatuses])

  // When selection changes or file completes, load cached logs or fetch from backend
  useEffect(() => {
    if (!selectedSourceId) {
      setCachedLogs([])
      return
    }

    if (isSelectedProcessing) {
      // Live logs will display via liveLogs — also set cached so they're ready on transition
      const cached = logCacheRef.current.get(selectedSourceId)
      if (cached && cached.length > 0) {
        setCachedLogs(cached)
      }
      return
    }

    // Not processing — show from cache or fetch from backend
    const cached = logCacheRef.current.get(selectedSourceId)
    if (cached && cached.length > 0) {
      setCachedLogs(cached)
      return
    }

    // Fetch from backend
    let cancelled = false
    fetchProcessingLogs(selectedSourceId).then(logs => {
      if (!cancelled) {
        setCachedLogs(logs)
        if (logs.length > 0) {
          logCacheRef.current.set(selectedSourceId, logs)
        }
      }
    }).catch(() => {
      if (!cancelled) setCachedLogs([])
    })

    return () => { cancelled = true }
  }, [selectedSourceId, isSelectedProcessing, selectedFile?.status])

  // Fetch full text for selected file when it changes
  useEffect(() => {
    if (!selectedSourceId) {
      setSelectedFullText(null)
      return
    }

    // Only fetch text for completed sources
    if (selectedFile?.status !== 'completed') {
      setSelectedFullText(null)
      return
    }

    let cancelled = false
    setLoadingText(true)
    sourcesApi.get(selectedSourceId).then(data => {
      if (!cancelled) {
        setSelectedFullText(data.full_text)
        setLoadingText(false)
      }
    }).catch(() => {
      if (!cancelled) {
        setSelectedFullText(null)
        setLoadingText(false)
      }
    })

    return () => { cancelled = true }
  }, [selectedSourceId, selectedFile?.status])

  // Fetch chunks for selected file when it completes
  useEffect(() => {
    setImageBaseUrl(null)

    if (!selectedSourceId) {
      setSelectedChunks([])
      return
    }

    if (selectedFile?.status !== 'completed') {
      setSelectedChunks([])
      return
    }

    let cancelled = false
    setLoadingChunks(true)
    sourcesApi.getChunks(selectedSourceId).then(data => {
      if (!cancelled) {
        setSelectedChunks(data.chunks as ChunkEntry[])
        setLoadingChunks(false)
      }
    }).catch(() => {
      if (!cancelled) {
        setSelectedChunks([])
        setLoadingChunks(false)
      }
    })

    // Resolve image base URL for this source
    sourcesApi.getImageUrl(selectedSourceId, '').then(url => {
      if (!cancelled) {
        // Strip trailing slash — we'll append the filename later
        setImageBaseUrl(url.replace(/\/$/, ''))
      }
    }).catch(() => {})

    return () => { cancelled = true }
  }, [selectedSourceId, selectedFile?.status])

  // When switching files, clear live logs for the new file
  const handleFileSelect = useCallback((index: number) => {
    setSelectedIndex(index)
    clearLiveLogs()
  }, [clearLiveLogs])

  // Determine which logs to show — prefer live when processing, cached otherwise
  // Fall back to whichever has content to handle transition moments
  const displayLogs = isSelectedProcessing
    ? (liveLogs.length > 0 ? liveLogs : cachedLogs)
    : (cachedLogs.length > 0 ? cachedLogs : liveLogs)

  // Check if all files are done
  const allDone = files.length > 0 && files.every(f => f.status === 'completed' || f.status === 'failed')

  // Derive table and image chunks
  const tableChunks = selectedChunks.filter(c => c.element_type === 'table')
  const imageChunks = selectedChunks.filter(c => c.element_type === 'image')

  // Count stats
  const completedCount = files.filter(f => f.status === 'completed').length
  const failedCount = files.filter(f => f.status === 'failed').length

  return (
    <Card className="flex-1 flex flex-col min-h-0">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 shrink-0">
        <CardTitle className="text-base font-medium flex items-center gap-2">
          <FileText className="h-4 w-4" />
          Text Extraction
        </CardTitle>
        <div className="flex items-center gap-2">
          {files.length > 1 && (
            <Badge variant="secondary" className="text-xs">
              {completedCount}/{files.length} done
              {failedCount > 0 && ` (${failedCount} failed)`}
            </Badge>
          )}
          <StatusBadge status={overallStatus} />
        </div>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col min-h-0">
        {isActive ? (
          <div className="flex-1 flex flex-col min-h-0">
            <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr_2fr] gap-3 flex-1 min-h-0">
              {/* Left pane: clickable file list */}
              <FileListPane
                files={files}
                selectedIndex={selectedIndex}
                onSelect={handleFileSelect}
              />

              {/* Middle pane: processing log for selected file */}
              <LogPane
                logs={displayLogs}
                status={
                  isSelectedProcessing ? 'running'
                    : selectedFile?.status === 'completed' ? 'completed'
                    : selectedFile?.status === 'failed' ? 'failed'
                    : 'pending'
                }
              />

              {/* Right pane: extraction results with tabs */}
              <ResultsPane
                fullText={selectedFullText}
                tableChunks={tableChunks}
                imageChunks={imageChunks}
                status={
                  selectedFile?.status === 'completed' ? 'completed'
                    : selectedFile?.status === 'failed' ? 'failed'
                    : isSelectedProcessing ? 'running'
                    : 'pending'
                }
                loadingText={loadingText}
                loadingChunks={loadingChunks}
                imageBaseUrl={imageBaseUrl}
              />
            </div>

            {/* Continue button */}
            {allDone && onContinue && (
              <div className="flex justify-end mt-4 shrink-0">
                <Button onClick={onContinue} className="gap-2">
                  Continue to Summarization
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </div>
            )}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground py-6">
            Text will be extracted after submission.
          </p>
        )}
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
/* Left pane — clickable file list                                     */
/* ------------------------------------------------------------------ */

function FileListPane({
  files,
  selectedIndex,
  onSelect,
}: {
  files: FileEntry[]
  selectedIndex: number
  onSelect: (index: number) => void
}) {
  return (
    <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
      <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
        <p className="text-xs font-medium text-muted-foreground">Files</p>
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="p-1.5 space-y-0.5">
          {files.map((file, i) => (
            <button
              key={i}
              type="button"
              onClick={() => onSelect(i)}
              className={cn(
                "flex items-center gap-2 px-2 py-1.5 rounded text-xs w-full text-left transition-colors",
                i === selectedIndex
                  ? "bg-primary/10 text-primary font-medium"
                  : "hover:bg-muted/50"
              )}
            >
              <FileStatusIcon status={file.status} />
              <span className="truncate flex-1" title={file.name}>
                {file.name}
              </span>
            </button>
          ))}
          {files.length === 0 && (
            <p className="text-xs text-muted-foreground px-2 py-3">No files</p>
          )}
        </div>
      </div>
    </div>
  )
}

function FileStatusIcon({ status }: { status: FileEntry['status'] }) {
  switch (status) {
    case 'completed':
      return <CheckCircle2 className="h-3.5 w-3.5 text-green-500 shrink-0" />
    case 'processing':
    case 'creating':
      return <RotateCw className="h-3.5 w-3.5 animate-spin text-primary shrink-0" />
    case 'failed':
      return <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />
    default:
      return <Clock className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
  }
}

/* ------------------------------------------------------------------ */
/* Middle pane — processing output                                     */
/* ------------------------------------------------------------------ */

function LogPane({
  logs,
  status,
}: {
  logs: LogEntry[]
  status: StepStatus
}) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs.length])

  const levelColor = (level: string) => {
    switch (level.toUpperCase()) {
      case 'ERROR': return 'text-red-400'
      case 'WARNING': return 'text-yellow-400'
      case 'INFO': return 'text-green-400'
      default: return 'text-zinc-400'
    }
  }

  return (
    <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
      <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
        <p className="text-xs font-medium text-muted-foreground">Processing Output</p>
      </div>
      <div className="bg-zinc-950 flex-1 p-3 font-mono text-xs overflow-y-auto min-h-0">
        {logs.length === 0 && status === 'running' && (
          <div className="flex items-center gap-2 text-zinc-500">
            <Loader2 className="h-3 w-3 animate-spin" />
            <span>Waiting for output...</span>
          </div>
        )}
        {logs.map((log, i) => (
          <div key={i} className="leading-5">
            <span className={levelColor(log.level)}>[{log.level}]</span>{' '}
            <span className="text-zinc-300">{log.message}</span>
          </div>
        ))}
        {status === 'completed' && logs.length === 0 && (
          <div className="text-green-400">Extraction completed successfully.</div>
        )}
        {status === 'failed' && logs.length === 0 && (
          <div className="text-red-400">Extraction failed.</div>
        )}
        <div ref={endRef} />
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Right pane — extraction results with tabs                           */
/* ------------------------------------------------------------------ */

function ResultsPane({
  fullText,
  tableChunks,
  imageChunks,
  status,
  loadingText,
  loadingChunks,
  imageBaseUrl,
}: {
  fullText?: string | null
  tableChunks: ChunkEntry[]
  imageChunks: ChunkEntry[]
  status: StepStatus
  loadingText?: boolean
  loadingChunks?: boolean
  imageBaseUrl?: string | null
}) {
  return (
    <Tabs defaultValue="text" className="border rounded-md overflow-hidden flex flex-col min-h-0 gap-0">
      <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
        <TabsList className="h-7">
          <TabsTrigger value="text" className="text-xs h-6 px-2.5 gap-1">
            <FileText className="h-3 w-3" />
            Text
          </TabsTrigger>
          <TabsTrigger value="tables" className="text-xs h-6 px-2.5 gap-1">
            <TableIcon className="h-3 w-3" />
            Tables{status === 'completed' && ` (${tableChunks.length})`}
          </TabsTrigger>
          <TabsTrigger value="images" className="text-xs h-6 px-2.5 gap-1">
            <ImageIcon className="h-3 w-3" />
            Images{status === 'completed' && ` (${imageChunks.length})`}
          </TabsTrigger>
        </TabsList>
      </div>

      <TabsContent value="text" className="flex-1 min-h-0 overflow-y-auto mt-0">
        <div className="p-3">
            {(status === 'running' || loadingText) && !fullText && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                <span>{loadingText ? 'Loading text...' : 'Extracting...'}</span>
              </div>
            )}
            {fullText ? (
              <p className="text-xs text-muted-foreground whitespace-pre-wrap leading-relaxed">
                {fullText.slice(0, 3000)}
                {fullText.length > 3000 && (
                  <span className="text-muted-foreground/60">
                    {'\n\n'}... ({(fullText.length / 1000).toFixed(0)}k characters total)
                  </span>
                )}
              </p>
            ) : status === 'completed' && !loadingText ? (
              <p className="text-xs text-muted-foreground italic">No text content available.</p>
            ) : status === 'failed' ? (
              <p className="text-xs text-destructive italic">Extraction failed.</p>
            ) : null}
          </div>
        </TabsContent>

        <TabsContent value="tables" className="flex-1 min-h-0 overflow-y-auto mt-0">
          <div className="p-3 space-y-4">
            {(status === 'running' || loadingChunks) && tableChunks.length === 0 && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                <span>{loadingChunks ? 'Loading tables...' : 'Extracting...'}</span>
              </div>
            )}
            {status === 'completed' && !loadingChunks && tableChunks.length === 0 && (
              <p className="text-xs text-muted-foreground italic">No tables found.</p>
            )}
            {status === 'failed' && tableChunks.length === 0 && (
              <p className="text-xs text-destructive italic">Extraction failed.</p>
            )}
            {tableChunks.map((chunk, i) => (
              <div key={chunk.id || i} className="space-y-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium text-muted-foreground">
                    Table {i + 1}
                  </span>
                  {chunk.printed_page != null && (
                    <Badge variant="outline" className="text-[10px] h-4 px-1">
                      p. {chunk.printed_page}
                    </Badge>
                  )}
                  {chunk.metadata?.num_rows != null && chunk.metadata?.num_cols != null && (
                    <span className="text-[10px] text-muted-foreground/60">
                      {String(chunk.metadata.num_rows)} rows x {String(chunk.metadata.num_cols)} cols
                    </span>
                  )}
                </div>
                <pre className="text-[11px] text-muted-foreground bg-muted/30 rounded p-2 overflow-x-auto whitespace-pre leading-relaxed">
                  {chunk.text}
                </pre>
              </div>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="images" className="flex-1 min-h-0 overflow-y-auto mt-0">
          <ImageGrid
            imageChunks={imageChunks}
            status={status}
            loadingChunks={loadingChunks}
            imageBaseUrl={imageBaseUrl}
          />
        </TabsContent>
    </Tabs>
  )
}

/* ------------------------------------------------------------------ */
/* Image grid with thumbnail → dialog                                  */
/* ------------------------------------------------------------------ */

function ImageGrid({
  imageChunks,
  status,
  loadingChunks,
  imageBaseUrl,
}: {
  imageChunks: ChunkEntry[]
  status: StepStatus
  loadingChunks?: boolean
  imageBaseUrl?: string | null
}) {
  const [selectedImage, setSelectedImage] = useState<{ chunk: ChunkEntry; index: number } | null>(null)

  const getImageSrc = (chunk: ChunkEntry) => {
    const filename = chunk.metadata?.image_file
    if (!imageBaseUrl || typeof filename !== 'string') return null
    return `${imageBaseUrl}/${filename}`
  }

  return (
    <>
      <div className="p-3">
        {(status === 'running' || loadingChunks) && imageChunks.length === 0 && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>{loadingChunks ? 'Loading...' : 'Extracting...'}</span>
          </div>
        )}
        {status === 'completed' && !loadingChunks && imageChunks.length === 0 && (
          <p className="text-xs text-muted-foreground italic">No images found.</p>
        )}
        {status === 'failed' && imageChunks.length === 0 && (
          <p className="text-xs text-destructive italic">Extraction failed.</p>
        )}

        {imageChunks.length > 0 && (
          <div className="grid grid-cols-4 gap-1.5">
            {imageChunks.map((chunk, i) => {
              const src = getImageSrc(chunk)
              return (
                <button
                  key={chunk.id || i}
                  type="button"
                  onClick={() => setSelectedImage({ chunk, index: i })}
                  className="group relative border rounded overflow-hidden bg-muted/30 aspect-[4/3] flex items-center justify-center hover:ring-2 hover:ring-primary/50 transition-all cursor-pointer"
                >
                  {src ? (
                    <img
                      src={src}
                      alt={chunk.text || `Image ${i + 1}`}
                      className="w-full h-full object-contain bg-white"
                    />
                  ) : (
                    <ImageIcon className="h-6 w-6 text-muted-foreground/40" />
                  )}
                  <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/60 to-transparent px-1.5 py-1">
                    <div className="flex items-center gap-1 flex-wrap">
                      <span className="text-[10px] text-white font-medium">#{i + 1}</span>
                      {chunk.printed_page != null && (
                        <span className="text-[10px] text-white/70">p.{chunk.printed_page}</span>
                      )}
                      {chunk.metadata?.classification != null && String(chunk.metadata.classification) !== 'unknown' && (
                        <span className="text-[10px] text-white/90 bg-white/20 rounded px-1">{String(chunk.metadata.classification)}</span>
                      )}
                    </div>
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </div>

      {/* Full-size image dialog */}
      <Dialog open={!!selectedImage} onOpenChange={(open) => { if (!open) setSelectedImage(null) }}>
        <DialogContent className="max-w-3xl max-h-[90vh] flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-sm">
              Image {selectedImage ? selectedImage.index + 1 : ''}
              {selectedImage?.chunk.printed_page != null && (
                <Badge variant="outline" className="text-xs">
                  Page {selectedImage.chunk.printed_page}
                </Badge>
              )}
              {selectedImage?.chunk.metadata?.classification != null && String(selectedImage.chunk.metadata.classification) !== 'unknown' && (
                <Badge variant="secondary" className="text-xs">
                  {String(selectedImage.chunk.metadata.classification)}
                </Badge>
              )}
              {selectedImage?.chunk.metadata?.width != null && selectedImage?.chunk.metadata?.height != null && (
                <span className="text-xs text-muted-foreground">
                  {String(selectedImage.chunk.metadata.width)}&times;{String(selectedImage.chunk.metadata.height)}
                </span>
              )}
            </DialogTitle>
            {selectedImage?.chunk.text && !/^\[.*\]$/.test(selectedImage.chunk.text) && (
              <DialogDescription className="text-sm leading-relaxed">
                {selectedImage.chunk.text}
              </DialogDescription>
            )}
          </DialogHeader>

          {selectedImage && (() => {
            const src = getImageSrc(selectedImage.chunk)
            return src ? (
              <div className="flex-1 min-h-0 flex items-center justify-center overflow-auto">
                <img
                  src={src}
                  alt={selectedImage.chunk.text || `Image ${selectedImage.index + 1}`}
                  className="max-w-full max-h-[60vh] object-contain rounded border bg-white"
                />
              </div>
            ) : null
          })()}
        </DialogContent>
      </Dialog>
    </>
  )
}

/* ------------------------------------------------------------------ */
/* Status badge                                                        */
/* ------------------------------------------------------------------ */

function StatusBadge({ status }: { status: StepStatus }) {
  switch (status) {
    case 'running':
      return <Badge variant="default" className="bg-primary">Running</Badge>
    case 'completed':
      return <Badge variant="default" className="bg-green-600">Completed</Badge>
    case 'failed':
      return <Badge variant="destructive">Failed</Badge>
    case 'skipped':
      return <Badge variant="secondary">Skipped</Badge>
    default:
      return <Badge variant="outline">Pending</Badge>
  }
}
