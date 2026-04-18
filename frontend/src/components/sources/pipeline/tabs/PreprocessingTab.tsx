'use client'

import { useState } from 'react'
import {
  ScanSearch,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  ArrowRight,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { useSourceChunks } from '@/lib/hooks/use-sources'
import { PdfChunkViewer } from '@/components/source/PdfChunkViewer'
import type { StepStatus } from '../PipelineStepper'
import type { FileEntry } from './ExtractionTab'

interface PreprocessingTabProps {
  status: StepStatus
  extractionComplete?: boolean
  sourceId?: string
  files?: FileEntry[]
  onContinue?: () => void
}

export function PreprocessingTab({
  status,
  extractionComplete,
  sourceId,
  files = [],
  onContinue,
}: PreprocessingTabProps) {
  const [selectedFileIndex, setSelectedFileIndex] = useState(0)

  const selectedFile = files[selectedFileIndex] || null
  const selectedSourceId = selectedFile?.sourceId || sourceId

  const { data: chunksData, isLoading: chunksLoading } = useSourceChunks(
    selectedSourceId || '',
    !!selectedSourceId && !!extractionComplete
  )

  // Locked state
  if (!extractionComplete) {
    return (
      <Card className="flex-1 flex flex-col min-h-0">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 shrink-0">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <ScanSearch className="h-4 w-4" />
            Chunk Review
          </CardTitle>
          <Badge variant="outline">Locked</Badge>
        </CardHeader>
        <CardContent className="flex items-center justify-center flex-1">
          <p className="text-sm text-muted-foreground">
            Chunk review available after extraction completes.
          </p>
        </CardContent>
      </Card>
    )
  }

  const chunks = chunksData?.chunks || []

  return (
    <Card className="flex-1 flex flex-col min-h-0">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 shrink-0">
        <div className="flex items-center gap-2">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <ScanSearch className="h-4 w-4" />
            Chunk Review
          </CardTitle>
          <StatusBadge status={status} />
        </div>
        {onContinue && (
          <Button type="button" onClick={onContinue} size="sm" className="gap-2">
            Continue to Classification
            <ArrowRight className="h-4 w-4" />
          </Button>
        )}
      </CardHeader>
      <CardContent className="flex-1 flex min-h-0 p-0 px-6 pb-6">
        <div className="flex gap-3 flex-1 min-h-0 w-full">
          <FileListPane
            files={files}
            selectedIndex={selectedFileIndex}
            onSelect={setSelectedFileIndex}
          />
          {/* Stop submit events from PdfChunkViewer buttons bubbling to pipeline form */}
          <div className="flex-1 min-h-0" onSubmit={e => e.stopPropagation()}>
            {chunksLoading ? (
              <div className="flex h-full items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : selectedSourceId && chunks.length > 0 ? (
              <PdfChunkViewer sourceId={selectedSourceId} chunks={chunks} />
            ) : (
              <div className="flex h-full items-center justify-center">
                <p className="text-sm text-muted-foreground">
                  {selectedSourceId ? 'No chunks found for this document.' : 'Select a file to review chunks.'}
                </p>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
/* File list pane                                                      */
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
    <div className="w-48 min-w-[160px] border rounded-md overflow-hidden flex flex-col min-h-0 shrink-0">
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
      return <Loader2 className="h-3.5 w-3.5 animate-spin text-primary shrink-0" />
    case 'failed':
      return <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />
    default:
      return <Clock className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
  }
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
    default:
      return <Badge variant="outline">Pending</Badge>
  }
}
