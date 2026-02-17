'use client'

import { Database, Loader2, CheckCircle2, XCircle, Play } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { StepStatus } from '../PipelineStepper'

interface EmbeddingTabProps {
  status: StepStatus
  embeddedChunks?: number
  errorMessage?: string
  extractionComplete?: boolean
  onStart?: () => void
}

export function EmbeddingTab({
  status,
  embeddedChunks,
  errorMessage,
  extractionComplete,
  onStart,
}: EmbeddingTabProps) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-base font-medium flex items-center gap-2">
          <Database className="h-4 w-4" />
          Embedding Generation
        </CardTitle>
        <StatusBadge status={status} />
      </CardHeader>
      <CardContent>
        {status === 'pending' && extractionComplete && onStart && (
          <div className="py-6 space-y-3">
            <p className="text-sm text-muted-foreground">
              Generate vector embeddings to enable semantic search.
            </p>
            <Button onClick={onStart} size="sm" className="gap-2">
              <Play className="h-4 w-4" />
              Start Embedding
            </Button>
          </div>
        )}

        {status === 'running' && (
          <div className="flex items-center gap-3 py-6">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <span className="text-sm text-muted-foreground">
              Generating vector embeddings for search...
            </span>
          </div>
        )}

        {status === 'completed' && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-sm text-green-600 dark:text-green-400">
              <CheckCircle2 className="h-4 w-4" />
              Embedding generation completed
            </div>

            {embeddedChunks !== undefined && embeddedChunks > 0 && (
              <Badge variant="secondary">{embeddedChunks} chunks embedded</Badge>
            )}
          </div>
        )}

        {status === 'skipped' && (
          <div className="py-6">
            <p className="text-sm text-muted-foreground">
              Embedding was not enabled for this source. Vector search will not be available.
            </p>
          </div>
        )}

        {status === 'failed' && (
          <div className="flex items-center gap-2 py-6 text-destructive">
            <XCircle className="h-4 w-4" />
            <span className="text-sm">{errorMessage || 'Embedding generation failed'}</span>
          </div>
        )}

        {(status === 'pending' && !extractionComplete) && (
          <p className="text-sm text-muted-foreground py-6">
            Embeddings will be available after text extraction completes.
          </p>
        )}

        {status === 'locked' && (
          <p className="text-sm text-muted-foreground py-6">
            Embeddings will be available after text extraction completes.
          </p>
        )}
      </CardContent>
    </Card>
  )
}

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
