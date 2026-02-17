'use client'

import { useRouter } from 'next/navigation'
import { CheckCircle2, FileText, Network, Database, AlignLeft, Clock, AlertCircle, Loader2 } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

interface SourceSummary {
  id: string
  title: string
  status: 'processing' | 'completed' | 'failed'
}

interface CompletionTabProps {
  sourceTitle?: string
  sourceType?: string
  chunkCount?: number
  entityCount?: number
  relationCount?: number
  embeddedChunks?: number
  insightsCount?: number
  embedEnabled?: boolean
  summariesEnabled?: boolean
  sources?: SourceSummary[]
}

export function CompletionTab({
  sourceTitle,
  sourceType,
  chunkCount,
  entityCount,
  relationCount,
  embeddedChunks,
  insightsCount,
  embedEnabled,
  summariesEnabled,
  sources,
}: CompletionTabProps) {
  const router = useRouter()

  // Multi-source variant
  if (sources && sources.length > 1) {
    const completedCount = sources.filter(s => s.status === 'completed').length
    const failedCount = sources.filter(s => s.status === 'failed').length
    const processingCount = sources.filter(s => s.status === 'processing').length
    const allDone = processingCount === 0

    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-12 h-12 rounded-full bg-green-100 dark:bg-green-900/30">
            <CheckCircle2 className="h-6 w-6 text-green-600 dark:text-green-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">
              {allDone ? 'Sources Created' : 'Creating Sources...'}
            </h3>
            <p className="text-sm text-muted-foreground">
              {completedCount} of {sources.length} sources created
              {failedCount > 0 && ` (${failedCount} failed)`}
            </p>
          </div>
        </div>

        <div className="space-y-2">
          {sources.map((source, idx) => (
            <div
              key={source.id || idx}
              className="flex items-center justify-between rounded-md border p-3"
            >
              <div className="flex items-center gap-3 min-w-0">
                <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                <span className="text-sm truncate">{source.title}</span>
              </div>
              <SourceStatusBadge status={source.status} />
            </div>
          ))}
        </div>

        <div className="flex gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={() => router.push('/sources')}
          >
            View All Sources
          </Button>
        </div>
      </div>
    )
  }

  // Single source variant (original)
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center w-12 h-12 rounded-full bg-green-100 dark:bg-green-900/30">
          <CheckCircle2 className="h-6 w-6 text-green-600 dark:text-green-400" />
        </div>
        <div>
          <h3 className="text-lg font-semibold">Source Created Successfully</h3>
          <p className="text-sm text-muted-foreground">
            {sourceTitle || 'Your source'} has been processed.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <StatCard
          icon={FileText}
          label="Source Type"
          value={sourceType || 'Unknown'}
        />
        <StatCard
          icon={FileText}
          label="Chunks Extracted"
          value={chunkCount !== undefined ? String(chunkCount) : '0'}
        />
        <StatCard
          icon={Network}
          label="Entities"
          value={entityCount !== undefined ? String(entityCount) : '0'}
          detail={relationCount ? `${relationCount} relations` : undefined}
        />
        <StatCard
          icon={Database}
          label="Embeddings"
          value={embedEnabled ? String(embeddedChunks || 0) : 'Skipped'}
        />
        <StatCard
          icon={AlignLeft}
          label="Summaries"
          value={summariesEnabled ? String(insightsCount || 0) : 'None'}
        />
        <StatCard
          icon={Clock}
          label="Status"
          value="Complete"
          badge
        />
      </div>
    </div>
  )
}

function SourceStatusBadge({ status }: { status: SourceSummary['status'] }) {
  switch (status) {
    case 'completed':
      return (
        <Badge variant="default" className="bg-green-600 gap-1">
          <CheckCircle2 className="h-3 w-3" />
          Created
        </Badge>
      )
    case 'failed':
      return (
        <Badge variant="destructive" className="gap-1">
          <AlertCircle className="h-3 w-3" />
          Failed
        </Badge>
      )
    case 'processing':
      return (
        <Badge variant="secondary" className="gap-1">
          <Loader2 className="h-3 w-3 animate-spin" />
          Processing
        </Badge>
      )
  }
}

function StatCard({
  icon: Icon,
  label,
  value,
  detail,
  badge,
}: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  value: string
  detail?: string
  badge?: boolean
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
          <Icon className="h-3.5 w-3.5" />
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {badge ? (
          <Badge variant="default" className="bg-green-600">{value}</Badge>
        ) : (
          <p className="text-xl font-semibold">{value}</p>
        )}
        {detail && (
          <p className="text-xs text-muted-foreground mt-1">{detail}</p>
        )}
      </CardContent>
    </Card>
  )
}
