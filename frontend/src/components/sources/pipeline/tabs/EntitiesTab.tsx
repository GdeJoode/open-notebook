'use client'

import { Network, Loader2, CheckCircle2, XCircle, Play } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { StepStatus } from '../PipelineStepper'

interface EntityGroup {
  type: string
  count: number
}

interface EntitiesTabProps {
  status: StepStatus
  entityCount?: number
  relationCount?: number
  entityGroups?: EntityGroup[]
  errorMessage?: string
  extractionComplete?: boolean
  onStart?: () => void
}

export function EntitiesTab({
  status,
  entityCount,
  relationCount,
  entityGroups,
  errorMessage,
  extractionComplete,
  onStart,
}: EntitiesTabProps) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-base font-medium flex items-center gap-2">
          <Network className="h-4 w-4" />
          Entity Extraction & Filtering
        </CardTitle>
        <StatusBadge status={status} />
      </CardHeader>
      <CardContent>
        {status === 'pending' && extractionComplete && onStart && (
          <div className="py-6 space-y-3">
            <p className="text-sm text-muted-foreground">
              Extract entities using the ontology-driven pipeline.
            </p>
            <Button onClick={onStart} size="sm" className="gap-2">
              <Play className="h-4 w-4" />
              Start Entity Extraction
            </Button>
          </div>
        )}

        {status === 'running' && (
          <div className="flex items-center gap-3 py-6">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <span className="text-sm text-muted-foreground">
              Extracting entities using ontology-driven pipeline...
            </span>
          </div>
        )}

        {status === 'completed' && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-sm text-green-600 dark:text-green-400">
              <CheckCircle2 className="h-4 w-4" />
              Entity extraction completed
            </div>

            <div className="flex items-center gap-3">
              {entityCount !== undefined && (
                <Badge variant="secondary">{entityCount} entities</Badge>
              )}
              {relationCount !== undefined && relationCount > 0 && (
                <Badge variant="secondary">{relationCount} relations</Badge>
              )}
            </div>

            {entityGroups && entityGroups.length > 0 && (
              <div className="mt-3">
                <p className="text-xs font-medium text-muted-foreground mb-2">By Type</p>
                <div className="flex flex-wrap gap-2">
                  {entityGroups.map((group) => (
                    <Badge key={group.type} variant="outline" className="text-xs">
                      {group.type} ({group.count})
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {status === 'failed' && (
          <div className="flex items-center gap-2 py-6 text-destructive">
            <XCircle className="h-4 w-4" />
            <span className="text-sm">{errorMessage || 'Entity extraction failed'}</span>
          </div>
        )}

        {(status === 'pending' && !extractionComplete) && (
          <p className="text-sm text-muted-foreground py-6">
            Entities will be available after text extraction completes.
          </p>
        )}

        {status === 'locked' && (
          <p className="text-sm text-muted-foreground py-6">
            Entities will be available after text extraction completes.
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
