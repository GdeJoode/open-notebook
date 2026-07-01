'use client'

import React, { useState, useEffect } from 'react'
import { SourceListResponse } from '@/lib/types/api'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator
} from '@/components/ui/dropdown-menu'
import {
  FileText,
  ExternalLink,
  Upload,
  MoreVertical,
  Trash2,
  RefreshCw,
  Unlink
} from 'lucide-react'
import { useSourcePipeline } from '@/lib/hooks/use-sources'
import {
  isPollableStage,
  isTerminalStage,
} from '@/lib/pipeline/processing-stage'
import { toPipelineCounts } from '@/lib/pipeline/source-counts'
import type { PipelineNodeAction } from '@/lib/pipeline/pipeline-stages'
import { PipelineStatus } from '@/components/sources/pipeline'
import { cn } from '@/lib/utils'
import { ContextToggle } from '@/components/common/ContextToggle'
import { ContextMode } from '@/app/(dashboard)/notebooks/[id]/page'

interface SourceCardProps {
  source: SourceListResponse
  onDelete?: (sourceId: string) => void
  onRetry?: (sourceId: string) => void
  onRemoveFromNotebook?: (sourceId: string) => void
  onClick?: (sourceId: string) => void
  onRefresh?: () => void
  className?: string
  showRemoveFromNotebook?: boolean
  contextMode?: ContextMode
  onContextModeChange?: (mode: ContextMode) => void
}

const SOURCE_TYPE_ICONS = {
  link: ExternalLink,
  upload: Upload,
  text: FileText,
} as const

function getSourceType(source: SourceListResponse): 'link' | 'upload' | 'text' {
  // Determine type based on asset information
  if (source.asset?.url) return 'link'
  if (source.asset?.file_path) return 'upload'
  return 'text'
}

export function SourceCard({
  source,
  onDelete,
  onRetry,
  onRemoveFromNotebook,
  onClick,
  onRefresh,
  className,
  showRemoveFromNotebook = false,
  contextMode,
  onContextModeChange
}: SourceCardProps) {

  const sourceWithStatus = source as SourceListResponse & { command_id?: string; status?: string }

  // Retained to fall back to the job axis for legacy/list-only rows that carry
  // no `processing_stage` yet (pre-backfill) but were recently created.
  const [wasProcessing, setWasProcessing] = useState(false)

  // The spine is fed from the card's OWN `processing_stage` (list payload). We
  // only fan out a per-card GET /sources/{id} while that stage can still advance
  // on its own; terminal (complete/failed) and gated (awaiting_schema_review)
  // cards read the spine straight from the prop and issue no repeating requests.
  // For a legacy row with no stage, fall back to the job-axis signals that mark
  // a recently-created, in-flight source (command_id / active status / a
  // previously-observed processing state).
  const propStage = source.processing_stage
  const jobActive =
    sourceWithStatus.status === 'new' ||
    sourceWithStatus.status === 'queued' ||
    sourceWithStatus.status === 'running'
  const pipelineEnabled =
    propStage !== undefined
      ? isPollableStage(propStage)
      : (!!sourceWithStatus.command_id || jobActive || wasProcessing)

  const { data: pipelineData } = useSourcePipeline(source.id, pipelineEnabled)

  // Prefer freshly-polled data when we have it; otherwise the list payload.
  const effectiveSource = pipelineData ?? source
  const effectiveStage = effectiveSource.processing_stage
  const jobStatus = (effectiveSource as { status?: string }).status

  // Refresh the surrounding list once an in-flight card settles, so its counts
  // (chunks/entities/insights) catch up without a manual reload. Replaces the
  // old useSourceStatus completion-detection.
  useEffect(() => {
    if (pipelineEnabled && effectiveStage !== undefined && isPollableStage(effectiveStage)) {
      setWasProcessing(true)
    }
    if (wasProcessing && isTerminalStage(effectiveStage)) {
      setWasProcessing(false)
      if (onRefresh) {
        // Store the id and clear it on unmount / re-run so a card that unmounts
        // within the 500ms window doesn't fire a refresh on a dead component.
        const timer = setTimeout(() => onRefresh(), 500)
        return () => clearTimeout(timer)
      }
    }
  }, [pipelineEnabled, effectiveStage, wasProcessing, onRefresh])

  const sourceType = getSourceType(source)
  const SourceTypeIcon = SOURCE_TYPE_ICONS[sourceType]

  const title = source.title || 'Untitled Source'

  const isFailedStage = effectiveStage === 'failed'
  // "Settled" cards (successfully complete, or legacy rows with no stage signal)
  // surface their enrichment badges; in-flight cards keep the row lean.
  const showEnrichmentBadges = effectiveStage === 'complete' || effectiveStage === undefined

  const pipelineCounts = toPipelineCounts(effectiveSource)

  const handleRetry = () => {
    onRetry?.(source.id)
  }

  const handleDelete = () => {
    onDelete?.(source.id)
  }

  const handleRemoveFromNotebook = () => {
    onRemoveFromNotebook?.(source.id)
  }

  const handleCardClick = () => {
    onClick?.(source.id)
  }

  // Gated ⇒ deep-link to the source detail (its entities/review view); failed ⇒
  // the existing retry path. Both actions stop propagation inside PipelineStatus
  // so they never trigger the card's own open-detail click.
  const handleNodeAction = (action: PipelineNodeAction) => {
    if (action === 'retry') {
      onRetry?.(source.id)
    } else if (action === 'review-schema') {
      onClick?.(source.id)
    }
  }

  return (
    <Card
      className={cn(
        'transition-all duration-200 hover:shadow-md group relative cursor-pointer border border-border/60 dark:border-border/40',
        className
      )}
      onClick={handleCardClick}
    >
      <CardContent className="px-3 py-2">
        {/* Header with the pipeline mini-bar */}
        <div className="flex items-start justify-between gap-3 mb-1">
          <div className="flex-1 min-w-0">
            {/* Pipeline spine (5-segment mini-bar + current-stage label). Fed
                from the card's own processing_stage; degrades to an all-pending
                bar when the stage is unavailable. */}
            <div className="mb-2" onClick={(e) => e.stopPropagation()}>
              <PipelineStatus
                variant="card"
                processingStage={effectiveStage}
                jobStatus={jobStatus}
                counts={pipelineCounts}
                onNodeAction={handleNodeAction}
              />
            </div>

            {/* Title */}
            <div className="mb-1.5">
              <h4
                className="text-sm font-medium leading-tight line-clamp-2"
                title={title}
              >
                {title}
              </h4>
            </div>

            {/* Metadata badges */}
            <div className="flex items-center gap-2 flex-wrap">
              {/* Source type badge */}
              <Badge variant="secondary" className="text-xs flex items-center gap-1">
                <SourceTypeIcon className="h-3 w-3" />
                {sourceType}
              </Badge>

              {showEnrichmentBadges && source.insights_count > 0 && (
                <Badge variant="outline" className="text-xs">
                  {source.insights_count} insights
                </Badge>
              )}
              {showEnrichmentBadges && source.topics && source.topics.length > 0 && (
                <>
                  {source.topics.slice(0, 2).map((topic, index) => (
                    <Badge key={index} variant="outline" className="text-xs">
                      {topic}
                    </Badge>
                  ))}
                  {source.topics.length > 2 && (
                    <Badge variant="outline" className="text-xs">
                      +{source.topics.length - 2}
                    </Badge>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Context toggle and actions */}
          <div className="flex items-center gap-1">
            {/* Context toggle - only show if handler provided */}
            {onContextModeChange && contextMode && (
              <ContextToggle
                mode={contextMode}
                hasInsights={source.insights_count > 0}
                onChange={onContextModeChange}
              />
            )}

            {/* Actions dropdown */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 w-8 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                  onClick={(e) => e.stopPropagation()}
                  aria-label="Source actions"
                >
                  <MoreVertical className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              {showRemoveFromNotebook && (
                <>
                  <DropdownMenuItem
                    onClick={(e) => {
                      e.stopPropagation()
                      handleRemoveFromNotebook()
                    }}
                    disabled={!onRemoveFromNotebook}
                  >
                    <Unlink className="h-4 w-4 mr-2" />
                    Remove from Notebook
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                </>
              )}

              {isFailedStage && (
                <>
                  <DropdownMenuItem
                    onClick={(e) => {
                      e.stopPropagation()
                      handleRetry()
                    }}
                    disabled={!onRetry}
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Retry Processing
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                </>
              )}

              <DropdownMenuItem
                onClick={(e) => {
                  e.stopPropagation()
                  handleDelete()
                }}
                disabled={!onDelete}
                className="text-red-600 focus:text-red-600"
              >
                <Trash2 className="h-4 w-4 mr-2" />
                Delete Source
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
