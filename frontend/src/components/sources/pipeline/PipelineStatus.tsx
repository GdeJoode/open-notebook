'use client'

import * as React from 'react'
import {
  Check,
  Circle,
  Loader2,
  AlertTriangle,
  XCircle,
  ChevronRight,
} from 'lucide-react'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import type { ProcessingStage } from '@/lib/pipeline/processing-stage'
import {
  derivePipelineNodes,
  type PipelineCounts,
  type PipelineDeepLinkTab,
  type PipelineJobStatus,
  type PipelineNode,
  type PipelineNodeAction,
  type PipelineNodeState,
} from '@/lib/pipeline/pipeline-stages'

export interface PipelineStatusProps {
  /**
   * `live` — full horizontal tracker (spinner on the active node) with a slot
   * for the streaming-log console the create-flow passes as `children`.
   * `card` — compact 5-segment mini-bar + current-stage label (list surface).
   * `detail` — collapsible spine row (source-detail surface).
   */
  variant: 'live' | 'card' | 'detail'
  processingStage: ProcessingStage | undefined
  jobStatus?: PipelineJobStatus | string
  counts?: PipelineCounts
  /** Fired when a node is activated; carries the node's deep-link tab. */
  onNodeClick?: (tab: PipelineDeepLinkTab | undefined) => void
  /** Fired when a node's recovery action (Review schema / Retry) is activated. */
  onNodeAction?: (action: PipelineNodeAction, node: PipelineNode) => void
  /** Streaming-log console (or any extra content) rendered under `live`. */
  children?: React.ReactNode
  /** Initial open state of the `detail` collapsible (default collapsed). */
  defaultOpen?: boolean
  className?: string
}

const STATE_DESCRIPTION: Record<PipelineNodeState, string> = {
  pending: 'not started',
  active: 'in progress',
  done: 'complete',
  gated: 'awaiting schema review',
  failed: 'failed',
}

const ACTION_LABEL: Record<PipelineNodeAction, string> = {
  'review-schema': 'Review schema',
  retry: 'Retry',
}

function nodeAriaLabel(node: PipelineNode): string {
  const base = `${node.label}: ${STATE_DESCRIPTION[node.state]}`
  return node.countLabel ? `${base}, ${node.countLabel}` : base
}

function StateIcon({
  state,
  className,
}: {
  state: PipelineNodeState
  className?: string
}) {
  switch (state) {
    case 'active':
      return <Loader2 className={cn('animate-spin text-blue-600', className)} aria-hidden />
    case 'done':
      return <Check className={cn('text-green-600', className)} aria-hidden />
    case 'gated':
      return <AlertTriangle className={cn('text-amber-600', className)} aria-hidden />
    case 'failed':
      return <XCircle className={cn('text-red-600', className)} aria-hidden />
    default:
      return <Circle className={cn('text-muted-foreground', className)} aria-hidden />
  }
}

const SEGMENT_STATE_CLASS: Record<PipelineNodeState, string> = {
  pending: 'bg-muted',
  active: 'bg-blue-500 animate-pulse',
  done: 'bg-green-500',
  gated: 'bg-amber-500',
  failed: 'bg-red-500',
}

/** The recovery action button (Review schema / Retry), a standalone control. */
function NodeAction({
  node,
  onNodeAction,
}: {
  node: PipelineNode
  onNodeAction?: (action: PipelineNodeAction, node: PipelineNode) => void
}) {
  if (!node.action) return null
  const action = node.action
  return (
    <Button
      type="button"
      size="sm"
      variant={action === 'retry' ? 'destructive' : 'outline'}
      aria-label={`${ACTION_LABEL[action]} — ${node.label}`}
      onClick={(e) => {
        e.stopPropagation()
        onNodeAction?.(action, node)
      }}
    >
      {ACTION_LABEL[action]}
    </Button>
  )
}

/** A full node control (icon + label + count) used by `live` and `detail`. */
function NodeControl({
  node,
  onNodeClick,
  onNodeAction,
  compact,
}: {
  node: PipelineNode
  onNodeClick?: (tab: PipelineDeepLinkTab | undefined) => void
  onNodeAction?: (action: PipelineNodeAction, node: PipelineNode) => void
  compact?: boolean
}) {
  return (
    <div
      className={cn(
        'flex items-center gap-2',
        node.parallel && 'ml-6 text-sm'
      )}
      data-node-key={node.key}
      data-node-state={node.state}
    >
      <button
        type="button"
        aria-label={nodeAriaLabel(node)}
        onClick={() => onNodeClick?.(node.deepLinkTab)}
        className={cn(
          'flex items-center gap-2 rounded-md px-2 py-1 text-left',
          'outline-none focus-visible:ring-2 focus-visible:ring-ring',
          'hover:bg-accent hover:text-accent-foreground transition-colors',
          node.state === 'pending' && 'text-muted-foreground'
        )}
      >
        <StateIcon state={node.state} className={compact ? 'size-4' : 'size-5'} />
        <span className="font-medium">{node.label}</span>
        {node.countLabel && (
          <span className="text-xs text-muted-foreground">{node.countLabel}</span>
        )}
      </button>
      <NodeAction node={node} onNodeAction={onNodeAction} />
    </div>
  )
}

/** Human label for the source's current stage (card sub-line). */
function currentStageLabel(spine: PipelineNode[]): string {
  const active = spine.find((n) => n.state === 'active')
  if (active) return `${active.label}…`
  const gated = spine.find((n) => n.state === 'gated')
  if (gated) return 'Awaiting schema review'
  const failed = spine.find((n) => n.state === 'failed')
  if (failed) return 'Failed'
  if (spine.every((n) => n.state === 'done')) return 'Complete'
  const lastDone = [...spine].reverse().find((n) => n.state === 'done')
  return lastDone ? `${lastDone.label} done` : 'Not started'
}

export function PipelineStatus({
  variant,
  processingStage,
  jobStatus,
  counts,
  onNodeClick,
  onNodeAction,
  children,
  defaultOpen = false,
  className,
}: PipelineStatusProps) {
  const nodes = React.useMemo(
    () => derivePipelineNodes({ processingStage, jobStatus, counts }),
    [processingStage, jobStatus, counts]
  )
  const spine = nodes.filter((n) => !n.parallel)
  const insights = nodes.find((n) => n.parallel)

  if (variant === 'card') {
    return (
      <div className={cn('flex flex-col gap-1', className)} data-variant="card">
        <div className="flex items-center gap-1" role="group" aria-label="Pipeline progress">
          {spine.map((node) => (
            <button
              key={node.key}
              type="button"
              aria-label={nodeAriaLabel(node)}
              onClick={() => onNodeClick?.(node.deepLinkTab)}
              data-node-key={node.key}
              data-node-state={node.state}
              className={cn(
                'h-1.5 flex-1 rounded-full outline-none focus-visible:ring-2 focus-visible:ring-ring',
                SEGMENT_STATE_CLASS[node.state]
              )}
            />
          ))}
        </div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-muted-foreground">{currentStageLabel(spine)}</span>
          {insights && insights.state === 'done' && insights.countLabel && (
            <button
              type="button"
              aria-label={nodeAriaLabel(insights)}
              onClick={() => onNodeClick?.(insights.deepLinkTab)}
              data-node-key="insights"
              data-node-state={insights.state}
              className="text-xs text-muted-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring rounded"
            >
              {insights.countLabel}
            </button>
          )}
        </div>
        {/* An inline recovery action for gated/failed cards. */}
        {spine
          .filter((n) => n.action)
          .map((n) => (
            <NodeAction key={n.key} node={n} onNodeAction={onNodeAction} />
          ))}
      </div>
    )
  }

  if (variant === 'detail') {
    const recoveryNodes = spine.filter((n) => n.action)
    return (
      <div className={cn('flex w-full flex-col gap-1', className)} data-variant="detail">
      <Collapsible defaultOpen={defaultOpen} className="w-full">
        <CollapsibleTrigger
          className={cn(
            'group flex w-full items-center gap-2 rounded-md px-2 py-1',
            'outline-none focus-visible:ring-2 focus-visible:ring-ring'
          )}
          aria-label="Toggle pipeline detail"
        >
          <ChevronRight
            className="size-4 transition-transform group-data-[state=open]:rotate-90"
            aria-hidden
          />
          <div className="flex flex-1 items-center gap-1">
            {spine.map((node) => (
              <span
                key={node.key}
                data-node-key={node.key}
                data-node-state={node.state}
                className={cn('h-1.5 flex-1 rounded-full', SEGMENT_STATE_CLASS[node.state])}
                aria-hidden
              />
            ))}
          </div>
          <span className="text-xs text-muted-foreground">{currentStageLabel(spine)}</span>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="flex flex-col gap-1 py-2">
            {spine.map((node) => (
              <NodeControl
                key={node.key}
                node={node}
                onNodeClick={onNodeClick}
                onNodeAction={onNodeAction}
                compact
              />
            ))}
            {insights && (
              <NodeControl
                node={insights}
                onNodeClick={onNodeClick}
                onNodeAction={onNodeAction}
                compact
              />
            )}
          </div>
        </CollapsibleContent>
      </Collapsible>
        {/* Recovery actions stay reachable even while the spine is collapsed. */}
        {recoveryNodes.map((n) => (
          <NodeAction key={n.key} node={n} onNodeAction={onNodeAction} />
        ))}
      </div>
    )
  }

  // variant === 'live'
  return (
    <div className={cn('flex flex-col gap-4', className)} data-variant="live">
      <div
        className="flex flex-wrap items-center gap-x-2 gap-y-3"
        role="group"
        aria-label="Pipeline progress"
      >
        {spine.map((node, i) => (
          <React.Fragment key={node.key}>
            {i > 0 && (
              <ChevronRight className="size-4 text-muted-foreground" aria-hidden />
            )}
            <NodeControl node={node} onNodeClick={onNodeClick} onNodeAction={onNodeAction} />
          </React.Fragment>
        ))}
      </div>
      {insights && (
        <NodeControl node={insights} onNodeClick={onNodeClick} onNodeAction={onNodeAction} />
      )}
      {children}
    </div>
  )
}

export default PipelineStatus
