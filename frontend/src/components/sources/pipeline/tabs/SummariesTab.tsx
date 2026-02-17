'use client'

import { useState, useEffect, useCallback } from 'react'
import {
  AlignLeft, Loader2, CheckCircle2, XCircle, Clock,
  Play, ChevronDown, ChevronRight,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { cn } from '@/lib/utils'
import { summariesApi, type StrategyInfo, type SummaryListItem, type SummaryDetail } from '@/lib/api/summaries'
import type { StepStatus } from '../PipelineStepper'
import type { FileEntry } from './ExtractionTab'

const STRATEGY_DESCRIPTIONS: Record<string, string> = {
  raptor: 'Recursive clustering & summarization',
  treekg: 'Structure-aware hierarchical summaries',
  naive: 'Single-pass baseline summary',
}

interface SummariesTabProps {
  status: StepStatus
  extractionComplete?: boolean
  sourceId?: string
  files?: FileEntry[]
}

export function SummariesTab({
  status,
  extractionComplete,
  sourceId,
  files = [],
}: SummariesTabProps) {
  // Left pane: selected file
  const [selectedFileIndex, setSelectedFileIndex] = useState(0)

  // Middle pane: strategies & selection (multi-select via checkboxes)
  const [strategies, setStrategies] = useState<StrategyInfo[]>([])
  const [selectedStrategies, setSelectedStrategies] = useState<Set<string>>(new Set())
  const [generating, setGenerating] = useState(false)
  const [currentlyGenerating, setCurrentlyGenerating] = useState<string | null>(null)
  const [generateError, setGenerateError] = useState<string | null>(null)

  // Right pane: results
  const [existingSummaries, setExistingSummaries] = useState<SummaryListItem[]>([])
  const [activeResultTab, setActiveResultTab] = useState<string>('')
  const [expandedDetail, setExpandedDetail] = useState<SummaryDetail | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [expandedNodes, setExpandedNodes] = useState(false)

  const selectedFile = files[selectedFileIndex] || null
  const selectedSourceId = selectedFile?.sourceId || sourceId

  // Fetch strategies once
  useEffect(() => {
    let cancelled = false
    summariesApi.listStrategies().then(data => {
      if (!cancelled) {
        setStrategies(data)
        // Pre-select all implemented strategies
        const implemented = data.filter(s => s.implemented).map(s => s.name)
        setSelectedStrategies(new Set(implemented))
      }
    }).catch(() => {})
    return () => { cancelled = true }
  }, [])

  // Fetch existing summaries when selected source changes
  const refreshSummaries = useCallback(async () => {
    if (!selectedSourceId) {
      setExistingSummaries([])
      return
    }
    try {
      const data = await summariesApi.list(selectedSourceId)
      setExistingSummaries(data)
    } catch {
      setExistingSummaries([])
    }
  }, [selectedSourceId])

  useEffect(() => {
    refreshSummaries()
    setExpandedDetail(null)
    setActiveResultTab('')
  }, [refreshSummaries])

  // Auto-select first result tab when summaries change
  useEffect(() => {
    if (existingSummaries.length > 0 && !activeResultTab) {
      setActiveResultTab(existingSummaries[0].strategy)
    }
  }, [existingSummaries, activeResultTab])

  // Load full detail when switching result tab
  useEffect(() => {
    if (!activeResultTab) {
      setExpandedDetail(null)
      return
    }
    const summary = existingSummaries.find(s => s.strategy === activeResultTab)
    if (!summary) {
      setExpandedDetail(null)
      return
    }
    let cancelled = false
    setLoadingDetail(true)
    summariesApi.get(summary.id).then(data => {
      if (!cancelled) {
        setExpandedDetail(data)
        setLoadingDetail(false)
      }
    }).catch(() => {
      if (!cancelled) {
        setExpandedDetail(null)
        setLoadingDetail(false)
      }
    })
    return () => { cancelled = true }
  }, [activeResultTab, existingSummaries])

  // Toggle a strategy checkbox
  const handleToggleStrategy = (name: string) => {
    setSelectedStrategies(prev => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  // Run all selected strategies sequentially
  const handleRun = async () => {
    if (!selectedSourceId || selectedStrategies.size === 0 || generating) return

    // Only run strategies that haven't already been completed
    const completedSet = new Set(existingSummaries.map(s => s.strategy))
    const toRun = [...selectedStrategies].filter(s => !completedSet.has(s))
    if (toRun.length === 0) return

    setGenerating(true)
    setGenerateError(null)
    let lastStrategy = ''

    for (const strategy of toRun) {
      setCurrentlyGenerating(strategy)
      lastStrategy = strategy
      try {
        await summariesApi.generate({
          source_id: selectedSourceId,
          strategy,
        })
        await refreshSummaries()
        setActiveResultTab(strategy)
      } catch (err) {
        setGenerateError(
          `${strategy}: ${err instanceof Error ? err.message : 'Generation failed'}`
        )
        break
      }
    }

    setCurrentlyGenerating(null)
    setGenerating(false)
  }

  // Strategies that already have results for this source
  const completedStrategies = new Set(existingSummaries.map(s => s.strategy))

  // Count how many selected strategies still need to run
  const runnableCount = [...selectedStrategies].filter(s => !completedStrategies.has(s)).length

  // If extraction not complete, show locked state
  if (!extractionComplete) {
    return (
      <Card className="flex-1 flex flex-col min-h-0">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 shrink-0">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <AlignLeft className="h-4 w-4" />
            Summarization
          </CardTitle>
          <Badge variant="outline">Locked</Badge>
        </CardHeader>
        <CardContent className="flex items-center justify-center flex-1">
          <p className="text-sm text-muted-foreground">
            Summaries available after extraction completes.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="flex-1 flex flex-col min-h-0">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 shrink-0">
        <CardTitle className="text-base font-medium flex items-center gap-2">
          <AlignLeft className="h-4 w-4" />
          Summarization
        </CardTitle>
        <StatusBadge status={status} />
      </CardHeader>
      <CardContent className="flex-1 flex flex-col min-h-0">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr_2fr] gap-3 flex-1 min-h-0">
          {/* Left pane: source list */}
          <SourceListPane
            files={files}
            selectedIndex={selectedFileIndex}
            onSelect={setSelectedFileIndex}
          />

          {/* Middle pane: strategy picker */}
          <StrategyPane
            strategies={strategies}
            selectedStrategies={selectedStrategies}
            onToggleStrategy={handleToggleStrategy}
            completedStrategies={completedStrategies}
            generating={generating}
            currentlyGenerating={currentlyGenerating}
            generateError={generateError}
            canRun={!!selectedSourceId && !generating && runnableCount > 0}
            runnableCount={runnableCount}
            onRun={handleRun}
          />

          {/* Right pane: tabbed results */}
          <ResultsPane
            summaries={existingSummaries}
            activeTab={activeResultTab}
            onTabChange={setActiveResultTab}
            detail={expandedDetail}
            loadingDetail={loadingDetail}
            expandedNodes={expandedNodes}
            onToggleNodes={() => setExpandedNodes(v => !v)}
            currentlyGenerating={currentlyGenerating}
          />
        </div>
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
/* Left pane — source list                                             */
/* ------------------------------------------------------------------ */

function SourceListPane({
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
        <p className="text-xs font-medium text-muted-foreground">Sources</p>
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
            <p className="text-xs text-muted-foreground px-2 py-3">No sources available</p>
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
/* Middle pane — strategy picker (checkboxes)                          */
/* ------------------------------------------------------------------ */

function StrategyPane({
  strategies,
  selectedStrategies,
  onToggleStrategy,
  completedStrategies,
  generating,
  currentlyGenerating,
  generateError,
  canRun,
  runnableCount,
  onRun,
}: {
  strategies: StrategyInfo[]
  selectedStrategies: Set<string>
  onToggleStrategy: (name: string) => void
  completedStrategies: Set<string>
  generating: boolean
  currentlyGenerating: string | null
  generateError: string | null
  canRun: boolean
  runnableCount: number
  onRun: () => void
}) {
  return (
    <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
      <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
        <p className="text-xs font-medium text-muted-foreground">Strategies</p>
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto p-3">
        {strategies.length === 0 ? (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin" />
            Loading strategies...
          </div>
        ) : (
          <div className="space-y-2">
            {strategies.map((strategy) => {
              const isCompleted = completedStrategies.has(strategy.name)
              const isGenerating = currentlyGenerating === strategy.name
              const isChecked = selectedStrategies.has(strategy.name)
              const isDisabled = !strategy.implemented || generating

              return (
                <div key={strategy.name} className="flex items-start space-x-2">
                  <Checkbox
                    id={`strategy-${strategy.name}`}
                    checked={isChecked}
                    onCheckedChange={() => onToggleStrategy(strategy.name)}
                    disabled={isDisabled}
                    className="mt-0.5"
                  />
                  <Label
                    htmlFor={`strategy-${strategy.name}`}
                    className={cn(
                      "text-xs leading-snug cursor-pointer flex-1",
                      !strategy.implemented && "text-muted-foreground/50 cursor-not-allowed"
                    )}
                  >
                    <span className="flex items-center gap-1.5">
                      <span className="font-medium">{strategy.name}</span>
                      {isCompleted && (
                        <CheckCircle2 className="h-3 w-3 text-green-500 shrink-0" />
                      )}
                      {isGenerating && (
                        <Loader2 className="h-3 w-3 animate-spin text-primary shrink-0" />
                      )}
                    </span>
                    <span className="block text-muted-foreground mt-0.5">
                      {strategy.implemented
                        ? (STRATEGY_DESCRIPTIONS[strategy.name] || 'Summarization strategy')
                        : '(coming soon)'}
                    </span>
                  </Label>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Run button */}
      <div className="px-3 py-2 border-t shrink-0 space-y-2">
        {generateError && (
          <p className="text-xs text-destructive">{generateError}</p>
        )}
        <Button
          type="button"
          size="sm"
          className="w-full gap-2"
          disabled={!canRun}
          onClick={onRun}
        >
          {generating ? (
            <>
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Running {currentlyGenerating}...
            </>
          ) : (
            <>
              <Play className="h-3.5 w-3.5" />
              {runnableCount > 0
                ? `Run ${runnableCount} ${runnableCount === 1 ? 'strategy' : 'strategies'}`
                : 'All selected already run'}
            </>
          )}
        </Button>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Right pane — tabbed results                                         */
/* ------------------------------------------------------------------ */

function ResultsPane({
  summaries,
  activeTab,
  onTabChange,
  detail,
  loadingDetail,
  expandedNodes,
  onToggleNodes,
  currentlyGenerating,
}: {
  summaries: SummaryListItem[]
  activeTab: string
  onTabChange: (tab: string) => void
  detail: SummaryDetail | null
  loadingDetail: boolean
  expandedNodes: boolean
  onToggleNodes: () => void
  currentlyGenerating: string | null
}) {
  if (summaries.length === 0 && !currentlyGenerating) {
    return (
      <div className="border rounded-md overflow-hidden flex flex-col min-h-0">
        <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
          <p className="text-xs font-medium text-muted-foreground">Results</p>
        </div>
        <div className="flex-1 flex items-center justify-center p-6">
          <p className="text-sm text-muted-foreground text-center">
            Select strategies and click Run to generate summaries.
          </p>
        </div>
      </div>
    )
  }

  // Build tab list: existing summaries + currently generating strategy
  const tabStrategies = [...new Set([
    ...summaries.map(s => s.strategy),
    ...(currentlyGenerating ? [currentlyGenerating] : []),
  ])]

  return (
    <Tabs
      value={activeTab || tabStrategies[0] || ''}
      onValueChange={onTabChange}
      className="border rounded-md overflow-hidden flex flex-col min-h-0 gap-0"
    >
      <div className="px-3 py-2 border-b bg-muted/50 shrink-0">
        <TabsList className="h-7">
          {tabStrategies.map(strategy => {
            const isGenerating = currentlyGenerating === strategy
            const summary = summaries.find(s => s.strategy === strategy)
            return (
              <TabsTrigger
                key={strategy}
                value={strategy}
                className="text-xs h-6 px-2.5 gap-1"
              >
                {isGenerating && !summary && (
                  <Loader2 className="h-3 w-3 animate-spin" />
                )}
                {strategy}
              </TabsTrigger>
            )
          })}
        </TabsList>
      </div>

      {tabStrategies.map(strategy => {
        const summary = summaries.find(s => s.strategy === strategy)
        const isGenerating = currentlyGenerating === strategy && !summary

        return (
          <TabsContent
            key={strategy}
            value={strategy}
            className="flex-1 min-h-0 overflow-y-auto mt-0"
          >
            {isGenerating ? (
              <div className="flex items-center gap-2 p-4 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Generating {strategy} summary...
              </div>
            ) : summary ? (
              <SummaryContent
                summary={summary}
                detail={activeTab === strategy ? detail : null}
                loadingDetail={activeTab === strategy ? loadingDetail : false}
                expandedNodes={expandedNodes}
                onToggleNodes={onToggleNodes}
              />
            ) : null}
          </TabsContent>
        )
      })}
    </Tabs>
  )
}

/* ------------------------------------------------------------------ */
/* Summary content — document summary + metadata + node tree            */
/* ------------------------------------------------------------------ */

function SummaryContent({
  summary,
  detail,
  loadingDetail,
  expandedNodes,
  onToggleNodes,
}: {
  summary: SummaryListItem
  detail: SummaryDetail | null
  loadingDetail: boolean
  expandedNodes: boolean
  onToggleNodes: () => void
}) {
  return (
    <div className="p-3 space-y-3">
      {/* Metadata badges */}
      <div className="flex items-center gap-1.5 flex-wrap">
        <Badge variant="secondary" className="text-[10px]">
          {summary.num_layers} layers
        </Badge>
        <Badge variant="secondary" className="text-[10px]">
          {summary.num_input_chunks} input chunks
        </Badge>
        <Badge variant="secondary" className="text-[10px]">
          {summary.node_count} nodes
        </Badge>
        <Badge variant="outline" className="text-[10px]">
          {new Date(summary.created).toLocaleDateString()}
        </Badge>
      </div>

      {/* Document summary */}
      {summary.document_summary ? (
        <div className="prose prose-sm dark:prose-invert max-w-none">
          <p className="text-xs text-muted-foreground whitespace-pre-wrap leading-relaxed">
            {summary.document_summary}
          </p>
        </div>
      ) : (
        <p className="text-xs text-muted-foreground italic">No document summary available.</p>
      )}

      {/* Node tree (collapsible) */}
      {loadingDetail ? (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin" />
          Loading details...
        </div>
      ) : detail && detail.summary_nodes.length > 0 ? (
        <div className="border-t pt-2">
          <button
            type="button"
            onClick={onToggleNodes}
            className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
          >
            {expandedNodes ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
            Summary Nodes ({detail.summary_nodes.length})
          </button>

          {expandedNodes && (
            <NodeTree nodes={detail.summary_nodes} />
          )}
        </div>
      ) : null}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Node tree — grouped by layer                                        */
/* ------------------------------------------------------------------ */

function NodeTree({ nodes }: { nodes: SummaryDetail['summary_nodes'] }) {
  // Group by layer
  const byLayer = new Map<number, typeof nodes>()
  for (const node of nodes) {
    const layer = node.layer
    if (!byLayer.has(layer)) byLayer.set(layer, [])
    byLayer.get(layer)!.push(node)
  }

  const sortedLayers = [...byLayer.keys()].sort((a, b) => a - b)

  return (
    <div className="mt-2 space-y-3">
      {sortedLayers.map(layer => (
        <div key={layer}>
          <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">
            Layer {layer} ({byLayer.get(layer)!.length} nodes)
          </p>
          <div className="space-y-1">
            {byLayer.get(layer)!.map((node, i) => (
              <div
                key={node.node_id || i}
                className="border rounded p-2 bg-muted/30"
              >
                {node.section_title && (
                  <p className="text-[10px] font-medium text-primary mb-0.5">
                    {node.section_title}
                  </p>
                )}
                <p className="text-[11px] text-muted-foreground leading-relaxed">
                  {node.text.slice(0, 500)}
                  {node.text.length > 500 && '...'}
                </p>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
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
