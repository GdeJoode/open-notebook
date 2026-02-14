'use client'

import { useState } from 'react'
import { AppShell } from '@/components/layout/AppShell'
import { EmptyState } from '@/components/common/EmptyState'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { useStrategies, useSummaries, useSummary } from '@/lib/hooks/use-summaries'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import { AlignLeft, ArrowLeft, ChevronDown } from 'lucide-react'
import type { SummaryListItem, StrategyInfo } from '@/lib/api/summaries'
import { formatDistanceToNow } from 'date-fns'

export default function SummariesPage() {
  const [selectedSummaryId, setSelectedSummaryId] = useState<string | null>(null)

  return (
    <AppShell>
      <div className="flex-1 overflow-y-auto">
        <div className="p-6">
          <div className="max-w-6xl">
            {selectedSummaryId ? (
              <SummaryDetail
                id={selectedSummaryId}
                onBack={() => setSelectedSummaryId(null)}
              />
            ) : (
              <SummaryListView onSelect={setSelectedSummaryId} />
            )}
          </div>
        </div>
      </div>
    </AppShell>
  )
}

function SummaryListView({ onSelect }: { onSelect: (id: string) => void }) {
  const { data: strategies, isLoading: loadingStrategies } = useStrategies()
  const { data: summaries, isLoading: loadingSummaries } = useSummaries()

  return (
    <>
      <h1 className="text-2xl font-bold mb-6">Summaries</h1>

      {/* Strategies section */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold mb-3">Available Strategies</h2>
        {loadingStrategies ? (
          <LoadingSpinner />
        ) : (
          <div className="grid gap-3 md:grid-cols-3 lg:grid-cols-4">
            {strategies?.map((s: StrategyInfo) => (
              <Card key={s.name} className="p-4">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-sm">{s.name}</span>
                  <Badge variant={s.implemented ? 'default' : 'secondary'} className="text-xs">
                    {s.implemented ? 'Available' : 'Coming soon'}
                  </Badge>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* Summaries list */}
      <h2 className="text-lg font-semibold mb-3">Generated Summaries</h2>
      {loadingSummaries ? (
        <div className="flex h-32 items-center justify-center">
          <LoadingSpinner />
        </div>
      ) : !summaries || summaries.length === 0 ? (
        <EmptyState
          icon={AlignLeft}
          title="No summaries yet"
          description="Generate summaries from your sources using the available strategies."
        />
      ) : (
        <div className="space-y-3">
          {summaries.map((s: SummaryListItem) => (
            <Card
              key={s.id}
              className="p-4 cursor-pointer hover:border-primary/50 transition-colors"
              onClick={() => onSelect(s.id)}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  <Badge variant="outline">{s.strategy}</Badge>
                  <span className="text-sm text-muted-foreground">
                    Source: {s.source_id}
                  </span>
                </div>
                <span className="text-xs text-muted-foreground">
                  {s.created && formatDistanceToNow(new Date(s.created), { addSuffix: true })}
                </span>
              </div>
              {s.document_summary && (
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {s.document_summary}
                </p>
              )}
              <div className="flex gap-3 mt-2 text-xs text-muted-foreground">
                <span>{s.node_count} nodes</span>
                <span>{s.num_layers} layers</span>
                <span>{s.num_input_chunks} input chunks</span>
              </div>
            </Card>
          ))}
        </div>
      )}
    </>
  )
}

function SummaryDetail({ id, onBack }: { id: string; onBack: () => void }) {
  const { data: summary, isLoading, error } = useSummary(id)

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <LoadingSpinner />
      </div>
    )
  }

  if (error || !summary) {
    return <p className="text-red-500">Failed to load summary</p>
  }

  // Group nodes by layer
  const nodesByLayer = new Map<number, typeof summary.summary_nodes>()
  for (const node of summary.summary_nodes) {
    const layer = node.layer
    if (!nodesByLayer.has(layer)) {
      nodesByLayer.set(layer, [])
    }
    nodesByLayer.get(layer)!.push(node)
  }
  const sortedLayers = [...nodesByLayer.keys()].sort((a, b) => a - b)

  return (
    <>
      <div className="flex items-center gap-3 mb-6">
        <Button variant="ghost" size="sm" onClick={onBack}>
          <ArrowLeft className="h-4 w-4 mr-1" />
          Back
        </Button>
        <h1 className="text-2xl font-bold">Summary Detail</h1>
        <Badge variant="outline">{summary.strategy}</Badge>
      </div>

      <div className="flex gap-3 mb-4 text-sm text-muted-foreground">
        <span>Source: {summary.source_id}</span>
        <span>{summary.num_layers} layers</span>
        <span>{summary.num_input_chunks} input chunks</span>
        <span>{summary.node_count} nodes</span>
      </div>

      {/* Document summary */}
      {summary.document_summary && (
        <Card className="p-5 mb-6">
          <h3 className="font-semibold mb-2">Document Summary</h3>
          <p className="text-sm leading-relaxed whitespace-pre-wrap">
            {summary.document_summary}
          </p>
        </Card>
      )}

      {/* Summary nodes by layer */}
      {sortedLayers.length > 0 && (
        <div>
          <h3 className="font-semibold mb-3">Summary Nodes</h3>
          {sortedLayers.map((layer) => (
            <div key={layer} className="mb-4">
              <h4 className="text-sm font-medium text-muted-foreground mb-2">
                Layer {layer} ({nodesByLayer.get(layer)!.length} nodes)
              </h4>
              <Accordion type="multiple" className="space-y-2">
                {nodesByLayer.get(layer)!.map((node, idx) => (
                  <AccordionItem
                    key={node.node_id || `${layer}-${idx}`}
                    value={node.node_id || `${layer}-${idx}`}
                    className="border rounded-lg px-4"
                  >
                    <AccordionTrigger className="hover:no-underline">
                      <div className="flex items-center gap-2 text-left">
                        {node.section_title && (
                          <span className="font-medium text-sm">{node.section_title}</span>
                        )}
                        <span className="text-xs text-muted-foreground truncate max-w-md">
                          {node.text.slice(0, 100)}...
                        </span>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent>
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">
                        {node.text}
                      </p>
                      {node.source_chunk_ids.length > 0 && (
                        <div className="mt-2 text-xs text-muted-foreground">
                          Source chunks: {node.source_chunk_ids.join(', ')}
                        </div>
                      )}
                    </AccordionContent>
                  </AccordionItem>
                ))}
              </Accordion>
            </div>
          ))}
        </div>
      )}
    </>
  )
}
