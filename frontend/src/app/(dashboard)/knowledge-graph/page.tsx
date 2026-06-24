'use client'

import { useState, useCallback } from 'react'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import { AppShell } from '@/components/layout/AppShell'
import { EmptyState } from '@/components/common/EmptyState'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { useEntities, useEntityTypes, useSearchEntities } from '@/lib/hooks/use-knowledge-graph'
import { knowledgeGraphApi } from '@/lib/api/knowledge-graph'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Share2, Search, X, ChevronLeft, ChevronRight, GitMerge, ClipboardList } from 'lucide-react'
import type { Entity, EntityDetail } from '@/lib/api/knowledge-graph'
import type { TriageStatus } from '@/lib/api/triage'
import { ResolutionLogTab } from './components/ResolutionLogTab'
import { ConfidenceBar } from '@/components/knowledge-graph/ConfidenceBar'
import { ConfidenceFilter } from '@/components/knowledge-graph/ConfidenceFilter'
import { StatusFilter } from '@/components/knowledge-graph/StatusFilter'
import { StatusBadge } from '@/components/knowledge-graph/StatusBadge'
import { ExternalIdBadges } from '@/components/resolution/ExternalIdBadges'
import { Switch } from '@/components/ui/switch'
import { useSetEntityOverride } from '@/lib/hooks/use-triage'

const SigmaGraphView = dynamic(
  () => import('./components/SigmaGraphView'),
  { ssr: false, loading: () => <div className="flex h-96 items-center justify-center"><LoadingSpinner /></div> }
)

export default function KnowledgeGraphPage() {
  const [entityTypeFilter, setEntityTypeFilter] = useState<string | undefined>(undefined)
  const [statusFilter, setStatusFilter] = useState<TriageStatus | undefined>(undefined)
  const [searchQuery, setSearchQuery] = useState('')
  const [page, setPage] = useState(0)
  const [selectedEntity, setSelectedEntity] = useState<EntityDetail | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const setOverride = useSetEntityOverride()
  // Confidence filter threshold. Driven by `<ConfidenceFilter>` which
  // persists the value to localStorage and pushes the restored value
  // back here on mount.
  const [confidenceThreshold, setConfidenceThreshold] = useState(0)
  const pageSize = 50

  const { data: entityTypes } = useEntityTypes()
  const { data: paginatedEntities, isLoading } = useEntities({
    limit: pageSize,
    offset: page * pageSize,
    entity_type: entityTypeFilter,
    status: statusFilter,
  })
  const { data: searchResults } = useSearchEntities(searchQuery)

  const handleEntityClick = useCallback(async (entityId: string) => {
    setLoadingDetail(true)
    try {
      const detail = await knowledgeGraphApi.getEntity(entityId)
      setSelectedEntity(detail)
    } catch {
      setSelectedEntity(null)
    } finally {
      setLoadingDetail(false)
    }
  }, [])

  const rawEntities = searchQuery ? searchResults : paginatedEntities?.items
  // Client-side confidence filter. Entities without a confidence field
  // (legacy rows) are kept when the threshold is 0; once the user raises
  // the slider they are filtered out, matching the "hide low / unknown"
  // expectation reviewers have voiced in retros.
  const displayEntities = rawEntities?.filter((e) => {
    if (confidenceThreshold <= 0) return true
    if (typeof e.confidence !== 'number') return false
    return e.confidence >= confidenceThreshold
  })
  const totalEntities = paginatedEntities?.total ?? 0
  const totalPages = Math.ceil(totalEntities / pageSize)

  return (
    <AppShell>
      <div className="flex h-full">
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="p-6 pb-0">
            <div className="flex items-center justify-between mb-4">
              <h1 className="text-2xl font-bold">Knowledge Graph</h1>
              <div className="flex items-center gap-2">
                <Link href="/knowledge-graph/triage">
                  <Button variant="outline" size="sm" className="gap-1.5">
                    <ClipboardList className="h-3.5 w-3.5" />
                    Triage queue
                  </Button>
                </Link>
                <Link href="/knowledge-graph/resolution">
                  <Button variant="outline" size="sm" className="gap-1.5">
                    <GitMerge className="h-3.5 w-3.5" />
                    Review duplicates
                  </Button>
                </Link>
              </div>
            </div>

            <div className="flex items-center gap-3 mb-4">
              <div className="relative flex-1 max-w-sm">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search entities..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
                {searchQuery && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="absolute right-1 top-1/2 -translate-y-1/2 h-6 w-6 p-0"
                    onClick={() => setSearchQuery('')}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                )}
              </div>

              <Select
                value={entityTypeFilter ?? 'all'}
                onValueChange={(v) => {
                  setEntityTypeFilter(v === 'all' ? undefined : v)
                  setPage(0)
                }}
              >
                <SelectTrigger className="w-48">
                  <SelectValue placeholder="All types" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All types</SelectItem>
                  {entityTypes?.map((t) => (
                    <SelectItem key={t.entity_type} value={t.entity_type}>
                      {t.entity_type} ({t.count})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <StatusFilter
                value={statusFilter}
                onChange={(v) => {
                  setStatusFilter(v)
                  setPage(0)
                }}
              />

              <ConfidenceFilter onChange={setConfidenceThreshold} />
            </div>
          </div>

          <div className="flex-1 overflow-hidden px-6 pb-6">
            <Tabs defaultValue="table" className="h-full flex flex-col">
              <TabsList>
                <TabsTrigger value="table">Table</TabsTrigger>
                <TabsTrigger value="graph">Graph</TabsTrigger>
                <TabsTrigger value="resolution">
                  <GitMerge className="h-3.5 w-3.5 mr-1.5" />
                  Resolution Log
                </TabsTrigger>
              </TabsList>

              <TabsContent value="table" className="flex-1 overflow-auto mt-4">
                {isLoading ? (
                  <div className="flex h-64 items-center justify-center">
                    <LoadingSpinner />
                  </div>
                ) : !displayEntities || displayEntities.length === 0 ? (
                  <EmptyState
                    icon={Share2}
                    title="No entities found"
                    description="No entities have been extracted yet. Run entity extraction on a source to populate the knowledge graph."
                  />
                ) : (
                  <>
                    <div className="border rounded-md overflow-auto">
                      <table className="w-full text-sm">
                        <thead className="sticky top-0 bg-background">
                          <tr className="border-b bg-muted/50">
                            <th className="text-left p-3 font-medium">Name</th>
                            <th className="text-left p-3 font-medium">Type</th>
                            <th className="text-left p-3 font-medium">Status</th>
                            <th className="text-left p-3 font-medium">Degree</th>
                            <th className="text-left p-3 font-medium">Docs</th>
                            <th className="text-left p-3 font-medium">Confidence</th>
                          </tr>
                        </thead>
                        <tbody>
                          {displayEntities.map((entity: Entity) => (
                            <tr
                              key={entity.id}
                              data-testid={`entity-row-${entity.id}`}
                              data-entity-name={entity.name}
                              className="border-b cursor-pointer hover:bg-muted/50 transition-colors"
                              onClick={() => handleEntityClick(entity.id)}
                            >
                              <td className="p-3 font-medium">{entity.name}</td>
                              <td className="p-3">
                                <Badge variant="secondary">{entity.entity_type}</Badge>
                              </td>
                              <td className="p-3">
                                {entity.status ? (
                                  <StatusBadge
                                    status={entity.status}
                                    manualOverride={entity.manual_override}
                                  />
                                ) : (
                                  <span className="text-xs text-muted-foreground">—</span>
                                )}
                              </td>
                              <td className="p-3 text-muted-foreground tabular-nums">
                                {typeof entity.structural_degree === 'number'
                                  ? entity.structural_degree
                                  : '—'}
                              </td>
                              <td className="p-3 text-muted-foreground tabular-nums">
                                {typeof entity.doc_count === 'number'
                                  ? entity.doc_count
                                  : '—'}
                              </td>
                              <td className="p-3">
                                {typeof entity.confidence === 'number' ? (
                                  <ConfidenceBar value={entity.confidence} />
                                ) : (
                                  <span className="text-xs text-muted-foreground">—</span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {!searchQuery && totalPages > 1 && (
                      <div className="flex items-center justify-between mt-4">
                        <span className="text-sm text-muted-foreground">
                          Page {page + 1} of {totalPages} ({totalEntities} total)
                        </span>
                        <div className="flex gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            disabled={page === 0}
                            onClick={() => setPage((p) => p - 1)}
                          >
                            <ChevronLeft className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            disabled={page >= totalPages - 1}
                            onClick={() => setPage((p) => p + 1)}
                          >
                            <ChevronRight className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </TabsContent>

              <TabsContent value="graph" className="flex-1 mt-4">
                <SigmaGraphView
                  entityTypeFilter={entityTypeFilter}
                  onNodeClick={handleEntityClick}
                />
              </TabsContent>

              <TabsContent value="resolution" className="flex-1 overflow-auto mt-4">
                <ResolutionLogTab />
              </TabsContent>
            </Tabs>
          </div>
        </div>

        {/* Entity Detail Side Panel */}
        {selectedEntity && (
          <div className="w-96 border-l overflow-y-auto bg-background">
            <div className="p-4">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold truncate">{selectedEntity.name}</h2>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedEntity(null)}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>

              <Badge variant="secondary" className="mb-4">
                {selectedEntity.entity_type}
              </Badge>

              {/* Triage: status + degree/doc-count + manual_override toggle (Q.5) */}
              {(() => {
                const detail = selectedEntity as Record<string, unknown>
                const status = detail.status as string | undefined
                const manualOverride = Boolean(detail.manual_override)
                const degree = detail.structural_degree as number | undefined
                const docCount = detail.doc_count as number | undefined
                if (!status && degree === undefined && docCount === undefined) {
                  return null
                }
                return (
                  <div className="mb-4 rounded-md border p-3 space-y-3" data-testid="triage-panel">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        {status && (
                          <StatusBadge status={status} manualOverride={manualOverride} />
                        )}
                      </div>
                      <div className="flex flex-col gap-3 text-xs text-muted-foreground">
                        {typeof degree === 'number' && (
                          <span>Degree: {degree}</span>
                        )}
                        {typeof docCount === 'number' && (
                          <span>Docs: {docCount}</span>
                        )}
                      </div>
                    </div>
                    <label className="flex items-center justify-between gap-3 text-sm">
                      <span id={`override-label-${selectedEntity.id}`}>
                        Pin status (manual override)
                      </span>
                      <Switch
                        checked={manualOverride}
                        disabled={setOverride.isPending}
                        aria-labelledby={`override-label-${selectedEntity.id}`}
                        data-testid="override-toggle"
                        onCheckedChange={(checked) => {
                          setOverride.mutate(
                            {
                              entityId: selectedEntity.id,
                              body: checked
                                ? {
                                    manual_override: true,
                                    status:
                                      status === 'active' ? 'active' : 'reference',
                                  }
                                : { manual_override: false },
                            },
                            {
                              onSuccess: () => {
                                // Optimistically reflect the toggle in the open panel.
                                setSelectedEntity((prev) =>
                                  prev
                                    ? ({
                                        ...prev,
                                        manual_override: checked,
                                      } as EntityDetail)
                                    : prev,
                                )
                              },
                            },
                          )
                        }}
                      />
                    </label>
                  </div>
                )
              })()}

              {/* External IDs (TOOI/DOI) from K.4 vocabulary reconciliation */}
              {(() => {
                const externalIds = (selectedEntity as Record<string, unknown>)
                  .external_ids as string[] | undefined
                return <ExternalIdBadges externalIds={externalIds} className="mb-3" />
              })()}

              {/* Alias count — full management on the resolution hub */}
              {(() => {
                const aliases = (selectedEntity as Record<string, unknown>)
                  .aliases as string[] | undefined
                if (!aliases || aliases.length === 0) return null
                return (
                  <p className="text-xs text-muted-foreground mb-3">
                    {aliases.length} alias{aliases.length === 1 ? '' : 'es'}:{' '}
                    {aliases.slice(0, 3).join(', ')}
                    {aliases.length > 3 ? '…' : ''}
                  </p>
                )
              })()}

              {/* Properties: weight, confidence */}
              <div className="mt-3 flex gap-3 text-xs text-muted-foreground">
                {selectedEntity.weight > 0 && (
                  <span>Weight: {selectedEntity.weight}</span>
                )}
                {typeof (selectedEntity as Record<string, unknown>).confidence === 'number' && (
                  <span>Confidence: {((selectedEntity as Record<string, unknown>).confidence as number * 100).toFixed(0)}%</span>
                )}
              </div>

              {/* Provenance: which sources contributed this entity */}
              {Array.isArray((selectedEntity as Record<string, unknown>).source_ids) &&
                ((selectedEntity as Record<string, unknown>).source_ids as string[]).length > 0 && (
                <div className="mt-4">
                  <h3 className="text-sm font-medium mb-2">Sources</h3>
                  <div className="space-y-1">
                    {((selectedEntity as Record<string, unknown>).source_ids as string[]).map((sid) => (
                      <a
                        key={sid}
                        href={`/sources/${sid}`}
                        className="block text-xs text-primary hover:underline truncate"
                      >
                        {sid}
                      </a>
                    ))}
                  </div>
                </div>
              )}

              {/* Merge history: surface forms that were merged into this entity */}
              {(() => {
                const props = (selectedEntity as Record<string, unknown>).properties as Record<string, unknown> | undefined
                const mergedFrom = props?.merged_from as string[] | undefined
                if (!mergedFrom || mergedFrom.length === 0) return null
                return (
                  <div className="mt-4">
                    <h3 className="text-sm font-medium mb-2">Merged From</h3>
                    <div className="flex flex-wrap gap-1">
                      {mergedFrom.map((variant) => (
                        <Badge key={variant} variant="outline" className="text-xs">
                          {variant}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )
              })()}

              {selectedEntity.relations && selectedEntity.relations.length > 0 && (
                <div className="mt-4">
                  <h3 className="text-sm font-medium mb-2">
                    Relations ({selectedEntity.relations.length})
                  </h3>
                  <div className="space-y-2">
                    {selectedEntity.relations.map((rel) => (
                      <Card
                        key={rel.id}
                        data-testid={`relation-row-${rel.id}`}
                        className="p-3"
                      >
                        <div className="text-xs text-muted-foreground mb-1">
                          {rel.relation_type}
                        </div>
                        <div className="text-sm">
                          {rel.source === selectedEntity.id ? (
                            <span
                              className="text-primary cursor-pointer hover:underline"
                              onClick={() => handleEntityClick(rel.target)}
                            >
                              {rel.target}
                            </span>
                          ) : (
                            <span
                              className="text-primary cursor-pointer hover:underline"
                              onClick={() => handleEntityClick(rel.source)}
                            >
                              {rel.source}
                            </span>
                          )}
                        </div>
                        {typeof rel.confidence === 'number' && (
                          <div className="mt-2">
                            <ConfidenceBar value={rel.confidence} />
                          </div>
                        )}
                      </Card>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {loadingDetail && (
          <div className="w-96 border-l flex items-center justify-center">
            <LoadingSpinner />
          </div>
        )}
      </div>
    </AppShell>
  )
}
