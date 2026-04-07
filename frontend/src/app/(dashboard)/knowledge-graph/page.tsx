'use client'

import { useState, useCallback } from 'react'
import dynamic from 'next/dynamic'
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
import { Share2, Search, X, ChevronLeft, ChevronRight, GitMerge } from 'lucide-react'
import type { Entity, EntityDetail } from '@/lib/api/knowledge-graph'
import { ResolutionLogTab } from './components/ResolutionLogTab'

const SigmaGraphView = dynamic(
  () => import('./components/SigmaGraphView'),
  { ssr: false, loading: () => <div className="flex h-96 items-center justify-center"><LoadingSpinner /></div> }
)

export default function KnowledgeGraphPage() {
  const [entityTypeFilter, setEntityTypeFilter] = useState<string | undefined>(undefined)
  const [searchQuery, setSearchQuery] = useState('')
  const [page, setPage] = useState(0)
  const [selectedEntity, setSelectedEntity] = useState<EntityDetail | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const pageSize = 50

  const { data: entityTypes } = useEntityTypes()
  const { data: paginatedEntities, isLoading } = useEntities({
    limit: pageSize,
    offset: page * pageSize,
    entity_type: entityTypeFilter,
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

  const displayEntities = searchQuery ? searchResults : paginatedEntities?.items
  const totalEntities = paginatedEntities?.total ?? 0
  const totalPages = Math.ceil(totalEntities / pageSize)

  return (
    <AppShell>
      <div className="flex h-full">
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="p-6 pb-0">
            <h1 className="text-2xl font-bold mb-4">Knowledge Graph</h1>

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
                            <th className="text-left p-3 font-medium">Weight</th>
                          </tr>
                        </thead>
                        <tbody>
                          {displayEntities.map((entity: Entity) => (
                            <tr
                              key={entity.id}
                              className="border-b cursor-pointer hover:bg-muted/50 transition-colors"
                              onClick={() => handleEntityClick(entity.id)}
                            >
                              <td className="p-3 font-medium">{entity.name}</td>
                              <td className="p-3">
                                <Badge variant="secondary">{entity.entity_type}</Badge>
                              </td>
                              <td className="p-3 text-muted-foreground">
                                {entity.weight}
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
                      <Card key={rel.id} className="p-3">
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
