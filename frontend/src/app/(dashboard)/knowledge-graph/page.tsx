'use client'

import { useState, useCallback, useRef } from 'react'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import { EmptyState } from '@/components/common/EmptyState'
import { PageBody } from '@/components/layout/PageBody'
import { PageHeader } from '@/components/layout/PageHeader'
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
import { Share2, Search, X, ChevronLeft, ChevronRight, GitMerge, ClipboardList, FileText, ArrowUpDown } from 'lucide-react'
import type { Entity, EntityDetail } from '@/lib/api/knowledge-graph'
import type { TriageStatus } from '@/lib/api/triage'
import { ConfidenceBar } from '@/components/knowledge-graph/ConfidenceBar'
import { ConfidenceFilter } from '@/components/knowledge-graph/ConfidenceFilter'
import { StatusFilter } from '@/components/knowledge-graph/StatusFilter'
import { StatusBadge } from '@/components/knowledge-graph/StatusBadge'
import { EntityTypeLabel } from '@/components/knowledge-graph/EntityTypeLabel'
import { ExternalIdBadges } from '@/components/resolution/ExternalIdBadges'
import { Switch } from '@/components/ui/switch'
import { useSetEntityOverride } from '@/lib/hooks/use-triage'
import { useMinWidth } from '@/lib/hooks/use-media-query'
import { Sheet, SheetContent, SheetDescription, SheetTitle } from '@/components/ui/sheet'

const SigmaGraphView = dynamic(
  () => import('./components/SigmaGraphView'),
  { ssr: false, loading: () => <div className="flex h-96 items-center justify-center"><LoadingSpinner /></div> }
)

const DocumentGraphView = dynamic(
  () => import('./components/DocumentGraphView'),
  { ssr: false, loading: () => <div className="flex h-96 items-center justify-center"><LoadingSpinner /></div> }
)

export default function KnowledgeGraphPage() {
  const [entityTypeFilter, setEntityTypeFilter] = useState<string | undefined>(undefined)
  // WB.3: open on the active graph, best connected first, not on the archive.
  const [statusFilter, setStatusFilter] = useState<TriageStatus | undefined>('active')
  const [sort, setSort] = useState<'name' | 'degree'>('degree')
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
  // WB.5: cards below md; the detail panel is a sheet below lg.
  const isMd = useMinWidth('md')
  const isLg = useMinWidth('lg')

  const { data: entityTypes } = useEntityTypes()
  const { data: paginatedEntities, isLoading } = useEntities({
    limit: pageSize,
    offset: page * pageSize,
    entity_type: entityTypeFilter,
    status: statusFilter,
    sort,
  })
  const { data: searchResults } = useSearchEntities(searchQuery)

  // The latest detail read; closing (or a newer click) moves it on, so an
  // answer that arrives later is dropped instead of reopening the details.
  const detailRequest = useRef(0)

  const handleEntityClick = useCallback(async (entityId: string) => {
    const request = ++detailRequest.current
    setLoadingDetail(true)
    try {
      const detail = await knowledgeGraphApi.getEntity(entityId)
      if (request === detailRequest.current) setSelectedEntity(detail)
    } catch {
      if (request === detailRequest.current) setSelectedEntity(null)
    } finally {
      if (request === detailRequest.current) setLoadingDetail(false)
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

  const nameSort = (
    <Button
      variant="ghost"
      size="sm"
      className="-ml-2 h-8 px-2"
      onClick={() => { setSort('name'); setPage(0) }}
      aria-pressed={isMd ? undefined : sort === 'name'}
      data-testid="kg-sort-name"
    >
      Name
      <ArrowUpDown className={`ml-2 h-3 w-3 ${sort === 'name' ? 'opacity-100' : 'opacity-30'}`} />
    </Button>
  )
  const degreeSort = (
    <Button
      variant="ghost"
      size="sm"
      className="-ml-2 h-8 px-2"
      onClick={() => { setSort('degree'); setPage(0) }}
      aria-pressed={isMd ? undefined : sort === 'degree'}
      data-testid="kg-sort-degree"
    >
      Degree
      <ArrowUpDown className={`ml-2 h-3 w-3 ${sort === 'degree' ? 'opacity-100' : 'opacity-30'}`} />
    </Button>
  )

  const closeDetail = () => {
    detailRequest.current += 1
    setSelectedEntity(null)
    setLoadingDetail(false)
  }

  // The panel's body, shown in the aside (lg and up) or the sheet (below).
  const detailBody = (selectedEntity: EntityDetail) => (
    <>
      <div className="mb-4">
        <EntityTypeLabel
          canonical={selectedEntity.entity_type}
          display={selectedEntity.primary_type}
        />
      </div>

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
    </>
  )


  return (
    <div className="flex min-h-0 flex-1">
      <PageBody fill>
        <PageHeader
          title="Knowledge Graph"
          actions={
            <>
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
            </>
          }
        />

        <div className="flex flex-wrap items-center gap-3">
          <div className="relative min-w-56 flex-1 max-w-sm">
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
                aria-label="Clear search"
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

          {/* max-w-full: on a phone the slider narrows to the row. */}
          <ConfidenceFilter onChange={setConfidenceThreshold} className="max-w-full" />
        </div>

        <div className="min-h-0 flex-1 overflow-hidden">
          <Tabs defaultValue="table" className="h-full flex flex-col">
            {/* Scrolls sideways when the three tabs do not fit (a phone). */}
            <TabsList className="max-w-full justify-start overflow-x-auto">
              <TabsTrigger value="table">Table</TabsTrigger>
              <TabsTrigger value="graph">Graph</TabsTrigger>
              <TabsTrigger value="documents">
                <FileText className="h-3.5 w-3.5 mr-1.5" />
                Document Graph
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
                  {isMd ? (
                  <div className="border rounded-md overflow-auto">
                    <table className="w-full text-sm">
                      <thead className="sticky top-0 bg-background">
                        <tr className="border-b bg-muted/50">
                          <th
                            className="text-left p-3 font-medium"
                            aria-sort={sort === 'name' ? 'ascending' : undefined}
                          >
                            {nameSort}
                          </th>
                          <th className="text-left p-3 font-medium">Type</th>
                          <th className="text-left p-3 font-medium">Status</th>
                          <th
                            className="text-left p-3 font-medium"
                            aria-sort={sort === 'degree' ? 'descending' : undefined}
                          >
                            {degreeSort}
                          </th>
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
                              <EntityTypeLabel canonical={entity.entity_type} display={entity.primary_type} />
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
                  ) : (
                  <div>
                    <div className="mb-2 flex flex-wrap items-center gap-2 pl-2">
                      <span className="text-sm text-muted-foreground">Sort by</span>
                      {nameSort}
                      {degreeSort}
                    </div>
                    <ul className="space-y-2" aria-label="Entities">
                      {displayEntities.map((entity: Entity) => (
                        <li key={entity.id}>
                          {/* A button: the card opens the details as a row does,
                              and by keyboard too. */}
                          <button
                            type="button"
                            data-testid={`entity-card-${entity.id}`}
                            data-entity-name={entity.name}
                            className="w-full space-y-2 rounded-md border p-3 text-left text-sm transition-colors hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                            onClick={() => handleEntityClick(entity.id)}
                          >
                            <span className="block font-medium break-words">{entity.name}</span>
                            <span className="flex flex-wrap items-center gap-2">
                              <EntityTypeLabel canonical={entity.entity_type} display={entity.primary_type} />
                              {entity.status && (
                                <StatusBadge
                                  status={entity.status}
                                  manualOverride={entity.manual_override}
                                />
                              )}
                            </span>
                            <span className="flex flex-wrap items-center gap-x-4 gap-y-1 text-muted-foreground tabular-nums">
                              <span>
                                Degree {typeof entity.structural_degree === 'number' ? entity.structural_degree : '—'}
                              </span>
                              <span>
                                Docs {typeof entity.doc_count === 'number' ? entity.doc_count : '—'}
                              </span>
                              {typeof entity.confidence === 'number' && (
                                <ConfidenceBar value={entity.confidence} />
                              )}
                            </span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                  )}

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
                          aria-label="Previous page"
                        >
                          <ChevronLeft className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={page >= totalPages - 1}
                          onClick={() => setPage((p) => p + 1)}
                          aria-label="Next page"
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

            <TabsContent value="documents" className="flex-1 overflow-hidden mt-4">
              <DocumentGraphView onNodeClick={handleEntityClick} />
            </TabsContent>

          </Tabs>
        </div>
      </PageBody>

      {/* Entity details: a side panel from lg up, a sheet below it (WB.5). */}
      {isLg ? (
        <>
          {selectedEntity && (
            <aside
              className="w-96 shrink-0 border-l overflow-y-auto bg-background"
              aria-label="Entity details"
              data-testid="entity-detail-panel"
            >
              <div className="p-4">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold truncate">{selectedEntity.name}</h2>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={closeDetail}
                    aria-label="Close details"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
                {detailBody(selectedEntity)}
              </div>
            </aside>
          )}

          {loadingDetail && (
            <div className="w-96 shrink-0 border-l flex items-center justify-center">
              <LoadingSpinner />
            </div>
          )}
        </>
      ) : (
        <Sheet
          open={!!selectedEntity || loadingDetail}
          onOpenChange={(open) => { if (!open) closeDetail() }}
        >
          <SheetContent
            side="right"
            className="w-full gap-0 overflow-y-auto p-4 sm:max-w-md"
            data-testid="entity-detail-sheet"
          >
            <SheetTitle className="mb-4 pr-10 text-lg break-words">
              {selectedEntity?.name ?? 'Loading entity'}
            </SheetTitle>
            <SheetDescription className="sr-only">Entity details</SheetDescription>
            {selectedEntity ? (
              detailBody(selectedEntity)
            ) : (
              <div className="flex flex-1 items-center justify-center">
                <LoadingSpinner />
              </div>
            )}
          </SheetContent>
        </Sheet>
      )}
    </div>
  )
}
