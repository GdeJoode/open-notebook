'use client'

import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { useInfiniteQuery, useQuery, useQueryClient } from '@tanstack/react-query'
import { sourcesApi } from '@/lib/api/sources'
import { QUERY_KEYS } from '@/lib/api/query-client'
import { ATTENTION_QUERY_KEY } from '@/lib/hooks/use-attention'
import { SourceListResponse } from '@/lib/types/api'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { EmptyState } from '@/components/common/EmptyState'
import { ConfirmDialog } from '@/components/common/ConfirmDialog'
import { PageBody } from '@/components/layout/PageBody'
import { PageHeader } from '@/components/layout/PageHeader'
import { FileText, Link as LinkIcon, Upload, AlignLeft, Trash2, ArrowUpDown, Plus } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { PipelineStatus } from '@/components/sources/pipeline/PipelineStatus'
import { StageChips } from '@/components/sources/StageChips'
import { useNotebooks } from '@/lib/hooks/use-notebooks'
import { activeChipKey } from '@/lib/sources/board'
import { displaySourceTitle } from '@/lib/utils/source-title'
import { useMinWidth } from '@/lib/hooks/use-media-query'

const PAGE_SIZE = 30

/**
 * `/sources` — the pipeline board (WB.3). Every source with its stage; the
 * ones that need the owner (failed, awaiting schema review, stopped: the
 * server's "needs you", the predicates The Tray uses) come first. Chips filter
 * by stage with the server's counts; the notebook is a filter.
 */
export default function SourcesPage() {
  const [selectedIndex, setSelectedIndex] = useState(0)
  // `attention`: needs you first, then the rest by last change.
  const [sortBy, setSortBy] = useState<'attention' | 'created' | 'updated'>('attention')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')
  const [deleteDialog, setDeleteDialog] = useState<{ open: boolean; source: SourceListResponse | null }>({
    open: false,
    source: null
  })
  const router = useRouter()
  const queryClient = useQueryClient()
  // WB.1: `?stage=` (repeatable) filters by processing stage, `?stalled=true`
  // to the sources that stopped (the Werkbak's group, same server predicate).
  const searchParams = useSearchParams()
  const stages = searchParams.getAll('stage')
  const stalled = searchParams.get('stalled') === 'true'
  const filtered = stages.length > 0 || stalled
  const filterLabel = [stalled ? 'stopped sources' : '', stages.join(', ')]
    .filter(Boolean)
    .join(' · ')
  const notebookId = searchParams.get('notebook') || undefined
  const { data: notebooks } = useNotebooks()
  const counts = useQuery({
    queryKey: [...QUERY_KEYS.sources(), 'counts', notebookId ?? null] as const,
    queryFn: () => sourcesApi.counts(notebookId ? { notebook_id: notebookId } : undefined),
  })
  // WB.5: the table from lg up; below it, where its columns squeeze the
  // title, cards. Either one is focusable and takes the arrow keys.
  const isLg = useMinWidth('lg')
  const tableRef = useRef<HTMLElement>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)

  const {
    data,
    isLoading,
    error,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = useInfiniteQuery({
    queryKey: [
      ...QUERY_KEYS.sources(), sortBy, sortOrder, stages.join(','), stalled, notebookId ?? null,
    ] as const,
    queryFn: ({ pageParam = 0 }) =>
      sourcesApi.list({
        limit: PAGE_SIZE,
        offset: pageParam as number,
        sort_by: sortBy,
        sort_order: sortOrder,
        stage: stages.length ? stages : undefined,
        stalled: stalled || undefined,
        notebook_id: notebookId,
      }),
    getNextPageParam: (lastPage, allPages) =>
      lastPage.length === PAGE_SIZE
        ? allPages.reduce((sum, page) => sum + page.length, 0)
        : undefined,
    initialPageParam: 0,
  })

  const sources = useMemo(
    () => data?.pages.flat() ?? [],
    [data]
  )

  useEffect(() => {
    if (sources.length > 0 && tableRef.current) {
      tableRef.current.focus()
    }
  }, [sources])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (sources.length === 0) return
      // The chips, the notebook filter and the row's delete button keep their
      // own keys; the table (focused on load) and the page body navigate rows.
      if (e.target !== document.body && e.target !== tableRef.current) return

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault()
          setSelectedIndex((prev) => {
            const newIndex = Math.min(prev + 1, sources.length - 1)
            setTimeout(() => scrollToSelectedRow(newIndex), 0)
            return newIndex
          })
          break
        case 'ArrowUp':
          e.preventDefault()
          setSelectedIndex((prev) => {
            const newIndex = Math.max(prev - 1, 0)
            setTimeout(() => scrollToSelectedRow(newIndex), 0)
            return newIndex
          })
          break
        case 'Enter':
          e.preventDefault()
          if (sources[selectedIndex]) {
            router.push(`/sources/${sources[selectedIndex].id}`)
          }
          break
        case 'Home':
          e.preventDefault()
          setSelectedIndex(0)
          setTimeout(() => scrollToSelectedRow(0), 0)
          break
        case 'End':
          e.preventDefault()
          const lastIndex = sources.length - 1
          setSelectedIndex(lastIndex)
          setTimeout(() => scrollToSelectedRow(lastIndex), 0)
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [sources, selectedIndex, router])

  const scrollToSelectedRow = (index: number) => {
    const scrollContainer = scrollContainerRef.current
    if (!scrollContainer) return

    const rows = scrollContainer.querySelectorAll('[data-source-row]')
    const selectedRow = rows[index] as HTMLElement
    if (!selectedRow) return

    const containerRect = scrollContainer.getBoundingClientRect()
    const rowRect = selectedRow.getBoundingClientRect()

    if (rowRect.top < containerRect.top) {
      selectedRow.scrollIntoView({ behavior: 'smooth', block: 'start' })
    } else if (rowRect.bottom > containerRect.bottom) {
      selectedRow.scrollIntoView({ behavior: 'smooth', block: 'end' })
    }
  }

  // Infinite scroll trigger
  useEffect(() => {
    const scrollContainer = scrollContainerRef.current
    if (!scrollContainer) return

    let scrollTimeout: NodeJS.Timeout | null = null

    const handleScroll = () => {
      if (scrollTimeout) clearTimeout(scrollTimeout)

      scrollTimeout = setTimeout(() => {
        if (!scrollContainerRef.current) return

        const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current
        const distanceFromBottom = scrollHeight - scrollTop - clientHeight

        if (distanceFromBottom < 200 && hasNextPage && !isFetchingNextPage) {
          fetchNextPage()
        }
      }, 100)
    }

    scrollContainer.addEventListener('scroll', handleScroll)
    handleScroll()

    return () => {
      scrollContainer.removeEventListener('scroll', handleScroll)
      if (scrollTimeout) clearTimeout(scrollTimeout)
    }
  }, [fetchNextPage, hasNextPage, isFetchingNextPage])

  const setNotebook = (value: string) => {
    const params = new URLSearchParams(searchParams.toString())
    if (value === 'all') params.delete('notebook')
    else params.set('notebook', value)
    const query = params.toString()
    router.push(query ? `/sources?${query}` : '/sources')
  }

  const toggleSort = (field: 'created' | 'updated') => {
    if (sortBy === field) {
      setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc')
    } else {
      setSortBy(field)
      setSortOrder('desc')
    }
  }

  const getSourceIcon = (source: SourceListResponse) => {
    if (source.asset?.url) return <LinkIcon className="h-4 w-4" />
    if (source.asset?.file_path) return <Upload className="h-4 w-4" />
    return <AlignLeft className="h-4 w-4" />
  }

  const getSourceType = (source: SourceListResponse) => {
    if (source.asset?.url) return 'Link'
    if (source.asset?.file_path) return 'File'
    return 'Text'
  }

  const handleRowClick = useCallback((index: number, sourceId: string) => {
    setSelectedIndex(index)
    router.push(`/sources/${sourceId}`)
  }, [router])

  const handleDeleteClick = useCallback((e: React.MouseEvent, source: SourceListResponse) => {
    e.stopPropagation()
    setDeleteDialog({ open: true, source })
  }, [])

  const handleDeleteConfirm = async () => {
    if (!deleteDialog.source) return

    try {
      await sourcesApi.delete(deleteDialog.source.id)
      toast.success('Source deleted successfully')
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sources() })
      queryClient.invalidateQueries({ queryKey: ATTENTION_QUERY_KEY })
      setDeleteDialog({ open: false, source: null })
    } catch (err) {
      console.error('Failed to delete source:', err)
      toast.error('Failed to delete source')
    }
  }

  const activeKey = activeChipKey(stages, stalled)
  const notebookName = notebooks?.find((n) => n.id === notebookId)?.name

  const header = (
    <PageHeader
      title="All Sources"
      subtitle="The pipeline board: sources that need you come first. Use arrow keys to navigate and Enter to open."
      actions={
        <Button asChild>
          <Link href="/sources/new" data-testid="sources-add">
            <Plus className="h-4 w-4 mr-2" />
            Add source
          </Link>
        </Button>
      }
    >
      {filtered && (
        <p className="text-sm" data-testid="sources-stage-filter">
          Showing only: <span className="font-medium">{filterLabel}</span>
          {' · '}
          <Link
            href={notebookId ? `/sources?notebook=${encodeURIComponent(notebookId)}` : '/sources'}
            className="underline underline-offset-2"
          >
            Show all
          </Link>
        </p>
      )}
    </PageHeader>
  )

  if (!isLoading && !error && sources.length === 0 && !filtered && !notebookId) {
    return (
      <PageBody>
        {header}
        <EmptyState
          icon={FileText}
          title="No sources yet"
          description="Upload a document, paste a link, or enter text to get started. Sources are the foundation of your knowledge base."
          action={
            <Button onClick={() => router.push('/sources/new')}>
              <Upload className="h-4 w-4 mr-2" />
              Create your first source
            </Button>
          }
        />
      </PageBody>
    )
  }

  // The pieces a row and a card share: the same data and actions either way.
  const typeCell = (source: SourceListResponse) => (
    <div className="flex items-center gap-2">
      {getSourceIcon(source)}
      <Badge variant="secondary" className="text-xs">
        {getSourceType(source)}
      </Badge>
      {source.needs_you && <span className="sr-only">Needs you</span>}
    </div>
  )

  const stageCell = (source: SourceListResponse) => (
    <div className="flex items-center gap-2">
      <PipelineStatus
        variant="card"
        readOnly
        processingStage={source.processing_stage}
        jobStatus={source.status}
        className="w-40 shrink-0"
      />
      {source.stalled && (
        <Badge
          variant="outline"
          className="border-amber-500 text-xs text-amber-800 dark:text-amber-300"
          data-testid="source-stopped-badge"
        >
          Stopped
        </Badge>
      )}
    </div>
  )

  const deleteButton = (source: SourceListResponse) => (
    <Button
      variant="ghost"
      size="icon"
      onClick={(e) => handleDeleteClick(e, source)}
      className="text-destructive hover:text-destructive"
      aria-label={`Delete ${displaySourceTitle(source.title) || 'source'}`}
    >
      <Trash2 className="h-4 w-4" />
    </Button>
  )

  const stageSort = (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => {
        setSortBy('attention')
        setSortOrder('desc')
      }}
      className="h-8 px-2 hover:bg-muted"
      aria-label="Sort by stage: needs you first"
      data-testid="sources-sort-attention"
    >
      Stage
      <ArrowUpDown className={cn(
        "ml-2 h-3 w-3",
        sortBy === 'attention' ? 'opacity-100' : 'opacity-30'
      )} />
    </Button>
  )

  const createdSort = (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => toggleSort('created')}
      className="h-8 px-2 hover:bg-muted"
      data-testid="sources-sort-created"
    >
      Created
      <ArrowUpDown className={cn(
        "ml-2 h-3 w-3",
        sortBy === 'created' ? 'opacity-100' : 'opacity-30'
      )} />
      {sortBy === 'created' && (
        <span className="ml-1 text-xs">
          {sortOrder === 'asc' ? '↑' : '↓'}
        </span>
      )}
    </Button>
  )

  let body: React.ReactNode
  if (isLoading) {
    body = (
      <div className="flex flex-1 items-center justify-center" role="status" aria-live="polite">
        <LoadingSpinner />
      </div>
    )
  } else if (error) {
    body = (
      <div className="flex flex-1 items-center justify-center" role="alert">
        <p className="text-red-500">Failed to load sources</p>
      </div>
    )
  } else if (sources.length === 0) {
    body = (
      <EmptyState
        icon={FileText}
        title="No sources match this filter"
        description={`Showing only: ${[filterLabel, notebookName ?? notebookId].filter(Boolean).join(' · ')}.`}
        action={
          <Button asChild variant="outline">
            <Link href="/sources">Show all sources</Link>
          </Button>
        }
      />
    )
  } else if (!isLg) {
    body = (
      <div ref={scrollContainerRef} className="flex-1 overflow-auto">
        <div className="mb-2 flex flex-wrap items-center gap-2" data-testid="sources-card-sort">
          <span className="text-sm text-muted-foreground">Sort by</span>
          {stageSort}
          {createdSort}
        </div>
        <ul
          ref={tableRef as React.RefObject<HTMLUListElement>}
          tabIndex={0}
          aria-label="Sources"
          className="space-y-2 outline-none"
        >
          {sources.map((source, index) => (
            <li
              key={source.id}
              data-source-row
              data-testid={`source-card-${source.id}`}
              data-needs-you={source.needs_you ? 'true' : 'false'}
              onMouseEnter={() => setSelectedIndex(index)}
              className={cn(
                'rounded-md border border-l-4 p-3',
                source.needs_you ? 'border-l-amber-500' : 'border-l-transparent',
                selectedIndex === index && 'bg-accent'
              )}
            >
              <div className="flex items-start gap-2">
                <div className="min-w-0 flex-1 space-y-2">
                  <Link
                    href={`/sources/${source.id}`}
                    onClick={() => setSelectedIndex(index)}
                    className="flex min-h-10 items-center font-medium break-words hover:underline"
                  >
                    {displaySourceTitle(source.title) || 'Untitled Source'}
                  </Link>
                  {source.asset?.url && (
                    <span className="block truncate text-xs text-muted-foreground">
                      {source.asset.url}
                    </span>
                  )}
                  {stageCell(source)}
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                    {typeCell(source)}
                    <span>Created {formatDistanceToNow(new Date(source.created), { addSuffix: true })}</span>
                    <span>{source.insights_count || 0} insights</span>
                    <span>Embedded: {source.embedded ? 'Yes' : 'No'}</span>
                  </div>
                </div>
                {deleteButton(source)}
              </div>
            </li>
          ))}
        </ul>
        {isFetchingNextPage && (
          <div className="flex h-16 items-center justify-center">
            <LoadingSpinner />
            <span className="ml-2 text-muted-foreground">Loading more sources...</span>
          </div>
        )}
      </div>
    )
  } else {
    body = (
      <div ref={scrollContainerRef} className="flex-1 rounded-md border overflow-auto">
        <table
          ref={tableRef as React.RefObject<HTMLTableElement>}
          tabIndex={0}
          aria-label="Sources"
          className="w-full outline-none table-fixed"
        >
          {/* WB.4: Stage and Created fit their common values on one line at
              1366 px ("Awaiting schema review", "in about 8 hours"). WB.5:
              below xl, Type, Insights and Embedded give their room to the
              title; the cards below lg show them all. */}
          <thead className="sticky top-0 bg-background z-10">
            <tr className="border-b bg-muted/50">
              <th className="hidden w-[110px] h-12 px-4 text-left align-middle font-medium text-muted-foreground xl:table-cell">
                Type
              </th>
              <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">
                Title
              </th>
              <th
                className="w-[260px] h-12 px-4 text-left align-middle font-medium text-muted-foreground"
                aria-sort={sortBy === 'attention' ? 'descending' : undefined}
              >
                {stageSort}
              </th>
              <th
                className="w-[170px] h-12 px-4 text-left align-middle font-medium text-muted-foreground"
                aria-sort={sortBy === 'created' ? (sortOrder === 'asc' ? 'ascending' : 'descending') : undefined}
              >
                {createdSort}
              </th>
              <th className="w-[100px] h-12 px-4 text-center align-middle font-medium text-muted-foreground hidden xl:table-cell">
                Insights
              </th>
              <th className="w-[100px] h-12 px-4 text-center align-middle font-medium text-muted-foreground hidden xl:table-cell">
                Embedded
              </th>
              <th className="w-[72px] h-12 px-4 text-right align-middle font-medium text-muted-foreground">
                <span className="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {sources.map((source, index) => (
              <tr
                key={source.id}
                onClick={() => handleRowClick(index, source.id)}
                onMouseEnter={() => setSelectedIndex(index)}
                data-source-row
                data-testid={`source-row-${source.id}`}
                data-needs-you={source.needs_you ? 'true' : 'false'}
                className={cn(
                  "border-b transition-colors cursor-pointer",
                  selectedIndex === index
                    ? "bg-accent"
                    : "hover:bg-muted/50"
                )}
              >
                {/* The needs-you edge sits on the first cell shown: Type from
                    xl up, Title below. */}
                <td
                  className={cn(
                    "hidden h-12 px-4 border-l-4 xl:table-cell",
                    source.needs_you ? "border-l-amber-500" : "border-l-transparent"
                  )}
                >
                  {typeCell(source)}
                </td>
                <td
                  className={cn(
                    "h-12 px-4 border-l-4 xl:border-l-0",
                    source.needs_you ? "border-l-amber-500" : "border-l-transparent"
                  )}
                >
                  <div className="flex flex-col overflow-hidden">
                    <span className="font-medium truncate" data-testid="source-title">
                      {displaySourceTitle(source.title) || 'Untitled Source'}
                    </span>
                    {source.asset?.url && (
                      <span className="text-xs text-muted-foreground truncate">
                        {source.asset.url}
                      </span>
                    )}
                  </div>
                </td>
                <td className="h-12 px-4" data-testid="source-stage">
                  {stageCell(source)}
                </td>
                <td className="h-12 px-4 text-muted-foreground text-sm" data-testid="source-created">
                  {formatDistanceToNow(new Date(source.created), { addSuffix: true })}
                </td>
                <td className="h-12 px-4 text-center hidden xl:table-cell">
                  <span className="text-sm font-medium">{source.insights_count || 0}</span>
                </td>
                <td className="h-12 px-4 text-center hidden xl:table-cell">
                  <Badge variant={source.embedded ? "default" : "secondary"} className="text-xs">
                    {source.embedded ? "Yes" : "No"}
                  </Badge>
                </td>
                <td className="h-12 px-4 text-right">
                  {deleteButton(source)}
                </td>
              </tr>
            ))}
            {isFetchingNextPage && (
              <tr>
                <td colSpan={7} className="h-16 text-center">
                  <div className="flex items-center justify-center">
                    <LoadingSpinner />
                    <span className="ml-2 text-muted-foreground">Loading more sources...</span>
                  </div>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    )
  }

  return (
    <>
      <PageBody fill>
        {header}

        <div className="flex flex-shrink-0 flex-wrap items-center justify-between gap-3">
          <StageChips
            counts={counts.data}
            countsLoading={counts.isLoading}
            countsError={counts.isError}
            activeKey={activeKey}
            notebookId={notebookId}
          />
          <Select value={notebookId ?? 'all'} onValueChange={setNotebook}>
            <SelectTrigger
              className="w-56"
              aria-label="Filter by notebook"
              data-testid="sources-notebook-filter"
            >
              <SelectValue placeholder="All notebooks" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All notebooks</SelectItem>
              {notebookId && !notebookName && (
                <SelectItem value={notebookId}>{notebookId}</SelectItem>
              )}
              {notebooks?.map((nb) => (
                <SelectItem key={nb.id} value={nb.id}>
                  {nb.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {body}
      </PageBody>

      <ConfirmDialog
        open={deleteDialog.open}
        onOpenChange={(open) => setDeleteDialog({ open, source: deleteDialog.source })}
        title="Delete Source"
        description={`Are you sure you want to delete "${displaySourceTitle(deleteDialog.source?.title) || 'this source'}"? This action cannot be undone.`}
        confirmText="Delete"
        confirmVariant="destructive"
        onConfirm={handleDeleteConfirm}
      />
    </>
  )
}
