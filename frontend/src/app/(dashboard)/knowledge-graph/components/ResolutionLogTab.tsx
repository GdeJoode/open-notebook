'use client'

import { useState } from 'react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { EmptyState } from '@/components/common/EmptyState'
import {
  useResolutionLog,
  useResolutionLogStats,
  useResolutionLogMethods,
  useUpdateResolutionStatus,
  useBulkUpdateResolutionStatus,
} from '@/lib/hooks/use-resolution-log'
import type { ResolutionLogEntry } from '@/lib/api/resolution-log'
import {
  ChevronLeft,
  ChevronRight,
  Check,
  X,
  GitMerge,
  Filter,
} from 'lucide-react'

const STATUS_COLORS: Record<string, string> = {
  pending: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
  accepted: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
  rejected: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
  auto_accepted: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
}

function ConfidenceBar({ confidence }: { confidence: number }) {
  const pct = Math.round(confidence * 100)
  const color = pct >= 80 ? 'bg-green-500' : pct >= 50 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-2 rounded-full bg-muted overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-muted-foreground">{pct}%</span>
    </div>
  )
}

export function ResolutionLogTab() {
  const [page, setPage] = useState(0)
  const [statusFilter, setStatusFilter] = useState<string | undefined>(undefined)
  const [methodFilter, setMethodFilter] = useState<string | undefined>(undefined)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const pageSize = 25

  const { data: stats } = useResolutionLogStats()
  const { data: methods } = useResolutionLogMethods()
  const { data, isLoading } = useResolutionLog({
    limit: pageSize,
    offset: page * pageSize,
    status: statusFilter,
    match_method: methodFilter,
  })

  const updateStatus = useUpdateResolutionStatus()
  const bulkUpdate = useBulkUpdateResolutionStatus()

  const items = data?.items ?? []
  const total = data?.total ?? 0
  const totalPages = Math.ceil(total / pageSize)

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const toggleSelectAll = () => {
    if (selectedIds.size === items.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(items.map((i) => i.id)))
    }
  }

  const handleBulkAction = (status: 'accepted' | 'rejected') => {
    bulkUpdate.mutate(
      { ids: Array.from(selectedIds), status },
      { onSuccess: () => setSelectedIds(new Set()) }
    )
  }

  return (
    <div className="space-y-4">
      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <Card className="p-3 text-center">
            <div className="text-2xl font-bold">{stats.total}</div>
            <div className="text-xs text-muted-foreground">Total</div>
          </Card>
          <Card className="p-3 text-center">
            <div className="text-2xl font-bold text-yellow-600">{stats.pending}</div>
            <div className="text-xs text-muted-foreground">Pending</div>
          </Card>
          <Card className="p-3 text-center">
            <div className="text-2xl font-bold text-green-600">{stats.accepted}</div>
            <div className="text-xs text-muted-foreground">Accepted</div>
          </Card>
          <Card className="p-3 text-center">
            <div className="text-2xl font-bold text-red-600">{stats.rejected}</div>
            <div className="text-xs text-muted-foreground">Rejected</div>
          </Card>
          <Card className="p-3 text-center">
            <div className="text-2xl font-bold text-blue-600">{stats.auto_accepted}</div>
            <div className="text-xs text-muted-foreground">Auto-accepted</div>
          </Card>
        </div>
      )}

      {/* Filters and bulk actions */}
      <div className="flex items-center gap-3 flex-wrap">
        <Filter className="h-4 w-4 text-muted-foreground" />
        <Select
          value={statusFilter ?? 'all'}
          onValueChange={(v) => { setStatusFilter(v === 'all' ? undefined : v); setPage(0) }}
        >
          <SelectTrigger className="w-40">
            <SelectValue placeholder="All statuses" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All statuses</SelectItem>
            <SelectItem value="pending">Pending</SelectItem>
            <SelectItem value="accepted">Accepted</SelectItem>
            <SelectItem value="rejected">Rejected</SelectItem>
            <SelectItem value="auto_accepted">Auto-accepted</SelectItem>
          </SelectContent>
        </Select>

        <Select
          value={methodFilter ?? 'all'}
          onValueChange={(v) => { setMethodFilter(v === 'all' ? undefined : v); setPage(0) }}
        >
          <SelectTrigger className="w-40">
            <SelectValue placeholder="All methods" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All methods</SelectItem>
            {methods?.map((m) => (
              <SelectItem key={m} value={m}>{m}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        {selectedIds.size > 0 && (
          <div className="flex items-center gap-2 ml-auto">
            <span className="text-sm text-muted-foreground">{selectedIds.size} selected</span>
            <Button
              size="sm"
              variant="outline"
              className="text-green-600"
              onClick={() => handleBulkAction('accepted')}
              disabled={bulkUpdate.isPending}
            >
              <Check className="h-3 w-3 mr-1" />
              Accept
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="text-red-600"
              onClick={() => handleBulkAction('rejected')}
              disabled={bulkUpdate.isPending}
            >
              <X className="h-3 w-3 mr-1" />
              Reject
            </Button>
          </div>
        )}
      </div>

      {/* Table */}
      {isLoading ? (
        <div className="flex h-64 items-center justify-center">
          <LoadingSpinner />
        </div>
      ) : items.length === 0 ? (
        <EmptyState
          icon={GitMerge}
          title="No resolution log entries"
          description="Run entity extraction with LLM matching to generate match candidates for review."
        />
      ) : (
        <>
          <div className="border rounded-md overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-background">
                <tr className="border-b bg-muted/50">
                  <th className="p-3 w-8">
                    <Checkbox
                      checked={selectedIds.size === items.length && items.length > 0}
                      onCheckedChange={toggleSelectAll}
                    />
                  </th>
                  <th className="text-left p-3 font-medium">Entity A</th>
                  <th className="text-left p-3 font-medium">Entity B</th>
                  <th className="text-left p-3 font-medium">Match</th>
                  <th className="text-left p-3 font-medium">Confidence</th>
                  <th className="text-left p-3 font-medium">Method</th>
                  <th className="text-left p-3 font-medium">Status</th>
                  <th className="text-left p-3 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {items.map((entry: ResolutionLogEntry) => (
                  <>
                    <tr
                      key={entry.id}
                      className="border-b cursor-pointer hover:bg-muted/50 transition-colors"
                      onClick={() => setExpandedId(expandedId === entry.id ? null : entry.id)}
                    >
                      <td className="p-3" onClick={(e) => e.stopPropagation()}>
                        <Checkbox
                          checked={selectedIds.has(entry.id)}
                          onCheckedChange={() => toggleSelect(entry.id)}
                        />
                      </td>
                      <td className="p-3">
                        <div className="font-medium">{entry.entity_a_text}</div>
                        <Badge variant="outline" className="text-xs mt-1">{entry.entity_a_label}</Badge>
                      </td>
                      <td className="p-3">
                        <div className="font-medium">{entry.entity_b_text}</div>
                        <Badge variant="outline" className="text-xs mt-1">{entry.entity_b_label}</Badge>
                      </td>
                      <td className="p-3">
                        {entry.match ? (
                          <Badge className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">Yes</Badge>
                        ) : (
                          <Badge variant="secondary">No</Badge>
                        )}
                      </td>
                      <td className="p-3">
                        <ConfidenceBar confidence={entry.confidence} />
                      </td>
                      <td className="p-3">
                        <Badge variant="secondary" className="text-xs">{entry.match_method}</Badge>
                      </td>
                      <td className="p-3">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${STATUS_COLORS[entry.status] ?? ''}`}>
                          {entry.status}
                        </span>
                      </td>
                      <td className="p-3" onClick={(e) => e.stopPropagation()}>
                        {(entry.status === 'pending' || entry.status === 'auto_accepted') && (
                          <div className="flex gap-1">
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 w-7 p-0 text-green-600 hover:text-green-700"
                              onClick={() => updateStatus.mutate({ id: entry.id, status: 'accepted' })}
                              disabled={updateStatus.isPending}
                            >
                              <Check className="h-3.5 w-3.5" />
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 w-7 p-0 text-red-600 hover:text-red-700"
                              onClick={() => updateStatus.mutate({ id: entry.id, status: 'rejected' })}
                              disabled={updateStatus.isPending}
                            >
                              <X className="h-3.5 w-3.5" />
                            </Button>
                          </div>
                        )}
                      </td>
                    </tr>
                    {expandedId === entry.id && (
                      <tr key={`${entry.id}-detail`} className="bg-muted/30">
                        <td colSpan={8} className="p-4">
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            {entry.match_reasoning && (
                              <div className="col-span-2">
                                <span className="font-medium">Reasoning: </span>
                                <span className="text-muted-foreground">{entry.match_reasoning}</span>
                              </div>
                            )}
                            {entry.matched_by_model && (
                              <div>
                                <span className="font-medium">Model: </span>
                                <span className="text-muted-foreground">{entry.matched_by_model}</span>
                              </div>
                            )}
                            {entry.iterations > 1 && (
                              <div>
                                <span className="font-medium">Iterations: </span>
                                <span className="text-muted-foreground">{entry.iterations}</span>
                              </div>
                            )}
                            {entry.source_section_a && (
                              <div>
                                <span className="font-medium">Section A: </span>
                                <span className="text-muted-foreground">{entry.source_section_a}</span>
                              </div>
                            )}
                            {entry.source_section_b && (
                              <div>
                                <span className="font-medium">Section B: </span>
                                <span className="text-muted-foreground">{entry.source_section_b}</span>
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">
                Page {page + 1} of {totalPages} ({total} total)
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
    </div>
  )
}
