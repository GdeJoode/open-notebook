'use client'

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { useRoutingSummary } from '@/lib/hooks/use-model-routing'
import { Cloud, HardDrive, Repeat } from 'lucide-react'

interface RoutingSummaryCardProps {
  /** Scope the summary to one notebook; omit for the global summary. */
  notebookId?: string
}

/**
 * Track J.6: a "recent routing" summary surfacing how routed LLM work split
 * across cloud vs local plus how often a provider was failed over. Reads the
 * `routing.served` telemetry via the summary endpoint. Renders default,
 * loading, error, and empty (no events yet) states.
 */
export function RoutingSummaryCard({ notebookId }: RoutingSummaryCardProps) {
  const { data, isLoading, isError } = useRoutingSummary(notebookId)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Routing</CardTitle>
        <CardDescription>
          {notebookId
            ? 'How this notebook’s LLM work split across cloud and local recently.'
            : 'How recent LLM work split across cloud and local across all notebooks.'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-16 w-full" aria-busy="true" />
        ) : isError ? (
          <p className="text-sm text-muted-foreground">
            Routing summary unavailable.
          </p>
        ) : !data || data.total_events === 0 ? (
          <p className="text-sm text-muted-foreground">
            No routed LLM activity recorded yet.
          </p>
        ) : (
          <div className="grid grid-cols-3 gap-4">
            <SummaryStat
              icon={<Cloud className="h-4 w-4" aria-hidden="true" />}
              label="Cloud"
              value={data.cloud_count}
            />
            <SummaryStat
              icon={<HardDrive className="h-4 w-4" aria-hidden="true" />}
              label="Local"
              value={data.local_count}
            />
            <SummaryStat
              icon={<Repeat className="h-4 w-4" aria-hidden="true" />}
              label="Fallbacks"
              value={data.fallback_count}
            />
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function SummaryStat({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode
  label: string
  value: number
}) {
  return (
    <div
      className="flex flex-col items-center gap-1 rounded-md border p-3"
      aria-label={`${label}: ${value}`}
    >
      <div className="flex items-center gap-1 text-muted-foreground text-xs">
        {icon}
        <span>{label}</span>
      </div>
      <span className="text-2xl font-semibold tabular-nums">{value}</span>
    </div>
  )
}
