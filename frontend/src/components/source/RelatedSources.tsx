'use client'

import Link from 'next/link'
import { AlertCircle, Network, Sparkles, Compass } from 'lucide-react'

import { useRelatedSources } from '@/lib/hooks/use-sources'
import type { RelatedSourceHybrid, KGSharedEntity } from '@/lib/types/api'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

interface RelatedSourcesProps {
  sourceId: string
}

/**
 * "Related sources" panel on the source detail page (Track R.5).
 *
 * Driven by the hybrid retriever (`GET /sources/{id}/related-hybrid`, R.3),
 * which fuses the dense embedding signal and the KG-proximity signal. Each row
 * links to the related source and explains *why* it matched: the shared KG
 * entities (names) and/or that it's a dense text-similarity match — derived
 * from the per-signal provenance (`dense`/`kg`) plus `kg_entities`.
 *
 * Renders loading / error / empty states explicitly. An empty list is the
 * normal response for a source with no aggregate embedding and no shared
 * entities — surfaced as a friendly message, never an error.
 */
export function RelatedSources({ sourceId }: RelatedSourcesProps) {
  const { data, isLoading, isError, error } = useRelatedSources(sourceId, {
    k: 8,
    preset: 'kg-heavy',
  })

  return (
    <Card role="region" aria-labelledby="related-sources-heading">
      <CardHeader>
        <CardTitle
          id="related-sources-heading"
          className="flex items-center gap-2"
        >
          <Compass className="h-5 w-5" aria-hidden="true" />
          Related sources
        </CardTitle>
        <CardDescription>
          Other sources ranked by shared knowledge-graph entities and text
          similarity.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <RelatedSourcesSkeleton />
        ) : isError ? (
          <Alert variant="destructive" role="alert">
            <AlertCircle className="h-4 w-4" aria-hidden="true" />
            <AlertTitle>Failed to load related sources</AlertTitle>
            <AlertDescription>
              {error instanceof Error
                ? error.message
                : 'An unexpected error occurred while finding related sources.'}
            </AlertDescription>
          </Alert>
        ) : !data || data.length === 0 ? (
          <RelatedSourcesEmpty />
        ) : (
          <ul
            className="space-y-3"
            aria-label="Related sources, most relevant first"
          >
            {data.map((item, index) => (
              <RelatedSourceRow key={item.id} item={item} rank={index + 1} />
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  )
}

function RelatedSourcesSkeleton() {
  return (
    <div
      className="space-y-3"
      role="status"
      aria-live="polite"
      aria-label="Loading related sources"
    >
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="rounded-lg border bg-muted/30 p-4 animate-pulse"
          aria-hidden="true"
        >
          <div className="h-4 w-1/2 rounded bg-muted" />
          <div className="mt-3 h-3 w-3/4 rounded bg-muted" />
        </div>
      ))}
      <span className="sr-only">Loading related sources…</span>
    </div>
  )
}

function RelatedSourcesEmpty() {
  return (
    <div className="text-center py-8 text-muted-foreground">
      <Compass className="h-12 w-12 mx-auto mb-3 opacity-50" aria-hidden="true" />
      <p className="text-sm">No related sources yet</p>
      <p className="text-xs mt-1">
        Related sources appear once this source has been embedded and its
        knowledge-graph entities extracted, and other sources share content or
        entities with it.
      </p>
    </div>
  )
}

/**
 * Build the human-readable "why matched" explanation for one result from its
 * provenance. KG-prominent by default, so shared entities lead; the
 * text-similarity note is appended (or stands alone) when the dense signal
 * contributed. Returned as a plain string so it is fully screen-reader
 * readable (not conveyed by colour or icon alone).
 */
function buildWhyMatched(item: RelatedSourceHybrid): string {
  const parts: string[] = []

  if (item.kg.present && item.kg_entities.length > 0) {
    const names = item.kg_entities.slice(0, 4).map((e) => e.name)
    const extra = item.kg_entities.length - names.length
    const list = names.join(', ') + (extra > 0 ? `, +${extra} more` : '')
    parts.push(`Shares ${item.kg_entities.length === 1 ? 'entity' : 'entities'}: ${list}`)
  }

  if (item.dense.present) {
    parts.push('Similar text content')
  }

  if (parts.length === 0) {
    // Defensive: a fused result always has at least one present signal, but
    // keep a sensible fallback rather than rendering an empty explanation.
    return 'Related by hybrid retrieval'
  }

  return parts.join(' · ')
}

function RelatedSourceRow({
  item,
  rank,
}: {
  item: RelatedSourceHybrid
  rank: number
}) {
  const title = item.title?.trim() || 'Untitled source'
  const why = buildWhyMatched(item)
  const sourceHref = `/sources/${encodeURIComponent(item.id)}`
  const sharedEntities: KGSharedEntity[] = item.kg.present ? item.kg_entities : []

  return (
    <li
      data-testid={`related-source-row-${item.id}`}
      className="rounded-lg border bg-background p-4 focus-within:ring-2 focus-within:ring-ring"
    >
      <div className="flex items-start justify-between gap-3">
        <Link
          href={sourceHref}
          className="font-medium text-blue-600 hover:underline focus:underline focus:outline-none"
          aria-label={`Open related source ${title}. Why matched: ${why}`}
          data-testid="related-source-title"
        >
          {title}
        </Link>
        <Badge
          variant="secondary"
          className="shrink-0 tabular-nums"
          aria-label={`Relevance rank ${rank}, fused score ${item.fused_score.toFixed(3)}`}
        >
          {item.fused_score.toFixed(3)}
        </Badge>
      </div>

      {/* "Why matched" provenance — plain text so screen readers convey it,
          not colour/icon alone. Icons are aria-hidden decoration. */}
      <p
        className="mt-2 flex flex-wrap items-center gap-x-1.5 gap-y-1 text-sm text-muted-foreground"
        data-testid="related-source-why"
      >
        <span className="sr-only">Why matched: </span>
        {sharedEntities.length > 0 && (
          <span className="inline-flex items-center gap-1">
            <Network className="h-3.5 w-3.5" aria-hidden="true" />
            <span>
              Shares {sharedEntities.length === 1 ? 'entity' : 'entities'}:{' '}
            </span>
          </span>
        )}
        {sharedEntities.slice(0, 4).map((entity) => (
          <Badge
            key={entity.entity_id}
            variant="outline"
            className="font-normal"
          >
            {entity.name}
          </Badge>
        ))}
        {sharedEntities.length > 4 && (
          <span className="text-xs">+{sharedEntities.length - 4} more</span>
        )}
        {item.dense.present && (
          <span className="inline-flex items-center gap-1">
            <Sparkles className="h-3.5 w-3.5" aria-hidden="true" />
            <span>{sharedEntities.length > 0 ? 'and similar text' : 'Similar text content'}</span>
          </span>
        )}
        {sharedEntities.length === 0 && !item.dense.present && (
          <span>Related by hybrid retrieval</span>
        )}
      </p>
    </li>
  )
}
