'use client'

import { Badge } from '@/components/ui/badge'
import { ExternalLink } from 'lucide-react'

import { providerLabel, toHref } from '@/lib/utils/external-ids'

interface ExternalIdBadgesProps {
  /** The entity's `external_ids` (stable URIs from K.4 vocabulary reconciliation). */
  externalIds?: string[]
  className?: string
}

/**
 * Render `entity.external_ids` as linked provider badges (TOOI / DOI / …).
 *
 * Renders nothing when there are no ids — no empty-badge artifact (AC6).
 */
export function ExternalIdBadges({
  externalIds,
  className,
}: ExternalIdBadgesProps) {
  const ids = (externalIds ?? []).filter((id) => id && id.trim())
  if (ids.length === 0) return null

  return (
    <div className={className} data-testid="external-id-badges">
      <div className="flex flex-wrap gap-1.5">
        {ids.map((id) => {
          const label = providerLabel(id)
          const href = toHref(id)
          return (
            <a
              key={id}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              title={id}
              aria-label={`${label}: ${id}`}
              className="inline-flex"
            >
              <Badge
                variant="outline"
                className="gap-1 text-[10px] hover:bg-muted cursor-pointer"
              >
                {label}
                <ExternalLink className="h-2.5 w-2.5" aria-hidden="true" />
              </Badge>
            </a>
          )
        })}
      </div>
    </div>
  )
}
