import { Badge } from '@/components/ui/badge'
import { entityTypeLabels } from '@/lib/utils/entity-type'

interface EntityTypeLabelProps {
  /** `entity_type`: the canonical type. */
  canonical: string | null | undefined
  /** `primary_type` (or a triage row's stored label): shown second. */
  display?: string | null
}

/** The canonical type as a badge, the display label after it in muted text. */
export function EntityTypeLabel({ canonical, display }: EntityTypeLabelProps) {
  const t = entityTypeLabels(canonical, display)
  return (
    <span className="inline-flex flex-wrap items-center gap-1.5" data-testid="entity-type">
      <Badge variant="secondary" data-testid="entity-type-canonical">
        {t.canonical || '—'}
      </Badge>
      {t.label && (
        <span className="text-xs text-muted-foreground" data-testid="entity-type-label">
          {t.label}
        </span>
      )}
    </span>
  )
}
