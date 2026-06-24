'use client'

import { AlertTriangle } from 'lucide-react'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import type { BackboneWarning } from '@/lib/api/triage'
import { totalWeakEdges } from '@/lib/utils/triage'

export interface BackboneWarningsProps {
  warnings: BackboneWarning[]
  /** Render nothing instead of an empty state when there are no warnings. */
  hideWhenEmpty?: boolean
  className?: string
}

/**
 * Backbone warnings (Q.5 AC5): for degree>=3 entities resting on weak
 * (`RELATED`) edges, list the weak edges that prop up the entity's connectivity
 * so the operator can see which "backbone" actors stand on shaky links. Renders
 * a summary headline + one row per affected entity. Empty state is explicit
 * (unless suppressed) so the panel reads as "checked, nothing to flag".
 */
export function BackboneWarnings({
  warnings,
  hideWhenEmpty = false,
  className,
}: BackboneWarningsProps) {
  if (warnings.length === 0) {
    if (hideWhenEmpty) return null
    return (
      <p
        className={`text-xs text-muted-foreground ${className ?? ''}`}
        data-testid="backbone-warnings-empty"
      >
        No backbone warnings — no well-connected entities rest on weak edges.
      </p>
    )
  }

  const weakTotal = totalWeakEdges(warnings)

  return (
    <Alert
      variant="default"
      className={className}
      data-testid="backbone-warnings"
      aria-label={`${warnings.length} backbone warnings`}
    >
      <AlertTriangle className="h-4 w-4" aria-hidden="true" />
      <AlertTitle>Backbone warnings</AlertTitle>
      <AlertDescription>
        <span className="mb-2 block">
          {weakTotal} weak edge{weakTotal === 1 ? '' : 's'} prop up{' '}
          {warnings.length} well-connected{' '}
          {warnings.length === 1 ? 'entity' : 'entities'}.
        </span>
        <ul className="space-y-2">
          {warnings.map((w) => (
            <li
              key={w.entity_id}
              data-testid={`backbone-warning-${w.entity_id}`}
              className="text-xs"
            >
              <div className="flex items-center gap-2">
                <span className="font-medium truncate">{w.entity_id}</span>
                <Badge variant="outline" className="shrink-0">
                  degree {w.structural_degree}
                </Badge>
              </div>
              <div className="mt-1 flex flex-wrap gap-1">
                {w.weak_edges.map((e, i) => (
                  <Badge
                    key={`${e.other}-${e.relation_type}-${i}`}
                    variant="secondary"
                    className="font-normal"
                  >
                    {e.relation_type} → {e.other} ({e.doc_count} doc
                    {e.doc_count === 1 ? '' : 's'})
                  </Badge>
                ))}
              </div>
            </li>
          ))}
        </ul>
      </AlertDescription>
    </Alert>
  )
}
