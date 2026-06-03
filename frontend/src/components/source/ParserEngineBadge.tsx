'use client'

/**
 * Provenance badge for the source-detail page.
 *
 * Four visual states, driven by Source.metadata persisted in A.1c:
 *  1. Legacy / pure-docling success → renders nothing (keeps the badge
 *     row uncluttered for the common case).
 *  2. MinerU (manual selection) → blue badge with WandSparkles.
 *  3. MinerU (auto-fallback) → amber badge with AlertCircle. Tooltip
 *     shows the per-signal confidence breakdown.
 *  4. Extraction failed (metadata absent AND source.status === 'error')
 *     → red badge with XCircle.
 *
 * The "extraction failed" detection is intentionally conservative: a
 * source with `status === 'error'` and no metadata block almost
 * certainly hit a hard parser failure (before the metadata write
 * happens). Sources that genuinely lack a parser_engine_used (e.g.
 * non-document types like notes-sourced text) just render nothing.
 */

import { AlertCircle, WandSparkles, XCircle } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { SourceMetadata } from '@/lib/types/api'

interface ParserEngineBadgeProps {
  metadata?: SourceMetadata
  sourceStatus?: string
}

function formatSignals(signals?: Record<string, number>): string {
  if (!signals || Object.keys(signals).length === 0) return ''
  return Object.entries(signals)
    .map(([key, value]) => `${key}: ${value.toFixed(2)}`)
    .join(' · ')
}

export function ParserEngineBadge({
  metadata,
  sourceStatus,
}: ParserEngineBadgeProps) {
  // Legacy "extraction failed" case: no metadata AND source is in error
  // state. We render the red badge.
  if (!metadata) {
    if (sourceStatus === 'error') {
      return (
        <Badge
          variant="destructive"
          className="gap-1 bg-red-700 hover:bg-red-700"
          data-testid="parser-engine-badge"
          data-state="extraction-failed"
        >
          <XCircle className="h-3 w-3" aria-hidden="true" />
          extraction failed
        </Badge>
      )
    }
    return null
  }

  const engine = metadata.parser_engine_used
  const fallback = metadata.extraction_fallback_triggered === true
  const confidence = metadata.extraction_confidence

  // Default docling success: render nothing — keep the badge bar clean
  // for the overwhelming majority case.
  if (engine === 'docling' && !fallback) {
    return null
  }

  // Auto-fallback: docling produced low confidence, the orchestrator
  // re-parsed with MinerU. Amber badge + tooltip with signal breakdown.
  if (fallback) {
    const confLabel = typeof confidence === 'number' ? confidence.toFixed(2) : '—'
    const signalText = formatSignals(metadata.extraction_confidence_signals)
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            className="gap-1 bg-amber-600 hover:bg-amber-600 text-white cursor-help"
            data-testid="parser-engine-badge"
            data-state="mineru-auto-fallback"
          >
            <AlertCircle className="h-3 w-3" aria-hidden="true" />
            MinerU (auto-fallback, conf {confLabel})
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          {signalText
            ? `Confidence signals — ${signalText}`
            : 'Docling confidence below threshold; MinerU re-parsed the document.'}
        </TooltipContent>
      </Tooltip>
    )
  }

  // MinerU (manual selection or parser_engine=mineru): blue badge.
  if (engine === 'mineru') {
    return (
      <Badge
        className="gap-1 bg-blue-700 hover:bg-blue-700 text-white"
        data-testid="parser-engine-badge"
        data-state="mineru-manual"
      >
        <WandSparkles className="h-3 w-3" aria-hidden="true" />
        MinerU
      </Badge>
    )
  }

  // Metadata exists but engine isn't recognised — fail closed and
  // render nothing rather than mislabel.
  return null
}
